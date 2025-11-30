"""Calculate betting edge by comparing model predictions to market odds."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

from ..database import get_session
from ..database.models import Prop, Player, Game, Prediction, Edge

logger = logging.getLogger(__name__)


class EdgeCalculator:
    """Calculate and identify betting edges."""

    def __init__(self):
        self.min_edge = 0.03  # Minimum 3% edge to consider
        self.min_confidence = 0.65  # Minimum model confidence

    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    def american_to_probability(self, american_odds: int) -> float:
        """Convert American odds to implied probability."""
        decimal = self.american_to_decimal(american_odds)
        return 1 / decimal

    def remove_vig(self, over_prob: float, under_prob: float) -> Tuple[float, float]:
        """
        Remove vig (bookmaker margin) to get true probabilities.

        Uses multiplicative method to de-vig odds.
        """
        total_prob = over_prob + under_prob
        vig_multiplier = 1 / total_prob

        fair_over = over_prob * vig_multiplier
        fair_under = under_prob * vig_multiplier

        return fair_over, fair_under

    def calculate_edge(
        self,
        model_prediction: float,
        model_confidence: float,
        market_line: float,
        over_odds: int,
        under_odds: int
    ) -> Dict:
        """
        Calculate betting edge for over/under.

        Returns:
            Dict with edge analysis for both over and under
        """
        # Convert odds to probabilities
        over_prob_raw = self.american_to_probability(over_odds)
        under_prob_raw = self.american_to_probability(under_odds)

        # Remove vig to get fair probabilities
        fair_over_prob, fair_under_prob = self.remove_vig(over_prob_raw, under_prob_raw)

        # Calculate model's probability of going over/under
        # Simple approach: use prediction vs line with confidence as certainty
        # More sophisticated: use prediction distribution

        # For now, simple distance-based probability
        yards_above_line = model_prediction - market_line

        # Normalize by typical variance (e.g., 20 yards for receiving)
        z_score = yards_above_line / 20.0

        # Convert to probability using normal CDF
        from scipy.stats import norm
        model_over_prob = norm.cdf(z_score)

        # Adjust by model confidence (if low confidence, regress to 0.5)
        adjusted_over_prob = (model_over_prob * model_confidence) + (0.5 * (1 - model_confidence))
        model_under_prob = 1 - adjusted_over_prob

        # Calculate edge (model probability - fair market probability)
        edge_over = adjusted_over_prob - fair_over_prob
        edge_under = model_under_prob - fair_under_prob

        # Calculate expected value (EV)
        # EV = (win_prob × profit) - (loss_prob × stake)
        decimal_over = self.american_to_decimal(over_odds)
        decimal_under = self.american_to_decimal(under_odds)

        ev_over = (adjusted_over_prob * (decimal_over - 1)) - (model_under_prob * 1)
        ev_under = (model_under_prob * (decimal_under - 1)) - (adjusted_over_prob * 1)

        return {
            'prediction': model_prediction,
            'line': market_line,
            'yards_above_line': yards_above_line,
            'model_confidence': model_confidence,
            'over': {
                'market_probability': fair_over_prob,
                'model_probability': adjusted_over_prob,
                'edge': edge_over,
                'edge_pct': edge_over * 100,
                'ev': ev_over,
                'ev_pct': ev_over * 100,
                'odds': over_odds,
                'should_bet': edge_over > self.min_edge and model_confidence > self.min_confidence,
            },
            'under': {
                'market_probability': fair_under_prob,
                'model_probability': model_under_prob,
                'edge': edge_under,
                'edge_pct': edge_under * 100,
                'ev': ev_under,
                'ev_pct': ev_under * 100,
                'odds': under_odds,
                'should_bet': edge_under > self.min_edge and model_confidence > self.min_confidence,
            },
        }

    def find_edges_for_week(
        self,
        week: int,
        season: int,
        min_edge: Optional[float] = None
    ) -> List[Dict]:
        """
        Find all betting edges for a specific week.

        Compares model predictions to market odds and identifies +EV bets.

        Returns:
            List of edges sorted by expected value
        """
        if min_edge is not None:
            self.min_edge = min_edge

        edges = []

        with get_session() as session:
            # Get all games for this week
            games = session.query(Game).filter_by(
                season=season,
                week=week,
                season_type='REG'
            ).all()

            for game in games:
                # Get props for this game
                props = session.query(Prop).filter_by(
                    game_id=game.id,
                    is_active=True
                ).filter(
                    Prop.timestamp >= datetime.now().replace(hour=0, minute=0, second=0)
                ).all()

                for prop in props:
                    # Get model prediction - match by player and market (predictions may have different game_id)
                    prediction = session.query(Prediction).filter_by(
                        player_id=prop.player_id,
                        market=prop.market
                    ).order_by(Prediction.created_at.desc()).first()

                    if not prediction:
                        continue

                    # Calculate edge
                    edge_analysis = self.calculate_edge(
                        model_prediction=prediction.prediction,
                        model_confidence=prediction.confidence,
                        market_line=prop.line,
                        over_odds=prop.over_odds,
                        under_odds=prop.under_odds
                    )

                    # Check if either side has an edge
                    has_edge = (edge_analysis['over']['should_bet'] or
                               edge_analysis['under']['should_bet'])

                    if has_edge:
                        player = session.query(Player).get(prop.player_id)

                        edge_info = {
                            'game': f"{game.away_team.abbreviation} @ {game.home_team.abbreviation}",
                            'game_date': game.game_date,
                            'player': player.name,
                            'position': player.position,
                            'market': prop.market,
                            **edge_analysis,
                        }

                        edges.append(edge_info)

        # Sort by absolute EV (best bets first)
        edges.sort(
            key=lambda x: max(abs(x['over']['ev']), abs(x['under']['ev'])),
            reverse=True
        )

        logger.info(f"Found {len(edges)} edges for week {week}")

        return edges

    def store_edges_in_database(self, edges: List[Dict]):
        """Store identified edges in the database for tracking."""
        with get_session() as session:
            stored_count = 0
            updated_count = 0

            for edge_data in edges:
                # Determine which side to bet
                if edge_data['over']['should_bet']:
                    bet_side = 'over'
                    edge_value = edge_data['over']['edge']
                    ev = edge_data['over']['ev']
                    odds = edge_data['over']['odds']
                    fair_prob = edge_data['over']['market_probability']
                elif edge_data['under']['should_bet']:
                    bet_side = 'under'
                    edge_value = edge_data['under']['edge']
                    ev = edge_data['under']['ev']
                    odds = edge_data['under']['odds']
                    fair_prob = edge_data['under']['market_probability']
                else:
                    continue

                # Find the prop
                player = session.query(Player).filter_by(
                    name=edge_data['player']
                ).first()

                if not player:
                    continue

                prop = session.query(Prop).filter_by(
                    player_id=player.id,
                    market=edge_data['market']
                ).order_by(Prop.timestamp.desc()).first()

                if not prop:
                    continue

                # Calculate Kelly fraction (simplified: edge / (odds - 1))
                decimal_odds = self.american_to_decimal(odds)
                kelly = edge_value / (decimal_odds - 1) if decimal_odds > 1 else 0

                # Check if edge already exists
                existing = session.query(Edge).filter_by(
                    game_id=prop.game_id,
                    player_id=prop.player_id,
                    book_id=prop.book_id,
                    market=prop.market,
                    side=bet_side,
                ).first()

                if existing:
                    # Update existing edge
                    existing.offered_line = edge_data['line']
                    existing.offered_odds = odds
                    existing.fair_line = edge_data['prediction']
                    existing.fair_probability = fair_prob
                    existing.expected_value = ev
                    existing.kelly_fraction = kelly
                    existing.confidence = edge_data['model_confidence']
                    existing.reasoning = f"Model predicts {edge_data['prediction']:.1f} vs line {edge_data['line']}"
                    existing.created_at = datetime.now()
                    updated_count += 1
                else:
                    # Store new edge
                    edge = Edge(
                        game_id=prop.game_id,
                        player_id=prop.player_id,
                        book_id=prop.book_id,
                        market=prop.market,
                        side=bet_side,
                        offered_line=edge_data['line'],
                        offered_odds=odds,
                        fair_line=edge_data['prediction'],
                        fair_probability=fair_prob,
                        expected_value=ev,
                        kelly_fraction=kelly,
                        confidence=edge_data['model_confidence'],
                        reasoning=f"Model predicts {edge_data['prediction']:.1f} vs line {edge_data['line']}",
                    )
                    session.add(edge)
                    stored_count += 1

            session.commit()
            logger.info(f"Stored {stored_count} new edges, updated {updated_count} existing")

    def format_edge_report(self, edges: List[Dict]) -> str:
        """Format edges into a readable report."""
        if not edges:
            return "No edges found."

        report = ["=" * 80]
        report.append(f"BETTING EDGES REPORT - {len(edges)} opportunities found")
        report.append("=" * 80)
        report.append("")

        for i, edge in enumerate(edges, 1):
            # Determine best side
            if edge['over']['ev'] > edge['under']['ev']:
                best_side = 'OVER'
                ev = edge['over']['ev_pct']
                edge_pct = edge['over']['edge_pct']
                odds = edge['over']['odds']
            else:
                best_side = 'UNDER'
                ev = edge['under']['ev_pct']
                edge_pct = edge['under']['edge_pct']
                odds = edge['under']['odds']

            report.append(f"{i}. {edge['player']} - {edge['market'].replace('_', ' ').title()}")
            report.append(f"   Game: {edge['game']}")
            report.append(f"   Line: {edge['line']}")
            report.append(f"   Model Prediction: {edge['prediction']:.1f}")
            report.append(f"   Confidence: {edge['model_confidence']:.1%}")
            report.append(f"   ")
            report.append(f"   >>> BET {best_side} {odds:+d}")
            report.append(f"   >>> Edge: {edge_pct:+.1f}%")
            report.append(f"   >>> Expected Value: {ev:+.1f}%")
            report.append("")

        return "\n".join(report)
