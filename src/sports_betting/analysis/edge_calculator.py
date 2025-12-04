"""Calculate betting edge by comparing model predictions to market odds."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..database import get_session
from ..database.models import Prop, Player, Game, Prediction, Edge
from ..data.weather import get_weather_service, GameWeather
from ..data.injuries import get_injury_service, InjuryStatus

logger = logging.getLogger(__name__)


class EdgeCalculator:
    """Calculate and identify betting edges."""

    # Minimum line thresholds by market to filter out low-volume players
    # Week 13 showed betting on players with very low lines (0.5, 1.5 yards) is risky
    MIN_LINE_THRESHOLDS = {
        'player_reception_yds': 15.0,  # Skip props below 15 receiving yards
        'player_rush_yds': 10.0,       # Skip props below 10 rushing yards
        'player_pass_yds': 150.0,      # Skip props below 150 passing yards
        'player_receptions': 2.0,      # Skip props below 2 receptions
    }

    # ==========================================================================
    # CONFIGURABLE BET FILTERS (Option A)
    # Based on Week 13 analysis - these are static thresholds that filter out
    # bet types with historically poor win rates.
    #
    # TODO: Convert to dynamic filters (Option B) - auto-adjust based on
    # historical win rates per market/position/side combination. Track rolling
    # performance and dynamically enable/disable filters when sample size
    # reaches statistical significance (n>=30 bets per category).
    # ==========================================================================
    BET_FILTERS = {
        # Skip reception lines > 70 yards (Week 13: 10% win rate, -$1,674)
        # High lines typically on elite WRs who have high variance
        'max_reception_line': 70.0,

        # Skip TE OVER bets (Week 13: 33% win rate, -$398)
        # TEs have inconsistent target share and red zone dependency
        'skip_te_over': True,

        # Additional elite WRs to add to UNDER filter (beyond base list)
        # Week 13 showed Rashee Rice crushed UNDER bets
        'additional_elite_wrs': {'Rashee Rice'},

        # Future filters to consider based on more data:
        # 'skip_under_all': False,  # Week 13: UNDER overall 21.7% win rate
        # 'min_rb_over_edge': 0.05,  # RB OVER performed well (71.4%)
    }

    def __init__(self):
        self.min_edge = 0.03  # Minimum 3% edge to consider
        self.min_confidence = 0.65  # Minimum model confidence

    def generate_reasoning(
        self,
        player: str,
        position: str,
        market: str,
        prediction: float,
        line: float,
        side: str,
        edge_pct: float,
        ev_pct: float,
        confidence: float,
        weather: Optional[str] = None,
        weather_warning: Optional[str] = None,
        injury_warning: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Generate human-readable reasoning for an edge.

        Returns:
            Dict with 'short' (1-line) and 'detailed' (multi-line) reasoning
        """
        # Market display name
        market_names = {
            'player_pass_yds': 'passing yards',
            'player_rush_yds': 'rushing yards',
            'player_reception_yds': 'receiving yards',
            'player_receptions': 'receptions',
            'player_pass_tds': 'passing TDs',
        }
        market_name = market_names.get(market, market.replace('player_', '').replace('_', ' '))

        # Calculate difference
        diff = prediction - line
        diff_pct = (diff / line * 100) if line > 0 else 0

        # Build short reasoning (1-line for notifications)
        direction = "above" if diff > 0 else "below"
        short_parts = [
            f"Model: {prediction:.1f} vs line {line:.1f} ({abs(diff_pct):.0f}% {direction})"
        ]

        if injury_warning:
            short_parts.append(f"🏥 {injury_warning}")

        if weather_warning:
            short_parts.append(f"⚠️ {weather_warning}")

        short = " | ".join(short_parts)

        # Build detailed reasoning (for Discord embed)
        detailed_lines = []

        # Primary reasoning
        if side == 'over':
            detailed_lines.append(
                f"📈 **OVER {line}** {market_name}"
            )
            detailed_lines.append(
                f"Model predicts **{prediction:.1f}** ({diff_pct:+.1f}% vs line)"
            )
        else:
            detailed_lines.append(
                f"📉 **UNDER {line}** {market_name}"
            )
            detailed_lines.append(
                f"Model predicts **{prediction:.1f}** ({diff_pct:+.1f}% vs line)"
            )

        # Edge strength
        if ev_pct >= 10:
            strength = "🔥 Strong"
        elif ev_pct >= 5:
            strength = "✅ Good"
        else:
            strength = "📊 Moderate"
        detailed_lines.append(f"{strength} edge: **{ev_pct:+.1f}% EV**")

        # Confidence
        conf_emoji = "🎯" if confidence >= 0.8 else "📍" if confidence >= 0.65 else "⚠️"
        detailed_lines.append(f"{conf_emoji} Model confidence: {confidence:.0%}")

        # Injury status
        if injury_warning:
            detailed_lines.append(f"🏥 Injury: {injury_warning} (confidence adjusted)")

        # Weather impact
        if weather_warning and "nan" not in str(weather_warning).lower():
            detailed_lines.append(f"🌨️ Weather: {weather_warning} (confidence adjusted)")
        elif weather and weather != "Dome" and "nan" not in str(weather).lower() and weather != "Outdoor":
            detailed_lines.append(f"🌤️ Weather: {weather}")

        # Position context
        position_context = {
            'QB': 'Quarterback consistency varies with game script',
            'RB': 'Volume dependent on game flow and score',
            'WR': 'Target share drives receiving production',
            'TE': 'Red zone usage affects ceiling',
        }
        if position in position_context:
            detailed_lines.append(f"ℹ️ {position_context[position]}")

        return {
            'short': short,
            'detailed': '\n'.join(detailed_lines),
        }

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

    # Player Tier System - affects confidence adjustments and bet filtering
    # Tier 1 (Elite): Avoid UNDERs, high variance players
    # Tier 2 (Starter): Normal treatment
    # Tier 3 (Backup/Depth): Require higher edge to bet
    PLAYER_TIERS = {
        # Tier 1 - Elite (avoid UNDERs on receiving)
        'elite_wr': {
            "Ja'Marr Chase", "Ja'marr Chase", "CeeDee Lamb", "Justin Jefferson", "Tyreek Hill",
            "A.J. Brown", "Amon-Ra St. Brown", "Davante Adams", "Stefon Diggs",
            "DK Metcalf", "Chris Olave", "Garrett Wilson", "Nico Collins",
            "Puka Nacua", "Mike Evans", "Deebo Samuel", "Jaylen Waddle",
        },
        'elite_rb': {
            "Derrick Henry", "Saquon Barkley", "Breece Hall", "Bijan Robinson",
            "Jonathan Taylor", "Josh Jacobs", "De'Von Achane", "Jahmyr Gibbs",
            "Kyren Williams", "Joe Mixon", "Alvin Kamara", "James Cook",
        },
        'elite_te': {
            "Travis Kelce", "George Kittle", "Mark Andrews", "Sam LaPorta",
            "Trey McBride", "T.J. Hockenson", "David Njoku", "Dalton Kincaid",
        },
        'elite_qb': {
            "Patrick Mahomes", "Josh Allen", "Lamar Jackson", "Jalen Hurts",
            "Joe Burrow", "Dak Prescott", "Tua Tagovailoa", "C.J. Stroud",
        },
    }

    # For backwards compatibility - combine base elite WRs with additional from filters
    @property
    def ELITE_WRS(self):
        """Get combined elite WRs (base list + additional from BET_FILTERS)."""
        return self.PLAYER_TIERS['elite_wr'] | self.BET_FILTERS.get('additional_elite_wrs', set())

    # Confidence adjustment by tier (lower = more conservative)
    TIER_CONFIDENCE_MULTIPLIER = {
        'elite': 0.95,    # Slightly reduce confidence for elite (high variance)
        'starter': 1.0,   # Normal confidence
        'backup': 0.85,   # Reduce confidence for backups (inconsistent usage)
    }

    def get_player_tier(self, player_name: str, position: str) -> str:
        """
        Determine a player's tier based on name and position.

        Returns: 'elite', 'starter', or 'backup'
        """
        # Check if player is in any elite tier
        position_to_tier_key = {
            'WR': 'elite_wr',
            'RB': 'elite_rb',
            'TE': 'elite_te',
            'QB': 'elite_qb',
        }

        tier_key = position_to_tier_key.get(position)
        if tier_key and player_name in self.PLAYER_TIERS.get(tier_key, set()):
            return 'elite'

        # Default to starter (we don't have backup detection yet)
        return 'starter'

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
        # TODO: Week 14 - Track player bets to avoid duplicates (deduplication)
        seen_player_bets = {}  # {(player_id, market, side): best_edge}

        # Get weather data for the week
        weather_service = get_weather_service()
        week_weather = weather_service.get_weather_for_week(season, week)

        # Get injury service and pre-fetch ESPN data
        injury_service = get_injury_service()
        injury_service.fetch_all_espn_injuries()  # Pre-populate cache

        with get_session() as session:
            # Get all games for this week
            games = session.query(Game).filter_by(
                season=season,
                week=week,
                season_type='REG'
            ).all()

            for game in games:
                # Get weather for this game
                game_key = f"{game.away_team.abbreviation}@{game.home_team.abbreviation}"
                weather = week_weather.get(game_key)

                # Get props for this game (include props from last 48 hours)
                cutoff_time = datetime.now() - timedelta(hours=48)
                props = session.query(Prop).filter_by(
                    game_id=game.id,
                    is_active=True
                ).filter(
                    Prop.timestamp >= cutoff_time
                ).all()

                for prop in props:
                    # Skip props below minimum line threshold (low-volume players)
                    min_line = self.MIN_LINE_THRESHOLDS.get(prop.market, 0)
                    if prop.line < min_line:
                        logger.debug(f"Skipping low-volume prop: {prop.market} line {prop.line} < {min_line}")
                        continue

                    # Skip reception lines above max threshold (high variance, poor win rate)
                    max_rec_line = self.BET_FILTERS.get('max_reception_line')
                    if (max_rec_line and
                        prop.market == 'player_reception_yds' and
                        prop.line > max_rec_line):
                        logger.debug(f"Skipping high reception line: {prop.line} > {max_rec_line}")
                        continue

                    # Get model prediction - match by player and market (predictions may have different game_id)
                    prediction = session.query(Prediction).filter_by(
                        player_id=prop.player_id,
                        market=prop.market
                    ).order_by(Prediction.created_at.desc()).first()

                    if not prediction:
                        continue

                    # Check player injury status
                    injury_status = injury_service.get_injury_status(
                        prop.player_id, season, week
                    )
                    injury_warning = None

                    # Skip players who are OUT
                    if injury_status.is_out:
                        logger.info(f"Skipping OUT player: {injury_status.player_name} ({injury_status.status})")
                        continue

                    # Apply injury adjustment to confidence
                    adjusted_confidence = prediction.confidence * injury_status.confidence_multiplier
                    if injury_status.confidence_multiplier < 1.0:
                        injury_warning = injury_status.warning_message

                    # Apply weather adjustment to confidence for passing/receiving
                    weather_warning = None
                    if weather and weather.is_bad_weather:
                        # Reduce confidence for weather-sensitive stats
                        if prop.market in ('player_pass_yds', 'player_reception_yds'):
                            adjusted_confidence *= weather.weather_impact
                            weather_warning = weather.summary

                    # Calculate edge
                    edge_analysis = self.calculate_edge(
                        model_prediction=prediction.prediction,
                        model_confidence=adjusted_confidence,
                        market_line=prop.line,
                        over_odds=prop.over_odds,
                        under_odds=prop.under_odds
                    )

                    # Check if either side has an edge
                    has_edge = (edge_analysis['over']['should_bet'] or
                               edge_analysis['under']['should_bet'])

                    if has_edge:
                        player = session.query(Player).get(prop.player_id)

                        # Skip UNDER bets on elite WRs for receiving yards
                        # Week 13 showed elite WRs (Chase, Lamb) crushed UNDER bets
                        if (prop.market == 'player_reception_yds' and
                            player.name in self.ELITE_WRS and
                            edge_analysis['under']['should_bet'] and
                            not edge_analysis['over']['should_bet']):
                            logger.info(f"Skipping UNDER bet on elite WR: {player.name}")
                            continue

                        # Skip TE OVER bets (inconsistent target share, red zone dependent)
                        # Week 13: 33% win rate, -$398
                        if (self.BET_FILTERS.get('skip_te_over') and
                            player.position == 'TE' and
                            prop.market in ('player_reception_yds', 'player_receptions') and
                            edge_analysis['over']['should_bet'] and
                            not edge_analysis['under']['should_bet']):
                            logger.info(f"Skipping OVER bet on TE: {player.name}")
                            continue

                        # Determine which side has the edge
                        if edge_analysis['over']['should_bet']:
                            bet_side = 'over'
                            edge_pct = edge_analysis['over']['edge_pct']
                            ev_pct = edge_analysis['over']['ev_pct']
                        else:
                            bet_side = 'under'
                            edge_pct = edge_analysis['under']['edge_pct']
                            ev_pct = edge_analysis['under']['ev_pct']

                        # Generate reasoning
                        reasoning = self.generate_reasoning(
                            player=player.name,
                            position=player.position,
                            market=prop.market,
                            prediction=edge_analysis['prediction'],
                            line=edge_analysis['line'],
                            side=bet_side,
                            edge_pct=edge_pct,
                            ev_pct=ev_pct,
                            confidence=adjusted_confidence,
                            weather=weather.summary if weather else None,
                            weather_warning=weather_warning,
                            injury_warning=injury_warning,
                        )

                        edge_info = {
                            'game': f"{game.away_team.abbreviation} @ {game.home_team.abbreviation}",
                            'game_date': game.game_date,
                            'player': player.name,
                            'team': player.team.abbreviation if player.team else None,
                            'position': player.position,
                            'market': prop.market,
                            'weather': weather.summary if weather else None,
                            'weather_warning': weather_warning,
                            'injury_warning': injury_warning,
                            'injury_status': injury_status.status if injury_status else None,
                            'reasoning': reasoning['short'],
                            'reasoning_detailed': reasoning['detailed'],
                            **edge_analysis,
                        }

                        edges.append(edge_info)

        # Sort by absolute EV (best bets first)
        edges.sort(
            key=lambda x: max(abs(x['over']['ev']), abs(x['under']['ev'])),
            reverse=True
        )

        # Deduplicate edges - keep only best line per player/market/side
        # This prevents having 9 bets on same player at slightly different lines
        original_count = len(edges)
        deduplicated = {}
        for edge in edges:
            bet_side = 'over' if edge['over']['should_bet'] else 'under'
            key = (edge['player'], edge['market'], bet_side)

            if key not in deduplicated:
                deduplicated[key] = edge
            else:
                # Keep the one with better EV
                existing_ev = deduplicated[key][bet_side]['ev']
                new_ev = edge[bet_side]['ev']
                if new_ev > existing_ev:
                    deduplicated[key] = edge

        edges = list(deduplicated.values())

        # Re-sort after deduplication
        edges.sort(
            key=lambda x: max(abs(x['over']['ev']), abs(x['under']['ev'])),
            reverse=True
        )

        if original_count != len(edges):
            logger.info(f"Deduplicated {original_count} edges to {len(edges)} unique player/market/side combinations")

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
                    existing.reasoning = edge_data.get('reasoning', f"Model predicts {edge_data['prediction']:.1f} vs line {edge_data['line']}")
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
                        reasoning=edge_data.get('reasoning', f"Model predicts {edge_data['prediction']:.1f} vs line {edge_data['line']}"),
                    )
                    session.add(edge)
                    stored_count += 1

            session.commit()
            logger.info(f"Stored {stored_count} new edges, updated {updated_count} existing")

    def format_edge_report(self, edges: List[Dict], top_n: int = 20) -> str:
        """Format edges into a readable report with compact table."""
        if not edges:
            return "No edges found."

        # Group by market for summary
        by_market = {}
        for edge in edges:
            market = edge['market']
            if market not in by_market:
                by_market[market] = []
            by_market[market].append(edge)

        # Collect weather warnings (any game with precipitation or weather_warning)
        weather_games = {}
        for edge in edges:
            weather_str = edge.get('weather', '')
            if edge.get('weather_warning') or any(w in str(weather_str).lower() for w in ['snow', 'rain']):
                game = edge['game']
                if game not in weather_games:
                    weather_games[game] = weather_str or edge.get('weather_warning')

        # Collect injury warnings
        injury_players = {}
        for edge in edges:
            if edge.get('injury_warning'):
                injury_players[edge['player']] = edge['injury_warning']

        report = []
        report.append(f"BETTING EDGES - {len(edges)} opportunities found")
        report.append("")

        # Injury warnings section
        if injury_players:
            report.append("🏥 INJURY CONCERNS:")
            for player, warning in sorted(injury_players.items()):
                report.append(f"  {player}: {warning}")
            report.append("  (Confidence adjusted for injury status)")
            report.append("")

        # Weather warnings section
        if weather_games:
            report.append("⚠️  WEATHER WARNINGS:")
            for game, weather in weather_games.items():
                report.append(f"  {game}: {weather}")
            report.append("  (Pass/rec confidence reduced for affected games)")
            report.append("")

        # Summary by market
        report.append("By Market:")
        for market, market_edges in sorted(by_market.items()):
            market_name = market.replace('player_', '').replace('_', ' ').title()
            report.append(f"  {market_name}: {len(market_edges)} edges")
        report.append("")

        # Prepare edges for table (sort by EV)
        table_edges = []
        for edge in edges:
            if edge['over']['ev'] > edge['under']['ev']:
                side = 'OVER'
                ev = edge['over']['ev_pct']
                edge_pct = edge['over']['edge_pct']
                odds = edge['over']['odds']
            else:
                side = 'UNDER'
                ev = edge['under']['ev_pct']
                edge_pct = edge['under']['edge_pct']
                odds = edge['under']['odds']

            # Add weather flag (show for any bad weather game)
            weather_str = edge.get('weather', '')
            has_bad_weather = any(w in str(weather_str).lower() for w in ['snow', 'rain']) or edge.get('weather_warning')

            # Add injury flag
            has_injury = edge.get('injury_warning') is not None

            # Combined flag column (wx = weather, inj = injury)
            flags = ''
            if has_injury:
                flags += '🏥'
            if has_bad_weather:
                flags += '❄'

            table_edges.append({
                'side': side,
                'player': edge['player'][:16],  # Truncate long names
                'market': edge['market'].replace('player_', '').replace('_yds', '').replace('_', ' ')[:8],
                'line': edge['line'],
                'pred': edge['prediction'],
                'ev': ev,
                'edge': edge_pct,
                'odds': odds,
                'conf': edge['model_confidence'],
                'flags': flags,
                'game': edge['game'],
            })

        # Sort by EV descending
        table_edges.sort(key=lambda x: x['ev'], reverse=True)

        # Print table header
        report.append(f"Top {min(top_n, len(table_edges))} Edges by EV:")
        report.append("")
        report.append(f"{'':3} {'Side':<6} {'Player':<16} {'Market':<8} {'Line':>6} {'Pred':>6} {'EV':>7} {'Edge':>6} {'Odds':>6}")
        report.append("-" * 76)

        # Print top edges
        for e in table_edges[:top_n]:
            report.append(
                f"{e['flags']:<3} {e['side']:<6} {e['player']:<16} {e['market']:<8} "
                f"{e['line']:>6.1f} {e['pred']:>6.1f} {e['ev']:>+6.1f}% {e['edge']:>+5.1f}% {e['odds']:>+5d}"
            )

        if len(table_edges) > top_n:
            report.append("")
            report.append(f"... and {len(table_edges) - top_n} more edges")

        return "\n".join(report)

    def edge_to_alert_params(self, edge: Dict) -> Dict:
        """
        Convert an edge dict to parameters suitable for send_edge_alert().

        Args:
            edge: Edge dict from find_edges_for_week()

        Returns:
            Dict of keyword arguments for send_edge_alert()
        """
        # Determine which side has the edge
        if edge['over']['should_bet']:
            direction = 'over'
            edge_pct = edge['over']['edge_pct']
            ev_pct = edge['over']['ev_pct']
        else:
            direction = 'under'
            edge_pct = edge['under']['edge_pct']
            ev_pct = edge['under']['ev_pct']

        return {
            'player_name': edge['player'],
            'stat_type': edge['market'],
            'line': edge['line'],
            'prediction': edge['prediction'],
            'edge_pct': edge_pct,
            'direction': direction,
            'confidence': edge['model_confidence'],
            'game': edge.get('game', ''),
            'ev_pct': ev_pct,
            'reasoning': edge.get('reasoning', ''),
            'reasoning_detailed': edge.get('reasoning_detailed', ''),
            'weather': edge.get('weather'),
            'weather_warning': edge.get('weather_warning'),
            'injury_warning': edge.get('injury_warning'),
            'injury_status': edge.get('injury_status'),
        }
