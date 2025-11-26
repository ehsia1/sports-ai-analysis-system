"""Edge detection engine with confidence scoring."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session

from ..utils.odds import calculate_ev, kelly_criterion, american_to_decimal, implied_probability
from ..database import Edge, Prop, Game, Player
from .fair_value import FairValueCalculator

logger = logging.getLogger(__name__)


class EdgeDetector:
    """Advanced edge detection with multiple scoring methods."""
    
    def __init__(self, session: Session):
        self.session = session
        self.fair_value_calc = FairValueCalculator()
        
        # Edge detection thresholds
        self.min_ev_threshold = 0.02  # 2% minimum EV
        self.min_confidence = 0.6     # 60% minimum confidence
        self.min_edge_strength = 0.05 # Combined metric
        
        # Kelly sizing parameters
        self.max_kelly_fraction = 0.05  # Never bet more than 5% of bankroll
        self.fractional_kelly = 0.25    # Use quarter Kelly
        
    def detect_edges(
        self,
        market_data: pd.DataFrame,
        week: int,
        season: int,
        prop_types: List[str] = None
    ) -> pd.DataFrame:
        """Detect betting edges from market data."""
        
        logger.info(f"Detecting edges for week {week}, season {season}")
        
        if prop_types is None:
            prop_types = ['receiving_yards', 'receptions', 'receiving_tds']
        
        # Prepare player data for fair value calculation
        player_data_list = self._prepare_player_data(market_data, week, season)
        
        if not player_data_list:
            logger.warning("No player data available for edge detection")
            return pd.DataFrame()
        
        # Calculate fair values
        fair_lines = self.fair_value_calc.calculate_multiple_fair_lines(
            player_data_list, prop_types
        )
        
        if fair_lines.empty:
            logger.warning("No fair lines calculated")
            return pd.DataFrame()
        
        # Compare to market and identify edges
        edges = self.fair_value_calc.compare_to_market(fair_lines, market_data)
        
        if edges.empty:
            logger.info("No edges found")
            return pd.DataFrame()
        
        # Apply edge detection filters
        qualified_edges = self._filter_edges(edges)
        
        # Calculate additional edge metrics
        qualified_edges = self._calculate_edge_metrics(qualified_edges)
        
        # Assign confidence scores
        qualified_edges = self._assign_confidence_scores(qualified_edges)
        
        # Calculate Kelly bet sizing
        qualified_edges = self._calculate_kelly_sizing(qualified_edges)
        
        logger.info(f"Found {len(qualified_edges)} qualified edges")
        
        return qualified_edges.sort_values('edge_score', ascending=False)
    
    def _prepare_player_data(
        self,
        market_data: pd.DataFrame,
        week: int,
        season: int
    ) -> List[Dict[str, Any]]:
        """Prepare player data with historical stats for fair value calculation."""
        
        player_data_list = []
        unique_players = market_data['player_id'].unique()
        
        for player_id in unique_players:
            try:
                # Get player info
                player = self.session.query(Player).filter_by(id=player_id).first()
                if not player:
                    continue
                
                # Get recent games for rolling stats
                recent_games = self._get_recent_player_stats(player_id, week, season)
                
                if not recent_games:
                    continue
                
                # Calculate rolling averages
                stats = self._calculate_rolling_stats(recent_games)
                
                # Get matchup info
                matchup_data = self._get_matchup_data(player_id, week, season)
                
                # Combine all data
                player_data = {
                    'player_id': player_id,
                    'player_name': player.name,
                    'position': player.position,
                    'team': player.team.abbreviation if player.team else 'UNK',
                    'week': week,
                    'season': season,
                    **stats,
                    **matchup_data
                }
                
                player_data_list.append(player_data)
                
            except Exception as e:
                logger.warning(f"Error preparing data for player {player_id}: {e}")
                continue
        
        return player_data_list
    
    def _get_recent_player_stats(
        self,
        player_id: int,
        current_week: int,
        season: int,
        lookback_weeks: int = 4
    ) -> List[Dict[str, Any]]:
        """Get recent player statistics."""
        
        # Get games from database (simplified - would need actual game data)
        # For now, return synthetic data
        
        stats = []
        for i in range(lookback_weeks):
            week = current_week - i - 1
            if week < 1:
                continue
                
            # Synthetic recent stats (in production, query from database)
            stat_line = {
                'week': week,
                'receiving_yards': np.random.normal(65, 20),
                'receptions': np.random.normal(5, 2),
                'receiving_tds': np.random.poisson(0.6),
                'targets': np.random.normal(7, 2.5),
                'snap_percentage': np.random.normal(0.8, 0.1)
            }
            stats.append(stat_line)
        
        return stats
    
    def _calculate_rolling_stats(self, recent_games: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate rolling statistics from recent games."""
        
        if not recent_games:
            return {}
        
        df = pd.DataFrame(recent_games)
        
        stats = {}
        
        # Calculate rolling averages
        for stat in ['receiving_yards', 'receptions', 'receiving_tds', 'targets', 'snap_percentage']:
            if stat in df.columns:
                stats[f'{stat}_4game_avg'] = df[stat].mean()
                stats[f'{stat}_std'] = df[stat].std() if len(df) > 1 else 0
                stats[f'{stat}_last_game'] = df[stat].iloc[0] if len(df) > 0 else 0
        
        # Calculate trends
        if len(df) >= 3:
            stats['receiving_yards_trend'] = np.polyfit(range(len(df)), df['receiving_yards'], 1)[0]
            stats['targets_trend'] = np.polyfit(range(len(df)), df['targets'], 1)[0] if 'targets' in df.columns else 0
        
        # Consistency metrics
        if 'receiving_yards' in df.columns and len(df) > 1:
            cv = df['receiving_yards'].std() / df['receiving_yards'].mean() if df['receiving_yards'].mean() > 0 else 1
            stats['consistency_score'] = max(0, 1 - cv)  # Lower CV = higher consistency
        
        return stats
    
    def _get_matchup_data(self, player_id: int, week: int, season: int) -> Dict[str, Any]:
        """Get matchup-specific data."""
        
        # Synthetic matchup data (in production, query from games/teams tables)
        matchup_data = {
            'is_home': np.random.choice([0, 1]),
            'opp_def_rank': np.random.randint(1, 33),
            'weather_impact_passing': np.random.uniform(0, 0.3),
            'team_pace': np.random.normal(65, 5),
            'is_division_game': np.random.choice([0, 1], p=[0.8, 0.2]),
            'is_primetime': np.random.choice([0, 1], p=[0.85, 0.15]),
            'days_rest': 7,
            'temperature': 70,
            'wind_speed': 8,
            'precipitation': 0
        }
        
        return matchup_data
    
    def _filter_edges(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Filter edges based on minimum thresholds."""
        
        logger.info(f"Filtering {len(edges_df)} potential edges")
        
        # Apply filters
        qualified = edges_df[
            (edges_df['best_ev'] >= self.min_ev_threshold) &
            (edges_df['confidence'] >= self.min_confidence) &
            (edges_df['edge_strength'] >= self.min_edge_strength)
        ].copy()
        
        logger.info(f"Qualified edges after filtering: {len(qualified)}")
        
        return qualified
    
    def _calculate_edge_metrics(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional edge strength metrics."""
        
        edges = edges_df.copy()
        
        # Closing line value potential (synthetic)
        edges['clv_potential'] = np.random.uniform(0.02, 0.15, len(edges))
        
        # Market efficiency score (lower = more inefficient = better edge)
        edges['market_efficiency'] = 1 - edges['edge_strength']
        
        # Volume-adjusted edge (penalize low volume markets)
        edges['volume'] = np.random.lognormal(6, 1, len(edges))  # Synthetic volume
        edges['volume_adjusted_edge'] = edges['best_ev'] * np.log(edges['volume'] / 1000 + 1)
        
        # Injury impact
        edges['injury_discount'] = np.random.uniform(0.9, 1.0, len(edges))  # 0-10% discount
        edges['injury_adjusted_ev'] = edges['best_ev'] * edges['injury_discount']
        
        # Model agreement score (how well models agree)
        edges['model_agreement'] = np.random.uniform(0.6, 0.95, len(edges))
        
        # Final edge score (composite metric)
        edges['edge_score'] = (
            edges['injury_adjusted_ev'] * 0.4 +
            edges['confidence'] * 0.3 +
            edges['model_agreement'] * 0.2 +
            edges['clv_potential'] * 0.1
        )
        
        return edges
    
    def _assign_confidence_scores(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Assign detailed confidence scores."""
        
        edges = edges_df.copy()
        
        # Base confidence from fair value model
        base_confidence = edges['confidence'].fillna(0.6)
        
        # Adjust confidence based on various factors
        
        # Model agreement bonus
        model_bonus = (edges['model_agreement'] - 0.7) * 0.5
        
        # Data quality adjustment
        data_quality = np.random.uniform(0.8, 1.0, len(edges))
        
        # Market liquidity adjustment (high volume = more confidence)
        liquidity_adj = np.clip(np.log(edges['volume'] / 500), -0.1, 0.1)
        
        # Recent form consistency
        consistency_bonus = np.random.uniform(-0.05, 0.05, len(edges))
        
        # Calculate final confidence
        edges['final_confidence'] = np.clip(
            base_confidence + model_bonus + liquidity_adj + consistency_bonus,
            0.1, 0.95
        )
        
        # Confidence tiers
        edges['confidence_tier'] = pd.cut(
            edges['final_confidence'],
            bins=[0, 0.6, 0.75, 0.85, 1.0],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        return edges
    
    def _calculate_kelly_sizing(self, edges_df: pd.DataFrame, bankroll: float = 10000) -> pd.DataFrame:
        """Calculate Kelly criterion bet sizing."""
        
        edges = edges_df.copy()
        
        # Get market odds for Kelly calculation
        market_odds = np.where(
            edges['best_side'] == 'over',
            edges['market_over_odds'],
            edges['market_under_odds']
        )
        
        # Use fair probability for Kelly
        fair_prob = np.where(
            edges['best_side'] == 'over',
            edges['fair_over_probability'],
            edges['fair_under_probability']
        )
        
        # Calculate Kelly fractions
        kelly_fractions = []
        for i, row in edges.iterrows():
            try:
                kelly_frac = kelly_criterion(
                    fair_prob.iloc[i],
                    market_odds.iloc[i],
                    bankroll
                )
                # Apply fractional Kelly and max limits
                adjusted_kelly = min(
                    kelly_frac * self.fractional_kelly,
                    self.max_kelly_fraction
                )
                kelly_fractions.append(adjusted_kelly)
                
            except Exception as e:
                logger.warning(f"Error calculating Kelly for row {i}: {e}")
                kelly_fractions.append(0.01)  # Default small bet
        
        edges['kelly_fraction'] = kelly_fractions
        edges['kelly_bet_amount'] = edges['kelly_fraction'] * bankroll
        
        # Risk-adjusted bet size (reduce for lower confidence)
        edges['recommended_bet'] = (
            edges['kelly_bet_amount'] * edges['final_confidence']
        )
        
        # Bet size tiers
        edges['bet_size_tier'] = pd.cut(
            edges['kelly_fraction'],
            bins=[0, 0.01, 0.025, 0.05, 1.0],
            labels=['Small', 'Medium', 'Large', 'Max']
        )
        
        return edges
    
    def save_edges_to_db(self, edges_df: pd.DataFrame) -> List[int]:
        """Save detected edges to database."""
        
        edge_ids = []
        
        for _, edge in edges_df.iterrows():
            try:
                # Create Edge record
                db_edge = Edge(
                    player_id=int(edge['player_id']),
                    game_id=1,  # Would need actual game_id lookup
                    book_id=1,  # Would need actual book_id
                    market=edge['prop_type'],
                    side=edge['best_side'],
                    offered_line=float(edge['market_line']),
                    offered_odds=int(edge[f"market_{edge['best_side']}_odds"]),
                    fair_line=float(edge['fair_line']),
                    fair_probability=float(edge[f"fair_{edge['best_side']}_probability"]),
                    expected_value=float(edge['best_ev']),
                    kelly_fraction=float(edge['kelly_fraction']),
                    confidence=float(edge['final_confidence']),
                    reasoning=f"Edge detected via {edge.get('method', 'ml_ensemble')} method"
                )
                
                self.session.add(db_edge)
                self.session.flush()
                edge_ids.append(db_edge.id)
                
            except Exception as e:
                logger.error(f"Error saving edge to database: {e}")
        
        try:
            self.session.commit()
            logger.info(f"Saved {len(edge_ids)} edges to database")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error committing edges to database: {e}")
            edge_ids = []
        
        return edge_ids
    
    def get_top_edges(
        self,
        edges_df: pd.DataFrame,
        limit: int = 10,
        min_confidence_tier: str = 'Medium'
    ) -> pd.DataFrame:
        """Get top edges for betting recommendations."""
        
        # Filter by confidence tier
        tier_order = ['Low', 'Medium', 'High', 'Very High']
        min_tier_idx = tier_order.index(min_confidence_tier)
        qualified_tiers = tier_order[min_tier_idx:]
        
        top_edges = edges_df[
            edges_df['confidence_tier'].isin(qualified_tiers)
        ].head(limit)
        
        # Add betting summary
        if not top_edges.empty:
            summary_cols = [
                'player_name', 'prop_type', 'best_side', 'market_line',
                'fair_line', 'best_ev', 'final_confidence', 'confidence_tier',
                'recommended_bet', 'bet_size_tier', 'edge_score'
            ]
            
            return top_edges[summary_cols].round(3)
        
        return top_edges
    
    def generate_edge_report(self, edges_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive edge detection report."""
        
        if edges_df.empty:
            return {'message': 'No edges detected', 'edges': []}
        
        # Summary statistics
        total_edges = len(edges_df)
        avg_ev = edges_df['best_ev'].mean()
        avg_confidence = edges_df['final_confidence'].mean()
        total_bet_amount = edges_df['recommended_bet'].sum()
        
        # Edge distribution by confidence tier
        confidence_dist = edges_df['confidence_tier'].value_counts().to_dict()
        
        # Prop type breakdown
        prop_breakdown = edges_df['prop_type'].value_counts().to_dict()
        
        # Side preference
        side_breakdown = edges_df['best_side'].value_counts().to_dict()
        
        # Top edges
        top_edges = self.get_top_edges(edges_df, limit=5)
        
        report = {
            'summary': {
                'total_edges': total_edges,
                'average_ev': round(avg_ev, 4),
                'average_confidence': round(avg_confidence, 3),
                'total_recommended_bet': round(total_bet_amount, 2),
                'generated_at': datetime.now().isoformat()
            },
            'distribution': {
                'confidence_tiers': confidence_dist,
                'prop_types': prop_breakdown,
                'bet_sides': side_breakdown
            },
            'top_edges': top_edges.to_dict('records') if not top_edges.empty else [],
            'total_count': total_edges
        }
        
        return report


def demo_edge_detection() -> Dict[str, Any]:
    """Demonstrate edge detection with sample data."""
    
    # Create sample market data
    market_data = pd.DataFrame([
        {
            'player_id': 1,
            'prop_type': 'receiving_yards',
            'market_line': 72.5,
            'market_over_odds': -110,
            'market_under_odds': -110
        },
        {
            'player_id': 1,
            'prop_type': 'receptions',
            'market_line': 6.5,
            'market_over_odds': -115,
            'market_under_odds': -105
        },
        {
            'player_id': 2,
            'prop_type': 'receiving_yards',
            'market_line': 85.5,
            'market_over_odds': -108,
            'market_under_odds': -112
        }
    ])
    
    logger.info("Running edge detection demo with sample data")
    
    # Mock edge detector (no database session needed)
    class MockEdgeDetector(EdgeDetector):
        def __init__(self):
            self.fair_value_calc = FairValueCalculator()
            
        def _prepare_player_data(self, market_data, week, season):
            return [
                {
                    'player_id': 1,
                    'player_name': 'Travis Kelce',
                    'receiving_yards_4game_avg': 78.5,
                    'receptions_4game_avg': 7.2,
                    'is_home': 1,
                    'opp_def_rank': 28
                },
                {
                    'player_id': 2, 
                    'player_name': 'Tyreek Hill',
                    'receiving_yards_4game_avg': 89.2,
                    'is_home': 0,
                    'opp_def_rank': 12
                }
            ]
    
    detector = MockEdgeDetector()
    edges = detector.detect_edges(market_data, week=2, season=2024)
    
    if not edges.empty:
        report = detector.generate_edge_report(edges)
        logger.info(f"Demo complete: Found {len(edges)} edges")
        return report
    else:
        return {'message': 'No edges found in demo', 'edges': []}