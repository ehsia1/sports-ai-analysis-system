"""Parlay builder with correlation analysis and optimization."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from itertools import combinations
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from ..database import Parlay, Edge, Game, Player
from ..utils.odds import american_to_decimal, calculate_ev
from .edge_detector import EdgeDetector

logger = logging.getLogger(__name__)


class ParlayBuilder:
    """Build optimized parlays with correlation analysis."""
    
    def __init__(self, session: Session):
        self.session = session
        self.edge_detector = EdgeDetector(session)
        
        # Parlay building parameters
        self.max_legs = 4           # Maximum legs per parlay
        self.min_legs = 2           # Minimum legs per parlay
        self.min_parlay_ev = 0.05   # 5% minimum expected value
        self.max_negative_correlation = -0.3  # Max negative correlation allowed
        
        # Correlation matrix (simplified - in practice would be learned from data)
        self.correlation_matrix = self._build_correlation_matrix()
        
    def _build_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build correlation matrix between different prop types."""
        
        # Simplified correlation matrix based on NFL knowledge
        # In production, this would be calculated from historical data
        correlations = {
            'receiving_yards': {
                'receiving_yards': 1.0,
                'receptions': 0.75,        # Strong positive correlation
                'receiving_tds': 0.45,     # Moderate positive correlation
                'passing_yards': 0.55,     # QB success helps WR
                'rushing_yards': -0.15,    # Negative correlation (game script)
                'rushing_tds': -0.10
            },
            'receptions': {
                'receiving_yards': 0.75,
                'receptions': 1.0,
                'receiving_tds': 0.35,
                'passing_yards': 0.60,
                'rushing_yards': -0.20,
                'rushing_tds': -0.15
            },
            'receiving_tds': {
                'receiving_yards': 0.45,
                'receptions': 0.35,
                'receiving_tds': 1.0,
                'passing_yards': 0.30,
                'rushing_yards': -0.25,
                'rushing_tds': -0.40      # Strong negative (red zone usage)
            },
            'passing_yards': {
                'receiving_yards': 0.55,
                'receptions': 0.60,
                'receiving_tds': 0.30,
                'passing_yards': 1.0,
                'rushing_yards': -0.35,
                'rushing_tds': -0.25
            },
            'rushing_yards': {
                'receiving_yards': -0.15,
                'receptions': -0.20,
                'receiving_tds': -0.25,
                'passing_yards': -0.35,
                'rushing_yards': 1.0,
                'rushing_tds': 0.50
            },
            'rushing_tds': {
                'receiving_yards': -0.10,
                'receptions': -0.15,
                'receiving_tds': -0.40,
                'passing_yards': -0.25,
                'rushing_yards': 0.50,
                'rushing_tds': 1.0
            }
        }
        
        return correlations
        
    def build_parlays(
        self,
        edges_df: pd.DataFrame,
        max_parlays: int = 10,
        same_game_only: bool = False
    ) -> pd.DataFrame:
        """Build optimized parlays from detected edges."""
        
        logger.info(f"Building parlays from {len(edges_df)} edges")
        
        if edges_df.empty:
            logger.warning("No edges provided for parlay building")
            return pd.DataFrame()
        
        # Filter edges for parlay eligibility
        eligible_edges = self._filter_edges_for_parlays(edges_df)
        
        if len(eligible_edges) < self.min_legs:
            logger.warning("Not enough eligible edges for parlay building")
            return pd.DataFrame()
        
        # Generate parlay combinations
        parlay_combinations = self._generate_combinations(
            eligible_edges, 
            same_game_only=same_game_only
        )
        
        if not parlay_combinations:
            logger.warning("No valid parlay combinations found")
            return pd.DataFrame()
        
        # Evaluate each parlay
        parlays = []
        
        for combination in parlay_combinations[:max_parlays * 3]:  # Generate more than needed
            try:
                parlay_data = self._evaluate_parlay(combination, eligible_edges)
                if parlay_data and parlay_data['expected_value'] >= self.min_parlay_ev:
                    parlays.append(parlay_data)
            except Exception as e:
                logger.warning(f"Error evaluating parlay combination: {e}")
        
        if not parlays:
            logger.info("No parlays met minimum EV threshold")
            return pd.DataFrame()
        
        # Convert to DataFrame and sort by EV
        parlays_df = pd.DataFrame(parlays)
        parlays_df = parlays_df.sort_values('expected_value', ascending=False).head(max_parlays)
        
        logger.info(f"Built {len(parlays_df)} optimal parlays")
        
        return parlays_df
        
    def _filter_edges_for_parlays(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Filter edges suitable for parlays."""
        
        # Parlay eligibility criteria
        eligible = edges_df[
            (edges_df['final_confidence'] >= 0.65) &  # Higher confidence for parlays
            (edges_df['best_ev'] >= 0.03) &           # Higher EV threshold
            (edges_df['edge_score'] >= 0.06)          # Strong edge score
        ].copy()
        
        # Add parlay-specific features
        eligible['parlay_weight'] = (
            eligible['best_ev'] * eligible['final_confidence'] * eligible['edge_score']
        )
        
        return eligible.sort_values('parlay_weight', ascending=False)
        
    def _generate_combinations(
        self,
        edges_df: pd.DataFrame,
        same_game_only: bool = False
    ) -> List[List[int]]:
        """Generate valid parlay combinations."""
        
        combinations_list = []
        edge_indices = edges_df.index.tolist()
        
        # Generate 2-leg to max_legs combinations
        for num_legs in range(self.min_legs, self.max_legs + 1):
            for combo in combinations(edge_indices, num_legs):
                combo_edges = edges_df.loc[list(combo)]
                
                # Check if combination is valid
                if self._is_valid_combination(combo_edges, same_game_only):
                    combinations_list.append(list(combo))
        
        # Sort by potential value (sum of individual EVs as proxy)
        combinations_list.sort(
            key=lambda combo: sum(edges_df.loc[combo, 'best_ev']),
            reverse=True
        )
        
        return combinations_list
        
    def _is_valid_combination(
        self,
        combo_edges: pd.DataFrame,
        same_game_only: bool = False
    ) -> bool:
        """Check if a combination of edges is valid for a parlay."""
        
        # Same game restriction
        if same_game_only:
            if len(combo_edges['game_id'].unique()) > 1:
                return False
        
        # No same player, same market conflicts
        player_market_pairs = combo_edges[['player_id', 'prop_type']].drop_duplicates()
        if len(player_market_pairs) != len(combo_edges):
            return False
        
        # Check correlations
        if not self._check_correlation_constraints(combo_edges):
            return False
        
        # Avoid highly correlated same-side bets
        if self._has_excessive_correlation(combo_edges):
            return False
        
        return True
        
    def _check_correlation_constraints(self, combo_edges: pd.DataFrame) -> bool:
        """Check if the combination meets correlation constraints."""
        
        prop_types = combo_edges['prop_type'].tolist()
        
        # Check all pairs for excessive negative correlation
        for i, prop1 in enumerate(prop_types):
            for j, prop2 in enumerate(prop_types[i+1:], i+1):
                correlation = self.correlation_matrix.get(prop1, {}).get(prop2, 0)
                
                if correlation < self.max_negative_correlation:
                    return False
        
        return True
        
    def _has_excessive_correlation(self, combo_edges: pd.DataFrame) -> bool:
        """Check for excessive positive correlation that reduces diversification."""
        
        prop_types = combo_edges['prop_type'].tolist()
        high_correlation_count = 0
        
        for i, prop1 in enumerate(prop_types):
            for j, prop2 in enumerate(prop_types[i+1:], i+1):
                correlation = self.correlation_matrix.get(prop1, {}).get(prop2, 0)
                
                if correlation > 0.8:  # Very high correlation
                    high_correlation_count += 1
        
        # Don't allow more than 1 highly correlated pair in a parlay
        return high_correlation_count > 1
        
    def _evaluate_parlay(
        self,
        combination: List[int],
        edges_df: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a specific parlay combination."""
        
        combo_edges = edges_df.loc[combination]
        
        if combo_edges.empty:
            return None
        
        # Calculate parlay odds
        individual_odds = []
        for _, edge in combo_edges.iterrows():
            market_odds = edge[f"market_{edge['best_side']}_odds"]
            individual_odds.append(american_to_decimal(market_odds))
        
        # Parlay odds (multiply decimal odds)
        parlay_decimal_odds = np.prod(individual_odds)
        
        # Calculate joint probability with correlation adjustment
        joint_probability = self._calculate_joint_probability(combo_edges)
        
        # Calculate expected value
        parlay_payout = parlay_decimal_odds - 1
        expected_value = (joint_probability * parlay_payout) - (1 - joint_probability)
        
        # Risk metrics
        individual_probabilities = combo_edges[f'fair_{combo_edges["best_side"]}_probability'].tolist()
        
        # Kelly fraction for parlay (more conservative)
        if joint_probability > 0 and parlay_decimal_odds > 1:
            kelly_fraction = max(0, (joint_probability * parlay_decimal_odds - 1) / (parlay_decimal_odds - 1))
            kelly_fraction *= 0.1  # Very conservative for parlays
        else:
            kelly_fraction = 0
        
        # Build parlay data
        parlay_data = {
            'legs': combination,
            'num_legs': len(combination),
            'parlay_type': 'same_game' if len(combo_edges['game_id'].unique()) == 1 else 'multi_game',
            'individual_odds': individual_odds,
            'parlay_odds': parlay_decimal_odds,
            'joint_probability': joint_probability,
            'expected_value': expected_value,
            'kelly_fraction': kelly_fraction,
            'recommended_bet': kelly_fraction * 10000,  # Assuming $10k bankroll
            'legs_summary': combo_edges[['player_name', 'prop_type', 'best_side', 'market_line']].to_dict('records'),
            'correlation_score': self._calculate_correlation_score(combo_edges),
            'confidence_score': combo_edges['final_confidence'].mean(),
            'total_individual_ev': combo_edges['best_ev'].sum(),
            'parlay_boost': expected_value - combo_edges['best_ev'].sum(),  # EV boost from parlay
        }
        
        return parlay_data
        
    def _calculate_joint_probability(self, combo_edges: pd.DataFrame) -> float:
        """Calculate joint probability with correlation adjustments."""
        
        individual_probs = []
        prop_types = []
        
        for _, edge in combo_edges.iterrows():
            prob = edge[f'fair_{edge["best_side"]}_probability']
            individual_probs.append(prob)
            prop_types.append(edge['prop_type'])
        
        # Start with independent assumption
        joint_prob = np.prod(individual_probs)
        
        # Apply correlation adjustments
        if len(prop_types) == 2:
            correlation = self.correlation_matrix.get(prop_types[0], {}).get(prop_types[1], 0)
            
            # Adjust for correlation (simplified approach)
            if correlation > 0:
                # Positive correlation increases joint probability slightly
                adjustment = 1 + (correlation * 0.1)
            else:
                # Negative correlation decreases joint probability
                adjustment = 1 + (correlation * 0.1)
                
            joint_prob *= adjustment
            
        elif len(prop_types) > 2:
            # For multi-leg parlays, apply a general correlation discount
            avg_correlation = self._calculate_average_correlation(prop_types)
            correlation_adjustment = 1 + (avg_correlation * 0.05)  # Smaller adjustment for multi-leg
            joint_prob *= correlation_adjustment
        
        return max(0.001, min(0.999, joint_prob))  # Keep within valid probability bounds
        
    def _calculate_average_correlation(self, prop_types: List[str]) -> float:
        """Calculate average correlation between all pairs of prop types."""
        
        correlations = []
        
        for i, prop1 in enumerate(prop_types):
            for prop2 in prop_types[i+1:]:
                correlation = self.correlation_matrix.get(prop1, {}).get(prop2, 0)
                correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0
        
    def _calculate_correlation_score(self, combo_edges: pd.DataFrame) -> float:
        """Calculate a correlation score for the parlay (higher is better diversification)."""
        
        prop_types = combo_edges['prop_type'].tolist()
        avg_correlation = self._calculate_average_correlation(prop_types)
        
        # Score from -1 to 1, where 0 is perfectly uncorrelated
        # Penalize high positive or negative correlations
        correlation_penalty = abs(avg_correlation)
        diversity_score = 1 - correlation_penalty
        
        return max(0, diversity_score)
        
    def optimize_parlay_selection(
        self,
        parlays_df: pd.DataFrame,
        bankroll: float = 10000,
        max_risk: float = 0.05  # 5% of bankroll max risk
    ) -> pd.DataFrame:
        """Optimize parlay selection for portfolio construction."""
        
        if parlays_df.empty:
            return parlays_df
        
        # Risk budgeting
        parlays = parlays_df.copy()
        
        # Calculate risk-adjusted bet sizes
        max_bet = bankroll * max_risk
        
        parlays['risk_adjusted_bet'] = np.minimum(
            parlays['recommended_bet'],
            max_bet / len(parlays)  # Spread risk across parlays
        )
        
        # Calculate portfolio metrics
        total_risk = parlays['risk_adjusted_bet'].sum()
        portfolio_ev = (parlays['expected_value'] * parlays['risk_adjusted_bet']).sum() / total_risk if total_risk > 0 else 0
        
        # Diversification score
        parlays['diversification_score'] = parlays['correlation_score']
        
        # Final optimization score
        parlays['optimization_score'] = (
            parlays['expected_value'] * 0.4 +
            parlays['confidence_score'] * 0.3 +
            parlays['diversification_score'] * 0.2 +
            (parlays['kelly_fraction'] / parlays['kelly_fraction'].max()) * 0.1  # Size consistency
        )
        
        # Add portfolio context
        parlays['portfolio_allocation'] = parlays['risk_adjusted_bet'] / total_risk if total_risk > 0 else 0
        parlays['portfolio_ev_contribution'] = parlays['expected_value'] * parlays['portfolio_allocation']
        
        return parlays.sort_values('optimization_score', ascending=False)
        
    def save_parlays_to_db(self, parlays_df: pd.DataFrame) -> List[int]:
        """Save parlays to database."""
        
        parlay_ids = []
        
        for _, parlay in parlays_df.iterrows():
            try:
                db_parlay = Parlay(
                    legs=parlay['legs'],
                    parlay_type=parlay['parlay_type'],
                    offered_odds=None,  # Would get from sportsbook
                    fair_odds=float(parlay['parlay_odds']),
                    joint_probability=float(parlay['joint_probability']),
                    expected_value=float(parlay['expected_value']),
                    correlation_matrix={}  # Would include detailed correlation data
                )
                
                self.session.add(db_parlay)
                self.session.flush()
                parlay_ids.append(db_parlay.id)
                
            except Exception as e:
                logger.error(f"Error saving parlay to database: {e}")
        
        try:
            self.session.commit()
            logger.info(f"Saved {len(parlay_ids)} parlays to database")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Error committing parlays to database: {e}")
            parlay_ids = []
        
        return parlay_ids
        
    def generate_parlay_report(self, parlays_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive parlay analysis report."""
        
        if parlays_df.empty:
            return {'message': 'No parlays generated', 'parlays': []}
        
        # Summary statistics
        total_parlays = len(parlays_df)
        avg_ev = parlays_df['expected_value'].mean()
        avg_legs = parlays_df['num_legs'].mean()
        total_recommended_bet = parlays_df['risk_adjusted_bet'].sum()
        
        # Best parlay
        best_parlay = parlays_df.iloc[0].to_dict() if len(parlays_df) > 0 else {}
        
        # Distribution analysis
        legs_distribution = parlays_df['num_legs'].value_counts().to_dict()
        type_distribution = parlays_df['parlay_type'].value_counts().to_dict()
        
        # Risk analysis
        risk_metrics = {
            'total_risk_dollars': total_recommended_bet,
            'average_parlay_bet': parlays_df['risk_adjusted_bet'].mean(),
            'max_single_risk': parlays_df['risk_adjusted_bet'].max(),
            'portfolio_ev': (parlays_df['expected_value'] * parlays_df['portfolio_allocation']).sum()
        }
        
        report = {
            'summary': {
                'total_parlays': total_parlays,
                'average_ev': round(avg_ev, 4),
                'average_legs': round(avg_legs, 1),
                'total_recommended_bet': round(total_recommended_bet, 2),
                'generated_at': datetime.now().isoformat()
            },
            'distribution': {
                'legs_count': legs_distribution,
                'parlay_types': type_distribution
            },
            'risk_analysis': risk_metrics,
            'best_parlay': best_parlay,
            'parlays': parlays_df.head(5).to_dict('records')  # Top 5 parlays
        }
        
        return report


def demo_parlay_building() -> Dict[str, Any]:
    """Demonstrate parlay building with sample edges."""
    
    # Create sample edges data
    sample_edges = pd.DataFrame([
        {
            'player_id': 1, 'player_name': 'Travis Kelce', 'game_id': 1,
            'prop_type': 'receiving_yards', 'best_side': 'over', 'market_line': 72.5,
            'market_over_odds': -110, 'market_under_odds': -110,
            'fair_line': 78.2, 'fair_over_probability': 0.58, 'fair_under_probability': 0.42,
            'best_ev': 0.08, 'final_confidence': 0.75, 'edge_score': 0.12
        },
        {
            'player_id': 1, 'player_name': 'Travis Kelce', 'game_id': 1,
            'prop_type': 'receptions', 'best_side': 'over', 'market_line': 6.5,
            'market_over_odds': -115, 'market_under_odds': -105,
            'fair_line': 7.2, 'fair_over_probability': 0.61, 'fair_under_probability': 0.39,
            'best_ev': 0.09, 'final_confidence': 0.78, 'edge_score': 0.14
        },
        {
            'player_id': 2, 'player_name': 'Tyreek Hill', 'game_id': 2,
            'prop_type': 'receiving_yards', 'best_side': 'over', 'market_line': 85.5,
            'market_over_odds': -108, 'market_under_odds': -112,
            'fair_line': 92.1, 'fair_over_probability': 0.59, 'fair_under_probability': 0.41,
            'best_ev': 0.07, 'final_confidence': 0.72, 'edge_score': 0.10
        },
        {
            'player_id': 3, 'player_name': 'Josh Allen', 'game_id': 2,
            'prop_type': 'passing_yards', 'best_side': 'over', 'market_line': 267.5,
            'market_over_odds': -110, 'market_under_odds': -110,
            'fair_line': 275.8, 'fair_over_probability': 0.55, 'fair_under_probability': 0.45,
            'best_ev': 0.06, 'final_confidence': 0.70, 'edge_score': 0.09
        }
    ])
    
    logger.info("Running parlay building demo with sample edges")
    
    # Mock parlay builder (no database session needed)
    class MockParlayBuilder(ParlayBuilder):
        def __init__(self):
            self.correlation_matrix = self._build_correlation_matrix()
    
    builder = MockParlayBuilder()
    parlays = builder.build_parlays(sample_edges, max_parlays=5)
    
    if not parlays.empty:
        optimized_parlays = builder.optimize_parlay_selection(parlays)
        report = builder.generate_parlay_report(optimized_parlays)
        logger.info(f"Demo complete: Built {len(parlays)} parlays")
        return report
    else:
        return {'message': 'No parlays built in demo', 'parlays': []}