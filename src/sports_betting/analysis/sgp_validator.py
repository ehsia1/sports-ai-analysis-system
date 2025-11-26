"""Same-game parlay validator with advanced correlation modeling."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SGPValidator:
    """Validate and optimize same-game parlays with correlation modeling."""
    
    def __init__(self):
        # Advanced correlation matrix for same-game scenarios
        self.sgp_correlations = self._build_sgp_correlation_matrix()
        
        # Validation rules
        self.max_sgp_legs = 6
        self.min_sgp_legs = 2
        self.max_player_props = 3  # Max props per player
        
        # Book-specific restrictions (varies by sportsbook)
        self.book_restrictions = {
            'draftkings': {
                'max_legs': 8,
                'allows_opposing_players': True,
                'allows_team_total_with_player': False,
                'restricted_combinations': [
                    ('passing_yards', 'receiving_yards', 'same_team'),  # QB + own WR restricted
                ]
            },
            'fanduel': {
                'max_legs': 6,
                'allows_opposing_players': True,
                'allows_team_total_with_player': True,
                'restricted_combinations': []
            }
        }
        
    def _build_sgp_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Build correlation matrix specifically for same-game parlays."""
        
        # Enhanced correlations for same-game scenarios
        sgp_correlations = {
            # QB correlations
            'passing_yards': {
                'passing_tds': 0.70,           # Strong positive
                'interceptions': -0.35,        # Negative
                'team_total_points': 0.60,     # Team success
                'receiving_yards_same_team': 0.45,  # Own WR benefits
                'receiving_yards_opponent': -0.20,  # Game script
                'rushing_yards_same_team': -0.40,   # Game script conflict
            },
            
            # WR correlations (same team)
            'receiving_yards_same_team': {
                'receptions_same_player': 0.85,     # Very strong
                'receiving_tds_same_player': 0.50,  # Moderate
                'passing_yards_qb': 0.45,           # QB success helps
                'team_total_points': 0.35,          # Team offense
                'receiving_yards_teammate': -0.25,   # Target share conflict
            },
            
            # WR correlations (opponent)
            'receiving_yards_opponent': {
                'receiving_yards_same_team': -0.15,  # Slight negative (game flow)
                'passing_yards_opponent_qb': 0.40,   # Opponent QB success
                'team_total_points_opponent': 0.30,  # Opponent offense
            },
            
            # RB correlations
            'rushing_yards': {
                'rushing_tds': 0.55,              # Strong positive
                'receptions_same_player': 0.20,   # Catching RB
                'passing_yards_qb': -0.40,        # Game script conflict
                'team_total_points': 0.40,        # Team success
                'rushing_yards_opponent': -0.30,  # Game flow
            },
            
            # TD correlations
            'receiving_tds': {
                'receiving_yards_same_player': 0.50,
                'passing_tds_qb': 0.65,           # QB throwing TD
                'team_total_points': 0.70,        # Strong team correlation
                'receiving_tds_teammate': -0.35,   # Red zone target share
            },
            
            'rushing_tds': {
                'rushing_yards_same_player': 0.55,
                'team_total_points': 0.65,
                'passing_tds_qb': -0.30,          # Red zone usage conflict
                'receiving_tds_teammate': -0.40,   # Goal line usage
            },
            
            # Team totals
            'team_total_points': {
                'passing_yards_qb': 0.60,
                'rushing_yards_rb': 0.40,
                'receiving_yards_wr': 0.35,
                'total_tds': 0.85,                # Very strong
                'opponent_total_points': -0.10,   # Slight negative (defensive impact)
            },
            
            # Game totals
            'game_total_points': {
                'team_total_points': 0.50,
                'passing_yards_combined': 0.55,
                'total_tds': 0.80,
                'total_turnovers': -0.25,
            }
        }
        
        return sgp_correlations
        
    def validate_sgp(
        self,
        legs: List[Dict[str, Any]],
        sportsbook: str = 'draftkings'
    ) -> Dict[str, Any]:
        """Validate a same-game parlay configuration."""
        
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'correlation_score': 0.0,
            'risk_assessment': 'low',
            'optimization_suggestions': []
        }
        
        # Basic validation
        if len(legs) < self.min_sgp_legs:
            validation_result['errors'].append(f"Minimum {self.min_sgp_legs} legs required")
            validation_result['is_valid'] = False
            
        if len(legs) > self.max_sgp_legs:
            validation_result['errors'].append(f"Maximum {self.max_sgp_legs} legs allowed")
            validation_result['is_valid'] = False
        
        # Sportsbook-specific validation
        book_config = self.book_restrictions.get(sportsbook, self.book_restrictions['draftkings'])
        
        if len(legs) > book_config['max_legs']:
            validation_result['errors'].append(f"{sportsbook} allows maximum {book_config['max_legs']} legs")
            validation_result['is_valid'] = False
        
        # Check for duplicate markets
        player_markets = [(leg['player_id'], leg['prop_type']) for leg in legs]
        if len(set(player_markets)) != len(player_markets):
            validation_result['errors'].append("Duplicate player/market combinations not allowed")
            validation_result['is_valid'] = False
        
        # Player prop limits
        player_prop_counts = {}
        for leg in legs:
            player_id = leg['player_id']
            player_prop_counts[player_id] = player_prop_counts.get(player_id, 0) + 1
            
        for player_id, count in player_prop_counts.items():
            if count > self.max_player_props:
                validation_result['warnings'].append(f"Player {player_id} has {count} props (max recommended: {self.max_player_props})")
        
        # Correlation analysis
        correlation_analysis = self._analyze_sgp_correlations(legs)
        validation_result['correlation_score'] = correlation_analysis['overall_score']
        validation_result['risk_assessment'] = correlation_analysis['risk_level']
        
        # Check for restricted combinations
        restricted_combos = self._check_restricted_combinations(legs, book_config)
        if restricted_combos:
            validation_result['errors'].extend(restricted_combos)
            validation_result['is_valid'] = False
        
        # Generate optimization suggestions
        optimization_suggestions = self._generate_optimization_suggestions(legs, correlation_analysis)
        validation_result['optimization_suggestions'] = optimization_suggestions
        
        return validation_result
        
    def _analyze_sgp_correlations(self, legs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between SGP legs."""
        
        correlations = []
        prop_types = [leg['prop_type'] for leg in legs]
        player_ids = [leg['player_id'] for leg in legs]
        teams = [leg.get('team', 'unknown') for leg in legs]
        
        # Calculate pairwise correlations
        for i, leg1 in enumerate(legs):
            for j, leg2 in enumerate(legs[i+1:], i+1):
                correlation = self._get_sgp_correlation(leg1, leg2)
                correlations.append({
                    'leg1_index': i,
                    'leg2_index': j,
                    'correlation': correlation,
                    'leg1': f"{leg1['player_name']} {leg1['prop_type']}",
                    'leg2': f"{leg2['player_name']} {leg2['prop_type']}"
                })
        
        if not correlations:
            return {'overall_score': 0.5, 'risk_level': 'low', 'details': []}
        
        # Calculate overall correlation metrics
        correlation_values = [c['correlation'] for c in correlations]
        avg_correlation = np.mean(correlation_values)
        max_correlation = max(correlation_values)
        min_correlation = min(correlation_values)
        
        # Risk assessment
        if max_correlation > 0.7:
            risk_level = 'high'
        elif max_correlation > 0.5:
            risk_level = 'medium'
        elif min_correlation < -0.4:
            risk_level = 'high'  # High negative correlation also risky
        else:
            risk_level = 'low'
        
        # Overall score (0 to 1, where 0.5 is neutral)
        # Penalize both high positive and high negative correlations
        overall_score = 0.5 + (avg_correlation * 0.3) - (abs(avg_correlation) * 0.2)
        overall_score = max(0, min(1, overall_score))
        
        return {
            'overall_score': overall_score,
            'average_correlation': avg_correlation,
            'max_correlation': max_correlation,
            'min_correlation': min_correlation,
            'risk_level': risk_level,
            'details': correlations
        }
        
    def _get_sgp_correlation(self, leg1: Dict[str, Any], leg2: Dict[str, Any]) -> float:
        """Get correlation between two SGP legs."""
        
        # Same player different props
        if leg1['player_id'] == leg2['player_id']:
            prop1, prop2 = leg1['prop_type'], leg2['prop_type']
            
            # High correlation for same player stats
            if prop1 == 'receiving_yards' and prop2 == 'receptions':
                return 0.85
            elif prop1 == 'receiving_yards' and prop2 == 'receiving_tds':
                return 0.50
            elif prop1 == 'rushing_yards' and prop2 == 'rushing_tds':
                return 0.55
            elif prop1 == 'passing_yards' and prop2 == 'passing_tds':
                return 0.70
            else:
                return 0.30  # Default same-player correlation
        
        # Same team different players
        if leg1.get('team') == leg2.get('team') and leg1.get('team') != 'unknown':
            return self._get_same_team_correlation(leg1, leg2)
        
        # Different teams (opponents)
        if leg1.get('opponent') == leg2.get('team') or leg1.get('team') == leg2.get('opponent'):
            return self._get_opponent_correlation(leg1, leg2)
        
        # Default (different games - shouldn't happen in SGP)
        return 0.0
        
    def _get_same_team_correlation(self, leg1: Dict[str, Any], leg2: Dict[str, Any]) -> float:
        """Get correlation between same-team players."""
        
        prop1, prop2 = leg1['prop_type'], leg2['prop_type']
        pos1, pos2 = leg1.get('position', ''), leg2.get('position', '')
        
        # QB to WR
        if pos1 == 'QB' and pos2 == 'WR':
            if prop1 == 'passing_yards' and prop2 == 'receiving_yards':
                return 0.45
            elif prop1 == 'passing_tds' and prop2 == 'receiving_tds':
                return 0.65
        
        # QB to RB (negative for passing vs rushing)
        if pos1 == 'QB' and pos2 == 'RB':
            if prop1 == 'passing_yards' and prop2 == 'rushing_yards':
                return -0.40
            elif prop1 == 'passing_tds' and prop2 == 'rushing_tds':
                return -0.30
        
        # WR to WR (target share conflict)
        if pos1 == 'WR' and pos2 == 'WR':
            if prop1 == 'receiving_yards' and prop2 == 'receiving_yards':
                return -0.25
            elif prop1 == 'targets' and prop2 == 'targets':
                return -0.40
        
        # RB to RB (snap share conflict)
        if pos1 == 'RB' and pos2 == 'RB':
            return -0.50
        
        return 0.10  # Default same-team correlation
        
    def _get_opponent_correlation(self, leg1: Dict[str, Any], leg2: Dict[str, Any]) -> float:
        """Get correlation between opposing players."""
        
        # Opposing QBs
        if leg1.get('position') == 'QB' and leg2.get('position') == 'QB':
            return -0.15  # Slight negative (game flow)
        
        # Game script correlations
        if leg1['prop_type'] == 'passing_yards' and leg2['prop_type'] == 'rushing_yards':
            return -0.20  # Passing vs rushing game script
        
        return -0.10  # Default opponent correlation
        
    def _check_restricted_combinations(
        self,
        legs: List[Dict[str, Any]],
        book_config: Dict[str, Any]
    ) -> List[str]:
        """Check for sportsbook-specific restricted combinations."""
        
        errors = []
        restricted = book_config.get('restricted_combinations', [])
        
        for restriction in restricted:
            prop1, prop2, condition = restriction
            
            # Check if this restriction applies
            matching_legs = []
            for leg in legs:
                if leg['prop_type'] in [prop1, prop2]:
                    matching_legs.append(leg)
            
            if len(matching_legs) >= 2:
                if condition == 'same_team':
                    same_team_count = 0
                    for i, leg_a in enumerate(matching_legs):
                        for leg_b in matching_legs[i+1:]:
                            if leg_a.get('team') == leg_b.get('team'):
                                same_team_count += 1
                    
                    if same_team_count > 0:
                        errors.append(f"Restricted combination: {prop1} + {prop2} for same team")
        
        return errors
        
    def _generate_optimization_suggestions(
        self,
        legs: List[Dict[str, Any]],
        correlation_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate suggestions to optimize the SGP."""
        
        suggestions = []
        
        # High correlation warnings
        for detail in correlation_analysis.get('details', []):
            if detail['correlation'] > 0.7:
                suggestions.append(
                    f"Consider removing highly correlated legs: {detail['leg1']} and {detail['leg2']} (r={detail['correlation']:.2f})"
                )
            elif detail['correlation'] < -0.5:
                suggestions.append(
                    f"High negative correlation detected: {detail['leg1']} and {detail['leg2']} (r={detail['correlation']:.2f})"
                )
        
        # Diversification suggestions
        teams = [leg.get('team') for leg in legs]
        unique_teams = set(team for team in teams if team)
        
        if len(unique_teams) == 1:
            suggestions.append("Consider adding props from opposing team for better diversification")
        
        # Position diversity
        positions = [leg.get('position') for leg in legs if leg.get('position')]
        if len(set(positions)) < len(legs) // 2:
            suggestions.append("Consider diversifying across different player positions")
        
        return suggestions
        
    def calculate_sgp_fair_odds(
        self,
        legs: List[Dict[str, Any]],
        individual_probabilities: List[float]
    ) -> Dict[str, Any]:
        """Calculate fair odds for SGP with correlation adjustments."""
        
        if len(individual_probabilities) != len(legs):
            raise ValueError("Number of probabilities must match number of legs")
        
        # Independent probability (baseline)
        independent_prob = np.prod(individual_probabilities)
        
        # Correlation adjustment
        correlation_analysis = self._analyze_sgp_correlations(legs)
        avg_correlation = correlation_analysis['average_correlation']
        
        # Adjust joint probability based on average correlation
        if avg_correlation > 0:
            # Positive correlation increases joint probability
            correlation_adjustment = 1 + (avg_correlation * 0.15)
        else:
            # Negative correlation decreases joint probability
            correlation_adjustment = 1 + (avg_correlation * 0.10)
        
        adjusted_prob = independent_prob * correlation_adjustment
        adjusted_prob = max(0.001, min(0.999, adjusted_prob))  # Bound probability
        
        # Convert to odds
        if adjusted_prob > 0:
            fair_decimal_odds = 1 / adjusted_prob
            fair_american_odds = self._decimal_to_american(fair_decimal_odds)
        else:
            fair_decimal_odds = 1000  # Very unlikely
            fair_american_odds = 99900
        
        return {
            'independent_probability': independent_prob,
            'adjusted_probability': adjusted_prob,
            'correlation_adjustment': correlation_adjustment,
            'fair_decimal_odds': fair_decimal_odds,
            'fair_american_odds': fair_american_odds,
            'correlation_analysis': correlation_analysis
        }
        
    def _decimal_to_american(self, decimal_odds: float) -> int:
        """Convert decimal odds to American odds."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
            
    def optimize_sgp_construction(
        self,
        available_edges: List[Dict[str, Any]],
        target_legs: int = 3,
        max_correlation: float = 0.6
    ) -> Dict[str, Any]:
        """Optimize SGP construction from available edges."""
        
        from itertools import combinations
        
        if len(available_edges) < target_legs:
            return {'error': 'Not enough edges for target legs', 'sgp': None}
        
        best_sgp = None
        best_score = -1
        
        # Try all combinations of target_legs size
        for combo in combinations(available_edges, target_legs):
            legs = list(combo)
            
            # Validate combination
            validation = self.validate_sgp(legs)
            if not validation['is_valid']:
                continue
            
            # Check correlation constraint
            correlation_analysis = validation.get('correlation_score', 0)
            if correlation_analysis > max_correlation:
                continue
            
            # Calculate combined EV and probability
            individual_probs = [leg.get('fair_probability', 0.5) for leg in legs]
            sgp_analysis = self.calculate_sgp_fair_odds(legs, individual_probs)
            
            # Score based on EV, correlation, and confidence
            individual_evs = [leg.get('expected_value', 0) for leg in legs]
            combined_ev = np.prod([1 + ev for ev in individual_evs]) - 1  # Compound EV approximation
            
            score = (
                combined_ev * 0.5 +
                correlation_analysis * 0.3 +
                np.mean([leg.get('confidence', 0.5) for leg in legs]) * 0.2
            )
            
            if score > best_score:
                best_score = score
                best_sgp = {
                    'legs': legs,
                    'validation': validation,
                    'sgp_analysis': sgp_analysis,
                    'score': score,
                    'combined_ev': combined_ev
                }
        
        return best_sgp if best_sgp else {'error': 'No valid SGP found', 'sgp': None}


def demo_sgp_validation() -> Dict[str, Any]:
    """Demonstrate SGP validation with sample legs."""
    
    # Sample SGP legs
    sample_legs = [
        {
            'player_id': 1,
            'player_name': 'Travis Kelce',
            'prop_type': 'receiving_yards',
            'position': 'TE',
            'team': 'KC',
            'opponent': 'DEN',
            'market_line': 72.5,
            'side': 'over'
        },
        {
            'player_id': 1,
            'player_name': 'Travis Kelce', 
            'prop_type': 'receptions',
            'position': 'TE',
            'team': 'KC',
            'opponent': 'DEN',
            'market_line': 6.5,
            'side': 'over'
        },
        {
            'player_id': 2,
            'player_name': 'Patrick Mahomes',
            'prop_type': 'passing_yards',
            'position': 'QB',
            'team': 'KC', 
            'opponent': 'DEN',
            'market_line': 267.5,
            'side': 'over'
        }
    ]
    
    validator = SGPValidator()
    
    logger.info("Running SGP validation demo")
    
    # Validate the SGP
    validation_result = validator.validate_sgp(sample_legs, sportsbook='draftkings')
    
    # Calculate fair odds
    individual_probs = [0.58, 0.61, 0.55]  # Sample probabilities
    sgp_analysis = validator.calculate_sgp_fair_odds(sample_legs, individual_probs)
    
    demo_result = {
        'sample_sgp': sample_legs,
        'validation': validation_result,
        'fair_odds_analysis': sgp_analysis,
        'recommendation': 'Valid SGP with moderate correlation' if validation_result['is_valid'] else 'Invalid SGP'
    }
    
    logger.info(f"SGP validation complete: {'Valid' if validation_result['is_valid'] else 'Invalid'}")
    
    return demo_result