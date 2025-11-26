"""Fair value calculator using ML predictions."""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from ..models import XGBoostPropsModel, NeuralNetModel
from ..utils.odds import american_to_decimal, decimal_to_american, implied_probability
from ..database import Game, Player, Prop
from ..features.ml_features import MLFeatureEngineer

logger = logging.getLogger(__name__)


class FairValueCalculator:
    """Calculate fair value lines using ML model predictions."""
    
    def __init__(self):
        self.models = {}
        self.ml_engineer = MLFeatureEngineer()
        self.model_weights = {
            'xgboost': 0.6,
            'neural_net': 0.4
        }
        
    def load_models(self, model_paths: Dict[str, str]) -> None:
        """Load trained ML models."""
        for prop_type, path in model_paths.items():
            try:
                # Load XGBoost model
                xgb_model = XGBoostPropsModel(prop_type)
                xgb_model.load_model(path)
                
                # Load Neural Network model (if available)
                nn_model = NeuralNetModel(prop_type)
                # nn_model.load_model(path.replace('xgboost', 'neural'))
                
                self.models[prop_type] = {
                    'xgboost': xgb_model,
                    'neural_net': nn_model
                }
                
                logger.info(f"Loaded models for {prop_type}")
                
            except Exception as e:
                logger.warning(f"Could not load model for {prop_type}: {e}")
                
    def calculate_fair_line(
        self,
        player_data: Dict[str, Any],
        prop_type: str,
        confidence_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """Calculate fair value line for a player prop."""
        
        if prop_type not in self.models:
            logger.warning(f"No model available for {prop_type}")
            return self._fallback_fair_line(player_data, prop_type)
        
        try:
            # Convert to DataFrame for feature engineering
            df = pd.DataFrame([player_data])
            
            # Create ML features
            feature_df = self.ml_engineer.create_ml_features(df, prop_type)
            
            predictions = {}
            confidences = {}
            
            # Get predictions from each model
            for model_name, model in self.models[prop_type].items():
                if model.is_trained:
                    try:
                        pred_df = model.predict(feature_df)
                        pred_col = f'{prop_type}_{model_name}_prediction' if model_name == 'neural_net' else f'{prop_type}_prediction'
                        conf_col = f'{prop_type}_{model_name}_confidence' if model_name == 'neural_net' else f'{prop_type}_confidence'
                        
                        if pred_col in pred_df.columns:
                            predictions[model_name] = pred_df[pred_col].iloc[0]
                            confidences[model_name] = pred_df[conf_col].iloc[0] if conf_col in pred_df.columns else 0.7
                            
                    except Exception as e:
                        logger.warning(f"Error getting prediction from {model_name}: {e}")
            
            if not predictions:
                return self._fallback_fair_line(player_data, prop_type)
            
            # Ensemble prediction using weighted average
            total_weight = 0
            weighted_prediction = 0
            weighted_confidence = 0
            
            for model_name, prediction in predictions.items():
                weight = self.model_weights.get(model_name, 0.5)
                confidence = confidences.get(model_name, 0.7)
                
                # Weight by model confidence
                adjusted_weight = weight * confidence
                
                weighted_prediction += prediction * adjusted_weight
                weighted_confidence += confidence * adjusted_weight
                total_weight += adjusted_weight
            
            if total_weight > 0:
                fair_line = weighted_prediction / total_weight
                ensemble_confidence = weighted_confidence / total_weight
            else:
                return self._fallback_fair_line(player_data, prop_type)
            
            # Calculate fair odds (both sides)
            fair_odds = self._line_to_fair_odds(fair_line, prop_type)
            
            result = {
                'prop_type': prop_type,
                'fair_line': round(fair_line, 1),
                'fair_over_odds': fair_odds['over_odds'],
                'fair_under_odds': fair_odds['under_odds'],
                'fair_over_probability': fair_odds['over_prob'],
                'fair_under_probability': fair_odds['under_prob'],
                'confidence': round(ensemble_confidence, 3),
                'model_predictions': predictions,
                'model_confidences': confidences,
                'method': 'ml_ensemble'
            }
            
            logger.info(f"Calculated fair line for {prop_type}: {fair_line} (confidence: {ensemble_confidence:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating fair line for {prop_type}: {e}")
            return self._fallback_fair_line(player_data, prop_type)
    
    def _line_to_fair_odds(self, fair_line: float, prop_type: str) -> Dict[str, Any]:
        """Convert fair line prediction to fair odds on both sides."""
        
        # Use historical variance to estimate probability distribution
        if prop_type == 'receiving_yards':
            # Typical standard deviation for receiving yards
            std_dev = max(15.0, fair_line * 0.25)
        elif prop_type == 'receiving_tds':
            std_dev = max(0.5, fair_line * 0.4)
        elif prop_type == 'receptions':
            std_dev = max(2.0, fair_line * 0.3)
        else:
            # Default 25% coefficient of variation
            std_dev = max(1.0, fair_line * 0.25)
        
        # Calculate probabilities using normal distribution
        # P(X > line) for over
        from scipy.stats import norm
        over_prob = 1 - norm.cdf(fair_line, loc=fair_line, scale=std_dev)
        under_prob = norm.cdf(fair_line, loc=fair_line, scale=std_dev)
        
        # Adjust for typical betting lines (slightly favor under)
        over_prob *= 0.48  # Account for juice/vig
        under_prob *= 0.48
        
        # Convert to fair odds (no vig)
        if over_prob > 0:
            fair_over_odds = decimal_to_american(1 / over_prob)
        else:
            fair_over_odds = 200  # Default if calculation fails
            
        if under_prob > 0:
            fair_under_odds = decimal_to_american(1 / under_prob)
        else:
            fair_under_odds = 200
        
        return {
            'over_prob': round(over_prob, 4),
            'under_prob': round(under_prob, 4),
            'over_odds': fair_over_odds,
            'under_odds': fair_under_odds
        }
    
    def _fallback_fair_line(self, player_data: Dict[str, Any], prop_type: str) -> Dict[str, Any]:
        """Fallback method when ML models aren't available."""
        logger.info(f"Using fallback method for {prop_type}")
        
        # Simple heuristic-based fair line calculation
        if prop_type == 'receiving_yards':
            # Use recent averages with basic adjustments
            recent_avg = player_data.get('receiving_yards_4game_avg', 60)
            matchup_adj = self._get_matchup_adjustment(player_data)
            fair_line = recent_avg * matchup_adj
            
        elif prop_type == 'receptions':
            recent_avg = player_data.get('receptions_4game_avg', 4.5)
            matchup_adj = self._get_matchup_adjustment(player_data)
            fair_line = recent_avg * matchup_adj
            
        elif prop_type == 'receiving_tds':
            recent_avg = player_data.get('receiving_tds_4game_avg', 0.5)
            matchup_adj = self._get_matchup_adjustment(player_data, td_prop=True)
            fair_line = recent_avg * matchup_adj
            
        else:
            fair_line = 50.0  # Default fallback
        
        fair_odds = self._line_to_fair_odds(fair_line, prop_type)
        
        return {
            'prop_type': prop_type,
            'fair_line': round(fair_line, 1),
            'fair_over_odds': fair_odds['over_odds'],
            'fair_under_odds': fair_odds['under_odds'],
            'fair_over_probability': fair_odds['over_prob'],
            'fair_under_probability': fair_odds['under_prob'],
            'confidence': 0.5,  # Lower confidence for fallback
            'model_predictions': {},
            'model_confidences': {},
            'method': 'heuristic_fallback'
        }
    
    def _get_matchup_adjustment(self, player_data: Dict[str, Any], td_prop: bool = False) -> float:
        """Calculate matchup adjustment factor."""
        adjustment = 1.0
        
        # Home field advantage
        if player_data.get('is_home', 0):
            adjustment *= 1.05
        
        # Opponent strength
        opp_rank = player_data.get('opp_def_rank', 16)
        if opp_rank > 24:  # Weak defense
            adjustment *= 1.1
        elif opp_rank < 8:  # Strong defense
            adjustment *= 0.9
        
        # Weather impact (for passing props)
        weather_impact = player_data.get('weather_impact_passing', 0)
        if weather_impact > 0.3:
            adjustment *= 0.95
        
        # Pace adjustment
        pace = player_data.get('team_pace', 65)
        if pace > 70:  # Fast pace
            adjustment *= 1.05
        elif pace < 60:  # Slow pace
            adjustment *= 0.95
        
        # TD props get additional red zone adjustments
        if td_prop:
            red_zone_eff = player_data.get('team_red_zone_efficiency', 0.5)
            adjustment *= (0.8 + red_zone_eff * 0.4)
        
        return adjustment
    
    def calculate_multiple_fair_lines(
        self,
        players_data: List[Dict[str, Any]],
        prop_types: List[str]
    ) -> pd.DataFrame:
        """Calculate fair lines for multiple players and prop types."""
        
        results = []
        
        for player_data in players_data:
            player_id = player_data.get('player_id', 'unknown')
            player_name = player_data.get('player_name', 'Unknown')
            
            for prop_type in prop_types:
                fair_line_data = self.calculate_fair_line(player_data, prop_type)
                
                result = {
                    'player_id': player_id,
                    'player_name': player_name,
                    **fair_line_data
                }
                
                results.append(result)
        
        return pd.DataFrame(results)
    
    def compare_to_market(
        self,
        fair_lines: pd.DataFrame,
        market_lines: pd.DataFrame
    ) -> pd.DataFrame:
        """Compare fair lines to market lines to identify edges."""
        
        # Merge fair lines with market lines
        comparison = fair_lines.merge(
            market_lines,
            on=['player_id', 'prop_type'],
            how='inner',
            suffixes=('_fair', '_market')
        )
        
        if comparison.empty:
            logger.warning("No matching data between fair lines and market lines")
            return pd.DataFrame()
        
        # Calculate edge metrics
        comparison['line_difference'] = (
            comparison['fair_line'] - comparison['market_line']
        )
        
        comparison['over_edge'] = (
            comparison['fair_over_probability'] - 
            implied_probability(comparison['market_over_odds'])
        )
        
        comparison['under_edge'] = (
            comparison['fair_under_probability'] - 
            implied_probability(comparison['market_under_odds'])
        )
        
        # Expected value calculations
        comparison['over_ev'] = (
            comparison['fair_over_probability'] * 
            (american_to_decimal(comparison['market_over_odds']) - 1) -
            (1 - comparison['fair_over_probability'])
        )
        
        comparison['under_ev'] = (
            comparison['fair_under_probability'] * 
            (american_to_decimal(comparison['market_under_odds']) - 1) -
            (1 - comparison['fair_under_probability'])
        )
        
        # Best bet side
        comparison['best_side'] = np.where(
            comparison['over_ev'] > comparison['under_ev'], 'over', 'under'
        )
        
        comparison['best_ev'] = np.maximum(
            comparison['over_ev'], comparison['under_ev']
        )
        
        # Edge strength score (combines EV and confidence)
        comparison['edge_strength'] = (
            comparison['best_ev'] * comparison['confidence']
        )
        
        logger.info(f"Compared {len(comparison)} fair lines to market lines")
        
        return comparison.sort_values('edge_strength', ascending=False)


def create_sample_fair_value_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create sample data for fair value testing."""
    
    # Sample player data
    players_data = [
        {
            'player_id': 1,
            'player_name': 'Travis Kelce',
            'receiving_yards_4game_avg': 75.2,
            'receptions_4game_avg': 6.8,
            'receiving_tds_4game_avg': 0.8,
            'is_home': 1,
            'opp_def_rank': 28,
            'weather_impact_passing': 0.1,
            'team_pace': 68
        },
        {
            'player_id': 2,
            'player_name': 'Tyreek Hill',
            'receiving_yards_4game_avg': 82.5,
            'receptions_4game_avg': 7.2,
            'receiving_tds_4game_avg': 0.6,
            'is_home': 0,
            'opp_def_rank': 12,
            'weather_impact_passing': 0.0,
            'team_pace': 71
        }
    ]
    
    # Sample market data
    market_data = [
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
    ]
    
    return pd.DataFrame(players_data), pd.DataFrame(market_data)