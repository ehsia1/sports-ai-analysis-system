"""XGBoost model for NFL player prop predictions."""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    xgb = None
    print("XGBoost not available. Install with: pip install xgboost scikit-learn")

from ..base.predictor import BasePredictor

logger = logging.getLogger(__name__)


class XGBoostPropsModel(BasePredictor):
    """XGBoost model for predicting NFL player prop values."""
    
    def __init__(self, prop_type: str = "receiving_yards"):
        super().__init__(f"xgboost_{prop_type}", "1.0")
        self.prop_type = prop_type
        self.xgb_params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for XGBoost training/prediction."""
        logger.info(f"Preparing features for {self.prop_type} predictions")
        
        # Create rolling averages for key stats
        feature_df = df.copy()
        
        # Player-level rolling stats (last 4 games)
        player_stats = ['receiving_yards', 'receiving_tds', 'targets', 'receptions', 
                       'rushing_yards', 'rushing_tds', 'passing_yards', 'passing_tds']
        
        for stat in player_stats:
            if stat in feature_df.columns:
                # 4-game rolling average
                feature_df[f'{stat}_4game_avg'] = (
                    feature_df.groupby('player_id')[stat]
                    .rolling(window=4, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Season average
                feature_df[f'{stat}_season_avg'] = (
                    feature_df.groupby(['player_id', 'season'])[stat]
                    .expanding(min_periods=1)
                    .mean()
                    .reset_index([0, 1], drop=True)
                )
        
        # Team-level features
        team_stats = ['points_scored', 'points_allowed', 'total_yards', 'passing_yards_allowed']
        for stat in team_stats:
            if stat in feature_df.columns:
                feature_df[f'team_{stat}_4game_avg'] = (
                    feature_df.groupby('team')[stat]
                    .rolling(window=4, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
        
        # Opponent strength features
        if 'opponent' in feature_df.columns:
            # Opponent's defensive ranking (points allowed)
            opp_def_rank = (
                feature_df.groupby(['opponent', 'season'])['points_allowed']
                .mean()
                .rank(ascending=True)
                .to_dict()
            )
            feature_df['opp_def_rank'] = feature_df.apply(
                lambda x: opp_def_rank.get((x['opponent'], x['season']), 16), axis=1
            )
        
        # Game context features
        if 'game_date' in feature_df.columns:
            feature_df['is_home'] = (feature_df['home_team'] == feature_df['team']).astype(int)
            feature_df['week'] = feature_df['week']  # Already exists
            feature_df['is_primetime'] = (
                (feature_df['game_date'].dt.dayofweek == 6) |  # Sunday night
                (feature_df['game_date'].dt.dayofweek == 0) |  # Monday night  
                (feature_df['game_date'].dt.dayofweek == 3)    # Thursday night
            ).astype(int)
        
        # Weather features (if available)
        weather_features = ['temperature', 'wind_speed', 'precipitation']
        for feature in weather_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 70  # Default values
        
        # Select final feature set
        base_features = [
            'week', 'is_home', 'is_primetime',
            'temperature', 'wind_speed', 'precipitation'
        ]
        
        # Add rolling averages
        rolling_features = [col for col in feature_df.columns if '_4game_avg' in col or '_season_avg' in col]
        context_features = ['opp_def_rank'] if 'opp_def_rank' in feature_df.columns else []
        
        self.feature_columns = base_features + rolling_features + context_features
        
        # Ensure all features exist
        for feature in self.feature_columns:
            if feature not in feature_df.columns:
                logger.warning(f"Feature {feature} not found, setting to 0")
                feature_df[feature] = 0
        
        # Handle missing values
        feature_df[self.feature_columns] = feature_df[self.feature_columns].fillna(0)
        
        logger.info(f"Prepared {len(self.feature_columns)} features: {self.feature_columns}")
        
        return feature_df[self.feature_columns + [self.prop_type] if self.prop_type in feature_df.columns else self.feature_columns]
        
    def train(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """Train XGBoost model on player prop data."""
        if xgb is None:
            raise ImportError("XGBoost not available. Install with: pip install xgboost scikit-learn")
            
        if target_column is None:
            target_column = self.prop_type
            
        self.target_column = target_column
        
        logger.info(f"Training XGBoost model for {target_column}")
        
        # Prepare features
        feature_df = self.prepare_features(df)
        
        if target_column not in feature_df.columns:
            raise ValueError(f"Target column {target_column} not found in data")
        
        # Split features and target
        X = feature_df[self.feature_columns]
        y = feature_df[target_column]
        
        # Remove rows with missing target values
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            raise ValueError("No valid training data after removing missing values")
        
        logger.info(f"Training on {len(X)} samples with {len(self.feature_columns)} features")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(self.feature_columns)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_mean_absolute_error')
        metrics['cv_mae'] = -cv_scores.mean()
        metrics['cv_mae_std'] = cv_scores.std()
        
        self.is_trained = True
        
        logger.info(f"Model training complete. MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}")
        
        return metrics
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        logger.info(f"Making predictions for {len(df)} samples")
        
        # Prepare features
        feature_df = self.prepare_features(df)
        self.validate_features(feature_df)
        
        X = feature_df[self.feature_columns]
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create results dataframe
        results = df.copy()
        results[f'{self.prop_type}_prediction'] = predictions
        
        # Add confidence intervals (simple approach using feature importance)
        if hasattr(self.model, 'feature_importances_'):
            # Simple confidence based on prediction value and model uncertainty
            prediction_std = np.std(predictions)
            results[f'{self.prop_type}_confidence'] = np.clip(
                1.0 - (np.abs(predictions - predictions.mean()) / (2 * prediction_std)),
                0.1, 0.9
            )
        else:
            results[f'{self.prop_type}_confidence'] = 0.7  # Default confidence
            
        # Add prediction bounds
        results[f'{self.prop_type}_lower'] = predictions * 0.9
        results[f'{self.prop_type}_upper'] = predictions * 1.1
        
        logger.info(f"Predictions complete. Mean prediction: {predictions.mean():.2f}")
        
        return results[[col for col in results.columns if 'prediction' in col or 'confidence' in col or 'lower' in col or 'upper' in col]]
        
    def get_prop_line_prediction(self, player_stats: Dict[str, Any]) -> Tuple[float, float]:
        """Get a single prop line prediction with confidence."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        # Convert to DataFrame
        df = pd.DataFrame([player_stats])
        
        # Make prediction
        prediction_df = self.predict(df)
        
        prediction = prediction_df[f'{self.prop_type}_prediction'].iloc[0]
        confidence = prediction_df[f'{self.prop_type}_confidence'].iloc[0]
        
        return float(prediction), float(confidence)


def create_sample_training_data() -> pd.DataFrame:
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'player_id': np.random.randint(1, 101, n_samples),
        'season': np.random.choice([2022, 2023], n_samples),
        'week': np.random.randint(1, 19, n_samples),
        'team': np.random.choice(['KC', 'BUF', 'CIN', 'PHI', 'SF', 'DAL', 'MIA', 'MIN'], n_samples),
        'opponent': np.random.choice(['KC', 'BUF', 'CIN', 'PHI', 'SF', 'DAL', 'MIA', 'MIN'], n_samples),
        'home_team': np.random.choice(['KC', 'BUF', 'CIN', 'PHI', 'SF', 'DAL', 'MIA', 'MIN'], n_samples),
        'game_date': pd.date_range('2022-09-01', periods=n_samples, freq='D'),
        'receiving_yards': np.random.normal(60, 25, n_samples).clip(0, 200),
        'receiving_tds': np.random.poisson(0.5, n_samples),
        'targets': np.random.normal(6, 3, n_samples).clip(0, 15),
        'receptions': np.random.normal(4, 2, n_samples).clip(0, 12),
        'rushing_yards': np.random.normal(8, 10, n_samples).clip(0, 50),
        'rushing_tds': np.random.poisson(0.1, n_samples),
        'passing_yards': np.random.normal(250, 60, n_samples).clip(0, 450),
        'passing_tds': np.random.poisson(1.5, n_samples),
        'points_scored': np.random.normal(24, 8, n_samples).clip(0, 50),
        'points_allowed': np.random.normal(22, 7, n_samples).clip(0, 45),
        'total_yards': np.random.normal(350, 80, n_samples).clip(200, 600),
        'passing_yards_allowed': np.random.normal(240, 50, n_samples).clip(150, 400),
    }
    
    return pd.DataFrame(data)