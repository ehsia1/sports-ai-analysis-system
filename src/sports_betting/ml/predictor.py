"""
Enhanced model predictor for receiving yards.

Uses the improved XGBoost model with proper feature engineering.

V2.0 - Adaptive Predictions:
- Automatically uses best available data source
- Adjusts confidence based on data quality
- Blends historical and current season appropriately
- Handles early season, mid-season, and new players
"""
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
import logging

from .feature_engineering import ReceivingYardsFeatureEngineer
from .data_sources import DataQuality

logger = logging.getLogger(__name__)


class ReceivingYardsPredictor:
    """Generate receiving yards predictions using enhanced model."""

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to model pickle file. If None, uses default enhanced model.
        """
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent.parent / 'models' / 'receiving_yards_enhanced.pkl'

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run: python scripts/train_enhanced_model.py"
            )

        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.features = model_data['features']
        self.engineer = model_data['engineer']
        self.metrics = model_data['metrics']

        print(f"✓ Loaded enhanced model from {model_path}")
        print(f"  Validation R²: {self.metrics['val_r2']:.4f}")
        print(f"  Test R²: {self.metrics['test_r2']:.4f}")
        print(f"  Features: {len(self.features)}")

    def predict_for_week(self, season: int, week: int) -> pd.DataFrame:
        """
        Generate predictions for all players in a given week.

        Args:
            season: NFL season year
            week: Week number

        Returns:
            DataFrame with predictions for each player
        """
        print(f"\nGenerating predictions for {season} Week {week}...")

        # Build features - need historical data to calculate rolling averages
        # For future seasons (2025+), only use data that's actually available
        print("Building features (this may take a minute)...")

        # Determine which seasons to load
        current_year = 2024  # Last year with complete data in nfl_data_py
        if season > current_year:
            # Predicting future season - use historical data up to current year
            seasons_to_load = [2020, 2021, 2022, 2023, 2024]
            print(f"⚠️  {season} data not in nfl_data_py yet, using 2020-2024 for features")
        else:
            # Predicting past season - can include target season
            seasons_to_load = [2020, 2021, 2022, 2023, 2024]
            if season not in seasons_to_load:
                seasons_to_load.append(season)

        df = self.engineer.build_features(seasons=seasons_to_load)

        # For future seasons, we need to manually calculate recent stats
        # from database since nfl_data_py doesn't have the current season yet
        if season > current_year:
            print(f"\n⚠️  For {season} predictions, using most recent 2024 stats as baseline")
            print("    (Real predictions would need live {season} data through Week {week-1})")

            # Get latest stats for each player from 2024
            latest_2024 = df[df['season'] == 2024].copy()

            # Group by player and get their latest stats
            latest_stats = latest_2024.groupby('player_id').last().reset_index()

            # Create pseudo-records for target week with carried-forward features
            target_df = latest_stats.copy()
            target_df['season'] = season
            target_df['week'] = week

            # Note: This is a limitation - we're using end-of-2024 stats
            # Real predictions need actual stats through current week
        else:
            # Filter to target week
            target_df = df[(df['season'] == season) & (df['week'] == week)].copy()

        if len(target_df) == 0:
            print(f"❌ No games found for {season} Week {week}")
            return pd.DataFrame()

        print(f"✓ Found {len(target_df)} player-games")

        # Check for missing features
        df_clean = target_df[self.features + ['player_name', 'player_id', 'position', 'recent_team']].dropna()

        if len(df_clean) < len(target_df):
            print(f"⚠️  Dropped {len(target_df) - len(df_clean)} records with missing features")

        if len(df_clean) == 0:
            print("❌ No complete feature data available")
            return pd.DataFrame()

        # Generate predictions
        X = df_clean[self.features]
        predictions = self.model.predict(X.values)

        # Build results dataframe
        results = df_clean[['player_name', 'player_id', 'position', 'recent_team']].copy()
        results['predicted_receiving_yards'] = predictions

        # Add key features for context
        results['targets_last_5'] = df_clean['targets_last_5']
        results['target_share_last_5'] = df_clean['target_share_last_5']
        results['rec_yards_last_5'] = df_clean['rec_yards_last_5']

        # Sort by prediction (highest first)
        results = results.sort_values('predicted_receiving_yards', ascending=False)

        print(f"✓ Generated {len(results)} predictions")

        if season > current_year:
            print(f"\n⚠️  NOTE: These predictions use end-of-2024 stats")
            print(f"    For accurate {season} Week {week} predictions, need actual stats through Week {week-1}")

        return results

    def predict_single_player(
        self,
        player_id: str,
        season: int,
        week: int
    ) -> Optional[float]:
        """
        Predict receiving yards for a single player.

        Args:
            player_id: Player's GSIS ID
            season: NFL season
            week: Week number

        Returns:
            Predicted receiving yards, or None if prediction not available
        """
        predictions = self.predict_for_week(season, week)

        if len(predictions) == 0:
            return None

        player_pred = predictions[predictions['player_id'] == player_id]

        if len(player_pred) == 0:
            return None

        return player_pred['predicted_receiving_yards'].iloc[0]

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            'features': self.features,
            'metrics': self.metrics,
            'num_features': len(self.features)
        }

    # =========================================================================
    # V2.0 ADAPTIVE PREDICTIONS
    # =========================================================================

    def predict_adaptive(
        self,
        season: int,
        week: int,
        use_current_season: bool = True
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Generate predictions using adaptive data sources.

        This method:
        1. Automatically determines best data source
        2. Builds features from available data
        3. Generates predictions with confidence scores
        4. Returns metadata about data quality

        Args:
            season: NFL season year
            week: Week number to predict
            use_current_season: Whether to use current season data

        Returns:
            Tuple of (predictions_df, metadata_dict)
        """
        print(f"\n{'=' * 80}")
        print(f"ADAPTIVE PREDICTION: {season} Week {week}")
        print(f"{'=' * 80}")

        # Build features adaptively
        df, metadata = self.engineer.build_features_adaptive(
            season=season,
            week=week,
            include_current_season=use_current_season
        )

        if len(df) == 0:
            print("❌ No features could be built")
            return pd.DataFrame(), metadata

        # Check which features we have
        available_features = [f for f in self.features if f in df.columns]
        missing_features = [f for f in self.features if f not in df.columns]

        if missing_features:
            print(f"\n⚠️ Missing features: {missing_features}")
            # Fill missing with zeros (will hurt predictions but allows fallback)
            for f in missing_features:
                df[f] = 0

        # Drop rows with NaN in required features
        feature_cols = self.features
        df_clean = df.dropna(subset=feature_cols)

        if len(df_clean) < len(df):
            dropped = len(df) - len(df_clean)
            print(f"  Dropped {dropped} rows with missing features")
            metadata['rows_dropped'] = dropped

        if len(df_clean) == 0:
            print("❌ No complete data for predictions")
            return pd.DataFrame(), metadata

        # Generate predictions
        X = df_clean[feature_cols]
        predictions = self.model.predict(X.values)

        # Build results
        results = df_clean[['player_name', 'player_id', 'position', 'recent_team']].copy()
        results['predicted_yards'] = predictions

        # Add confidence based on data quality
        if 'data_quality' in df_clean.columns:
            results['confidence'] = df_clean['data_quality'].values
        else:
            # Default confidence based on missing features
            base_confidence = len(available_features) / len(self.features)
            results['confidence'] = base_confidence

        # Add key context features
        for col in ['rec_yards_last_5', 'targets_last_5', 'target_share_last_5']:
            if col in df_clean.columns:
                results[col] = df_clean[col].values

        # Sort by prediction
        results = results.sort_values('predicted_yards', ascending=False)

        # Update metadata
        metadata['predictions_generated'] = len(results)
        metadata['avg_confidence'] = results['confidence'].mean()

        print(f"\n✓ Generated {len(results)} predictions")
        print(f"  Average confidence: {results['confidence'].mean():.2%}")

        return results, metadata

    def predict_with_confidence_bands(
        self,
        season: int,
        week: int
    ) -> pd.DataFrame:
        """
        Generate predictions with confidence intervals (p10, p50, p90).

        Uses data quality to widen bands for uncertain predictions.
        """
        predictions, metadata = self.predict_adaptive(season, week)

        if len(predictions) == 0:
            return predictions

        # Base variance from model training (if available)
        base_std = self.metrics.get('test_rmse', 20.0)

        # Adjust variance by confidence
        # Lower confidence = wider bands
        predictions['adjusted_std'] = base_std / predictions['confidence'].clip(lower=0.3)

        # Calculate percentiles using normal distribution
        from scipy import stats

        predictions['p50'] = predictions['predicted_yards']
        predictions['p10'] = predictions['predicted_yards'] - 1.28 * predictions['adjusted_std']
        predictions['p90'] = predictions['predicted_yards'] + 1.28 * predictions['adjusted_std']

        # Clip to non-negative
        predictions['p10'] = predictions['p10'].clip(lower=0)

        return predictions

    def get_prediction_for_player(
        self,
        player_name: str,
        season: int,
        week: int
    ) -> Optional[dict]:
        """
        Get detailed prediction for a specific player.

        Args:
            player_name: Player name (partial match supported)
            season: NFL season
            week: Week number

        Returns:
            Dict with prediction details, or None if not found
        """
        predictions, metadata = self.predict_adaptive(season, week)

        if len(predictions) == 0:
            return None

        # Find player (partial match)
        mask = predictions['player_name'].str.contains(player_name, case=False, na=False)
        player_preds = predictions[mask]

        if len(player_preds) == 0:
            return None

        row = player_preds.iloc[0]

        return {
            'player_name': row['player_name'],
            'team': row['recent_team'],
            'position': row['position'],
            'predicted_yards': row['predicted_yards'],
            'confidence': row['confidence'],
            'rec_yards_last_5': row.get('rec_yards_last_5', 0),
            'targets_last_5': row.get('targets_last_5', 0),
            'data_quality': 'high' if row['confidence'] > 0.7 else 'medium' if row['confidence'] > 0.4 else 'low',
            'metadata': metadata,
        }


if __name__ == "__main__":
    # Test the predictor
    predictor = ReceivingYardsPredictor()

    # Generate predictions for Week 14 2025
    predictions = predictor.predict_for_week(season=2025, week=14)

    if len(predictions) > 0:
        print("\n" + "=" * 80)
        print("TOP 20 PREDICTED RECEIVING YARDS (Week 14 2025)")
        print("=" * 80)

        top_20 = predictions.head(20)
        for idx, row in top_20.iterrows():
            print(f"{row['player_name']:20s} ({row['position']}, {row['recent_team']}) - {row['predicted_receiving_yards']:.1f} yards")

        print("\nPredictor ready to use!")
