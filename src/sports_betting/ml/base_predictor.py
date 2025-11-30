"""
Base predictor class for all stat types.

Provides common functionality for loading models and generating predictions.
"""
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import logging

import nfl_data_py as nfl

logger = logging.getLogger(__name__)


class BaseStatPredictor(ABC):
    """Abstract base class for stat predictors."""

    # Subclasses must define these
    stat_type: str = "base"
    model_filename: str = "base_v2.pkl"
    target_column: str = "target"
    position_filter: List[str] = []

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize predictor with trained model.

        Args:
            model_path: Path to model pickle file. If None, uses default.
        """
        if model_path is None:
            model_path = (
                Path(__file__).parent.parent.parent.parent
                / "models"
                / self.model_filename
            )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run: python scripts/train_models_v2.py"
            )

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.features = model_data["features"]
        self.metrics = model_data.get("metrics", {})
        self.model_path = model_path

        logger.info(f"Loaded {self.stat_type} model from {model_path.name}")
        logger.info(f"  Test R²: {self.metrics.get('test_r2', 'N/A'):.4f}")
        logger.info(f"  Features: {len(self.features)}")

    @abstractmethod
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build prediction features from raw data.

        Subclasses implement stat-specific feature engineering.
        Must match the features used during training.

        Args:
            df: Raw weekly data from nfl_data_py

        Returns:
            DataFrame with features added
        """
        pass

    def load_weekly_data(self, seasons: List[int]) -> pd.DataFrame:
        """Load weekly player data from nfl_data_py."""
        logger.info(f"Loading weekly data for seasons {seasons}")
        weekly = nfl.import_weekly_data(seasons)
        logger.info(f"  Loaded {len(weekly)} records")
        return weekly

    def predict_for_week(
        self, season: int, week: int, seasons_for_history: Optional[List[int]] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Generate predictions for all players in a given week.

        Args:
            season: NFL season year
            week: Week number
            seasons_for_history: Seasons to load for building historical features

        Returns:
            Tuple of (predictions DataFrame, metadata dict)
        """
        logger.info(f"Generating {self.stat_type} predictions for {season} Week {week}")

        # Determine which seasons to load for historical data
        if seasons_for_history is None:
            # Use last few seasons for feature building
            current_year = 2024  # Last year with complete data
            seasons_for_history = [2022, 2023, 2024]
            if season <= current_year and season not in seasons_for_history:
                seasons_for_history.append(season)

        # Load data
        weekly = self.load_weekly_data(seasons_for_history)

        # Build features
        df = self.build_features(weekly)

        if len(df) == 0:
            logger.warning(f"No data after feature building for {self.stat_type}")
            return pd.DataFrame(), {"error": "No data available"}

        # For predictions, we need to create pseudo-records for the target week
        # using the most recent available data
        predictions_df = self._prepare_prediction_data(df, season, week)

        if len(predictions_df) == 0:
            logger.warning(f"No players to predict for {self.stat_type}")
            return pd.DataFrame(), {"error": "No players to predict"}

        # Ensure all features are present
        missing_features = [f for f in self.features if f not in predictions_df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            for f in missing_features:
                predictions_df[f] = 0

        # Generate predictions
        X = predictions_df[self.features].fillna(0)
        predictions = self.model.predict(X.values)

        # Build results - use player_display_name for full names (matches our Player table)
        name_col = "player_display_name" if "player_display_name" in predictions_df.columns else "player_name"
        results = predictions_df[
            [name_col, "player_id", "position", "recent_team"]
        ].copy()
        results = results.rename(columns={name_col: "player_name"})
        results["predicted"] = predictions
        results["stat_type"] = self.stat_type

        # Add confidence based on data recency
        results["confidence"] = predictions_df.get("data_confidence", 0.7)

        # Sort by prediction
        results = results.sort_values("predicted", ascending=False)

        metadata = {
            "stat_type": self.stat_type,
            "season": season,
            "week": week,
            "predictions_count": len(results),
            "model_r2": self.metrics.get("test_r2", 0),
        }

        logger.info(f"Generated {len(results)} {self.stat_type} predictions")

        return results, metadata

    def _prepare_prediction_data(
        self, df: pd.DataFrame, season: int, week: int
    ) -> pd.DataFrame:
        """
        Prepare data for making predictions.

        Uses most recent data for each player to build prediction features.
        """
        # Get most recent record for each player
        # This contains their rolling averages up to their last game
        df_sorted = df.sort_values(["player_id", "season", "week"])
        latest = df_sorted.groupby("player_id").last().reset_index()

        # Filter to relevant positions
        if self.position_filter:
            latest = latest[latest["position"].isin(self.position_filter)]

        # Filter to players who have played recently (within last season)
        max_season = df["season"].max()
        latest = latest[latest["season"] >= max_season - 1]

        # Update week_num for the target week
        if "week_num" in latest.columns:
            latest["week_num"] = week

        # Set confidence based on how recent the data is
        latest["data_confidence"] = np.where(
            latest["season"] == max_season,
            0.8,  # Current season data
            0.5,  # Older data
        )

        return latest

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "stat_type": self.stat_type,
            "model_file": self.model_filename,
            "features": self.features,
            "metrics": self.metrics,
            "num_features": len(self.features),
        }
