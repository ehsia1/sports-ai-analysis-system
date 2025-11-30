"""
Stat-specific predictor implementations.

Each predictor class handles feature engineering for its specific stat type,
matching the features used during model training.
"""
import pandas as pd
import numpy as np
from typing import List
import logging

from .base_predictor import BaseStatPredictor

logger = logging.getLogger(__name__)


class RushingYardsPredictor(BaseStatPredictor):
    """Predictor for rushing yards."""

    stat_type = "rushing_yards"
    model_filename = "rushing_yards_v2.pkl"
    target_column = "rushing_yards"
    position_filter = ["RB", "QB", "WR"]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build rushing yards features from weekly data."""
        logger.info("Building rushing yards features...")

        # Filter to players with rushing stats
        df = df[df["rushing_yards"].notna() & (df["rushing_yards"] >= 0)].copy()
        if self.position_filter:
            df = df[df["position"].isin(self.position_filter)].copy()

        if len(df) == 0:
            return df

        # Sort for rolling calculations
        df = df.sort_values(["player_id", "season", "week"])

        # Rolling rushing yards (shifted - use past games only)
        for window in [3, 5]:
            df[f"rush_yards_last_{window}"] = df.groupby("player_id")[
                "rushing_yards"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f"rush_yards_std_{window}"] = df.groupby("player_id")[
                "rushing_yards"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())

        # Rolling carries
        df["carries"] = df["carries"].fillna(0)
        for window in [3, 5]:
            df[f"carries_last_{window}"] = df.groupby("player_id")["carries"].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )

        # Yards per carry (historical)
        df["ypc_career"] = (
            df.groupby("player_id")
            .apply(
                lambda x: x["rushing_yards"].shift(1).expanding().sum()
                / x["carries"].shift(1).expanding().sum().clip(lower=1),
                include_groups=False,
            )
            .reset_index(level=0, drop=True)
        )
        df["ypc_career"] = df["ypc_career"].fillna(4.0)

        # Position encoding
        pos_map = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
        df["position_encoded"] = df["position"].map(pos_map).fillna(1)

        # Week number
        df["week_num"] = df["week"]

        # Snap percentage (default if not available)
        if "snap_pct" not in df.columns:
            df["snap_pct"] = 0.5

        # Fill NaN in rolling columns
        for col in df.columns:
            if "last_" in col or "_std_" in col:
                df[col] = df[col].fillna(0)

        logger.info(f"Built features for {len(df)} rushing records")
        return df


class PassingYardsPredictor(BaseStatPredictor):
    """Predictor for passing yards."""

    stat_type = "passing_yards"
    model_filename = "passing_yards_v2.pkl"
    target_column = "passing_yards"
    position_filter = ["QB"]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build passing yards features from weekly data."""
        logger.info("Building passing yards features...")

        # Filter to QBs with passing stats
        df = df[df["passing_yards"].notna() & (df["passing_yards"] > 0)].copy()
        df = df[df["position"] == "QB"].copy()

        if len(df) == 0:
            return df

        df = df.sort_values(["player_id", "season", "week"])

        # Rolling passing yards
        for window in [3, 5]:
            df[f"pass_yards_last_{window}"] = df.groupby("player_id")[
                "passing_yards"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f"pass_yards_std_{window}"] = df.groupby("player_id")[
                "passing_yards"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())

        # Rolling attempts
        df["attempts"] = df["attempts"].fillna(0)
        for window in [3, 5]:
            df[f"attempts_last_{window}"] = df.groupby("player_id")[
                "attempts"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Historical completion percentage
        df["completions"] = df["completions"].fillna(0)
        df["comp_pct_last_5"] = (
            df.groupby("player_id")
            .apply(
                lambda x: x["completions"].shift(1).rolling(5, min_periods=1).sum()
                / x["attempts"].shift(1).rolling(5, min_periods=1).sum().clip(lower=1),
                include_groups=False,
            )
            .reset_index(level=0, drop=True)
        )
        df["comp_pct_last_5"] = df["comp_pct_last_5"].fillna(0.65)

        # Yards per attempt (historical)
        df["ypa_career"] = (
            df.groupby("player_id")
            .apply(
                lambda x: x["passing_yards"].shift(1).expanding().sum()
                / x["attempts"].shift(1).expanding().sum().clip(lower=1),
                include_groups=False,
            )
            .reset_index(level=0, drop=True)
        )
        df["ypa_career"] = df["ypa_career"].fillna(7.0)

        # Position encoded (all QBs = 0)
        df["position_encoded"] = 0

        df["week_num"] = df["week"]

        # Fill NaN
        for col in df.columns:
            if "last_" in col or "_std_" in col:
                df[col] = df[col].fillna(0)

        logger.info(f"Built features for {len(df)} passing records")
        return df


class ReceptionsPredictor(BaseStatPredictor):
    """Predictor for receptions."""

    stat_type = "receptions"
    model_filename = "receptions_v2.pkl"
    target_column = "receptions"
    position_filter = ["WR", "TE", "RB"]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build receptions features from weekly data."""
        logger.info("Building receptions features...")

        # Filter to pass catchers
        df = df[df["receptions"].notna() & (df["receptions"] >= 0)].copy()
        if self.position_filter:
            df = df[df["position"].isin(self.position_filter)].copy()

        if len(df) == 0:
            return df

        df = df.sort_values(["player_id", "season", "week"])

        # Rolling receptions
        for window in [3, 5]:
            df[f"receptions_last_{window}"] = df.groupby("player_id")[
                "receptions"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
            df[f"receptions_std_{window}"] = df.groupby("player_id")[
                "receptions"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())

        # Rolling targets
        df["targets"] = df["targets"].fillna(0)
        for window in [3, 5]:
            df[f"targets_last_{window}"] = df.groupby("player_id")[
                "targets"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Historical target share
        df["target_share"] = df["target_share"].fillna(0)
        for window in [3, 5]:
            df[f"target_share_last_{window}"] = df.groupby("player_id")[
                "target_share"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

        # Historical catch rate
        df["catch_rate_last_5"] = (
            df.groupby("player_id")
            .apply(
                lambda x: x["receptions"].shift(1).rolling(5, min_periods=1).sum()
                / x["targets"].shift(1).rolling(5, min_periods=1).sum().clip(lower=1),
                include_groups=False,
            )
            .reset_index(level=0, drop=True)
        )
        df["catch_rate_last_5"] = df["catch_rate_last_5"].fillna(0.65)

        # Position encoding
        pos_map = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
        df["position_encoded"] = df["position"].map(pos_map).fillna(2)

        df["week_num"] = df["week"]

        # Snap percentage
        if "snap_pct" not in df.columns:
            df["snap_pct"] = 0.5

        # Fill NaN
        for col in df.columns:
            if "last_" in col or "_std_" in col:
                df[col] = df[col].fillna(0)

        logger.info(f"Built features for {len(df)} reception records")
        return df


# Registry of all predictors
PREDICTOR_REGISTRY = {
    "rushing_yards": RushingYardsPredictor,
    "passing_yards": PassingYardsPredictor,
    "receptions": ReceptionsPredictor,
}


def get_predictor(stat_type: str) -> BaseStatPredictor:
    """Get a predictor instance by stat type."""
    if stat_type not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"Unknown stat type: {stat_type}. "
            f"Available: {list(PREDICTOR_REGISTRY.keys())}"
        )
    return PREDICTOR_REGISTRY[stat_type]()


def list_predictors() -> List[str]:
    """List available predictor stat types."""
    return list(PREDICTOR_REGISTRY.keys())
