"""Base feature engineering framework."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering."""

    def __init__(self, lookback_weeks: int = 10):
        self.lookback_weeks = lookback_weeks

    @abstractmethod
    def create_features(
        self,
        session: Session,
        player_id: int,
        game_id: int,
        target_week: int,
        season: int,
    ) -> Dict[str, Any]:
        """Create features for a player in a specific game."""
        pass

    def create_rolling_stats(
        self,
        data: pd.DataFrame,
        value_col: str,
        windows: List[int] = [3, 5, 10],
        prefix: str = "",
    ) -> Dict[str, float]:
        """Create rolling statistics for a given metric."""
        features = {}
        
        for window in windows:
            if len(data) >= window:
                window_data = data[value_col].tail(window)
                
                features[f"{prefix}avg_{window}g"] = window_data.mean()
                features[f"{prefix}std_{window}g"] = window_data.std()
                features[f"{prefix}min_{window}g"] = window_data.min()
                features[f"{prefix}max_{window}g"] = window_data.max()
                features[f"{prefix}trend_{window}g"] = self._calculate_trend(window_data)
            else:
                # Insufficient data
                features[f"{prefix}avg_{window}g"] = 0.0
                features[f"{prefix}std_{window}g"] = 0.0
                features[f"{prefix}min_{window}g"] = 0.0
                features[f"{prefix}max_{window}g"] = 0.0
                features[f"{prefix}trend_{window}g"] = 0.0
        
        return features

    def create_opponent_adjusted_stats(
        self,
        player_stats: pd.DataFrame,
        opponent_stats: pd.DataFrame,
        metric: str,
    ) -> Dict[str, float]:
        """Create opponent-adjusted statistics."""
        features = {}
        
        if len(player_stats) > 0 and len(opponent_stats) > 0:
            player_avg = player_stats[metric].mean()
            opponent_avg_allowed = opponent_stats[f"{metric}_allowed"].mean()
            league_avg = opponent_stats[f"{metric}_allowed"].mean()  # Simplified
            
            # Opponent adjustment factor
            adjustment_factor = opponent_avg_allowed / league_avg if league_avg > 0 else 1.0
            
            features[f"{metric}_adj"] = player_avg * adjustment_factor
            features[f"{metric}_vs_avg"] = player_avg - league_avg
            features[f"opp_{metric}_rank"] = self._calculate_rank(
                opponent_stats[f"{metric}_allowed"]
            )
        else:
            features[f"{metric}_adj"] = 0.0
            features[f"{metric}_vs_avg"] = 0.0
            features[f"opp_{metric}_rank"] = 16.0  # Middle rank
        
        return features

    def create_situational_features(
        self,
        game_data: Dict[str, Any],
        team_data: Dict[str, Any],
    ) -> Dict[str, float]:
        """Create situational features based on game context."""
        features = {}
        
        # Home/away
        features["is_home"] = float(game_data.get("is_home", 0))
        
        # Weather features
        features["temperature"] = game_data.get("temperature", 70.0)
        features["wind_speed"] = game_data.get("wind_speed", 0.0)
        features["precipitation"] = game_data.get("precipitation", 0.0)
        features["is_dome"] = float(game_data.get("is_dome", 0))
        
        # Game script features (if available)
        features["spread"] = game_data.get("spread", 0.0)
        features["total"] = game_data.get("total", 45.0)
        features["implied_pace"] = features["total"] / 60.0  # Rough pace estimate
        
        # Team strength
        features["team_strength"] = team_data.get("elo_rating", 1500.0)
        features["opp_team_strength"] = team_data.get("opp_elo_rating", 1500.0)
        features["strength_diff"] = features["team_strength"] - features["opp_team_strength"]
        
        return features

    def create_usage_features(
        self,
        player_stats: pd.DataFrame,
        team_stats: pd.DataFrame,
    ) -> Dict[str, float]:
        """Create usage and role-based features."""
        features = {}
        
        if len(player_stats) > 0 and len(team_stats) > 0:
            # Target/touch share
            features["target_share"] = (
                player_stats["targets"].sum() / team_stats["total_targets"].sum()
                if team_stats["total_targets"].sum() > 0 else 0.0
            )
            
            features["snap_share"] = (
                player_stats["snaps"].sum() / team_stats["total_snaps"].sum()
                if team_stats["total_snaps"].sum() > 0 else 0.0
            )
            
            # Red zone usage
            features["rz_target_share"] = (
                player_stats["rz_targets"].sum() / team_stats["total_rz_targets"].sum()
                if team_stats["total_rz_targets"].sum() > 0 else 0.0
            )
            
            # Air yards and depth
            features["avg_depth"] = player_stats["air_yards"].mean()
            features["adot"] = player_stats["air_yards"].sum() / player_stats["targets"].sum() if player_stats["targets"].sum() > 0 else 0.0
            
        else:
            features["target_share"] = 0.0
            features["snap_share"] = 0.0
            features["rz_target_share"] = 0.0
            features["avg_depth"] = 0.0
            features["adot"] = 0.0
        
        return features

    def create_efficiency_features(
        self,
        player_stats: pd.DataFrame,
    ) -> Dict[str, float]:
        """Create efficiency-based features."""
        features = {}
        
        if len(player_stats) > 0:
            # Catch rate
            total_targets = player_stats["targets"].sum()
            total_receptions = player_stats["receptions"].sum()
            features["catch_rate"] = total_receptions / total_targets if total_targets > 0 else 0.0
            
            # Yards per target/reception
            total_rec_yards = player_stats["receiving_yards"].sum()
            features["ypr"] = total_rec_yards / total_receptions if total_receptions > 0 else 0.0
            features["ypt"] = total_rec_yards / total_targets if total_targets > 0 else 0.0
            
            # Yards after catch
            features["yac_per_rec"] = (
                player_stats["yac"].sum() / total_receptions 
                if total_receptions > 0 and "yac" in player_stats.columns else 0.0
            )
            
            # TD efficiency
            total_tds = player_stats["receiving_tds"].sum()
            features["td_rate"] = total_tds / total_receptions if total_receptions > 0 else 0.0
            
        else:
            features["catch_rate"] = 0.0
            features["ypr"] = 0.0
            features["ypt"] = 0.0
            features["yac_per_rec"] = 0.0
            features["td_rate"] = 0.0
        
        return features

    def _calculate_trend(self, data: pd.Series) -> float:
        """Calculate trend using simple linear regression slope."""
        if len(data) < 2:
            return 0.0
        
        x = range(len(data))
        y = data.values
        
        # Simple linear regression
        n = len(data)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

    def _calculate_rank(self, data: pd.Series) -> float:
        """Calculate rank (1 = best, higher = worse)."""
        if len(data) == 0:
            return 16.0
        
        # Assume lower values are better (e.g., yards allowed)
        ranked = data.rank(ascending=True)
        return ranked.iloc[-1] if len(ranked) > 0 else 16.0