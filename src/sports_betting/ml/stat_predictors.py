"""
Stat-specific predictor implementations.

Each predictor class handles feature engineering for its specific stat type,
matching the features used during model training.
"""
import pandas as pd
import numpy as np
from typing import List
import logging

import nfl_data_py as nfl

from .base_predictor import BaseStatPredictor

logger = logging.getLogger(__name__)


def add_opponent_pass_defense(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Add opponent pass defense features (yards allowed per game)."""
    try:
        weekly = nfl.import_weekly_data(years=seasons)

        # Calculate pass yards allowed by each team (group by opponent_team)
        # This gets how many yards each team GAVE UP each week
        pass_allowed = weekly.groupby(['season', 'week', 'opponent_team']).agg({
            'passing_yards': 'sum'
        }).reset_index()
        pass_allowed.columns = ['season', 'week', 'defense_team', 'pass_yards_allowed']

        # Calculate rolling average of pass yards allowed (defense perspective)
        pass_allowed = pass_allowed.sort_values(['defense_team', 'season', 'week'])
        pass_allowed['opp_pass_yards_allowed_avg'] = pass_allowed.groupby('defense_team')[
            'pass_yards_allowed'
        ].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())

        # Fill NaN with league average (~230 yards)
        pass_allowed['opp_pass_yards_allowed_avg'] = pass_allowed['opp_pass_yards_allowed_avg'].fillna(230)

        # Merge with player data using opponent_team as the defense
        if 'opponent_team' in df.columns:
            df = df.merge(
                pass_allowed[['season', 'week', 'defense_team', 'opp_pass_yards_allowed_avg']],
                left_on=['season', 'week', 'opponent_team'],
                right_on=['season', 'week', 'defense_team'],
                how='left'
            )
            df['opp_pass_yards_allowed_avg'] = df['opp_pass_yards_allowed_avg'].fillna(230)

            # Create relative strength: above/below league avg
            df['opp_pass_defense_strength'] = df['opp_pass_yards_allowed_avg'] / 230 - 1
        else:
            df['opp_pass_yards_allowed_avg'] = 230
            df['opp_pass_defense_strength'] = 0

        logger.info(f"Added opponent pass defense features")

    except Exception as e:
        logger.warning(f"Could not add opponent defense features: {e}")
        df['opp_pass_yards_allowed_avg'] = 230
        df['opp_pass_defense_strength'] = 0

    return df


def add_vegas_total_features(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Add Vegas implied total (over/under) from schedule data."""
    try:
        schedule = nfl.import_schedules(seasons)

        # Get game totals
        game_totals = schedule[['season', 'week', 'home_team', 'away_team', 'total']].copy()

        # For home team players
        home_totals = game_totals.rename(columns={'home_team': 'recent_team'})
        home_totals = home_totals[['season', 'week', 'recent_team', 'total']]

        # For away team players
        away_totals = game_totals.rename(columns={'away_team': 'recent_team'})
        away_totals = away_totals[['season', 'week', 'recent_team', 'total']]

        # Combine
        all_totals = pd.concat([home_totals, away_totals]).drop_duplicates(
            subset=['season', 'week', 'recent_team']
        )
        all_totals.columns = ['season', 'week', 'recent_team', 'vegas_total']

        # Merge with player data
        df = df.merge(all_totals, on=['season', 'week', 'recent_team'], how='left')

        # Fill NaN with average total (~46)
        df['vegas_total'] = df['vegas_total'].fillna(46)

        # Normalize: (total - avg) / std, roughly
        df['vegas_total_normalized'] = (df['vegas_total'] - 46) / 10

        # High-scoring game indicator (top 25%)
        df['is_high_total'] = (df['vegas_total'] >= 50).astype(int)

        logger.info(f"Added Vegas total features: avg={df['vegas_total'].mean():.1f}")

    except Exception as e:
        logger.warning(f"Could not add Vegas total features: {e}")
        df['vegas_total'] = 46
        df['vegas_total_normalized'] = 0
        df['is_high_total'] = 0

    return df


def add_ngs_passing_features(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Add Next Gen Stats passing features for QBs."""
    try:
        # Import NGS passing data
        ngs = nfl.import_ngs_data('passing', years=seasons)

        # Get relevant columns - these are cumulative/season stats by week
        ngs_cols = [
            'season', 'week', 'player_display_name', 'team_abbr',
            'avg_time_to_throw', 'avg_intended_air_yards',
            'completion_percentage_above_expectation', 'aggressiveness'
        ]

        # Filter to available columns
        available_cols = [c for c in ngs_cols if c in ngs.columns]
        ngs_subset = ngs[available_cols].copy()

        # Merge on player name and team (NGS uses display name)
        if 'player_display_name' in df.columns:
            merge_key = 'player_display_name'
        elif 'player_name' in df.columns:
            merge_key = 'player_name'
            ngs_subset = ngs_subset.rename(columns={'player_display_name': 'player_name'})
        else:
            raise ValueError("No player name column found")

        # Rename columns to avoid conflicts
        rename_map = {
            'avg_time_to_throw': 'ngs_time_to_throw',
            'avg_intended_air_yards': 'ngs_air_yards',
            'completion_percentage_above_expectation': 'ngs_cpoe',
            'aggressiveness': 'ngs_aggressiveness'
        }
        ngs_subset = ngs_subset.rename(columns=rename_map)

        # Merge (use team_abbr -> recent_team for better matching)
        ngs_subset = ngs_subset.rename(columns={'team_abbr': 'recent_team'})

        df = df.merge(
            ngs_subset,
            on=['season', 'week', merge_key, 'recent_team'] if 'recent_team' in ngs_subset.columns else ['season', 'week', merge_key],
            how='left'
        )

        # Fill missing NGS values with league averages
        df['ngs_time_to_throw'] = df.get('ngs_time_to_throw', pd.Series([2.7] * len(df))).fillna(2.7)
        df['ngs_air_yards'] = df.get('ngs_air_yards', pd.Series([8.5] * len(df))).fillna(8.5)
        df['ngs_cpoe'] = df.get('ngs_cpoe', pd.Series([0] * len(df))).fillna(0)
        df['ngs_aggressiveness'] = df.get('ngs_aggressiveness', pd.Series([17] * len(df))).fillna(17)

        logger.info(f"Added NGS passing features for {(~df['ngs_time_to_throw'].isna()).sum()} records")

    except Exception as e:
        logger.warning(f"Could not add NGS passing features: {e}")
        df['ngs_time_to_throw'] = 2.7
        df['ngs_air_yards'] = 8.5
        df['ngs_cpoe'] = 0
        df['ngs_aggressiveness'] = 17

    return df


def add_opponent_rush_defense(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Add opponent rush defense features (yards allowed per game)."""
    try:
        # Calculate team rush defense from existing data
        team_rush_allowed = df.groupby(['season', 'week', 'recent_team']).agg({
            'rushing_yards': 'sum'
        }).reset_index()
        team_rush_allowed = team_rush_allowed.rename(columns={
            'recent_team': 'defense_team',
            'rushing_yards': 'rush_yards_allowed'
        })

        # Calculate rolling average (shifted to avoid leakage)
        team_rush_allowed = team_rush_allowed.sort_values(['defense_team', 'season', 'week'])
        team_rush_allowed['opp_rush_yards_allowed_avg'] = team_rush_allowed.groupby('defense_team')['rush_yards_allowed'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # Get opponent info from schedule
        schedule = nfl.import_schedules(seasons)
        sched_games = schedule[['season', 'week', 'home_team', 'away_team']].copy()

        # Create opponent lookup
        home_opp = sched_games.rename(columns={'home_team': 'recent_team', 'away_team': 'opponent'})
        away_opp = sched_games.rename(columns={'away_team': 'recent_team', 'home_team': 'opponent'})
        opp_lookup = pd.concat([
            home_opp[['season', 'week', 'recent_team', 'opponent']],
            away_opp[['season', 'week', 'recent_team', 'opponent']]
        ]).drop_duplicates()

        # Merge opponent info
        df = df.merge(opp_lookup, on=['season', 'week', 'recent_team'], how='left')

        # Merge opponent defense stats
        opp_defense = team_rush_allowed[['season', 'week', 'defense_team', 'opp_rush_yards_allowed_avg']].copy()
        opp_defense = opp_defense.rename(columns={'defense_team': 'opponent'})
        df = df.merge(opp_defense, on=['season', 'week', 'opponent'], how='left')

        # Fill missing with league average
        league_avg = df['opp_rush_yards_allowed_avg'].mean()
        df['opp_rush_yards_allowed_avg'] = df['opp_rush_yards_allowed_avg'].fillna(league_avg if not pd.isna(league_avg) else 115.0)

        # Normalize (higher = easier opponent)
        df['opp_rush_defense_strength'] = df['opp_rush_yards_allowed_avg'] / 115.0

        logger.info(f"Added opponent rush defense features: avg={df['opp_rush_yards_allowed_avg'].mean():.1f}")

    except Exception as e:
        logger.warning(f"Could not add opponent rush defense features: {e}")
        df['opp_rush_yards_allowed_avg'] = 115.0
        df['opp_rush_defense_strength'] = 1.0

    return df


def add_opponent_reception_defense(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Add opponent pass defense features for receptions."""
    try:
        # Calculate team targets/receptions allowed from existing data
        team_pass_allowed = df.groupby(['season', 'week', 'recent_team']).agg({
            'targets': 'sum',
            'receptions': 'sum'
        }).reset_index()
        team_pass_allowed = team_pass_allowed.rename(columns={
            'recent_team': 'defense_team',
            'targets': 'targets_allowed',
            'receptions': 'receptions_allowed'
        })

        # Calculate rolling averages (shifted to avoid leakage)
        team_pass_allowed = team_pass_allowed.sort_values(['defense_team', 'season', 'week'])
        team_pass_allowed['opp_targets_allowed_avg'] = team_pass_allowed.groupby('defense_team')['targets_allowed'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        team_pass_allowed['opp_receptions_allowed_avg'] = team_pass_allowed.groupby('defense_team')['receptions_allowed'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # Get opponent info from schedule
        schedule = nfl.import_schedules(seasons)
        sched_games = schedule[['season', 'week', 'home_team', 'away_team']].copy()

        home_opp = sched_games.rename(columns={'home_team': 'recent_team', 'away_team': 'opponent'})
        away_opp = sched_games.rename(columns={'away_team': 'recent_team', 'home_team': 'opponent'})
        opp_lookup = pd.concat([
            home_opp[['season', 'week', 'recent_team', 'opponent']],
            away_opp[['season', 'week', 'recent_team', 'opponent']]
        ]).drop_duplicates()

        df = df.merge(opp_lookup, on=['season', 'week', 'recent_team'], how='left')

        # Merge opponent defense stats
        opp_defense = team_pass_allowed[['season', 'week', 'defense_team', 'opp_targets_allowed_avg', 'opp_receptions_allowed_avg']].copy()
        opp_defense = opp_defense.rename(columns={'defense_team': 'opponent'})
        df = df.merge(opp_defense, on=['season', 'week', 'opponent'], how='left')

        # Fill with league averages
        df['opp_targets_allowed_avg'] = df['opp_targets_allowed_avg'].fillna(35.0)
        df['opp_receptions_allowed_avg'] = df['opp_receptions_allowed_avg'].fillna(22.0)

        # Normalize (higher = easier opponent)
        df['opp_target_defense_strength'] = df['opp_targets_allowed_avg'] / 35.0
        df['opp_reception_defense_strength'] = df['opp_receptions_allowed_avg'] / 22.0

        logger.info(f"Added opponent reception defense features")

    except Exception as e:
        logger.warning(f"Could not add opponent reception defense features: {e}")
        df['opp_targets_allowed_avg'] = 35.0
        df['opp_receptions_allowed_avg'] = 22.0
        df['opp_target_defense_strength'] = 1.0
        df['opp_reception_defense_strength'] = 1.0

    return df


def add_ngs_rushing_features(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Add Next Gen Stats rushing features for RBs."""
    try:
        ngs = nfl.import_ngs_data('rushing', years=seasons)

        ngs_cols = [
            'season', 'week', 'player_display_name', 'team_abbr',
            'avg_rush_yards', 'efficiency', 'rush_yards_over_expected_per_att',
            'percent_attempts_gte_eight_defenders'
        ]

        available_cols = [c for c in ngs_cols if c in ngs.columns]
        ngs_subset = ngs[available_cols].copy()

        # Rename for merge
        if 'player_name' in df.columns:
            ngs_subset = ngs_subset.rename(columns={'player_display_name': 'player_name'})

        rename_map = {
            'avg_rush_yards': 'ngs_avg_rush_yards',
            'efficiency': 'ngs_efficiency',
            'rush_yards_over_expected_per_att': 'ngs_ryoe',
            'percent_attempts_gte_eight_defenders': 'ngs_stacked_box_pct'
        }
        ngs_subset = ngs_subset.rename(columns=rename_map)
        ngs_subset = ngs_subset.rename(columns={'team_abbr': 'recent_team'})

        merge_key = 'player_name' if 'player_name' in df.columns else 'player_display_name'
        df = df.merge(
            ngs_subset,
            on=['season', 'week', merge_key, 'recent_team'] if 'recent_team' in ngs_subset.columns else ['season', 'week', merge_key],
            how='left'
        )

        # Fill with defaults
        df['ngs_avg_rush_yards'] = df.get('ngs_avg_rush_yards', pd.Series([4.5] * len(df))).fillna(4.5)
        df['ngs_efficiency'] = df.get('ngs_efficiency', pd.Series([50.0] * len(df))).fillna(50.0)
        df['ngs_ryoe'] = df.get('ngs_ryoe', pd.Series([0.0] * len(df))).fillna(0.0)
        df['ngs_stacked_box_pct'] = df.get('ngs_stacked_box_pct', pd.Series([25.0] * len(df))).fillna(25.0)

        logger.info(f"Added NGS rushing features")

    except Exception as e:
        logger.warning(f"Could not add NGS rushing features: {e}")
        df['ngs_avg_rush_yards'] = 4.5
        df['ngs_efficiency'] = 50.0
        df['ngs_ryoe'] = 0.0
        df['ngs_stacked_box_pct'] = 25.0

    return df


def add_ngs_receiving_features(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Add Next Gen Stats receiving features for WRs/TEs/RBs."""
    try:
        ngs = nfl.import_ngs_data('receiving', years=seasons)

        ngs_cols = [
            'season', 'week', 'player_display_name', 'team_abbr',
            'avg_separation', 'avg_cushion', 'avg_intended_air_yards', 'catch_percentage'
        ]

        available_cols = [c for c in ngs_cols if c in ngs.columns]
        ngs_subset = ngs[available_cols].copy()

        # Rename for merge
        if 'player_name' in df.columns:
            ngs_subset = ngs_subset.rename(columns={'player_display_name': 'player_name'})

        rename_map = {
            'avg_separation': 'ngs_separation',
            'avg_cushion': 'ngs_cushion',
            'avg_intended_air_yards': 'ngs_air_yards',
            'catch_percentage': 'ngs_catch_pct'
        }
        ngs_subset = ngs_subset.rename(columns=rename_map)
        ngs_subset = ngs_subset.rename(columns={'team_abbr': 'recent_team'})

        merge_key = 'player_name' if 'player_name' in df.columns else 'player_display_name'
        df = df.merge(
            ngs_subset,
            on=['season', 'week', merge_key, 'recent_team'] if 'recent_team' in ngs_subset.columns else ['season', 'week', merge_key],
            how='left'
        )

        # Fill with defaults
        df['ngs_separation'] = df.get('ngs_separation', pd.Series([2.5] * len(df))).fillna(2.5)
        df['ngs_cushion'] = df.get('ngs_cushion', pd.Series([6.0] * len(df))).fillna(6.0)
        df['ngs_air_yards'] = df.get('ngs_air_yards', pd.Series([8.0] * len(df))).fillna(8.0)
        df['ngs_catch_pct'] = df.get('ngs_catch_pct', pd.Series([65.0] * len(df))).fillna(65.0)

        logger.info(f"Added NGS receiving features")

    except Exception as e:
        logger.warning(f"Could not add NGS receiving features: {e}")
        df['ngs_separation'] = 2.5
        df['ngs_cushion'] = 6.0
        df['ngs_air_yards'] = 8.0
        df['ngs_catch_pct'] = 65.0

    return df


def add_weather_features(df: pd.DataFrame, seasons: List[int]) -> pd.DataFrame:
    """Add weather features from schedule data to player stats."""
    try:
        schedule = nfl.import_schedules(seasons)

        # Get weather columns from schedule
        weather_cols = ['season', 'week', 'home_team', 'away_team', 'roof', 'temp', 'wind']
        sched_weather = schedule[weather_cols].copy()

        # Create game-level weather lookup for home team players
        home_weather = sched_weather.rename(columns={'home_team': 'recent_team'})
        home_weather = home_weather[['season', 'week', 'recent_team', 'roof', 'temp', 'wind']]

        # For away team players
        away_weather = sched_weather.rename(columns={'away_team': 'recent_team'})
        away_weather = away_weather[['season', 'week', 'recent_team', 'roof', 'temp', 'wind']]

        # Combine home and away
        game_weather = pd.concat([home_weather, away_weather]).drop_duplicates(
            subset=['season', 'week', 'recent_team']
        )

        # Merge with player data
        df = df.merge(game_weather, on=['season', 'week', 'recent_team'], how='left')

        # Create derived features
        df['is_dome'] = df['roof'].isin(['dome', 'closed']).astype(int)
        df['is_outdoor'] = (~df['roof'].isin(['dome', 'closed'])).astype(int)

        # Temperature features (fill NaN with moderate temp for domes)
        df['game_temp'] = df['temp'].fillna(70)  # Dome default
        df['is_cold'] = (df['game_temp'] < 35).astype(int)
        df['is_very_cold'] = (df['game_temp'] < 25).astype(int)

        # Wind features
        df['game_wind'] = df['wind'].fillna(0)  # Dome default
        df['is_windy'] = (df['game_wind'] > 15).astype(int)
        df['is_very_windy'] = (df['game_wind'] > 20).astype(int)

        # For dome games, ensure zero weather impact
        df.loc[df['is_dome'] == 1, 'game_temp'] = 70
        df.loc[df['is_dome'] == 1, 'game_wind'] = 0
        df.loc[df['is_dome'] == 1, 'is_cold'] = 0
        df.loc[df['is_dome'] == 1, 'is_very_cold'] = 0
        df.loc[df['is_dome'] == 1, 'is_windy'] = 0
        df.loc[df['is_dome'] == 1, 'is_very_windy'] = 0

        logger.info(f"Added weather features: {df['is_dome'].sum()} dome games, "
                   f"{df['is_cold'].sum()} cold games, {df['is_windy'].sum()} windy games")

    except Exception as e:
        logger.warning(f"Could not add weather features: {e}")
        # Add default values
        df['is_dome'] = 0
        df['game_temp'] = 60
        df['game_wind'] = 5
        df['is_cold'] = 0
        df['is_very_cold'] = 0
        df['is_windy'] = 0
        df['is_very_windy'] = 0

    return df


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

        # Add weather features
        seasons = df['season'].unique().tolist()
        df = add_weather_features(df, seasons)

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

        # Add weather features (passing is most affected by weather)
        seasons = df['season'].unique().tolist()
        df = add_weather_features(df, seasons)

        logger.info(f"Built features for {len(df)} passing records")
        return df


class PassingYardsPredictorV3(BaseStatPredictor):
    """Enhanced predictor for passing yards with defense, vegas, and NGS features."""

    stat_type = "passing_yards"
    model_filename = "passing_yards_v3.pkl"
    target_column = "passing_yards"
    position_filter = ["QB"]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build enhanced passing yards features."""
        logger.info("Building enhanced passing yards features (V3)...")

        # Filter to QBs with passing stats
        df = df[df["passing_yards"].notna() & (df["passing_yards"] > 0)].copy()
        df = df[df["position"] == "QB"].copy()

        if len(df) == 0:
            return df

        df = df.sort_values(["player_id", "season", "week"])
        seasons = df['season'].unique().tolist()

        # === BASIC ROLLING FEATURES ===
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

        # Fill NaN in basic rolling columns
        for col in df.columns:
            if "last_" in col or "_std_" in col:
                df[col] = df[col].fillna(0)

        # === NEW ENHANCED FEATURES ===

        # 1. Opponent pass defense (yards allowed)
        df = add_opponent_pass_defense(df, seasons)

        # 2. Vegas implied total (game total over/under)
        df = add_vegas_total_features(df, seasons)

        # 3. NGS passing metrics (time to throw, air yards, CPOE)
        df = add_ngs_passing_features(df, seasons)

        # 4. Weather features (enhanced for passing)
        df = add_weather_features(df, seasons)

        # 5. Weather interaction features for passing
        # Cold + wind combination is especially bad for passing
        df['cold_and_windy'] = (df['is_cold'] * df['is_windy']).astype(int)
        df['very_cold_and_windy'] = (df['is_very_cold'] * df['is_windy']).astype(int)

        # Weather impact score (higher = worse for passing)
        df['weather_impact'] = (
            df['is_cold'] * 0.1 +
            df['is_very_cold'] * 0.15 +
            df['is_windy'] * 0.15 +
            df['is_very_windy'] * 0.1 +
            df['cold_and_windy'] * 0.1
        )

        # 6. Game environment score (combines vegas + defense)
        # Higher = more passing expected
        df['pass_environment_score'] = (
            df['vegas_total_normalized'] * 0.5 +
            df['opp_pass_defense_strength'] * 0.5
        )

        logger.info(f"Built enhanced features for {len(df)} passing records (V3)")
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

        # Add weather features (affects passing game)
        seasons = df['season'].unique().tolist()
        df = add_weather_features(df, seasons)

        logger.info(f"Built features for {len(df)} reception records")
        return df


class RushingYardsPredictorV3(BaseStatPredictor):
    """Enhanced predictor for rushing yards with defense, vegas, and weather features."""

    stat_type = "rushing_yards"
    model_filename = "rushing_yards_v3.pkl"
    target_column = "rushing_yards"
    position_filter = ["RB", "QB", "WR"]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build enhanced rushing yards features (V3)."""
        logger.info("Building enhanced rushing yards features (V3)...")

        # Filter to players with rushing stats
        df = df[df["rushing_yards"].notna() & (df["rushing_yards"] >= 0)].copy()
        df = df[df["position"].isin(["RB", "QB", "WR"])].copy()

        if len(df) == 0:
            return df

        df = df.sort_values(["player_id", "season", "week"])
        seasons = df['season'].unique().tolist()

        # === BASIC ROLLING FEATURES ===
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
            df[f"carries_last_{window}"] = df.groupby("player_id")[
                "carries"
            ].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

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

        df["week_num"] = df["week"]

        # Snap percentage
        if "snap_pct" not in df.columns:
            df["snap_pct"] = 0.5

        # Fill NaN
        for col in df.columns:
            if "last_" in col or "_std_" in col:
                df[col] = df[col].fillna(0)

        # === ADD WEATHER FEATURES ===
        df = add_weather_features(df, seasons)

        # === V3: OPPONENT RUSH DEFENSE ===
        df = add_opponent_rush_defense(df, seasons)

        # === V3: VEGAS TOTALS ===
        df = add_vegas_total_features(df, seasons)

        # Low totals favor rushing (opposite of passing)
        df['is_low_total'] = (df['vegas_total'] < 42).astype(int)

        # === V3: NGS RUSHING METRICS ===
        df = add_ngs_rushing_features(df, seasons)

        # === V3: WEATHER INTERACTION FEATURES ===
        df['cold_game'] = (df['is_cold'] == 1).astype(int)
        df['cold_and_windy'] = ((df['is_cold'] == 1) & (df['is_windy'] == 1)).astype(int)

        # Rush-friendly weather score
        df['rush_weather_score'] = (
            df['is_cold'].astype(float) * 0.3 +
            df['is_windy'].astype(float) * 0.3 +
            (1 - df['is_dome'].astype(float)) * 0.2 +
            df['cold_and_windy'].astype(float) * 0.2
        )

        # Rush environment score
        df['rush_environment_score'] = (
            df['opp_rush_defense_strength'] * 0.4 +
            (1 - df['vegas_total_normalized']) * 0.3 +
            df['rush_weather_score'] * 0.3
        )

        logger.info(f"Built V3 features for {len(df)} rushing records")
        return df


class ReceptionsPredictorV3(BaseStatPredictor):
    """Enhanced predictor for receptions with defense, vegas, and weather features."""

    stat_type = "receptions"
    model_filename = "receptions_v3.pkl"
    target_column = "receptions"
    position_filter = ["WR", "TE", "RB"]

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build enhanced receptions features (V3)."""
        logger.info("Building enhanced receptions features (V3)...")

        # Filter to pass catchers
        df = df[df["receptions"].notna() & (df["receptions"] >= 0)].copy()
        df = df[df["position"].isin(["WR", "TE", "RB"])].copy()

        if len(df) == 0:
            return df

        df = df.sort_values(["player_id", "season", "week"])
        seasons = df['season'].unique().tolist()

        # === BASIC ROLLING FEATURES ===
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

        # === ADD WEATHER FEATURES ===
        df = add_weather_features(df, seasons)

        # === V3: OPPONENT PASS DEFENSE ===
        df = add_opponent_reception_defense(df, seasons)

        # === V3: VEGAS TOTALS ===
        df = add_vegas_total_features(df, seasons)
        df['is_high_total'] = (df['vegas_total'] > 48).astype(int)

        # === V3: NGS RECEIVING METRICS ===
        df = add_ngs_receiving_features(df, seasons)

        # === V3: WEATHER INTERACTION FEATURES ===
        df['cold_and_windy'] = ((df['is_cold'] == 1) & (df['is_windy'] == 1)).astype(int)

        # Weather impact (higher = worse for receiving)
        df['weather_impact'] = (
            df['is_cold'].astype(float) * 0.25 +
            df['is_windy'].astype(float) * 0.4 +
            df['cold_and_windy'].astype(float) * 0.15 +
            (1 - df['is_dome'].astype(float)) * 0.2
        )

        # Reception environment score
        df['reception_environment_score'] = (
            df['opp_reception_defense_strength'] * 0.35 +
            df['vegas_total_normalized'] * 0.35 +
            (1 - df['weather_impact']) * 0.3
        )

        logger.info(f"Built V3 features for {len(df)} reception records")
        return df


# Registry of all predictors
PREDICTOR_REGISTRY = {
    "rushing_yards": RushingYardsPredictorV3,  # V3 with defense/vegas/weather features (default)
    "rushing_yards_v2": RushingYardsPredictor,  # V2 legacy
    "passing_yards": PassingYardsPredictorV3,  # V3 with defense/vegas/weather features (default)
    "passing_yards_v2": PassingYardsPredictor,  # V2 legacy
    "receptions": ReceptionsPredictorV3,  # V3 with defense/vegas/weather features (default)
    "receptions_v2": ReceptionsPredictor,  # V2 legacy
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
