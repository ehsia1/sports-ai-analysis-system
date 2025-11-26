"""Enhanced feature engineering for improved model accuracy."""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class EnhancedFeatureEngineer:
    """Enhanced feature engineering with defensive stats and game context."""

    def __init__(self):
        self.defensive_stats_cache = {}
        self.schedule_cache = {}

    def add_defensive_matchup_features(
        self,
        player_df: pd.DataFrame,
        all_weekly_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add opponent defensive strength features."""

        logger.info("Adding defensive matchup features...")

        # Calculate defensive stats by team and season
        # Group by opponent (defense) and calculate yards allowed
        defensive_stats = all_weekly_data.groupby(['opponent_team', 'season']).agg({
            'passing_yards': ['mean', 'std'],
            'rushing_yards': ['mean', 'std'],
            'receiving_yards': ['mean', 'std'],
            'targets': 'mean',
            'receptions': 'mean',
        }).round(2)

        # Flatten column names
        defensive_stats.columns = ['_'.join(col).strip() for col in defensive_stats.columns.values]
        defensive_stats = defensive_stats.reset_index()

        # Calculate defensive rankings (lower is better defense)
        for stat in ['passing_yards_mean', 'rushing_yards_mean', 'receiving_yards_mean']:
            rank_col = stat.replace('_mean', '_rank')
            defensive_stats[rank_col] = defensive_stats.groupby('season')[stat].rank(ascending=True)

        # Merge with player data
        player_df = player_df.merge(
            defensive_stats,
            left_on=['opponent_team', 'season'],
            right_on=['opponent_team', 'season'],
            how='left',
            suffixes=('', '_def')
        )

        # Create composite defensive strength score
        # Lower score = tougher defense
        player_df['def_strength_composite'] = (
            player_df['passing_yards_rank'].fillna(16) +
            player_df['rushing_yards_rank'].fillna(16) +
            player_df['receiving_yards_rank'].fillna(16)
        ) / 3

        # Normalize to 0-1 (1 = easiest matchup, 0 = hardest)
        player_df['def_matchup_advantage'] = 1 - (player_df['def_strength_composite'] / 32)

        logger.info(f"Added defensive features. Sample matchup advantages: {player_df['def_matchup_advantage'].describe()}")

        return player_df

    def add_game_context_features(
        self,
        player_df: pd.DataFrame,
        schedule_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add game context features from schedule data."""

        logger.info("Adding game context features...")

        # Prepare schedule data
        schedule_prep = schedule_df.copy()

        # Create home/away indicator for each team
        home_schedule = schedule_prep[['game_id', 'season', 'week', 'home_team',
                                       'temp', 'wind', 'roof', 'surface',
                                       'weekday', 'gametime']].copy()
        home_schedule['is_home'] = 1
        home_schedule['team'] = home_schedule['home_team']
        home_schedule['opponent'] = schedule_prep['away_team']

        away_schedule = schedule_prep[['game_id', 'season', 'week', 'away_team',
                                       'temp', 'wind', 'roof', 'surface',
                                       'weekday', 'gametime']].copy()
        away_schedule['is_home'] = 0
        away_schedule['team'] = away_schedule['away_team']
        away_schedule['opponent'] = schedule_prep['home_team']

        # Combine
        full_schedule = pd.concat([home_schedule, away_schedule], ignore_index=True)

        # Parse primetime games
        full_schedule['is_primetime'] = full_schedule['weekday'].isin(['Thursday', 'Sunday', 'Monday'])
        full_schedule['gametime_hour'] = pd.to_datetime(
            full_schedule['gametime'], format='%H:%M', errors='coerce'
        ).dt.hour
        full_schedule['is_primetime'] = (
            (full_schedule['weekday'] == 'Thursday') |  # Thursday Night
            ((full_schedule['weekday'] == 'Sunday') & (full_schedule['gametime_hour'] >= 20)) |  # SNF
            (full_schedule['weekday'] == 'Monday')  # MNF
        ).astype(int)

        # Weather categories
        full_schedule['is_cold'] = (full_schedule['temp'] < 40).astype(int)
        full_schedule['is_windy'] = (full_schedule['wind'] > 15).astype(int)
        full_schedule['is_dome'] = (full_schedule['roof'].isin(['dome', 'closed'])).astype(int)
        full_schedule['is_turf'] = (full_schedule['surface'].str.contains('turf', case=False, na=False)).astype(int)

        # Merge with player data
        merge_cols = ['season', 'week', 'team', 'opponent', 'is_home', 'temp', 'wind',
                     'is_primetime', 'is_cold', 'is_windy', 'is_dome', 'is_turf']

        player_df = player_df.merge(
            full_schedule[merge_cols],
            left_on=['season', 'week', 'recent_team'],
            right_on=['season', 'week', 'team'],
            how='left',
            suffixes=('', '_sched')
        )

        # Fill missing values
        player_df['is_home'] = player_df['is_home'].fillna(0.5)  # Unknown
        player_df['temp'] = player_df['temp'].fillna(70)  # Default temp
        player_df['wind'] = player_df['wind'].fillna(5)  # Default wind
        player_df['is_primetime'] = player_df['is_primetime'].fillna(0)
        player_df['is_cold'] = player_df['is_cold'].fillna(0)
        player_df['is_windy'] = player_df['is_windy'].fillna(0)
        player_df['is_dome'] = player_df['is_dome'].fillna(0)
        player_df['is_turf'] = player_df['is_turf'].fillna(0)

        logger.info("Added game context features")

        return player_df

    def add_recent_performance_features(
        self,
        player_df: pd.DataFrame,
        window_sizes: List[int] = [3, 5, 8]
    ) -> pd.DataFrame:
        """Add multiple rolling window features."""

        logger.info(f"Adding rolling features with windows: {window_sizes}")

        # Sort by player and date
        player_df = player_df.sort_values(['player_id', 'season', 'week'])

        stats_to_roll = ['receiving_yards', 'rushing_yards', 'targets', 'receptions',
                        'completions', 'attempts', 'passing_yards']

        for stat in stats_to_roll:
            if stat not in player_df.columns:
                continue

            for window in window_sizes:
                # Rolling mean
                player_df[f'{stat}_roll_{window}'] = (
                    player_df.groupby('player_id')[stat]
                    .rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )

                # Rolling std (consistency metric)
                player_df[f'{stat}_std_{window}'] = (
                    player_df.groupby('player_id')[stat]
                    .rolling(window=window, min_periods=2)
                    .std()
                    .reset_index(0, drop=True)
                ).fillna(0)

        # Trend features (is player improving or declining?)
        for stat in stats_to_roll:
            if stat not in player_df.columns:
                continue

            if f'{stat}_roll_3' in player_df.columns and f'{stat}_roll_8' in player_df.columns:
                # Positive trend = recent performance better than longer average
                player_df[f'{stat}_trend'] = (
                    player_df[f'{stat}_roll_3'] - player_df[f'{stat}_roll_8']
                ).fillna(0)

        logger.info("Added rolling performance features")

        return player_df

    def add_target_share_features(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """Add team target share and opportunity metrics."""

        logger.info("Adding target share features...")

        # Calculate team totals for each game
        team_totals = player_df.groupby(['recent_team', 'season', 'week']).agg({
            'targets': 'sum',
            'receiving_yards': 'sum',
            'completions': 'sum',
        }).reset_index()

        team_totals.columns = ['recent_team', 'season', 'week',
                               'team_targets', 'team_rec_yards', 'team_completions']

        # Merge back
        player_df = player_df.merge(
            team_totals,
            on=['recent_team', 'season', 'week'],
            how='left'
        )

        # Calculate shares
        player_df['target_share'] = (
            player_df['targets'] / player_df['team_targets'].replace(0, np.nan)
        ).fillna(0)

        player_df['yards_share'] = (
            player_df['receiving_yards'] / player_df['team_rec_yards'].replace(0, np.nan)
        ).fillna(0)

        # Rolling target share (more stable predictor)
        player_df = player_df.sort_values(['player_id', 'season', 'week'])
        player_df['target_share_roll_4'] = (
            player_df.groupby('player_id')['target_share']
            .rolling(window=4, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

        logger.info("Added target share features")

        return player_df

    def engineer_all_features(
        self,
        player_df: pd.DataFrame,
        all_weekly_data: pd.DataFrame,
        schedule_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply all feature engineering transformations."""

        logger.info("Starting comprehensive feature engineering...")

        # 1. Defensive matchup features
        player_df = self.add_defensive_matchup_features(player_df, all_weekly_data)

        # 2. Game context features
        player_df = self.add_game_context_features(player_df, schedule_df)

        # 3. Recent performance features
        player_df = self.add_recent_performance_features(player_df)

        # 4. Target share features
        player_df = self.add_target_share_features(player_df)

        logger.info("Feature engineering complete")

        return player_df


def get_enhanced_feature_list(position: str = None) -> List[str]:
    """Get list of enhanced features for modeling."""

    base_features = [
        # Game context
        'is_home', 'is_primetime', 'temp', 'wind',
        'is_cold', 'is_windy', 'is_dome', 'is_turf',

        # Defensive matchup
        'def_matchup_advantage',
        'passing_yards_rank', 'rushing_yards_rank', 'receiving_yards_rank',
    ]

    # Rolling performance features (3, 5, 8 game windows)
    rolling_features = []
    for stat in ['receiving_yards', 'rushing_yards', 'targets', 'receptions']:
        for window in [3, 5, 8]:
            rolling_features.extend([
                f'{stat}_roll_{window}',
                f'{stat}_std_{window}',
            ])
        rolling_features.append(f'{stat}_trend')

    # Target share features
    share_features = ['target_share', 'yards_share', 'target_share_roll_4']

    all_features = base_features + rolling_features + share_features

    return all_features
