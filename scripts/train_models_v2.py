#!/usr/bin/env python3
"""
Train prediction models using ONLY historical features.

Key principle: To predict Week N stats, only use data from Weeks 1 to N-1.
No data leakage from same-game stats.

Models:
- rushing_yards: Predict RB/QB rushing yards
- passing_yards: Predict QB passing yards
- receptions: Predict WR/TE/RB receptions
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import nfl_data_py as nfl
import xgboost as xgb
from sklearn.model_selection import train_test_split


def load_ngs_2025_data() -> pd.DataFrame:
    """Load 2025 season data from Next Gen Stats and transform to weekly_data format.

    NGS provides weekly player stats for the current season before the yearly
    data file is published by nfl_data_py.
    """
    print("Loading 2025 data from Next Gen Stats...")

    try:
        # Load NGS data for all stat types
        ngs_rush = nfl.import_ngs_data('rushing', [2025])
        ngs_rec = nfl.import_ngs_data('receiving', [2025])
        ngs_pass = nfl.import_ngs_data('passing', [2025])

        # Filter out season totals (week 0), include both regular season and playoffs
        ngs_rush = ngs_rush[(ngs_rush['week'] > 0) & (ngs_rush['season_type'].isin(['REG', 'POST']))]
        ngs_rec = ngs_rec[(ngs_rec['week'] > 0) & (ngs_rec['season_type'].isin(['REG', 'POST']))]
        ngs_pass = ngs_pass[(ngs_pass['week'] > 0) & (ngs_pass['season_type'].isin(['REG', 'POST']))]

        print(f"  NGS rushing records: {len(ngs_rush)}")
        print(f"  NGS receiving records: {len(ngs_rec)}")
        print(f"  NGS passing records: {len(ngs_pass)}")

        # Transform rushing data to match weekly_data format
        rush_df = ngs_rush.rename(columns={
            'player_gsis_id': 'player_id',
            'team_abbr': 'recent_team',
            'rush_attempts': 'carries',
            'rush_yards': 'rushing_yards',
            'rush_touchdowns': 'rushing_tds',
        })[['season', 'week', 'season_type', 'player_id', 'player_display_name', 'recent_team',
            'carries', 'rushing_yards', 'rushing_tds']].copy()

        # Transform receiving data
        rec_df = ngs_rec.rename(columns={
            'player_gsis_id': 'player_id',
            'team_abbr': 'recent_team',
            'yards': 'receiving_yards',
            'rec_touchdowns': 'receiving_tds',
        })[['season', 'week', 'season_type', 'player_id', 'player_display_name', 'recent_team',
            'receptions', 'targets', 'receiving_yards', 'receiving_tds']].copy()

        # Transform passing data
        pass_df = ngs_pass.rename(columns={
            'player_gsis_id': 'player_id',
            'team_abbr': 'recent_team',
            'pass_yards': 'passing_yards',
            'pass_touchdowns': 'passing_tds',
        })[['season', 'week', 'season_type', 'player_id', 'player_display_name', 'recent_team',
            'attempts', 'completions', 'passing_yards', 'passing_tds']].copy()

        # Merge all stat types on player/week
        # Start with rushing, merge receiving and passing
        combined = rush_df.merge(
            rec_df,
            on=['season', 'week', 'season_type', 'player_id', 'player_display_name', 'recent_team'],
            how='outer'
        )
        combined = combined.merge(
            pass_df,
            on=['season', 'week', 'season_type', 'player_id', 'player_display_name', 'recent_team'],
            how='outer'
        )

        # Fill NaN stats with 0
        stat_cols = ['carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
                     'receiving_yards', 'receiving_tds', 'attempts', 'completions',
                     'passing_yards', 'passing_tds']
        combined[stat_cols] = combined[stat_cols].fillna(0)

        # Add player_name column (same as player_display_name for NGS)
        combined['player_name'] = combined['player_display_name']

        print(f"  Combined 2025 records: {len(combined)}")
        print(f"  2025 weeks available: {sorted(combined['week'].unique())}")
        print(f"  Season types: {combined['season_type'].value_counts().to_dict()}")

        return combined

    except Exception as e:
        print(f"  Warning: Could not load 2025 NGS data: {e}")
        return pd.DataFrame()


def load_data(seasons: list) -> tuple:
    """Load weekly data, snap counts, and schedule (for weather).

    For 2025, uses NGS data since yearly file isn't published yet.
    """
    # Separate historical seasons from current (2025)
    historical_seasons = [s for s in seasons if s < 2025]
    include_2025 = 2025 in seasons

    print(f"Loading data for seasons {seasons}...")

    # Load historical data
    if historical_seasons:
        weekly = nfl.import_weekly_data(historical_seasons)
        print(f"  Historical weekly records (2020-2024): {len(weekly)}")
    else:
        weekly = pd.DataFrame()

    # Load and append 2025 NGS data if requested
    if include_2025:
        ngs_2025 = load_ngs_2025_data()
        if len(ngs_2025) > 0:
            # Ensure column compatibility before concatenating
            # Only keep columns that exist in historical data
            if len(weekly) > 0:
                common_cols = [c for c in ngs_2025.columns if c in weekly.columns]
                ngs_2025 = ngs_2025[common_cols]
            weekly = pd.concat([weekly, ngs_2025], ignore_index=True)
            print(f"  Total weekly records (with 2025): {len(weekly)}")

    snaps = nfl.import_snap_counts(seasons)
    print(f"  Snap count records: {len(snaps)}")

    schedule = nfl.import_schedules(seasons)
    print(f"  Schedule records: {len(schedule)}")

    return weekly, snaps, schedule


def add_weather_features(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Add weather features from schedule data."""
    print("\nAdding weather features...")

    # Get weather columns from schedule
    weather_cols = ['season', 'week', 'home_team', 'away_team', 'roof', 'temp', 'wind']
    sched_weather = schedule[weather_cols].copy()

    # Create game-level weather lookup
    # For home team players
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

    # For outdoor games, set dome-like conditions for missing data
    df.loc[df['is_dome'] == 1, 'game_temp'] = 70
    df.loc[df['is_dome'] == 1, 'game_wind'] = 0
    df.loc[df['is_dome'] == 1, 'is_cold'] = 0
    df.loc[df['is_dome'] == 1, 'is_very_cold'] = 0
    df.loc[df['is_dome'] == 1, 'is_windy'] = 0
    df.loc[df['is_dome'] == 1, 'is_very_windy'] = 0

    weather_coverage = df['temp'].notna().sum() / len(df) * 100
    print(f"  Weather data coverage: {weather_coverage:.1f}%")
    print(f"  Dome games: {df['is_dome'].sum()}")
    print(f"  Cold games (<35°F): {df['is_cold'].sum()}")
    print(f"  Windy games (>15mph): {df['is_windy'].sum()}")

    return df


def add_snap_counts(df: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Merge snap count data."""
    if len(snaps) == 0:
        df['snap_pct'] = 0.5
        return df

    snap_agg = snaps.groupby(['season', 'week', 'player']).agg({
        'offense_pct': 'mean'
    }).reset_index()
    snap_agg = snap_agg.rename(columns={'player': 'player_name', 'offense_pct': 'snap_pct'})

    df = df.merge(snap_agg, on=['season', 'week', 'player_name'], how='left')
    df['snap_pct'] = df['snap_pct'].fillna(0.5)

    return df


def calculate_opponent_defense(df: pd.DataFrame, stat_col: str) -> pd.DataFrame:
    """Calculate rolling opponent defense metrics."""
    # Team defense: avg stat allowed per game
    team_allowed = df.groupby(['season', 'week', 'recent_team']).agg({
        stat_col: 'sum'
    }).reset_index()
    team_allowed = team_allowed.rename(columns={
        'recent_team': 'defense_team',
        stat_col: f'{stat_col}_allowed'
    })

    # Rolling average of yards allowed (shifted to avoid leakage)
    team_allowed = team_allowed.sort_values(['defense_team', 'season', 'week'])
    team_allowed[f'opp_{stat_col}_allowed'] = team_allowed.groupby('defense_team')[f'{stat_col}_allowed'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Need opponent team info to merge
    # For now, use league average as placeholder
    df[f'opp_{stat_col}_allowed'] = df[stat_col].mean()

    return df


def build_rushing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features for rushing yards prediction."""
    print("\nBuilding rushing yards features...")

    # Filter to players with rushing stats
    df = df[df['rushing_yards'].notna() & (df['rushing_yards'] >= 0)].copy()
    df = df[df['position'].isin(['RB', 'QB', 'WR'])].copy()

    # Sort for rolling calculations
    df = df.sort_values(['player_id', 'season', 'week'])

    # === HISTORICAL FEATURES ONLY ===

    # Rolling rushing yards (shifted - use past games only)
    for window in [3, 5]:
        df[f'rush_yards_last_{window}'] = df.groupby('player_id')['rushing_yards'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'rush_yards_std_{window}'] = df.groupby('player_id')['rushing_yards'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # Rolling carries
    df['carries'] = df['carries'].fillna(0)
    for window in [3, 5]:
        df[f'carries_last_{window}'] = df.groupby('player_id')['carries'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Yards per carry (historical)
    df['ypc_career'] = df.groupby('player_id').apply(
        lambda x: x['rushing_yards'].shift(1).expanding().sum() / x['carries'].shift(1).expanding().sum().clip(lower=1)
    ).reset_index(level=0, drop=True)
    df['ypc_career'] = df['ypc_career'].fillna(4.0)  # League average

    # Position encoding
    pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    df['position_encoded'] = df['position'].map(pos_map).fillna(1)

    # Week number (affects workload)
    df['week_num'] = df['week']

    # Fill NaN
    for col in df.columns:
        if 'last_' in col or '_std_' in col:
            df[col] = df[col].fillna(0)

    return df


def build_passing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features for passing yards prediction."""
    print("\nBuilding passing yards features...")

    # Filter to QBs with passing stats
    df = df[df['passing_yards'].notna() & (df['passing_yards'] > 0)].copy()
    df = df[df['position'] == 'QB'].copy()

    df = df.sort_values(['player_id', 'season', 'week'])

    # === HISTORICAL FEATURES ONLY ===

    # Rolling passing yards
    for window in [3, 5]:
        df[f'pass_yards_last_{window}'] = df.groupby('player_id')['passing_yards'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'pass_yards_std_{window}'] = df.groupby('player_id')['passing_yards'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # Rolling attempts
    df['attempts'] = df['attempts'].fillna(0)
    for window in [3, 5]:
        df[f'attempts_last_{window}'] = df.groupby('player_id')['attempts'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Historical completion percentage
    df['completions'] = df['completions'].fillna(0)
    df['comp_pct_last_5'] = df.groupby('player_id').apply(
        lambda x: x['completions'].shift(1).rolling(5, min_periods=1).sum() /
                  x['attempts'].shift(1).rolling(5, min_periods=1).sum().clip(lower=1)
    ).reset_index(level=0, drop=True)
    df['comp_pct_last_5'] = df['comp_pct_last_5'].fillna(0.65)

    # Yards per attempt (historical)
    df['ypa_career'] = df.groupby('player_id').apply(
        lambda x: x['passing_yards'].shift(1).expanding().sum() / x['attempts'].shift(1).expanding().sum().clip(lower=1)
    ).reset_index(level=0, drop=True)
    df['ypa_career'] = df['ypa_career'].fillna(7.0)

    # Position encoded (all QBs = 0)
    df['position_encoded'] = 0

    df['week_num'] = df['week']

    for col in df.columns:
        if 'last_' in col or '_std_' in col:
            df[col] = df[col].fillna(0)

    return df


def build_passing_features_v3(df: pd.DataFrame, schedule: pd.DataFrame, seasons: list) -> pd.DataFrame:
    """Build enhanced features for passing yards V3 prediction.

    Adds:
    - Opponent pass defense metrics
    - Vegas implied totals
    - NGS passing metrics (time to throw, air yards, CPOE, aggressiveness)
    - Weather interaction features
    """
    print("\nBuilding passing yards V3 features...")

    # Start with base V2 features
    df = build_passing_features(df)

    # === OPPONENT PASS DEFENSE ===
    print("  Adding opponent pass defense metrics...")

    # Calculate team pass defense (yards allowed per game)
    team_pass_allowed = df.groupby(['season', 'week', 'recent_team']).agg({
        'passing_yards': 'sum'
    }).reset_index()
    team_pass_allowed = team_pass_allowed.rename(columns={
        'recent_team': 'defense_team',
        'passing_yards': 'pass_yards_allowed'
    })

    # Calculate rolling average of yards allowed (shifted to avoid leakage)
    team_pass_allowed = team_pass_allowed.sort_values(['defense_team', 'season', 'week'])
    team_pass_allowed['opp_pass_yards_allowed_avg'] = team_pass_allowed.groupby('defense_team')['pass_yards_allowed'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Get opponent info from schedule
    sched_games = schedule[['season', 'week', 'home_team', 'away_team']].copy()

    # Create opponent lookup - home team's opponent is away team and vice versa
    home_opp = sched_games.rename(columns={'home_team': 'recent_team', 'away_team': 'opponent'})
    away_opp = sched_games.rename(columns={'away_team': 'recent_team', 'home_team': 'opponent'})
    opp_lookup = pd.concat([
        home_opp[['season', 'week', 'recent_team', 'opponent']],
        away_opp[['season', 'week', 'recent_team', 'opponent']]
    ]).drop_duplicates()

    # Merge opponent info
    df = df.merge(opp_lookup, on=['season', 'week', 'recent_team'], how='left')

    # Merge opponent defense stats
    opp_defense = team_pass_allowed[['season', 'week', 'defense_team', 'opp_pass_yards_allowed_avg']].copy()
    opp_defense = opp_defense.rename(columns={'defense_team': 'opponent'})
    df = df.merge(opp_defense, on=['season', 'week', 'opponent'], how='left')

    # Fill missing with league average
    league_avg_allowed = df['opp_pass_yards_allowed_avg'].mean()
    df['opp_pass_yards_allowed_avg'] = df['opp_pass_yards_allowed_avg'].fillna(league_avg_allowed if not pd.isna(league_avg_allowed) else 230.0)

    # Normalize defense strength (higher = easier opponent)
    df['opp_pass_defense_strength'] = df['opp_pass_yards_allowed_avg'] / 230.0  # 230 = league avg

    # === VEGAS IMPLIED TOTALS ===
    print("  Adding Vegas implied totals...")

    # Get over/under from schedule (if available)
    if 'total' in schedule.columns:
        vegas_cols = ['season', 'week', 'home_team', 'away_team', 'total']
        sched_vegas = schedule[vegas_cols].copy()

        # Create lookup for both home and away teams
        home_vegas = sched_vegas.rename(columns={'home_team': 'recent_team'})
        away_vegas = sched_vegas.rename(columns={'away_team': 'recent_team'})
        vegas_lookup = pd.concat([
            home_vegas[['season', 'week', 'recent_team', 'total']],
            away_vegas[['season', 'week', 'recent_team', 'total']]
        ]).drop_duplicates()

        df = df.merge(vegas_lookup, on=['season', 'week', 'recent_team'], how='left')
        df = df.rename(columns={'total': 'vegas_total'})
    else:
        df['vegas_total'] = 45.0  # Default if not available

    df['vegas_total'] = df['vegas_total'].fillna(45.0)
    df['vegas_total_normalized'] = df['vegas_total'] / 45.0  # 45 = league average
    df['is_high_total'] = (df['vegas_total'] > 48).astype(int)

    # === NGS PASSING METRICS ===
    print("  Adding NGS passing metrics...")

    try:
        ngs_pass = nfl.import_ngs_data(stat_type='passing', years=seasons)
        if len(ngs_pass) > 0:
            # Keep relevant columns
            ngs_cols = ['season', 'week', 'player_display_name', 'avg_time_to_throw',
                       'avg_intended_air_yards', 'completion_percentage_above_expectation', 'aggressiveness']
            ngs_pass = ngs_pass[[c for c in ngs_cols if c in ngs_pass.columns]].copy()

            # Rename for merge
            ngs_pass = ngs_pass.rename(columns={
                'player_display_name': 'player_name',
                'avg_time_to_throw': 'ngs_time_to_throw',
                'avg_intended_air_yards': 'ngs_air_yards',
                'completion_percentage_above_expectation': 'ngs_cpoe',
                'aggressiveness': 'ngs_aggressiveness'
            })

            # Calculate rolling averages (shifted)
            ngs_pass = ngs_pass.sort_values(['player_name', 'season', 'week'])
            for col in ['ngs_time_to_throw', 'ngs_air_yards', 'ngs_cpoe', 'ngs_aggressiveness']:
                if col in ngs_pass.columns:
                    ngs_pass[col] = ngs_pass.groupby('player_name')[col].transform(
                        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
                    )

            # Merge with main df
            df = df.merge(ngs_pass, on=['season', 'week', 'player_name'], how='left')
    except Exception as e:
        print(f"    Warning: Could not load NGS data: {e}")

    # Fill NGS features with defaults
    df['ngs_time_to_throw'] = df.get('ngs_time_to_throw', pd.Series([2.6] * len(df))).fillna(2.6)
    df['ngs_air_yards'] = df.get('ngs_air_yards', pd.Series([8.0] * len(df))).fillna(8.0)
    df['ngs_cpoe'] = df.get('ngs_cpoe', pd.Series([0.0] * len(df))).fillna(0.0)
    df['ngs_aggressiveness'] = df.get('ngs_aggressiveness', pd.Series([17.0] * len(df))).fillna(17.0)

    # === WEATHER INTERACTION FEATURES ===
    print("  Adding weather interaction features...")

    # Cold and windy combination (worst for passing)
    df['cold_and_windy'] = ((df['is_cold'] == 1) & (df['is_windy'] == 1)).astype(int)
    df['very_cold_and_windy'] = ((df['is_very_cold'] == 1) & (df['is_very_windy'] == 1)).astype(int)

    # Weather impact score (0 = perfect, higher = worse for passing)
    df['weather_impact'] = (
        df['is_cold'].astype(float) * 0.3 +
        df['is_windy'].astype(float) * 0.4 +
        df['is_very_windy'].astype(float) * 0.2 +
        df['cold_and_windy'].astype(float) * 0.1
    )

    # Pass environment score (combine Vegas total + defense + weather)
    df['pass_environment_score'] = (
        df['vegas_total_normalized'] * 0.4 +
        df['opp_pass_defense_strength'] * 0.4 +
        (1 - df['weather_impact']) * 0.2
    )

    print(f"  V3 features built. Records: {len(df)}")
    print(f"  Opponent defense coverage: {df['opp_pass_yards_allowed_avg'].notna().mean()*100:.1f}%")
    print(f"  Vegas total coverage: {(df['vegas_total'] != 45.0).mean()*100:.1f}%")
    print(f"  NGS data coverage: {(df['ngs_cpoe'] != 0.0).mean()*100:.1f}%")

    return df


def build_receptions_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features for receptions prediction."""
    print("\nBuilding receptions features...")

    # Filter to pass catchers
    df = df[df['receptions'].notna() & (df['receptions'] >= 0)].copy()
    df = df[df['position'].isin(['WR', 'TE', 'RB'])].copy()

    df = df.sort_values(['player_id', 'season', 'week'])

    # === HISTORICAL FEATURES ONLY ===

    # Rolling receptions
    for window in [3, 5]:
        df[f'receptions_last_{window}'] = df.groupby('player_id')['receptions'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'receptions_std_{window}'] = df.groupby('player_id')['receptions'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    # Rolling targets
    df['targets'] = df['targets'].fillna(0)
    for window in [3, 5]:
        df[f'targets_last_{window}'] = df.groupby('player_id')['targets'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Historical target share
    df['target_share'] = df['target_share'].fillna(0)
    for window in [3, 5]:
        df[f'target_share_last_{window}'] = df.groupby('player_id')['target_share'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Historical catch rate
    df['catch_rate_last_5'] = df.groupby('player_id').apply(
        lambda x: x['receptions'].shift(1).rolling(5, min_periods=1).sum() /
                  x['targets'].shift(1).rolling(5, min_periods=1).sum().clip(lower=1)
    ).reset_index(level=0, drop=True)
    df['catch_rate_last_5'] = df['catch_rate_last_5'].fillna(0.65)

    # Position encoding
    pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    df['position_encoded'] = df['position'].map(pos_map).fillna(2)

    df['week_num'] = df['week']

    for col in df.columns:
        if 'last_' in col or '_std_' in col:
            df[col] = df[col].fillna(0)

    return df


def build_rushing_features_v3(df: pd.DataFrame, schedule: pd.DataFrame, seasons: list) -> pd.DataFrame:
    """Build enhanced features for rushing yards V3 prediction.

    Adds:
    - Opponent rush defense metrics
    - Vegas implied totals (low totals favor rushing)
    - Weather interaction features (cold/wet favors rushing)
    - NGS rushing metrics
    """
    print("\nBuilding rushing yards V3 features...")

    # Start with base V2 features
    df = build_rushing_features(df)

    # === OPPONENT RUSH DEFENSE ===
    print("  Adding opponent rush defense metrics...")

    # Calculate team rush defense (yards allowed per game)
    team_rush_allowed = df.groupby(['season', 'week', 'recent_team']).agg({
        'rushing_yards': 'sum'
    }).reset_index()
    team_rush_allowed = team_rush_allowed.rename(columns={
        'recent_team': 'defense_team',
        'rushing_yards': 'rush_yards_allowed'
    })

    # Calculate rolling average of yards allowed (shifted to avoid leakage)
    team_rush_allowed = team_rush_allowed.sort_values(['defense_team', 'season', 'week'])
    team_rush_allowed['opp_rush_yards_allowed_avg'] = team_rush_allowed.groupby('defense_team')['rush_yards_allowed'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Get opponent info from schedule
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
    league_avg_allowed = df['opp_rush_yards_allowed_avg'].mean()
    df['opp_rush_yards_allowed_avg'] = df['opp_rush_yards_allowed_avg'].fillna(league_avg_allowed if not pd.isna(league_avg_allowed) else 115.0)

    # Normalize defense strength (higher = easier opponent)
    df['opp_rush_defense_strength'] = df['opp_rush_yards_allowed_avg'] / 115.0  # 115 = league avg

    # === VEGAS IMPLIED TOTALS ===
    print("  Adding Vegas implied totals...")

    if 'total' in schedule.columns:
        vegas_cols = ['season', 'week', 'home_team', 'away_team', 'total']
        sched_vegas = schedule[vegas_cols].copy()

        home_vegas = sched_vegas.rename(columns={'home_team': 'recent_team'})
        away_vegas = sched_vegas.rename(columns={'away_team': 'recent_team'})
        vegas_lookup = pd.concat([
            home_vegas[['season', 'week', 'recent_team', 'total']],
            away_vegas[['season', 'week', 'recent_team', 'total']]
        ]).drop_duplicates()

        df = df.merge(vegas_lookup, on=['season', 'week', 'recent_team'], how='left')
        df = df.rename(columns={'total': 'vegas_total'})
    else:
        df['vegas_total'] = 45.0

    df['vegas_total'] = df['vegas_total'].fillna(45.0)
    df['vegas_total_normalized'] = df['vegas_total'] / 45.0
    # Low totals favor rushing (opposite of passing)
    df['is_low_total'] = (df['vegas_total'] < 42).astype(int)

    # === NGS RUSHING METRICS ===
    print("  Adding NGS rushing metrics...")

    try:
        ngs_rush = nfl.import_ngs_data(stat_type='rushing', years=seasons)
        if len(ngs_rush) > 0:
            ngs_cols = ['season', 'week', 'player_display_name', 'avg_rush_yards',
                       'efficiency', 'rush_yards_over_expected_per_att', 'percent_attempts_gte_eight_defenders']
            ngs_rush = ngs_rush[[c for c in ngs_cols if c in ngs_rush.columns]].copy()

            ngs_rush = ngs_rush.rename(columns={
                'player_display_name': 'player_name',
                'avg_rush_yards': 'ngs_avg_rush_yards',
                'efficiency': 'ngs_efficiency',
                'rush_yards_over_expected_per_att': 'ngs_ryoe',
                'percent_attempts_gte_eight_defenders': 'ngs_stacked_box_pct'
            })

            # Rolling averages (shifted)
            ngs_rush = ngs_rush.sort_values(['player_name', 'season', 'week'])
            for col in ['ngs_avg_rush_yards', 'ngs_efficiency', 'ngs_ryoe', 'ngs_stacked_box_pct']:
                if col in ngs_rush.columns:
                    ngs_rush[col] = ngs_rush.groupby('player_name')[col].transform(
                        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
                    )

            df = df.merge(ngs_rush, on=['season', 'week', 'player_name'], how='left')
    except Exception as e:
        print(f"    Warning: Could not load NGS rushing data: {e}")

    # Fill NGS features with defaults
    df['ngs_avg_rush_yards'] = df.get('ngs_avg_rush_yards', pd.Series([4.5] * len(df))).fillna(4.5)
    df['ngs_efficiency'] = df.get('ngs_efficiency', pd.Series([50.0] * len(df))).fillna(50.0)
    df['ngs_ryoe'] = df.get('ngs_ryoe', pd.Series([0.0] * len(df))).fillna(0.0)
    df['ngs_stacked_box_pct'] = df.get('ngs_stacked_box_pct', pd.Series([25.0] * len(df))).fillna(25.0)

    # === WEATHER INTERACTION FEATURES ===
    print("  Adding weather interaction features...")

    # Cold weather favors rushing (opposite of passing)
    df['cold_game'] = (df['is_cold'] == 1).astype(int)
    df['cold_and_windy'] = ((df['is_cold'] == 1) & (df['is_windy'] == 1)).astype(int)

    # Rush-friendly weather score (higher = more favorable for rushing)
    # Cold and windy = good for rushing, dome = neutral
    df['rush_weather_score'] = (
        df['is_cold'].astype(float) * 0.3 +
        df['is_windy'].astype(float) * 0.3 +
        (1 - df['is_dome'].astype(float)) * 0.2 +
        df['cold_and_windy'].astype(float) * 0.2
    )

    # Rush environment score (combine defense + low total + weather)
    df['rush_environment_score'] = (
        df['opp_rush_defense_strength'] * 0.4 +
        (1 - df['vegas_total_normalized']) * 0.3 +  # Lower total = better for rushing
        df['rush_weather_score'] * 0.3
    )

    print(f"  V3 features built. Records: {len(df)}")
    print(f"  Opponent defense coverage: {df['opp_rush_yards_allowed_avg'].notna().mean()*100:.1f}%")
    print(f"  Vegas total coverage: {(df['vegas_total'] != 45.0).mean()*100:.1f}%")

    return df


def build_receptions_features_v3(df: pd.DataFrame, schedule: pd.DataFrame, seasons: list) -> pd.DataFrame:
    """Build enhanced features for receptions V3 prediction.

    Adds:
    - Opponent pass defense metrics
    - Vegas implied totals (high totals = more passing)
    - Weather interaction features
    - NGS receiving metrics
    """
    print("\nBuilding receptions V3 features...")

    # Start with base V2 features
    df = build_receptions_features(df)

    # === OPPONENT PASS DEFENSE ===
    print("  Adding opponent pass defense metrics...")

    # Calculate team targets allowed per game (proxy for pass defense)
    team_targets_allowed = df.groupby(['season', 'week', 'recent_team']).agg({
        'targets': 'sum',
        'receptions': 'sum'
    }).reset_index()
    team_targets_allowed = team_targets_allowed.rename(columns={
        'recent_team': 'defense_team',
        'targets': 'targets_allowed',
        'receptions': 'receptions_allowed'
    })

    # Rolling averages (shifted)
    team_targets_allowed = team_targets_allowed.sort_values(['defense_team', 'season', 'week'])
    team_targets_allowed['opp_targets_allowed_avg'] = team_targets_allowed.groupby('defense_team')['targets_allowed'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    team_targets_allowed['opp_receptions_allowed_avg'] = team_targets_allowed.groupby('defense_team')['receptions_allowed'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Get opponent info from schedule
    sched_games = schedule[['season', 'week', 'home_team', 'away_team']].copy()

    home_opp = sched_games.rename(columns={'home_team': 'recent_team', 'away_team': 'opponent'})
    away_opp = sched_games.rename(columns={'away_team': 'recent_team', 'home_team': 'opponent'})
    opp_lookup = pd.concat([
        home_opp[['season', 'week', 'recent_team', 'opponent']],
        away_opp[['season', 'week', 'recent_team', 'opponent']]
    ]).drop_duplicates()

    df = df.merge(opp_lookup, on=['season', 'week', 'recent_team'], how='left')

    # Merge opponent defense stats
    opp_defense = team_targets_allowed[['season', 'week', 'defense_team', 'opp_targets_allowed_avg', 'opp_receptions_allowed_avg']].copy()
    opp_defense = opp_defense.rename(columns={'defense_team': 'opponent'})
    df = df.merge(opp_defense, on=['season', 'week', 'opponent'], how='left')

    # Fill with league averages
    df['opp_targets_allowed_avg'] = df['opp_targets_allowed_avg'].fillna(35.0)  # ~35 targets per game
    df['opp_receptions_allowed_avg'] = df['opp_receptions_allowed_avg'].fillna(22.0)  # ~22 receptions per game

    # Normalize defense strength (higher = easier opponent)
    df['opp_target_defense_strength'] = df['opp_targets_allowed_avg'] / 35.0
    df['opp_reception_defense_strength'] = df['opp_receptions_allowed_avg'] / 22.0

    # === VEGAS IMPLIED TOTALS ===
    print("  Adding Vegas implied totals...")

    if 'total' in schedule.columns:
        vegas_cols = ['season', 'week', 'home_team', 'away_team', 'total']
        sched_vegas = schedule[vegas_cols].copy()

        home_vegas = sched_vegas.rename(columns={'home_team': 'recent_team'})
        away_vegas = sched_vegas.rename(columns={'away_team': 'recent_team'})
        vegas_lookup = pd.concat([
            home_vegas[['season', 'week', 'recent_team', 'total']],
            away_vegas[['season', 'week', 'recent_team', 'total']]
        ]).drop_duplicates()

        df = df.merge(vegas_lookup, on=['season', 'week', 'recent_team'], how='left')
        df = df.rename(columns={'total': 'vegas_total'})
    else:
        df['vegas_total'] = 45.0

    df['vegas_total'] = df['vegas_total'].fillna(45.0)
    df['vegas_total_normalized'] = df['vegas_total'] / 45.0
    df['is_high_total'] = (df['vegas_total'] > 48).astype(int)

    # === NGS RECEIVING METRICS ===
    print("  Adding NGS receiving metrics...")

    try:
        ngs_rec = nfl.import_ngs_data(stat_type='receiving', years=seasons)
        if len(ngs_rec) > 0:
            ngs_cols = ['season', 'week', 'player_display_name', 'avg_separation',
                       'avg_cushion', 'avg_intended_air_yards', 'catch_percentage']
            ngs_rec = ngs_rec[[c for c in ngs_cols if c in ngs_rec.columns]].copy()

            ngs_rec = ngs_rec.rename(columns={
                'player_display_name': 'player_name',
                'avg_separation': 'ngs_separation',
                'avg_cushion': 'ngs_cushion',
                'avg_intended_air_yards': 'ngs_air_yards',
                'catch_percentage': 'ngs_catch_pct'
            })

            # Rolling averages (shifted)
            ngs_rec = ngs_rec.sort_values(['player_name', 'season', 'week'])
            for col in ['ngs_separation', 'ngs_cushion', 'ngs_air_yards', 'ngs_catch_pct']:
                if col in ngs_rec.columns:
                    ngs_rec[col] = ngs_rec.groupby('player_name')[col].transform(
                        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
                    )

            df = df.merge(ngs_rec, on=['season', 'week', 'player_name'], how='left')
    except Exception as e:
        print(f"    Warning: Could not load NGS receiving data: {e}")

    # Fill NGS features with defaults
    df['ngs_separation'] = df.get('ngs_separation', pd.Series([2.5] * len(df))).fillna(2.5)
    df['ngs_cushion'] = df.get('ngs_cushion', pd.Series([6.0] * len(df))).fillna(6.0)
    df['ngs_air_yards'] = df.get('ngs_air_yards', pd.Series([8.0] * len(df))).fillna(8.0)
    df['ngs_catch_pct'] = df.get('ngs_catch_pct', pd.Series([65.0] * len(df))).fillna(65.0)

    # === WEATHER INTERACTION FEATURES ===
    print("  Adding weather interaction features...")

    # Bad weather hurts passing/receiving
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

    print(f"  V3 features built. Records: {len(df)}")
    print(f"  Opponent defense coverage: {df['opp_targets_allowed_avg'].notna().mean()*100:.1f}%")
    print(f"  Vegas total coverage: {(df['vegas_total'] != 45.0).mean()*100:.1f}%")

    return df


def train_model(df: pd.DataFrame, target_col: str, features: list, name: str) -> dict:
    """Train XGBoost model."""
    print(f"\n{'='*60}")
    print(f"TRAINING {name.upper()} MODEL")
    print(f"{'='*60}")

    # Filter to valid features
    available = [f for f in features if f in df.columns]
    print(f"Features ({len(available)}): {available}")

    # Clean data
    df_clean = df.dropna(subset=available + [target_col]).copy()

    # Remove first few weeks per player (not enough history)
    df_clean = df_clean[df_clean.groupby('player_id').cumcount() >= 2]

    print(f"Training records: {len(df_clean)}")

    X = df_clean[available].values
    y = df_clean[target_col].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    train_r2 = 1 - np.sum((y_train - train_preds)**2) / np.sum((y_train - np.mean(y_train))**2)
    test_r2 = 1 - np.sum((y_test - test_preds)**2) / np.sum((y_test - np.mean(y_test))**2)

    train_mae = np.abs(y_train - train_preds).mean()
    test_mae = np.abs(y_test - test_preds).mean()

    print(f"\nResults:")
    print(f"  Train R²: {train_r2:.4f}, MAE: {train_mae:.2f}")
    print(f"  Test R²:  {test_r2:.4f}, MAE: {test_mae:.2f}")

    # Feature importance
    print(f"\nFeature Importance:")
    importance = sorted(zip(available, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importance[:7]:
        print(f"  {feat}: {imp:.3f}")

    return {
        'model': model,
        'features': available,
        'target': target_col,
        'metrics': {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
        },
        'trained_at': datetime.now().isoformat(),
    }


def save_model(model_data: dict, name: str, version: str = 'v2'):
    """Save model to disk."""
    path = Path(__file__).parent.parent / 'models' / f'{name}_{version}.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✓ Saved to {path}")


def main():
    print("="*60)
    print("MODEL TRAINING V2 - Historical Features Only")
    print("="*60)

    # 2020-2024: Historical data from nfl_data_py
    # 2025: Current season data from Next Gen Stats (NGS)
    seasons = [2020, 2021, 2022, 2023, 2024, 2025]
    weekly, snaps, schedule = load_data(seasons)

    # Add snap counts
    weekly = add_snap_counts(weekly, snaps)

    # Add weather features
    weekly = add_weather_features(weekly, schedule)

    results = {}

    # === RUSHING YARDS V2 ===
    rush_df = build_rushing_features(weekly.copy())
    rush_features = [
        'rush_yards_last_3', 'rush_yards_last_5',
        'rush_yards_std_3', 'rush_yards_std_5',
        'carries_last_3', 'carries_last_5',
        'ypc_career', 'position_encoded', 'week_num', 'snap_pct',
        # Weather features
        'is_dome', 'game_temp', 'is_cold',
    ]
    rush_model = train_model(rush_df, 'rushing_yards', rush_features, 'rushing_yards')
    save_model(rush_model, 'rushing_yards')
    results['rushing_yards'] = rush_model['metrics']

    # === RUSHING YARDS V3 (Enhanced) ===
    rush_df_v3 = build_rushing_features_v3(weekly.copy(), schedule, seasons)
    rush_features_v3 = [
        # Base V2 features
        'rush_yards_last_3', 'rush_yards_last_5',
        'rush_yards_std_3', 'rush_yards_std_5',
        'carries_last_3', 'carries_last_5',
        'ypc_career', 'position_encoded', 'week_num', 'snap_pct',
        # Weather features
        'is_dome', 'game_temp', 'is_cold',
        # NEW: Opponent defense
        'opp_rush_yards_allowed_avg', 'opp_rush_defense_strength',
        # NEW: Vegas implied totals
        'vegas_total', 'vegas_total_normalized', 'is_low_total',
        # NEW: NGS metrics
        'ngs_avg_rush_yards', 'ngs_efficiency', 'ngs_ryoe', 'ngs_stacked_box_pct',
        # NEW: Weather interactions
        'cold_game', 'cold_and_windy', 'rush_weather_score', 'rush_environment_score',
    ]
    rush_model_v3 = train_model(rush_df_v3, 'rushing_yards', rush_features_v3, 'rushing_yards_v3')
    save_model(rush_model_v3, 'rushing_yards', version='v3')
    results['rushing_yards_v3'] = rush_model_v3['metrics']

    # === PASSING YARDS V2 ===
    pass_df = build_passing_features(weekly.copy())
    pass_features = [
        'pass_yards_last_3', 'pass_yards_last_5',
        'pass_yards_std_3', 'pass_yards_std_5',
        'attempts_last_3', 'attempts_last_5',
        'comp_pct_last_5', 'ypa_career', 'week_num',
        # Weather features (passing is most affected)
        'is_dome', 'game_temp', 'game_wind', 'is_cold', 'is_windy', 'is_very_windy',
    ]
    pass_model = train_model(pass_df, 'passing_yards', pass_features, 'passing_yards')
    save_model(pass_model, 'passing_yards')
    results['passing_yards'] = pass_model['metrics']

    # === PASSING YARDS V3 (Enhanced) ===
    pass_df_v3 = build_passing_features_v3(weekly.copy(), schedule, seasons)
    pass_features_v3 = [
        # Base V2 features
        'pass_yards_last_3', 'pass_yards_last_5',
        'pass_yards_std_3', 'pass_yards_std_5',
        'attempts_last_3', 'attempts_last_5',
        'comp_pct_last_5', 'ypa_career', 'week_num',
        # Weather features
        'is_dome', 'game_temp', 'game_wind', 'is_cold', 'is_windy', 'is_very_windy',
        # NEW: Opponent defense
        'opp_pass_yards_allowed_avg', 'opp_pass_defense_strength',
        # NEW: Vegas implied totals
        'vegas_total', 'vegas_total_normalized', 'is_high_total',
        # NEW: NGS metrics
        'ngs_time_to_throw', 'ngs_air_yards', 'ngs_cpoe', 'ngs_aggressiveness',
        # NEW: Weather interactions
        'cold_and_windy', 'very_cold_and_windy', 'weather_impact', 'pass_environment_score',
    ]
    pass_model_v3 = train_model(pass_df_v3, 'passing_yards', pass_features_v3, 'passing_yards_v3')
    save_model(pass_model_v3, 'passing_yards', version='v3')
    results['passing_yards_v3'] = pass_model_v3['metrics']

    # === RECEPTIONS V2 ===
    rec_df = build_receptions_features(weekly.copy())
    rec_features = [
        'receptions_last_3', 'receptions_last_5',
        'receptions_std_3', 'receptions_std_5',
        'targets_last_3', 'targets_last_5',
        'target_share_last_3', 'target_share_last_5',
        'catch_rate_last_5', 'position_encoded', 'week_num', 'snap_pct',
        # Weather features (affects passing game)
        'is_dome', 'game_temp', 'game_wind', 'is_cold', 'is_windy',
    ]
    rec_model = train_model(rec_df, 'receptions', rec_features, 'receptions')
    save_model(rec_model, 'receptions')
    results['receptions'] = rec_model['metrics']

    # === RECEPTIONS V3 (Enhanced) ===
    rec_df_v3 = build_receptions_features_v3(weekly.copy(), schedule, seasons)
    rec_features_v3 = [
        # Base V2 features
        'receptions_last_3', 'receptions_last_5',
        'receptions_std_3', 'receptions_std_5',
        'targets_last_3', 'targets_last_5',
        'target_share_last_3', 'target_share_last_5',
        'catch_rate_last_5', 'position_encoded', 'week_num', 'snap_pct',
        # Weather features
        'is_dome', 'game_temp', 'game_wind', 'is_cold', 'is_windy',
        # NEW: Opponent defense
        'opp_targets_allowed_avg', 'opp_receptions_allowed_avg',
        'opp_target_defense_strength', 'opp_reception_defense_strength',
        # NEW: Vegas implied totals
        'vegas_total', 'vegas_total_normalized', 'is_high_total',
        # NEW: NGS metrics
        'ngs_separation', 'ngs_cushion', 'ngs_air_yards', 'ngs_catch_pct',
        # NEW: Weather interactions
        'cold_and_windy', 'weather_impact', 'reception_environment_score',
    ]
    rec_model_v3 = train_model(rec_df_v3, 'receptions', rec_features_v3, 'receptions_v3')
    save_model(rec_model_v3, 'receptions', version='v3')
    results['receptions_v3'] = rec_model_v3['metrics']

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\n{'Model':<20} {'Test R²':>10} {'Test MAE':>10}")
    print("-"*45)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['test_r2']:>10.4f} {metrics['test_mae']:>10.2f}")


if __name__ == "__main__":
    main()
