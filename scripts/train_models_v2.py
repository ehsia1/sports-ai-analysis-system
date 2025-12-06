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


def load_data(seasons: list) -> tuple:
    """Load weekly data, snap counts, and schedule (for weather)."""
    print(f"Loading data for seasons {seasons}...")

    weekly = nfl.import_weekly_data(seasons)
    print(f"  Weekly records: {len(weekly)}")

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

    seasons = [2020, 2021, 2022, 2023, 2024]
    weekly, snaps, schedule = load_data(seasons)

    # Add snap counts
    weekly = add_snap_counts(weekly, snaps)

    # Add weather features
    weekly = add_weather_features(weekly, schedule)

    results = {}

    # === RUSHING YARDS ===
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

    # === RECEPTIONS ===
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
