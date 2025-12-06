#!/usr/bin/env python3
"""
Train prediction models for multiple stat types.

Stat types supported:
- receiving_yards (WR/TE/RB)
- rushing_yards (RB/QB)
- passing_yards (QB)
- receptions (WR/TE/RB)
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


# Configuration for each stat type
STAT_CONFIGS = {
    'receiving_yards': {
        'target_col': 'receiving_yards',
        'positions': ['WR', 'TE', 'RB'],
        'min_value': 0,  # Minimum yards to include in training
        'features': [
            'target_share', 'targets', 'receptions', 'catch_rate',
            'rec_yards_last_3', 'rec_yards_last_5', 'targets_last_3',
            'opp_rec_yards_allowed', 'snap_pct', 'position_encoded'
        ]
    },
    'rushing_yards': {
        'target_col': 'rushing_yards',
        'positions': ['RB', 'QB', 'WR'],
        'min_value': 0,
        'features': [
            'carries', 'carry_share', 'rush_yards_last_3', 'rush_yards_last_5',
            'rush_attempts_last_3', 'opp_rush_yards_allowed', 'snap_pct',
            'position_encoded'
        ]
    },
    'passing_yards': {
        'target_col': 'passing_yards',
        'positions': ['QB'],
        'min_value': 0,
        'features': [
            'attempts', 'completions', 'completion_pct',
            'pass_yards_last_3', 'pass_yards_last_5', 'attempts_last_3',
            'opp_pass_yards_allowed', 'position_encoded'
        ]
    },
    'receptions': {
        'target_col': 'receptions',
        'positions': ['WR', 'TE', 'RB'],
        'min_value': 0,
        'features': [
            'target_share', 'targets', 'catch_rate',
            'receptions_last_3', 'receptions_last_5', 'targets_last_3',
            'opp_receptions_allowed', 'snap_pct', 'position_encoded'
        ]
    }
}


def load_weekly_data(seasons: list) -> pd.DataFrame:
    """Load weekly player data, using NGS for 2025+ since yearly file isn't available mid-season."""
    print(f"Loading weekly data for {seasons}...")

    historical_seasons = [s for s in seasons if s < 2025]
    include_2025 = 2025 in seasons
    dfs = []

    # Load historical data
    if historical_seasons:
        try:
            weekly = nfl.import_weekly_data(historical_seasons)
            dfs.append(weekly)
            print(f"  Historical records: {len(weekly)}")
        except Exception as e:
            print(f"  Warning: Could not load historical data: {e}")

    # Load 2025 from NGS (yearly file not available mid-season)
    if include_2025:
        try:
            ngs_rush = nfl.import_ngs_data('rushing', [2025])
            ngs_rec = nfl.import_ngs_data('receiving', [2025])
            ngs_pass = nfl.import_ngs_data('passing', [2025])

            # Filter out season totals
            ngs_rush = ngs_rush[(ngs_rush['week'] > 0) & (ngs_rush['season_type'] == 'REG')]
            ngs_rec = ngs_rec[(ngs_rec['week'] > 0) & (ngs_rec['season_type'] == 'REG')]
            ngs_pass = ngs_pass[(ngs_pass['week'] > 0) & (ngs_pass['season_type'] == 'REG')]

            # Transform rushing
            rush_df = ngs_rush.rename(columns={
                'player_gsis_id': 'player_id',
                'team_abbr': 'recent_team',
                'rush_attempts': 'carries',
                'rush_yards': 'rushing_yards',
                'rush_touchdowns': 'rushing_tds',
            })[['season', 'week', 'player_id', 'player_display_name', 'recent_team',
                'carries', 'rushing_yards', 'rushing_tds']].copy()

            # Transform receiving
            rec_df = ngs_rec.rename(columns={
                'player_gsis_id': 'player_id',
                'team_abbr': 'recent_team',
                'yards': 'receiving_yards',
                'rec_touchdowns': 'receiving_tds',
            })[['season', 'week', 'player_id', 'player_display_name', 'recent_team',
                'receptions', 'targets', 'receiving_yards', 'receiving_tds']].copy()

            # Transform passing
            pass_df = ngs_pass.rename(columns={
                'player_gsis_id': 'player_id',
                'team_abbr': 'recent_team',
                'pass_yards': 'passing_yards',
                'pass_touchdowns': 'passing_tds',
            })[['season', 'week', 'player_id', 'player_display_name', 'recent_team',
                'attempts', 'completions', 'passing_yards', 'passing_tds']].copy()

            # Merge all stats
            combined = rush_df.merge(
                rec_df,
                on=['season', 'week', 'player_id', 'player_display_name', 'recent_team'],
                how='outer'
            )
            combined = combined.merge(
                pass_df,
                on=['season', 'week', 'player_id', 'player_display_name', 'recent_team'],
                how='outer'
            )

            # Fill NaN stats with 0
            stat_cols = ['carries', 'rushing_yards', 'rushing_tds', 'receptions', 'targets',
                         'receiving_yards', 'receiving_tds', 'attempts', 'completions',
                         'passing_yards', 'passing_tds']
            for col in stat_cols:
                if col in combined.columns:
                    combined[col] = combined[col].fillna(0)

            combined['player_name'] = combined['player_display_name']
            combined['season_type'] = 'REG'

            dfs.append(combined)
            print(f"  2025 NGS records: {len(combined)}")

        except Exception as e:
            print(f"  Warning: Could not load 2025 NGS data: {e}")

    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        print(f"✓ Total: {len(result)} records")
        return result
    else:
        return pd.DataFrame()


def load_snap_counts(seasons: list) -> pd.DataFrame:
    """Load snap count data."""
    print(f"Loading snap counts...")
    snaps = nfl.import_snap_counts(seasons)
    print(f"✓ Loaded {len(snaps)} snap count records")
    return snaps


def build_features(df: pd.DataFrame, stat_type: str, snaps: pd.DataFrame) -> pd.DataFrame:
    """Build features for a specific stat type."""
    config = STAT_CONFIGS[stat_type]
    target_col = config['target_col']
    positions = config['positions']

    print(f"\nBuilding features for {stat_type}...")

    # Filter to relevant positions
    df = df[df['position'].isin(positions)].copy()

    # Filter to records with the target stat
    df = df[df[target_col].notna() & (df[target_col] >= config['min_value'])].copy()

    print(f"  Records with {target_col}: {len(df)}")

    # Sort for rolling calculations
    df = df.sort_values(['player_id', 'season', 'week'])

    # Position encoding
    pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    df['position_encoded'] = df['position'].map(pos_map).fillna(2)

    # === Common features ===
    # Catch rate (for receiving stats)
    if 'targets' in df.columns and 'receptions' in df.columns:
        df['catch_rate'] = np.where(df['targets'] > 0, df['receptions'] / df['targets'], 0)

    # === Stat-specific features ===
    if stat_type == 'receiving_yards':
        df = _add_receiving_features(df)
    elif stat_type == 'rushing_yards':
        df = _add_rushing_features(df)
    elif stat_type == 'passing_yards':
        df = _add_passing_features(df)
    elif stat_type == 'receptions':
        df = _add_receptions_features(df)

    # Add snap counts
    df = _add_snap_counts(df, snaps)

    # Add opponent defense metrics
    df = _add_opponent_defense(df, stat_type)

    return df


def _add_receiving_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add receiving-specific rolling features."""
    # Rolling receiving yards
    for window in [3, 5]:
        df[f'rec_yards_last_{window}'] = df.groupby('player_id')['receiving_yards'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'targets_last_{window}'] = df.groupby('player_id')['targets'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    # Fill NaN
    for col in df.columns:
        if 'last_' in col:
            df[col] = df[col].fillna(0)

    return df


def _add_rushing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rushing-specific rolling features."""
    # Carries/attempts
    if 'carries' not in df.columns:
        df['carries'] = df.get('rushing_attempts', df.get('attempts', 0))

    # Carry share (% of team carries)
    team_carries = df.groupby(['season', 'week', 'recent_team'])['carries'].transform('sum')
    df['carry_share'] = np.where(team_carries > 0, df['carries'] / team_carries, 0)

    # Rolling rushing yards
    for window in [3, 5]:
        df[f'rush_yards_last_{window}'] = df.groupby('player_id')['rushing_yards'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'rush_attempts_last_{window}'] = df.groupby('player_id')['carries'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    for col in df.columns:
        if 'last_' in col:
            df[col] = df[col].fillna(0)

    return df


def _add_passing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add passing-specific rolling features."""
    # Completions and attempts
    if 'completions' not in df.columns:
        df['completions'] = df.get('passing_completions', 0)
    if 'attempts' not in df.columns:
        df['attempts'] = df.get('passing_attempts', 0)

    # Completion percentage
    df['completion_pct'] = np.where(df['attempts'] > 0, df['completions'] / df['attempts'], 0)

    # Rolling passing yards
    for window in [3, 5]:
        df[f'pass_yards_last_{window}'] = df.groupby('player_id')['passing_yards'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'attempts_last_{window}'] = df.groupby('player_id')['attempts'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    for col in df.columns:
        if 'last_' in col:
            df[col] = df[col].fillna(0)

    return df


def _add_receptions_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add reception-specific rolling features."""
    # Rolling receptions
    for window in [3, 5]:
        df[f'receptions_last_{window}'] = df.groupby('player_id')['receptions'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f'targets_last_{window}'] = df.groupby('player_id')['targets'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    for col in df.columns:
        if 'last_' in col:
            df[col] = df[col].fillna(0)

    return df


def _add_snap_counts(df: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Add snap count percentage."""
    if len(snaps) == 0:
        df['snap_pct'] = 0.5
        return df

    # Calculate snap percentage
    snaps_agg = snaps.groupby(['season', 'week', 'player']).agg({
        'offense_pct': 'mean'
    }).reset_index()

    snaps_agg = snaps_agg.rename(columns={'player': 'player_name', 'offense_pct': 'snap_pct'})

    df = df.merge(snaps_agg, on=['season', 'week', 'player_name'], how='left')
    df['snap_pct'] = df['snap_pct'].fillna(0.5)

    return df


def _add_opponent_defense(df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    """Add opponent defense metrics."""
    config = STAT_CONFIGS[stat_type]
    target_col = config['target_col']

    # Calculate team defense averages
    if stat_type in ['receiving_yards', 'receptions']:
        defense_col = 'opp_rec_yards_allowed' if stat_type == 'receiving_yards' else 'opp_receptions_allowed'

        # Avg yards allowed per game by opponent
        opp_defense = df.groupby(['season', 'week', 'recent_team'])[target_col].sum().reset_index()
        opp_defense = opp_defense.rename(columns={target_col: defense_col, 'recent_team': 'opponent_team'})

        # Rolling average for defense
        opp_defense = opp_defense.sort_values(['opponent_team', 'season', 'week'])
        opp_defense[defense_col] = opp_defense.groupby('opponent_team')[defense_col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

        # Merge - need opponent info
        if 'opponent_team' in df.columns:
            df = df.merge(opp_defense[['season', 'week', 'opponent_team', defense_col]],
                         on=['season', 'week', 'opponent_team'], how='left')

        df[defense_col] = df.get(defense_col, df[target_col].mean()).fillna(df[target_col].mean())

    elif stat_type == 'rushing_yards':
        # Similar logic for rushing
        df['opp_rush_yards_allowed'] = df['rushing_yards'].mean()  # Simplified

    elif stat_type == 'passing_yards':
        df['opp_pass_yards_allowed'] = df['passing_yards'].mean()  # Simplified

    return df


def train_model(df: pd.DataFrame, stat_type: str) -> tuple:
    """Train XGBoost model for a stat type."""
    config = STAT_CONFIGS[stat_type]
    target_col = config['target_col']

    print(f"\n{'='*60}")
    print(f"TRAINING {stat_type.upper()} MODEL")
    print(f"{'='*60}")

    # Get available features
    available_features = [f for f in config['features'] if f in df.columns]
    print(f"Features: {available_features}")

    # Prepare data
    df_clean = df.dropna(subset=available_features + [target_col])
    print(f"Clean records: {len(df_clean)}")

    X = df_clean[available_features].values
    y = df_clean[target_col].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost
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
    print(f"\nTop Features:")
    importance = sorted(zip(available_features, model.feature_importances_), key=lambda x: -x[1])
    for feat, imp in importance[:5]:
        print(f"  {feat}: {imp:.3f}")

    return model, available_features, {'train_r2': train_r2, 'test_r2': test_r2, 'train_mae': train_mae, 'test_mae': test_mae}


def save_model(model, features: list, stat_type: str, metrics: dict):
    """Save model and metadata."""
    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(exist_ok=True)

    model_data = {
        'model': model,
        'features': features,
        'stat_type': stat_type,
        'metrics': metrics,
        'trained_at': datetime.now().isoformat(),
    }

    path = model_dir / f'{stat_type}_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"✓ Saved to {path}")
    return path


def main():
    print("="*60)
    print("MULTI-STAT MODEL TRAINING")
    print("="*60)

    # Configuration
    train_seasons = [2020, 2021, 2022, 2023, 2024]
    stat_types = ['receiving_yards', 'rushing_yards', 'passing_yards', 'receptions']

    # Load data once
    weekly = load_weekly_data(train_seasons)
    snaps = load_snap_counts(train_seasons)

    # Train each model
    results = {}
    for stat_type in stat_types:
        try:
            df = build_features(weekly.copy(), stat_type, snaps)
            model, features, metrics = train_model(df, stat_type)
            save_model(model, features, stat_type, metrics)
            results[stat_type] = metrics
        except Exception as e:
            print(f"❌ Failed to train {stat_type}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"\n{'Stat Type':<20} {'Test R²':>10} {'Test MAE':>10}")
    print("-"*45)
    for stat_type, metrics in results.items():
        print(f"{stat_type:<20} {metrics['test_r2']:>10.4f} {metrics['test_mae']:>10.2f}")

    print("\n✓ All models trained!")


if __name__ == "__main__":
    main()
