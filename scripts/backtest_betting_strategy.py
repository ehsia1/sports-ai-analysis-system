#!/usr/bin/env python3
"""
Comprehensive Backtesting Framework

Tests the model AND betting strategy on historical data.

Two types of backtests:
1. Model Accuracy - How good are predictions vs actuals?
2. Betting Simulation - Would we have made money betting edges?

For betting simulation, we need betting lines. Since we don't have historical
lines, we simulate them using the player's recent average (which is roughly
how books set lines).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

import nfl_data_py as nfl
from src.sports_betting.ml.feature_engineering import ReceivingYardsFeatureEngineer
import xgboost as xgb


def load_and_prepare_data(train_seasons: list, test_season: int):
    """Load data and prepare train/test split."""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    # Load all data
    all_seasons = train_seasons + [test_season]
    engineer = ReceivingYardsFeatureEngineer()
    df = engineer.build_features(seasons=all_seasons, include_snap_counts=False)

    # Get feature list
    features = engineer.get_feature_list(prediction_mode=True)

    return df, features, engineer


def train_model(df: pd.DataFrame, features: list, train_seasons: list):
    """Train model on historical data only."""
    print("\n" + "=" * 80)
    print("TRAINING MODEL")
    print("=" * 80)

    # Filter to training seasons
    train_df = df[df['season'].isin(train_seasons)].copy()

    # Prepare features
    X_train = train_df[features].values
    y_train = train_df['receiving_yards'].values

    print(f"Training on {len(train_df)} records from {train_seasons}")

    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Quick validation
    train_preds = model.predict(X_train)
    train_r2 = 1 - np.sum((y_train - train_preds)**2) / np.sum((y_train - np.mean(y_train))**2)
    print(f"Training R²: {train_r2:.4f}")

    return model


def generate_synthetic_lines(df: pd.DataFrame, method: str = 'recent_avg'):
    """
    Generate synthetic betting lines since we don't have historical lines.

    Methods:
    - 'recent_avg': Use player's last 3 game average (how books often set lines)
    - 'season_avg': Use player's season average
    - 'actual_minus_noise': Add noise to actual (for testing)
    """
    df = df.sort_values(['player_id', 'season', 'week'])

    if method == 'recent_avg':
        # Line = player's rolling 3-game average (shifted to avoid leakage)
        df['synthetic_line'] = df.groupby('player_id')['receiving_yards'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )
    elif method == 'season_avg':
        # Line = player's season average up to that point
        df['synthetic_line'] = df.groupby(['player_id', 'season'])['receiving_yards'].transform(
            lambda x: x.shift(1).expanding().mean()
        )
    else:
        # Fallback to recent_avg
        df['synthetic_line'] = df.groupby('player_id')['receiving_yards'].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        )

    # Fill NaN with position average
    position_avg = df.groupby('position')['receiving_yards'].transform('mean')
    df['synthetic_line'] = df['synthetic_line'].fillna(position_avg)

    return df


def backtest_week_by_week(
    df: pd.DataFrame,
    model,
    features: list,
    test_season: int,
    min_edge: float = 0.05,
    min_line: float = 20.0,
):
    """
    Walk forward through each week and simulate betting.

    Returns detailed results for analysis.
    """
    print("\n" + "=" * 80)
    print(f"BACKTESTING {test_season} SEASON")
    print("=" * 80)

    # Filter to test season
    test_df = df[df['season'] == test_season].copy()
    weeks = sorted(test_df['week'].unique())

    # Generate synthetic lines
    test_df = generate_synthetic_lines(test_df)

    all_results = []

    for week in weeks:
        week_df = test_df[test_df['week'] == week].copy()

        if len(week_df) == 0:
            continue

        # Filter to valid records
        week_df = week_df[week_df['synthetic_line'] >= min_line].copy()
        week_df = week_df.dropna(subset=features)

        if len(week_df) == 0:
            continue

        # Generate predictions
        X = week_df[features].values
        predictions = model.predict(X)
        week_df['prediction'] = predictions

        # Calculate edge
        week_df['edge'] = (week_df['prediction'] - week_df['synthetic_line']) / week_df['synthetic_line']
        week_df['abs_edge'] = week_df['edge'].abs()

        # Determine bet side
        week_df['bet_side'] = np.where(week_df['edge'] > 0, 'OVER', 'UNDER')

        # Determine actual result
        week_df['actual'] = week_df['receiving_yards']
        week_df['hit'] = np.where(
            week_df['bet_side'] == 'OVER',
            week_df['actual'] > week_df['synthetic_line'],
            week_df['actual'] < week_df['synthetic_line']
        )

        # Calculate profit (assuming -110 odds, bet $100)
        # Win: +$90.91, Lose: -$100
        week_df['profit'] = np.where(week_df['hit'], 90.91, -100)

        # Store results
        for _, row in week_df.iterrows():
            all_results.append({
                'season': test_season,
                'week': week,
                'player': row['player_name'],
                'position': row['position'],
                'team': row['recent_team'],
                'line': row['synthetic_line'],
                'prediction': row['prediction'],
                'actual': row['actual'],
                'edge': row['edge'],
                'bet_side': row['bet_side'],
                'hit': row['hit'],
                'profit': row['profit'],
            })

    results_df = pd.DataFrame(all_results)
    return results_df


def analyze_results(results_df: pd.DataFrame, min_edge: float = 0.05):
    """Analyze backtest results at various edge thresholds."""
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS ANALYSIS")
    print("=" * 80)

    # Overall model accuracy
    print("\n--- MODEL ACCURACY (All Predictions) ---")
    mae = np.abs(results_df['prediction'] - results_df['actual']).mean()
    rmse = np.sqrt(((results_df['prediction'] - results_df['actual'])**2).mean())

    # R² calculation
    ss_res = ((results_df['actual'] - results_df['prediction'])**2).sum()
    ss_tot = ((results_df['actual'] - results_df['actual'].mean())**2).sum()
    r2 = 1 - (ss_res / ss_tot)

    print(f"Total predictions: {len(results_df)}")
    print(f"MAE: {mae:.1f} yards")
    print(f"RMSE: {rmse:.1f} yards")
    print(f"R²: {r2:.4f}")

    # Betting results at different edge thresholds
    print("\n--- BETTING SIMULATION BY EDGE THRESHOLD ---")
    print(f"{'Threshold':<12} {'Bets':<8} {'Wins':<8} {'Win%':<8} {'Profit':<12} {'ROI':<8}")
    print("-" * 60)

    thresholds = [0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]

    for thresh in thresholds:
        bets = results_df[results_df['edge'].abs() >= thresh]

        if len(bets) == 0:
            continue

        wins = bets['hit'].sum()
        total = len(bets)
        win_rate = wins / total
        profit = bets['profit'].sum()
        roi = profit / (total * 100) * 100  # As percentage

        print(f"{thresh:>8.0%}      {total:<8} {wins:<8} {win_rate:<7.1%} ${profit:>+10.2f} {roi:>+6.1f}%")

    # Best edge threshold
    print("\n--- OPTIMAL THRESHOLD ANALYSIS ---")
    best_roi = -999
    best_thresh = 0

    for thresh in np.arange(0.01, 0.30, 0.01):
        bets = results_df[results_df['edge'].abs() >= thresh]
        if len(bets) >= 50:  # Minimum sample size
            roi = bets['profit'].sum() / (len(bets) * 100) * 100
            if roi > best_roi:
                best_roi = roi
                best_thresh = thresh

    print(f"Best threshold: {best_thresh:.0%} (ROI: {best_roi:+.1f}%)")

    # Breakdown by position
    print("\n--- RESULTS BY POSITION ---")
    for pos in ['WR', 'TE', 'RB']:
        pos_df = results_df[(results_df['position'] == pos) & (results_df['edge'].abs() >= min_edge)]
        if len(pos_df) > 0:
            win_rate = pos_df['hit'].mean()
            roi = pos_df['profit'].sum() / (len(pos_df) * 100) * 100
            print(f"{pos}: {len(pos_df)} bets, {win_rate:.1%} win rate, {roi:+.1f}% ROI")

    # Breakdown by bet side
    print("\n--- RESULTS BY BET SIDE ---")
    for side in ['OVER', 'UNDER']:
        side_df = results_df[(results_df['bet_side'] == side) & (results_df['edge'].abs() >= min_edge)]
        if len(side_df) > 0:
            win_rate = side_df['hit'].mean()
            roi = side_df['profit'].sum() / (len(side_df) * 100) * 100
            print(f"{side}: {len(side_df)} bets, {win_rate:.1%} win rate, {roi:+.1f}% ROI")

    # Weekly breakdown
    print("\n--- RESULTS BY WEEK (5%+ edge) ---")
    edge_bets = results_df[results_df['edge'].abs() >= 0.05]
    weekly = edge_bets.groupby('week').agg({
        'hit': ['count', 'sum', 'mean'],
        'profit': 'sum'
    }).round(2)
    weekly.columns = ['bets', 'wins', 'win_rate', 'profit']
    weekly['roi'] = (weekly['profit'] / (weekly['bets'] * 100) * 100).round(1)
    print(weekly.to_string())

    return results_df


def show_sample_bets(results_df: pd.DataFrame, n: int = 20):
    """Show sample of bets for inspection."""
    print("\n" + "=" * 80)
    print("SAMPLE BETS (5%+ edge)")
    print("=" * 80)

    sample = results_df[results_df['edge'].abs() >= 0.05].head(n)

    print(f"\n{'Week':<5} {'Player':<20} {'Pos':<4} {'Line':>6} {'Pred':>6} {'Actual':>6} {'Edge':>7} {'Bet':<5} {'Hit':<4} {'P&L':>8}")
    print("-" * 90)

    for _, row in sample.iterrows():
        hit_str = "✓" if row['hit'] else "✗"
        print(f"{row['week']:<5} {row['player']:<20} {row['position']:<4} {row['line']:>6.1f} {row['prediction']:>6.1f} {row['actual']:>6.1f} {row['edge']:>+6.1%} {row['bet_side']:<5} {hit_str:<4} ${row['profit']:>+7.2f}")


def main():
    print("=" * 80)
    print("COMPREHENSIVE BETTING STRATEGY BACKTEST")
    print("=" * 80)
    print("\nThis backtest:")
    print("1. Trains model on 2020-2023 data")
    print("2. Tests on 2024 season week-by-week")
    print("3. Simulates betting with synthetic lines")
    print("4. Analyzes profitability at various edge thresholds")
    print()

    # Configuration
    train_seasons = [2020, 2021, 2022, 2023]
    test_season = 2024

    # Load data
    df, features, engineer = load_and_prepare_data(train_seasons, test_season)

    # Train model
    model = train_model(df, features, train_seasons)

    # Backtest
    results = backtest_week_by_week(
        df=df,
        model=model,
        features=features,
        test_season=test_season,
        min_edge=0.03,
        min_line=20.0,
    )

    # Analyze
    analyze_results(results, min_edge=0.05)

    # Show samples
    show_sample_bets(results)

    # Save results
    output_path = Path(__file__).parent.parent / 'data' / 'backtest_results_2024.csv'
    output_path.parent.mkdir(exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")

    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
