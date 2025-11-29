#!/usr/bin/env python3
"""
Generate Week 13 predictions for all stat types.
Uses properly trained models (v2) with historical features.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import sqlite3
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import nfl_data_py as nfl


def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    if pd.isna(name):
        return ''
    name = str(name).lower().strip()
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', ' v']:
        name = name.replace(suffix, '')
    name = name.replace("'", "").replace(".", "").replace("-", " ")
    return ' '.join(name.split())


def load_model(name: str):
    """Load a trained model."""
    # Try v2 first, then original
    for suffix in ['_v2.pkl', '_enhanced.pkl', '.pkl']:
        path = Path(__file__).parent.parent / 'models' / f'{name}{suffix}'
        if path.exists():
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"  Loaded {path.name}")
            return data
    return None


def get_props_from_db(market: str) -> pd.DataFrame:
    """Get Week 13 props from database."""
    conn = sqlite3.connect('data/sports_betting.db')
    props = pd.read_sql(f"""
        SELECT
            p.name as player_name,
            p.position,
            pr.line,
            pr.over_odds,
            pr.under_odds,
            t1.abbreviation || ' @ ' || t2.abbreviation as matchup
        FROM props pr
        JOIN players p ON pr.player_id = p.id
        JOIN games g ON pr.game_id = g.id
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        WHERE pr.market = '{market}'
          AND g.game_date >= '2025-11-30'
          AND g.game_date <= '2025-12-02'
        GROUP BY p.name
        ORDER BY pr.line DESC
    """, conn)
    conn.close()
    return props


def predict_rushing_yards(pfr_data: pd.DataFrame, model_data: dict) -> pd.DataFrame:
    """Generate rushing yards predictions."""
    model = model_data['model']
    features = model_data['features']

    df = pfr_data.copy()

    # Map PFR columns -> features
    # PFR rush columns: att, yds, td, x1d, ybc, yac, brk_tkl
    games = df.get('g', pd.Series([12]*len(df)))
    if isinstance(games, (int, float)):
        games = pd.Series([games]*len(df))
    games = games.clip(lower=1)

    rush_yds = df.get('yds', 0)
    carries = df.get('att', 0)

    # Per-game averages as proxy for rolling features
    df['rush_yards_last_3'] = rush_yds / games
    df['rush_yards_last_5'] = rush_yds / games
    df['rush_yards_std_3'] = rush_yds / games * 0.3  # Estimate
    df['rush_yards_std_5'] = rush_yds / games * 0.3
    df['carries_last_3'] = carries / games
    df['carries_last_5'] = carries / games
    carries_safe = np.maximum(carries, 1)
    df['ypc_career'] = (rush_yds / carries_safe).fillna(4.0)
    df['week_num'] = 13
    df['snap_pct'] = 0.6

    pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    df['position_encoded'] = df.get('pos', 'RB').map(pos_map).fillna(1)

    # Predict
    X = df[features].fillna(0).values
    predictions = model.predict(X)

    return pd.DataFrame({
        'player_name': df['player'],
        'position': df.get('pos', 'RB'),
        'predicted': predictions
    })


def predict_passing_yards(ngs_data: pd.DataFrame, model_data: dict) -> pd.DataFrame:
    """Generate passing yards predictions using NGS data."""
    model = model_data['model']
    features = model_data['features']

    # Use season totals (week 0) from NGS
    df = ngs_data[ngs_data['week'] == 0].copy()
    if len(df) == 0:
        # Fallback to all data
        df = ngs_data.copy()

    # NGS columns: pass_yards, attempts, completions
    # Estimate games played from attempts (avg ~35 attempts/game for starters)
    df['games_est'] = (df['attempts'] / 35).clip(lower=1, upper=13)

    pass_yds = df['pass_yards'].fillna(0)
    attempts = df['attempts'].fillna(0)
    completions = df['completions'].fillna(0)
    games = df['games_est']

    # Build features from season totals
    df['pass_yards_last_3'] = pass_yds / games
    df['pass_yards_last_5'] = pass_yds / games
    # Estimate std as ~15% of mean for passing
    df['pass_yards_std_3'] = pass_yds / games * 0.15
    df['pass_yards_std_5'] = pass_yds / games * 0.15
    df['attempts_last_3'] = attempts / games
    df['attempts_last_5'] = attempts / games

    attempts_safe = np.maximum(attempts, 1)
    df['comp_pct_last_5'] = completions / attempts_safe
    df['comp_pct_last_5'] = df['comp_pct_last_5'].clip(0, 1).fillna(0.65)
    df['ypa_career'] = pass_yds / attempts_safe
    df['ypa_career'] = df['ypa_career'].clip(0, 15).fillna(7.0)
    df['week_num'] = 13

    X = df[features].fillna(0).values
    predictions = model.predict(X)

    return pd.DataFrame({
        'player_name': df['player_display_name'],
        'position': 'QB',
        'predicted': predictions
    })


def predict_receptions(pfr_data: pd.DataFrame, model_data: dict) -> pd.DataFrame:
    """Generate receptions predictions."""
    model = model_data['model']
    features = model_data['features']

    df = pfr_data.copy()

    games = df.get('g', pd.Series([12]*len(df)))
    if isinstance(games, (int, float)):
        games = pd.Series([games]*len(df))
    games = games.clip(lower=1)

    receptions = df.get('rec', 0)
    targets = df.get('tgt', 0)
    target_share = df.get('tgt_percent', 0) / 100

    df['receptions_last_3'] = receptions / games
    df['receptions_last_5'] = receptions / games
    df['receptions_std_3'] = receptions / games * 0.3
    df['receptions_std_5'] = receptions / games * 0.3
    df['targets_last_3'] = targets / games
    df['targets_last_5'] = targets / games
    df['target_share_last_3'] = target_share
    df['target_share_last_5'] = target_share
    targets_safe = np.maximum(targets, 1)
    df['catch_rate_last_5'] = (receptions / targets_safe).clip(0, 1).fillna(0.65)
    df['week_num'] = 13
    df['snap_pct'] = 0.6

    pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    df['position_encoded'] = df.get('pos', 'WR').map(pos_map).fillna(2)

    X = df[features].fillna(0).values
    predictions = model.predict(X)

    return pd.DataFrame({
        'player_name': df['player'],
        'position': df.get('pos', 'WR'),
        'predicted': predictions
    })


def predict_receiving_yards() -> pd.DataFrame:
    """Use existing adaptive predictor for receiving yards."""
    from src.sports_betting.ml import ReceivingYardsPredictor

    predictor = ReceivingYardsPredictor()
    preds, _ = predictor.predict_adaptive(2025, 13)

    return pd.DataFrame({
        'player_name': preds['player_name'],
        'position': preds.get('position', 'WR'),
        'predicted': preds['predicted_yards']
    })


def find_edges(predictions: pd.DataFrame, props: pd.DataFrame, stat_name: str) -> pd.DataFrame:
    """Find betting edges."""
    # Normalize names
    predictions['name_norm'] = predictions['player_name'].apply(normalize_name)
    props['name_norm'] = props['player_name'].apply(normalize_name)

    # Merge
    merged = props.merge(predictions[['name_norm', 'predicted']], on='name_norm', how='left')

    # Calculate edge
    merged['edge'] = (merged['predicted'] - merged['line']) / merged['line'] * 100
    merged['bet'] = np.where(merged['predicted'] > merged['line'], 'OVER', 'UNDER')
    merged['odds'] = np.where(merged['bet'] == 'OVER', merged['over_odds'], merged['under_odds'])
    merged['stat_type'] = stat_name

    # Filter to matched and reasonable edges
    matched = merged[merged['predicted'].notna()].copy()

    return matched


def main():
    print("="*70)
    print("WEEK 13 MULTI-STAT PREDICTIONS")
    print("="*70)

    # Load PFR data
    print("\nLoading 2025 PFR data...")
    pfr_rec = nfl.import_seasonal_pfr('rec', [2025])
    pfr_rush = nfl.import_seasonal_pfr('rush', [2025])
    pfr_pass = nfl.import_seasonal_pfr('pass', [2025])
    print(f"  Receiving: {len(pfr_rec)}, Rushing: {len(pfr_rush)}, Passing: {len(pfr_pass)}")

    all_edges = []

    # === RECEIVING YARDS (use original working predictor) ===
    print("\n--- RECEIVING YARDS ---")
    rec_preds = predict_receiving_yards()
    rec_props = get_props_from_db('player_reception_yds')
    print(f"  Predictions: {len(rec_preds)}, Props: {len(rec_props)}")
    rec_edges = find_edges(rec_preds, rec_props, 'receiving_yards')
    quality_rec = rec_edges[(rec_edges['edge'].abs() >= 5) & (rec_edges['edge'].abs() <= 50)]
    print(f"  Quality edges (5-50%): {len(quality_rec)}")
    all_edges.append(quality_rec)

    # === RUSHING YARDS ===
    print("\n--- RUSHING YARDS ---")
    rush_model = load_model('rushing_yards')
    if rush_model and len(pfr_rush) > 0:
        rush_preds = predict_rushing_yards(pfr_rush, rush_model)
        rush_props = get_props_from_db('player_rush_yds')
        print(f"  Predictions: {len(rush_preds)}, Props: {len(rush_props)}")
        rush_edges = find_edges(rush_preds, rush_props, 'rushing_yards')
        quality_rush = rush_edges[(rush_edges['edge'].abs() >= 5) & (rush_edges['edge'].abs() <= 50)]
        print(f"  Quality edges (5-50%): {len(quality_rush)}")
        all_edges.append(quality_rush)

    # === PASSING YARDS (use NGS data which has actual pass yards) ===
    print("\n--- PASSING YARDS ---")
    pass_model = load_model('passing_yards')
    if pass_model:
        # Load NGS passing data (has pass_yards, attempts, completions)
        ngs_pass = nfl.import_ngs_data('passing', [2025])
        print(f"  Loaded NGS passing data: {len(ngs_pass)} records")
        if len(ngs_pass) > 0:
            pass_preds = predict_passing_yards(ngs_pass, pass_model)
            pass_props = get_props_from_db('player_pass_yds')
            print(f"  Predictions: {len(pass_preds)}, Props: {len(pass_props)}")
            pass_edges = find_edges(pass_preds, pass_props, 'passing_yards')
            quality_pass = pass_edges[(pass_edges['edge'].abs() >= 5) & (pass_edges['edge'].abs() <= 50)]
            print(f"  Quality edges (5-50%): {len(quality_pass)}")
            all_edges.append(quality_pass)

    # === RECEPTIONS ===
    print("\n--- RECEPTIONS ---")
    rec_model = load_model('receptions')
    if rec_model and len(pfr_rec) > 0:
        catch_preds = predict_receptions(pfr_rec, rec_model)
        catch_props = get_props_from_db('player_receptions')
        print(f"  Predictions: {len(catch_preds)}, Props: {len(catch_props)}")
        catch_edges = find_edges(catch_preds, catch_props, 'receptions')
        quality_catch = catch_edges[(catch_edges['edge'].abs() >= 5) & (catch_edges['edge'].abs() <= 50)]
        print(f"  Quality edges (5-50%): {len(quality_catch)}")
        all_edges.append(quality_catch)

    # Combine all edges
    combined = pd.concat(all_edges, ignore_index=True)
    combined = combined.sort_values('edge', key=abs, ascending=False)

    # Display results
    print("\n" + "="*70)
    print("ALL QUALITY EDGES - WEEK 13")
    print("="*70)

    for stat in combined['stat_type'].unique():
        subset = combined[combined['stat_type'] == stat]
        over_count = (subset['bet'] == 'OVER').sum()
        under_count = (subset['bet'] == 'UNDER').sum()

        print(f"\n--- {stat.upper()} ({len(subset)} edges: {over_count} OVER, {under_count} UNDER) ---")
        print(f"{'Bet':<6} {'Player':<24} {'Line':>6} {'Pred':>6} {'Edge':>8} {'Odds':>6}")
        print("-"*65)

        for _, row in subset.head(12).iterrows():
            print(f"{row['bet']:<6} {row['player_name']:<24} {row['line']:>6.1f} {row['predicted']:>6.1f} {row['edge']:>+7.1f}% {row['odds']:>+6.0f}")

    # Save to markdown
    output = f"""# Week 13 (2025) All Stats Predictions

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary
- Total quality edges: {len(combined)}
"""
    for stat in combined['stat_type'].unique():
        count = len(combined[combined['stat_type'] == stat])
        output += f"- {stat.replace('_', ' ').title()}: {count} edges\n"

    for stat in combined['stat_type'].unique():
        subset = combined[combined['stat_type'] == stat]
        output += f"\n## {stat.replace('_', ' ').title()}\n\n"
        output += "| Bet | Player | Matchup | Line | Pred | Edge | Odds | Actual | Result |\n"
        output += "|-----|--------|---------|------|------|------|------|--------|--------|\n"

        for _, row in subset.head(15).iterrows():
            matchup = row.get('matchup', '')
            output += f"| {row['bet']} | {row['player_name']} | {matchup} | {row['line']:.1f} | {row['predicted']:.1f} | {row['edge']:+.1f}% | {row['odds']:+.0f} | | |\n"

    output += "\n## Post-Game Analysis\n*Fill in Monday Dec 2*\n"

    with open('docs/WEEK_13_ALL_PREDICTIONS.md', 'w') as f:
        f.write(output)

    print(f"\n✓ Saved to docs/WEEK_13_ALL_PREDICTIONS.md")
    print(f"\nTotal edges: {len(combined)}")


if __name__ == "__main__":
    main()
