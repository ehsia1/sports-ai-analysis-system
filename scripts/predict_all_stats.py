#!/usr/bin/env python3
"""
Generate predictions for all stat types and compare to odds.
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


# Market name mapping (API -> our stat type)
MARKET_TO_STAT = {
    'player_reception_yds': 'receiving_yards',
    'player_rush_yds': 'rushing_yards',
    'player_pass_yds': 'passing_yards',
    'player_receptions': 'receptions',
}

STAT_TO_MARKET = {v: k for k, v in MARKET_TO_STAT.items()}


def load_model(stat_type: str):
    """Load a trained model."""
    model_path = Path(__file__).parent.parent / 'models' / f'{stat_type}_model.pkl'

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None, None

    with open(model_path, 'rb') as f:
        data = pickle.load(f)

    return data['model'], data['features']


def load_current_season_data(season: int = 2025) -> pd.DataFrame:
    """Load current season data from PFR."""
    print(f"Loading {season} season data from PFR...")

    try:
        # Try PFR seasonal data
        pfr = nfl.import_seasonal_pfr('rec', [season])
        pfr_rush = nfl.import_seasonal_pfr('rush', [season])
        pfr_pass = nfl.import_seasonal_pfr('pass', [season])

        # Combine
        all_data = []

        if len(pfr) > 0:
            pfr['stat_type'] = 'receiving'
            all_data.append(pfr)

        if len(pfr_rush) > 0:
            pfr_rush['stat_type'] = 'rushing'
            all_data.append(pfr_rush)

        if len(pfr_pass) > 0:
            pfr_pass['stat_type'] = 'passing'
            all_data.append(pfr_pass)

        print(f"✓ Loaded PFR data: {len(pfr)} rec, {len(pfr_rush)} rush, {len(pfr_pass)} pass")

        return pfr, pfr_rush, pfr_pass

    except Exception as e:
        print(f"Failed to load PFR data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def generate_predictions(stat_type: str, player_data: pd.DataFrame) -> pd.DataFrame:
    """Generate predictions for a stat type."""
    model, features = load_model(stat_type)

    if model is None:
        return pd.DataFrame()

    print(f"\nGenerating {stat_type} predictions...")
    print(f"  Features needed: {features}")

    # Build features from PFR data
    df = player_data.copy()

    # Map PFR columns to our feature names
    if stat_type == 'receiving_yards':
        df['targets'] = df.get('tgt', df.get('targets', 0))
        df['receptions'] = df.get('rec', df.get('receptions', 0))
        df['receiving_yards'] = df.get('yds', df.get('receiving_yards', 0))
        df['target_share'] = df.get('tgt_percent', 0)
        df['catch_rate'] = np.where(df['targets'] > 0, df['receptions'] / df['targets'], 0)

        # Rolling features (use per-game averages)
        games = df.get('g', df.get('games', pd.Series([12]*len(df))))
        if isinstance(games, (int, float)):
            games = pd.Series([games]*len(df))
        games = games.clip(lower=1)
        df['rec_yards_last_3'] = df['receiving_yards'] / games
        df['rec_yards_last_5'] = df['receiving_yards'] / games
        df['targets_last_3'] = df['targets'] / games

    elif stat_type == 'rushing_yards':
        df['carries'] = df.get('att', df.get('carries', 0))
        df['rushing_yards'] = df.get('yds', df.get('rushing_yards', 0))
        df['carry_share'] = df.get('att_percent', 0)

        games = df.get('g', df.get('games', pd.Series([12]*len(df))))
        if isinstance(games, (int, float)):
            games = pd.Series([games]*len(df))
        games = games.clip(lower=1)
        df['rush_yards_last_3'] = df['rushing_yards'] / games
        df['rush_yards_last_5'] = df['rushing_yards'] / games
        df['rush_attempts_last_3'] = df['carries'] / games

    elif stat_type == 'passing_yards':
        df['attempts'] = df.get('att', df.get('attempts', 0))
        df['completions'] = df.get('cmp', df.get('completions', 0))
        df['passing_yards'] = df.get('yds', df.get('passing_yards', 0))
        df['completion_pct'] = np.where(df['attempts'] > 0, df['completions'] / df['attempts'], 0)

        games = df.get('g', df.get('games', pd.Series([12]*len(df))))
        if isinstance(games, (int, float)):
            games = pd.Series([games]*len(df))
        games = games.clip(lower=1)
        df['pass_yards_last_3'] = df['passing_yards'] / games
        df['pass_yards_last_5'] = df['passing_yards'] / games
        df['attempts_last_3'] = df['attempts'] / games

    elif stat_type == 'receptions':
        df['targets'] = df.get('tgt', df.get('targets', 0))
        df['receptions'] = df.get('rec', df.get('receptions', 0))
        df['target_share'] = df.get('tgt_percent', 0)
        df['catch_rate'] = np.where(df['targets'] > 0, df['receptions'] / df['targets'], 0)

        games = df.get('g', df.get('games', pd.Series([12]*len(df))))
        if isinstance(games, (int, float)):
            games = pd.Series([games]*len(df))
        games = games.clip(lower=1)
        df['receptions_last_3'] = df['receptions'] / games
        df['receptions_last_5'] = df['receptions'] / games
        df['targets_last_3'] = df['targets'] / games

    # Common features
    df['snap_pct'] = 0.6  # Default
    df['opp_rec_yards_allowed'] = 150  # Default
    df['opp_rush_yards_allowed'] = 100
    df['opp_pass_yards_allowed'] = 230
    df['opp_receptions_allowed'] = 15

    # Position encoding
    pos_map = {'QB': 0, 'RB': 1, 'WR': 2, 'TE': 3}
    pos_col = df['pos'] if 'pos' in df.columns else df.get('position', pd.Series(['WR']*len(df)))
    if isinstance(pos_col, str):
        pos_col = pd.Series([pos_col]*len(df))
    df['position_encoded'] = pos_col.map(pos_map).fillna(2)

    # Check which features we have
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        print(f"  Missing features (using defaults): {missing}")
        for f in missing:
            df[f] = 0

    # Generate predictions
    X = df[features].fillna(0).values
    predictions = model.predict(X)

    # Get player name
    name_col = 'player' if 'player' in df.columns else 'player_name'

    result = pd.DataFrame({
        'player_name': df[name_col],
        'position': df.get('pos', df.get('position', '')),
        'team': df.get('tm', df.get('team', '')),
        f'predicted_{stat_type}': predictions
    })

    print(f"  Generated {len(result)} predictions")

    return result


def get_props_from_db(market: str) -> pd.DataFrame:
    """Get props for a market from database."""
    conn = sqlite3.connect('data/sports_betting.db')

    props = pd.read_sql("""
        SELECT
            p.name as player_name,
            p.position,
            pr.line,
            pr.over_odds,
            pr.under_odds,
            g.game_date,
            t1.abbreviation || ' @ ' || t2.abbreviation as matchup
        FROM props pr
        JOIN players p ON pr.player_id = p.id
        JOIN games g ON pr.game_id = g.id
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        WHERE pr.market = ?
          AND g.game_date >= '2025-11-30'
          AND g.game_date <= '2025-12-02'
        GROUP BY p.name
        ORDER BY pr.line DESC
    """, conn, params=[market])

    conn.close()
    return props


def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    if pd.isna(name):
        return ''
    name = str(name).lower().strip()
    # Remove suffixes
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
        name = name.replace(suffix, '')
    # Remove punctuation
    name = name.replace("'", "").replace(".", "").replace("-", " ")
    # Normalize whitespace
    name = ' '.join(name.split())
    return name


def compare_predictions_to_odds(predictions: pd.DataFrame, props: pd.DataFrame, stat_type: str) -> pd.DataFrame:
    """Compare predictions to odds and find edges."""
    pred_col = f'predicted_{stat_type}'

    # Normalize names for matching
    predictions['name_norm'] = predictions['player_name'].apply(normalize_name)
    props['name_norm'] = props['player_name'].apply(normalize_name)

    # Merge on normalized name
    merged = props.merge(predictions[['name_norm', pred_col]], on='name_norm', how='left')

    # Calculate edge
    merged['edge'] = (merged[pred_col] - merged['line']) / merged['line'] * 100
    merged['bet'] = np.where(merged[pred_col] > merged['line'], 'OVER', 'UNDER')
    merged['odds'] = np.where(merged['bet'] == 'OVER', merged['over_odds'], merged['under_odds'])

    # Filter to matched
    matched = merged[merged[pred_col].notna()].copy()

    return matched


def main():
    print("="*70)
    print("MULTI-STAT PREDICTION SYSTEM")
    print("="*70)

    # Load current season data
    pfr_rec, pfr_rush, pfr_pass = load_current_season_data(2025)

    all_edges = []

    # Process each stat type
    stat_configs = [
        ('receiving_yards', pfr_rec, 'player_reception_yds'),
        ('rushing_yards', pfr_rush, 'player_rush_yds'),
        ('passing_yards', pfr_pass, 'player_pass_yds'),
        ('receptions', pfr_rec, 'player_receptions'),
    ]

    for stat_type, player_data, market in stat_configs:
        if len(player_data) == 0:
            print(f"\nSkipping {stat_type} - no data")
            continue

        # Generate predictions
        preds = generate_predictions(stat_type, player_data)

        if len(preds) == 0:
            continue

        # Get odds from database
        props = get_props_from_db(market)
        print(f"  Props in database: {len(props)}")

        if len(props) == 0:
            continue

        # Compare and find edges
        edges = compare_predictions_to_odds(preds, props, stat_type)
        edges['stat_type'] = stat_type
        edges['market'] = market

        # Filter to quality edges
        quality = edges[
            (edges['edge'].notna()) &
            (edges['edge'].abs() >= 5) &
            (edges['edge'].abs() <= 50)
        ].copy()

        print(f"  Quality edges (5-50%): {len(quality)}")

        all_edges.append(quality)

    # Combine all edges
    if all_edges:
        combined = pd.concat(all_edges, ignore_index=True)
        combined = combined.sort_values('edge', key=abs, ascending=False)

        print("\n" + "="*70)
        print("ALL QUALITY EDGES - WEEK 13")
        print("="*70)

        # Group by stat type
        for stat_type in combined['stat_type'].unique():
            subset = combined[combined['stat_type'] == stat_type]

            print(f"\n--- {stat_type.upper()} ({len(subset)} edges) ---")
            print(f"{'Bet':<6} {'Player':<22} {'Line':>6} {'Pred':>6} {'Edge':>8} {'Odds':>6}")
            print("-"*60)

            for _, row in subset.head(10).iterrows():
                pred_col = f'predicted_{stat_type}'
                pred = row[pred_col]
                print(f"{row['bet']:<6} {row['player_name']:<22} {row['line']:>6.1f} {pred:>6.1f} {row['edge']:>+7.1f}% {row['odds']:>+6.0f}")

        # Save to file
        output_path = Path('docs/WEEK_13_ALL_PREDICTIONS.md')

        md_content = f"""# Week 13 (2025) Multi-Stat Predictions

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Summary
- Total quality edges: {len(combined)}
- By stat type:
"""
        for stat_type in combined['stat_type'].unique():
            count = len(combined[combined['stat_type'] == stat_type])
            md_content += f"  - {stat_type}: {count}\n"

        for stat_type in combined['stat_type'].unique():
            subset = combined[combined['stat_type'] == stat_type].head(15)
            pred_col = f'predicted_{stat_type}'

            md_content += f"\n## {stat_type.replace('_', ' ').title()}\n\n"
            md_content += "| Bet | Player | Line | Pred | Edge | Odds | Actual | Result |\n"
            md_content += "|-----|--------|------|------|------|------|--------|--------|\n"

            for _, row in subset.iterrows():
                pred = row[pred_col]
                md_content += f"| {row['bet']} | {row['player_name']} | {row['line']:.1f} | {pred:.1f} | {row['edge']:+.1f}% | {row['odds']:+.0f} | | |\n"

        md_content += "\n## Post-Game Analysis\n*Fill in after games*\n"

        with open(output_path, 'w') as f:
            f.write(md_content)

        print(f"\n✓ Saved to {output_path}")
        print(f"\nTotal edges found: {len(combined)}")

    else:
        print("\nNo edges found")


if __name__ == "__main__":
    main()
