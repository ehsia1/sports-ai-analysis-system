#!/usr/bin/env python3
"""
Collect actual results and score Week 13 predictions.

Run this after games complete (Monday Dec 2, 2025 or later).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
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


def load_predictions_from_db() -> pd.DataFrame:
    """Load our predictions and props from database."""
    conn = sqlite3.connect('data/sports_betting.db')

    # Get all Week 13 props
    props = pd.read_sql("""
        SELECT
            p.name as player_name,
            p.position,
            pr.market,
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
        WHERE g.game_date >= '2025-11-30'
          AND g.game_date <= '2025-12-02'
    """, conn)
    conn.close()

    return props


def fetch_week_stats(week: int = 13) -> dict:
    """Fetch actual stats for a specific week from NGS."""
    print(f"\nFetching Week {week} actual stats from NGS...")

    stats = {}

    # NGS Receiving (has receptions and yards)
    try:
        ngs_rec = nfl.import_ngs_data('receiving', [2025])
        week_rec = ngs_rec[ngs_rec['week'] == week]
        if len(week_rec) > 0:
            stats['ngs_receiving'] = week_rec
            print(f"  ✓ NGS receiving: {len(week_rec)} records")
        else:
            print(f"  ⏳ NGS receiving: Week {week} not available yet")
    except Exception as e:
        print(f"  ✗ NGS receiving: {e}")

    # NGS Rushing (has rush_yards)
    try:
        ngs_rush = nfl.import_ngs_data('rushing', [2025])
        week_rush = ngs_rush[ngs_rush['week'] == week]
        if len(week_rush) > 0:
            stats['ngs_rushing'] = week_rush
            print(f"  ✓ NGS rushing: {len(week_rush)} records")
        else:
            print(f"  ⏳ NGS rushing: Week {week} not available yet")
    except Exception as e:
        print(f"  ✗ NGS rushing: {e}")

    # NGS Passing (has pass_yards)
    try:
        ngs_pass = nfl.import_ngs_data('passing', [2025])
        week_pass = ngs_pass[ngs_pass['week'] == week]
        if len(week_pass) > 0:
            stats['ngs_passing'] = week_pass
            print(f"  ✓ NGS passing: {len(week_pass)} records")
        else:
            print(f"  ⏳ NGS passing: Week {week} not available yet")
    except Exception as e:
        print(f"  ✗ NGS passing: {e}")

    return stats


def get_player_stat(player_name: str, market: str, stats: dict) -> float:
    """Get actual stat for a player from available data sources."""
    norm_name = normalize_name(player_name)

    # Map market to stat columns
    # NGS is primary source (has actual stats), PFR weekly only has advanced metrics
    market_config = {
        'player_reception_yds': {
            'sources': ['ngs_receiving'],
            'columns': {'ngs_receiving': 'yards'},
            'name_cols': {'ngs_receiving': 'player_display_name'}
        },
        'player_receptions': {
            'sources': ['ngs_receiving'],
            'columns': {'ngs_receiving': 'receptions'},
            'name_cols': {'ngs_receiving': 'player_display_name'}
        },
        'player_rush_yds': {
            'sources': ['ngs_rushing'],
            'columns': {'ngs_rushing': 'rush_yards'},
            'name_cols': {'ngs_rushing': 'player_display_name'}
        },
        'player_pass_yds': {
            'sources': ['ngs_passing'],
            'columns': {'ngs_passing': 'pass_yards'},
            'name_cols': {'ngs_passing': 'player_display_name'}
        }
    }

    if market not in market_config:
        return np.nan

    config = market_config[market]

    for source in config['sources']:
        if source not in stats:
            continue

        df = stats[source]
        name_col = config['name_cols'][source]
        stat_col = config['columns'][source]

        if name_col not in df.columns or stat_col not in df.columns:
            continue

        df['norm_name'] = df[name_col].apply(normalize_name)
        match = df[df['norm_name'] == norm_name]

        if len(match) > 0:
            return match[stat_col].iloc[0]

    return np.nan


def score_predictions(props: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Score each prediction against actual results."""
    results = []

    for _, row in props.iterrows():
        actual = get_player_stat(row['player_name'], row['market'], stats)

        if pd.isna(actual):
            result = 'NO_DATA'
            won = None
            profit = 0
        else:
            # Determine bet direction based on season average vs line
            # For now, we'll use the line to determine if it's over/under
            # In practice, we'd load our actual predictions

            # Placeholder - will be updated with actual prediction logic
            result = 'PENDING'
            won = None
            profit = 0

        results.append({
            'player_name': row['player_name'],
            'market': row['market'],
            'line': row['line'],
            'over_odds': row['over_odds'],
            'under_odds': row['under_odds'],
            'matchup': row['matchup'],
            'actual': actual,
            'result': result,
            'won': won,
            'profit': profit
        })

    return pd.DataFrame(results)


def load_our_bets() -> pd.DataFrame:
    """Load our actual betting recommendations from the analysis file."""
    # Read the analysis markdown and parse the betting tables
    # For now, regenerate predictions

    import pickle

    # Load PFR data
    pfr_rec = nfl.import_seasonal_pfr('rec', [2025])
    pfr_rush = nfl.import_seasonal_pfr('rush', [2025])

    def normalize(name):
        if pd.isna(name): return ''
        name = str(name).lower().strip()
        for s in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv']:
            name = name.replace(s, '')
        return name.replace("'", "").replace(".", "").replace("-", " ").strip()

    pfr_rec['norm'] = pfr_rec['player'].apply(normalize)
    pfr_rec['rec_per_game'] = pfr_rec['rec'] / pfr_rec['g'].clip(lower=1)
    pfr_rec['rec_ypg'] = pfr_rec['yds'] / pfr_rec['g'].clip(lower=1)
    pfr_rush['norm'] = pfr_rush['player'].apply(normalize)
    pfr_rush['rush_ypg'] = pfr_rush['yds'] / pfr_rush['g'].clip(lower=1)

    conn = sqlite3.connect('data/sports_betting.db')

    all_bets = []

    # Receptions
    props = pd.read_sql("""
        SELECT p.name, pr.line, pr.over_odds, pr.under_odds,
               t1.abbreviation || ' @ ' || t2.abbreviation as matchup
        FROM props pr
        JOIN players p ON pr.player_id = p.id
        JOIN games g ON pr.game_id = g.id
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        WHERE pr.market = 'player_receptions'
          AND g.game_date >= '2025-11-30' AND g.game_date <= '2025-12-02'
        GROUP BY p.name
    """, conn)
    props['norm'] = props['name'].apply(normalize)
    merged = props.merge(pfr_rec[['norm', 'rec_per_game', 'g']], on='norm', how='left')
    merged = merged.dropna(subset=['rec_per_game'])
    merged = merged[merged['g'] >= 3]
    merged['predicted'] = merged['rec_per_game']  # Using season avg as proxy
    merged['edge_pct'] = (merged['predicted'] - merged['line']) / merged['line'] * 100
    merged['bet'] = np.where(merged['predicted'] > merged['line'], 'OVER', 'UNDER')
    merged['odds'] = np.where(merged['bet'] == 'OVER', merged['over_odds'], merged['under_odds'])
    merged['market'] = 'player_receptions'
    all_bets.append(merged[['name', 'matchup', 'market', 'line', 'predicted', 'edge_pct', 'bet', 'odds']])

    # Rushing Yards
    props = pd.read_sql("""
        SELECT p.name, pr.line, pr.over_odds, pr.under_odds,
               t1.abbreviation || ' @ ' || t2.abbreviation as matchup
        FROM props pr
        JOIN players p ON pr.player_id = p.id
        JOIN games g ON pr.game_id = g.id
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        WHERE pr.market = 'player_rush_yds'
          AND g.game_date >= '2025-11-30' AND g.game_date <= '2025-12-02'
        GROUP BY p.name
    """, conn)
    props['norm'] = props['name'].apply(normalize)
    merged = props.merge(pfr_rush[['norm', 'rush_ypg', 'g']], on='norm', how='left')
    merged = merged.dropna(subset=['rush_ypg'])
    merged = merged[merged['g'] >= 3]
    merged['predicted'] = merged['rush_ypg']
    merged['edge_pct'] = (merged['predicted'] - merged['line']) / merged['line'] * 100
    merged['bet'] = np.where(merged['predicted'] > merged['line'], 'OVER', 'UNDER')
    merged['odds'] = np.where(merged['bet'] == 'OVER', merged['over_odds'], merged['under_odds'])
    merged['market'] = 'player_rush_yds'
    all_bets.append(merged[['name', 'matchup', 'market', 'line', 'predicted', 'edge_pct', 'bet', 'odds']])

    # Receiving Yards
    props = pd.read_sql("""
        SELECT p.name, pr.line, pr.over_odds, pr.under_odds,
               t1.abbreviation || ' @ ' || t2.abbreviation as matchup
        FROM props pr
        JOIN players p ON pr.player_id = p.id
        JOIN games g ON pr.game_id = g.id
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        WHERE pr.market = 'player_reception_yds'
          AND g.game_date >= '2025-11-30' AND g.game_date <= '2025-12-02'
        GROUP BY p.name
    """, conn)
    props['norm'] = props['name'].apply(normalize)
    merged = props.merge(pfr_rec[['norm', 'rec_ypg', 'g']], on='norm', how='left')
    merged = merged.dropna(subset=['rec_ypg'])
    merged = merged[merged['g'] >= 3]
    merged['predicted'] = merged['rec_ypg']
    merged['edge_pct'] = (merged['predicted'] - merged['line']) / merged['line'] * 100
    merged['bet'] = np.where(merged['predicted'] > merged['line'], 'OVER', 'UNDER')
    merged['odds'] = np.where(merged['bet'] == 'OVER', merged['over_odds'], merged['under_odds'])
    merged['market'] = 'player_reception_yds'
    all_bets.append(merged[['name', 'matchup', 'market', 'line', 'predicted', 'edge_pct', 'bet', 'odds']])

    conn.close()

    combined = pd.concat(all_bets, ignore_index=True)
    combined['abs_edge'] = combined['edge_pct'].abs()

    # Filter to quality bets (10%+ edge)
    quality = combined[combined['abs_edge'] >= 10].copy()

    return quality


def calculate_profit(odds: float, won: bool) -> float:
    """Calculate profit/loss for a bet."""
    if won is None:
        return 0

    if won:
        if odds > 0:
            return odds / 100  # +150 = 1.5 units profit
        else:
            return 100 / abs(odds)  # -110 = 0.91 units profit
    else:
        return -1  # Lost 1 unit


def score_bets(bets: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Score our bets against actual results."""
    results = []

    for _, row in bets.iterrows():
        actual = get_player_stat(row['name'], row['market'], stats)

        if pd.isna(actual):
            won = None
            result = 'NO_DATA'
        else:
            if row['bet'] == 'OVER':
                won = actual > row['line']
            else:
                won = actual < row['line']

            result = 'WIN' if won else 'LOSS'

        profit = calculate_profit(row['odds'], won)

        results.append({
            'player_name': row['name'],
            'matchup': row['matchup'],
            'market': row['market'],
            'bet': row['bet'],
            'line': row['line'],
            'predicted': row['predicted'],
            'edge_pct': row['edge_pct'],
            'odds': row['odds'],
            'actual': actual,
            'result': result,
            'profit': profit
        })

    return pd.DataFrame(results)


def generate_report(scored: pd.DataFrame) -> str:
    """Generate performance report."""
    # Filter to scored bets only
    scored_only = scored[scored['result'].isin(['WIN', 'LOSS'])]

    if len(scored_only) == 0:
        return "No results available yet. Run this after Week 13 games complete."

    wins = (scored_only['result'] == 'WIN').sum()
    losses = (scored_only['result'] == 'LOSS').sum()
    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0

    total_profit = scored_only['profit'].sum()
    roi = total_profit / total * 100 if total > 0 else 0

    report = f"""# Week 13 (2025) Results Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Overall Performance

| Metric | Value |
|--------|-------|
| Total Bets | {total} |
| Wins | {wins} |
| Losses | {losses} |
| Win Rate | {win_rate:.1f}% |
| Total Profit | {total_profit:+.2f} units |
| ROI | {roi:+.1f}% |

## Results by Stat Type

"""

    for market in scored_only['market'].unique():
        subset = scored_only[scored_only['market'] == market]
        m_wins = (subset['result'] == 'WIN').sum()
        m_total = len(subset)
        m_profit = subset['profit'].sum()
        m_wr = m_wins / m_total * 100 if m_total > 0 else 0

        market_name = market.replace('player_', '').replace('_', ' ').title()
        report += f"### {market_name}\n"
        report += f"- Record: {m_wins}-{m_total - m_wins} ({m_wr:.0f}%)\n"
        report += f"- Profit: {m_profit:+.2f} units\n\n"

    report += "## Results by Bet Direction\n\n"

    for direction in ['OVER', 'UNDER']:
        subset = scored_only[scored_only['bet'] == direction]
        d_wins = (subset['result'] == 'WIN').sum()
        d_total = len(subset)
        d_profit = subset['profit'].sum()
        d_wr = d_wins / d_total * 100 if d_total > 0 else 0

        report += f"### {direction}\n"
        report += f"- Record: {d_wins}-{d_total - d_wins} ({d_wr:.0f}%)\n"
        report += f"- Profit: {d_profit:+.2f} units\n\n"

    report += "## Detailed Results\n\n"
    report += "| Result | Bet | Player | Line | Pred | Actual | Edge | Odds | P/L |\n"
    report += "|--------|-----|--------|------|------|--------|------|------|-----|\n"

    for _, row in scored_only.sort_values('profit', ascending=False).iterrows():
        emoji = '✅' if row['result'] == 'WIN' else '❌'
        report += f"| {emoji} | {row['bet']} | {row['player_name']} | {row['line']:.1f} | {row['predicted']:.1f} | {row['actual']:.1f} | {row['edge_pct']:+.0f}% | {row['odds']:+.0f} | {row['profit']:+.2f} |\n"

    # Bets without data
    no_data = scored[scored['result'] == 'NO_DATA']
    if len(no_data) > 0:
        report += f"\n## Bets Without Data ({len(no_data)})\n\n"
        report += "These players may not have played or data isn't available yet:\n\n"
        for _, row in no_data.iterrows():
            report += f"- {row['player_name']} ({row['market']})\n"

    return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Collect and score Week 13 results')
    parser.add_argument('--week', type=int, default=13, help='Week number to collect (default: 13)')
    parser.add_argument('--test', action='store_true', help='Test with previous week data')
    args = parser.parse_args()

    week = args.week
    if args.test:
        week = 12  # Use Week 12 for testing

    print("="*60)
    print(f"WEEK {week} RESULTS COLLECTION")
    print("="*60)

    # Load our bets
    print("\nLoading our betting recommendations...")
    bets = load_our_bets()
    print(f"  Loaded {len(bets)} quality bets")

    # Fetch actual stats
    stats = fetch_week_stats(week)

    if not stats:
        print(f"\n⚠️  No Week {week} data available yet.")
        if week == 13:
            print("   Run this script after games complete (Monday Dec 2, 2025)")
        return

    # Score bets
    print("\nScoring bets...")
    scored = score_bets(bets, stats)

    scored_count = scored['result'].isin(['WIN', 'LOSS']).sum()
    print(f"  Scored: {scored_count}/{len(scored)} bets")

    if scored_count == 0:
        print(f"\n⚠️  No Week {week} results found in data.")
        print("   Data sources may not have updated yet.")
        if week == 13:
            print("   Try again later on Monday Dec 2, 2025")
        return

    # Generate report
    report = generate_report(scored)

    # Save report
    report_path = Path(f'docs/WEEK_{week}_RESULTS.md')
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n✓ Report saved to {report_path}")

    # Print summary
    scored_only = scored[scored['result'].isin(['WIN', 'LOSS'])]
    wins = (scored_only['result'] == 'WIN').sum()
    total = len(scored_only)
    profit = scored_only['profit'].sum()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Record: {wins}-{total-wins} ({wins/total*100:.1f}% win rate)")
    print(f"Profit: {profit:+.2f} units")
    print(f"ROI: {profit/total*100:+.1f}%")


if __name__ == "__main__":
    main()
