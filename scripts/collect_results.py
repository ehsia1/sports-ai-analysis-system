#!/usr/bin/env python3
"""
Collect actual results and score predictions.

Run this after games complete (typically Tuesday after Monday Night Football).
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.config import get_settings
from src.sports_betting.utils import get_logger

import nfl_data_py as nfl

logger = get_logger(__name__)
settings = get_settings()


def normalize_name(name: str) -> str:
    """Normalize player name for matching."""
    if pd.isna(name):
        return ''
    name = str(name).lower().strip()
    for suffix in [' jr.', ' jr', ' sr.', ' sr', ' iii', ' ii', ' iv', ' v']:
        name = name.replace(suffix, '')
    name = name.replace("'", "").replace(".", "").replace("-", " ")
    return ' '.join(name.split())


def get_current_week() -> tuple[int, int]:
    """Get current NFL season and week.

    TODO: Move to utils/nfl_schedule.py in Phase 2
    """
    season = settings.current_season
    week = 13  # Will be dynamic in Phase 2
    logger.debug(f"Using season {season}, week {week}")
    return season, week


def load_predictions_from_db(start_date: str, end_date: str) -> pd.DataFrame:
    """Load our predictions and props from database."""
    db_path = settings.db_path
    conn = sqlite3.connect(str(db_path))

    props = pd.read_sql(f"""
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
        WHERE g.game_date >= '{start_date}'
          AND g.game_date <= '{end_date}'
    """, conn)
    conn.close()

    logger.debug(f"Loaded {len(props)} props from database")
    return props


def fetch_week_stats(week: int, season: int) -> dict:
    """Fetch actual stats for a specific week from NGS."""
    logger.info(f"Fetching Week {week} actual stats from NGS...")

    stats = {}

    # NGS Receiving
    try:
        ngs_rec = nfl.import_ngs_data('receiving', [season])
        week_rec = ngs_rec[ngs_rec['week'] == week]
        if len(week_rec) > 0:
            stats['ngs_receiving'] = week_rec
            logger.info(f"  NGS receiving: {len(week_rec)} records")
        else:
            logger.warning(f"  NGS receiving: Week {week} not available yet")
    except Exception as e:
        logger.error(f"  NGS receiving failed: {e}")

    # NGS Rushing
    try:
        ngs_rush = nfl.import_ngs_data('rushing', [season])
        week_rush = ngs_rush[ngs_rush['week'] == week]
        if len(week_rush) > 0:
            stats['ngs_rushing'] = week_rush
            logger.info(f"  NGS rushing: {len(week_rush)} records")
        else:
            logger.warning(f"  NGS rushing: Week {week} not available yet")
    except Exception as e:
        logger.error(f"  NGS rushing failed: {e}")

    # NGS Passing
    try:
        ngs_pass = nfl.import_ngs_data('passing', [season])
        week_pass = ngs_pass[ngs_pass['week'] == week]
        if len(week_pass) > 0:
            stats['ngs_passing'] = week_pass
            logger.info(f"  NGS passing: {len(week_pass)} records")
        else:
            logger.warning(f"  NGS passing: Week {week} not available yet")
    except Exception as e:
        logger.error(f"  NGS passing failed: {e}")

    return stats


def get_player_stat(player_name: str, market: str, stats: dict) -> float:
    """Get actual stat for a player from available data sources."""
    norm_name = normalize_name(player_name)

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


def load_our_bets(start_date: str, end_date: str, season: int) -> pd.DataFrame:
    """Load our actual betting recommendations."""
    logger.info("Loading betting recommendations...")

    pfr_rec = nfl.import_seasonal_pfr('rec', [season])
    pfr_rush = nfl.import_seasonal_pfr('rush', [season])

    pfr_rec['norm'] = pfr_rec['player'].apply(normalize_name)
    pfr_rec['rec_per_game'] = pfr_rec['rec'] / pfr_rec['g'].clip(lower=1)
    pfr_rec['rec_ypg'] = pfr_rec['yds'] / pfr_rec['g'].clip(lower=1)
    pfr_rush['norm'] = pfr_rush['player'].apply(normalize_name)
    pfr_rush['rush_ypg'] = pfr_rush['yds'] / pfr_rush['g'].clip(lower=1)

    db_path = settings.db_path
    conn = sqlite3.connect(str(db_path))

    all_bets = []

    # Receptions
    props = pd.read_sql(f"""
        SELECT p.name, pr.line, pr.over_odds, pr.under_odds,
               t1.abbreviation || ' @ ' || t2.abbreviation as matchup
        FROM props pr
        JOIN players p ON pr.player_id = p.id
        JOIN games g ON pr.game_id = g.id
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        WHERE pr.market = 'player_receptions'
          AND g.game_date >= '{start_date}' AND g.game_date <= '{end_date}'
        GROUP BY p.name
    """, conn)
    props['norm'] = props['name'].apply(normalize_name)
    merged = props.merge(pfr_rec[['norm', 'rec_per_game', 'g']], on='norm', how='left')
    merged = merged.dropna(subset=['rec_per_game'])
    merged = merged[merged['g'] >= 3]
    merged['predicted'] = merged['rec_per_game']
    merged['edge_pct'] = (merged['predicted'] - merged['line']) / merged['line'] * 100
    merged['bet'] = np.where(merged['predicted'] > merged['line'], 'OVER', 'UNDER')
    merged['odds'] = np.where(merged['bet'] == 'OVER', merged['over_odds'], merged['under_odds'])
    merged['market'] = 'player_receptions'
    all_bets.append(merged[['name', 'matchup', 'market', 'line', 'predicted', 'edge_pct', 'bet', 'odds']])

    # Rushing Yards
    props = pd.read_sql(f"""
        SELECT p.name, pr.line, pr.over_odds, pr.under_odds,
               t1.abbreviation || ' @ ' || t2.abbreviation as matchup
        FROM props pr
        JOIN players p ON pr.player_id = p.id
        JOIN games g ON pr.game_id = g.id
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        WHERE pr.market = 'player_rush_yds'
          AND g.game_date >= '{start_date}' AND g.game_date <= '{end_date}'
        GROUP BY p.name
    """, conn)
    props['norm'] = props['name'].apply(normalize_name)
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
    props = pd.read_sql(f"""
        SELECT p.name, pr.line, pr.over_odds, pr.under_odds,
               t1.abbreviation || ' @ ' || t2.abbreviation as matchup
        FROM props pr
        JOIN players p ON pr.player_id = p.id
        JOIN games g ON pr.game_id = g.id
        JOIN teams t1 ON g.away_team_id = t1.id
        JOIN teams t2 ON g.home_team_id = t2.id
        WHERE pr.market = 'player_reception_yds'
          AND g.game_date >= '{start_date}' AND g.game_date <= '{end_date}'
        GROUP BY p.name
    """, conn)
    props['norm'] = props['name'].apply(normalize_name)
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

    # Filter to quality bets using settings threshold
    min_edge = settings.min_edge_pct
    quality = combined[combined['abs_edge'] >= min_edge].copy()

    logger.info(f"  Loaded {len(quality)} quality bets (>= {min_edge}% edge)")
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


def generate_report(scored: pd.DataFrame, week: int, season: int) -> str:
    """Generate performance report."""
    scored_only = scored[scored['result'].isin(['WIN', 'LOSS'])]

    if len(scored_only) == 0:
        return f"No results available yet for Week {week}."

    wins = (scored_only['result'] == 'WIN').sum()
    losses = (scored_only['result'] == 'LOSS').sum()
    total = wins + losses
    win_rate = wins / total * 100 if total > 0 else 0

    total_profit = scored_only['profit'].sum()
    roi = total_profit / total * 100 if total > 0 else 0

    report = f"""# Week {week} ({season}) Results Report

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
        if len(subset) == 0:
            continue
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
        emoji = 'WIN' if row['result'] == 'WIN' else 'LOSS'
        report += f"| {emoji} | {row['bet']} | {row['player_name']} | {row['line']:.1f} | {row['predicted']:.1f} | {row['actual']:.1f} | {row['edge_pct']:+.0f}% | {row['odds']:+.0f} | {row['profit']:+.2f} |\n"

    no_data = scored[scored['result'] == 'NO_DATA']
    if len(no_data) > 0:
        report += f"\n## Bets Without Data ({len(no_data)})\n\n"
        report += "These players may not have played or data isn't available yet:\n\n"
        for _, row in no_data.iterrows():
            report += f"- {row['player_name']} ({row['market']})\n"

    return report


def main():
    parser = argparse.ArgumentParser(description='Collect and score results')
    parser.add_argument('--week', type=int, default=13, help='Week number')
    parser.add_argument('--season', type=int, default=None, help='Season year')
    parser.add_argument('--test', action='store_true', help='Test with previous week data')
    args = parser.parse_args()

    season = args.season or settings.current_season
    week = args.week
    if args.test:
        week = week - 1  # Use previous week for testing
        logger.info(f"Test mode: using week {week}")

    logger.info("=" * 50)
    logger.info(f"WEEK {week} RESULTS COLLECTION")
    logger.info("=" * 50)

    # Calculate date range for the week
    # TODO: Move to Phase 2 utils/nfl_schedule.py
    # For now, use approximate dates
    start_date = '2025-11-30'
    end_date = '2025-12-02'
    if week == 12:
        start_date = '2025-11-23'
        end_date = '2025-11-28'

    # Load our bets
    bets = load_our_bets(start_date, end_date, season)

    if len(bets) == 0:
        logger.warning("No bets found for this week")
        return 1

    # Fetch actual stats
    stats = fetch_week_stats(week, season)

    if not stats:
        logger.warning(f"No Week {week} data available yet")
        logger.info("Run this script after games complete")
        return 1

    # Score bets
    logger.info("Scoring bets...")
    scored = score_bets(bets, stats)

    scored_count = scored['result'].isin(['WIN', 'LOSS']).sum()
    logger.info(f"  Scored: {scored_count}/{len(scored)} bets")

    if scored_count == 0:
        logger.warning(f"No Week {week} results found in data")
        logger.info("Data sources may not have updated yet")
        return 1

    # Generate report
    report = generate_report(scored, week, season)

    # Save report
    report_path = settings.data_dir.parent / 'docs' / f'WEEK_{week}_RESULTS.md'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report)

    logger.info(f"Report saved to {report_path}")

    # Print summary
    scored_only = scored[scored['result'].isin(['WIN', 'LOSS'])]
    wins = (scored_only['result'] == 'WIN').sum()
    total = len(scored_only)
    profit = scored_only['profit'].sum()

    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info(f"  Record: {wins}-{total-wins} ({wins/total*100:.1f}% win rate)")
    logger.info(f"  Profit: {profit:+.2f} units")
    logger.info(f"  ROI: {profit/total*100:+.1f}%")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
