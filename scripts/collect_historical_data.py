#!/usr/bin/env python3
"""
Collect historical NFL data (2022-2024) for model training.
"""
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.data.collectors.nfl_data import NFLDataCollector
from src.sports_betting.database import init_db


def main():
    """Collect historical data for training."""
    print("=" * 60)
    print("HISTORICAL NFL DATA COLLECTION (2022-2024)")
    print("=" * 60)
    print()

    # Initialize database
    init_db()

    # Create collector
    collector = NFLDataCollector()

    # Years to collect
    historical_years = [2022, 2023, 2024]

    try:
        # 1. Collect schedules
        print(f"[1/4] Collecting schedules for {historical_years}...")
        game_count = collector.collect_schedule(historical_years)
        print(f"✓ Collected {game_count} games")
        print()

        # 2. Collect rosters
        print(f"[2/4] Collecting rosters for {historical_years}...")
        roster_count = collector.collect_rosters(historical_years)
        print(f"✓ Collected {roster_count} player records")
        print()

        # 3. Collect player stats (weekly)
        print(f"[3/4] Collecting weekly player stats for {historical_years}...")
        print("(This may take a few minutes...)")
        stats_count = collector.collect_player_stats(historical_years, stat_type="weekly")
        print(f"✓ Collected {stats_count} player stat records")
        print()

        # 4. Collect injury reports
        print(f"[4/4] Collecting injury reports for {historical_years}...")
        injury_count = collector.collect_injury_reports(historical_years)
        print(f"✓ Collected {injury_count} injury reports")
        print()

        print("=" * 60)
        print("HISTORICAL DATA COLLECTION COMPLETE")
        print("=" * 60)
        print(f"Total games: {game_count}")
        print(f"Total player records: {roster_count}")
        print(f"Total player stats: {stats_count}")
        print(f"Total injury reports: {injury_count}")

    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
