"""Script to collect NFL data for 2024 season."""

import sys
from datetime import datetime

from src.sports_betting.data.collectors.nfl_data import NFLDataCollector

def main():
    print("=" * 60)
    print("NFL DATA COLLECTION SCRIPT")
    print("=" * 60)

    collector = NFLDataCollector()

    # 1. Collect 2024 schedule
    print("\n[1/5] Collecting 2024 NFL schedule...")
    games_count = collector.collect_schedule([2024])
    print(f"✓ Collected {games_count} games")

    # 2. Collect 2024 rosters (current)
    print("\n[2/5] Collecting 2024 NFL rosters...")
    roster_count = collector.collect_rosters([2024])
    print(f"✓ Collected {roster_count} player records")

    # 3. Collect 2024 injury reports
    print("\n[3/5] Collecting 2024 injury reports...")
    injury_count = collector.collect_injury_reports([2024])
    print(f"✓ Collected {injury_count} injury reports")

    # 4. Collect historical data for training (2022-2023)
    print("\n[4/5] Collecting historical rosters (2022-2023)...")
    historical_rosters = collector.collect_rosters([2022, 2023])
    print(f"✓ Collected {historical_rosters} historical player records")

    print("\n[5/5] Collecting historical injury data (2022-2023)...")
    historical_injuries = collector.collect_injury_reports([2022, 2023])
    print(f"✓ Collected {historical_injuries} historical injury reports")

    # Summary
    print("\n" + "=" * 60)
    print("DATA COLLECTION COMPLETE!")
    print("=" * 60)
    print(f"Games:              {games_count}")
    print(f"Players:            {roster_count}")
    print(f"Injury Reports:     {injury_count + historical_injuries}")
    print(f"Historical Players: {historical_rosters}")
    print("=" * 60)

    # Weekly roster refresh
    print("\n[BONUS] Running weekly roster refresh...")
    players_updated, changes = collector.weekly_roster_refresh()
    print(f"✓ Updated {players_updated} players, tracked {changes} roster changes")

    print("\n✅ All data collection complete!")
    print("Next steps:")
    print("  - Train XGBoost models on historical data")
    print("  - Generate parlays for upcoming week")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during data collection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
