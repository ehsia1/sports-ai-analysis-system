"""Script to collect NFL data for 2024 season."""

import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.data.collectors.nfl_data import NFLDataCollector

def main():
    print("=" * 60)
    print("NFL DATA COLLECTION SCRIPT")
    print("=" * 60)

    collector = NFLDataCollector()

    # 0. Collect teams (required first!)
    print("\n[0/5] Collecting NFL teams...")
    teams_count = collector.collect_teams()
    print(f"✓ Collected {teams_count} teams")

    # 1. Collect 2024-2025 schedule
    print("\n[1/5] Collecting 2024-2025 NFL schedule...")
    games_count = collector.collect_schedule([2024, 2025])
    print(f"✓ Collected {games_count} games")

    # 2. Collect 2024-2025 rosters (current)
    print("\n[2/5] Collecting 2024-2025 NFL rosters...")
    roster_count = collector.collect_rosters([2024, 2025])
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
    print(f"Teams:              {teams_count}")
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
