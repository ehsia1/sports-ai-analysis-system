"""Test injury data collection with fixed column mapping."""

from src.sports_betting.data.collectors.nfl_data import NFLDataCollector
from src.sports_betting.database import InjuryReport, Player, get_session

def main():
    print("=" * 60)
    print("TESTING INJURY DATA COLLECTION")
    print("=" * 60)

    collector = NFLDataCollector()

    # Collect 2024 injury reports
    print("\nCollecting 2024 injury reports...")
    injury_count = collector.collect_injury_reports([2024])
    print(f"✓ Collected {injury_count} injury reports")

    # Query some injury data
    print("\n" + "=" * 60)
    print("INJURY REPORT SAMPLES")
    print("=" * 60)

    with get_session() as session:
        # Get players with injuries
        injured_players = session.query(Player).join(InjuryReport).distinct().limit(10).all()

        print(f"\nFound {len(injured_players)} injured players (showing 10):\n")
        for player in injured_players:
            latest_injury = player.injury_reports[0] if player.injury_reports else None
            if latest_injury:
                practice = latest_injury.practice_friday or "Unknown"
                print(f"{player.name:25} | {latest_injury.injury_status:15} | {latest_injury.primary_injury or 'N/A':20} | Practice: {practice}")

        # Look for Aaron Rodgers specifically
        print("\n" + "=" * 60)
        print("AARON RODGERS INJURY STATUS")
        print("=" * 60)

        aaron = session.query(Player).filter(
            Player.name.like('%Rodgers%'),
            Player.position == 'QB'
        ).first()

        if aaron:
            print(f"\nPlayer: {aaron.name} ({aaron.position}) - {aaron.team.abbreviation if aaron.team else 'No team'}")
            print(f"Current Status: {aaron.current_status or 'Healthy'}")
            print(f"Is Active: {aaron.is_active}")

            if aaron.injury_reports:
                print(f"\nInjury Reports ({len(aaron.injury_reports)}):")
                for injury in aaron.injury_reports[:5]:  # Show up to 5 reports
                    print(f"  Week {injury.week}: {injury.injury_status:15} | {injury.primary_injury or 'N/A':20} | Practice: {injury.practice_friday or 'Unknown'}")
            else:
                print("\nNo injury reports found")
        else:
            print("\nAaron Rodgers not found in database")

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
