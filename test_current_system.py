"""Test script to verify current system functionality."""

import sys
from datetime import datetime

from src.sports_betting.database import (
    Game, InjuryReport, Player, RosterChange, Team, get_session
)
from src.sports_betting.data.collectors.nfl_data import NFLDataCollector

def test_database_connection():
    """Test database connectivity and tables."""
    print("=" * 60)
    print("TEST 1: Database Connection & Schema")
    print("=" * 60)

    try:
        with get_session() as session:
            # Count records in each table
            teams_count = session.query(Team).count()
            players_count = session.query(Player).count()
            games_count = session.query(Game).count()
            injuries_count = session.query(InjuryReport).count()
            roster_changes_count = session.query(RosterChange).count()

            print(f"✓ Database connected successfully")
            print(f"  - Teams: {teams_count}")
            print(f"  - Players: {players_count}")
            print(f"  - Games: {games_count}")
            print(f"  - Injury Reports: {injuries_count}")
            print(f"  - Roster Changes: {roster_changes_count}")

            return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False


def test_player_queries():
    """Test player queries with injury and roster data."""
    print("\n" + "=" * 60)
    print("TEST 2: Player Queries with Status")
    print("=" * 60)

    try:
        with get_session() as session:
            # Get some active players with their teams
            players = session.query(Player).join(Team).filter(
                Player.is_active == True
            ).limit(10).all()

            print(f"\n✓ Sample Active Players:")
            for player in players:
                status = player.current_status or "Healthy"
                injury_count = len(player.injury_reports) if player.injury_reports else 0
                print(f"  - {player.name:25} ({player.position:3}) | {player.team.abbreviation:3} | Status: {status:15} | Injuries: {injury_count}")

            # Get players with injury reports
            injured_players = session.query(Player).join(InjuryReport).distinct().limit(5).all()
            if injured_players:
                print(f"\n✓ Players with Injury Reports:")
                for player in injured_players:
                    latest_injury = player.injury_reports[0] if player.injury_reports else None
                    if latest_injury:
                        print(f"  - {player.name:25} | {latest_injury.injury_status:15} | {latest_injury.primary_injury or 'Unknown'}")

            # Get recent roster changes
            recent_changes = session.query(RosterChange).order_by(
                RosterChange.change_date.desc()
            ).limit(5).all()

            if recent_changes:
                print(f"\n✓ Recent Roster Changes:")
                for change in recent_changes:
                    player = session.query(Player).get(change.player_id)
                    if player:
                        print(f"  - {player.name:25} | {change.change_type:10} | {change.change_date.strftime('%Y-%m-%d')}")

            return True
    except Exception as e:
        print(f"✗ Player queries failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_data():
    """Test game data queries."""
    print("\n" + "=" * 60)
    print("TEST 3: Game Schedule Data")
    print("=" * 60)

    try:
        with get_session() as session:
            # Get some upcoming games
            games = session.query(Game).join(
                Team, Game.home_team_id == Team.id
            ).filter(
                Game.season == 2024,
                Game.is_completed == False
            ).limit(5).all()

            print(f"\n✓ Sample Upcoming Games (2024):")
            for game in games:
                matchup = f"{game.away_team.abbreviation} @ {game.home_team.abbreviation}"
                date = game.game_date.strftime('%Y-%m-%d')
                weather_info = ""
                if game.temperature:
                    weather_info = f" | {game.temperature}°F"
                if game.is_dome:
                    weather_info += " (Dome)"

                print(f"  - Week {game.week:2}: {matchup:15} | {date}{weather_info}")

            return True
    except Exception as e:
        print(f"✗ Game data queries failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test that feature engineering modules load."""
    print("\n" + "=" * 60)
    print("TEST 4: Feature Engineering Modules")
    print("=" * 60)

    try:
        from src.sports_betting.features.ml_features import MLFeatureEngineer
        from src.sports_betting.features.nfl_features import NFLFeatureEngineer

        ml_engineer = MLFeatureEngineer()
        nfl_engineer = NFLFeatureEngineer()

        print("✓ MLFeatureEngineer loaded successfully")
        print("  - Includes injury-aware features")
        print("  - Feature groups:", list(ml_engineer.get_feature_importance_groups().keys()))

        print("✓ NFLFeatureEngineer loaded successfully")

        return True
    except Exception as e:
        print(f"✗ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_collectors():
    """Test data collector modules."""
    print("\n" + "=" * 60)
    print("TEST 5: Data Collector Modules")
    print("=" * 60)

    try:
        collector = NFLDataCollector()

        print(f"✓ NFLDataCollector initialized")
        print(f"  - Current season: {collector.current_season}")
        print(f"  - Has weekly_roster_refresh: {hasattr(collector, 'weekly_roster_refresh')}")
        print(f"  - Has collect_injury_reports: {hasattr(collector, 'collect_injury_reports')}")

        return True
    except Exception as e:
        print(f"✗ Data collector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n🏈 NFL SPORTS BETTING SYSTEM - FUNCTIONALITY TEST")
    print("=" * 60)
    print(f"Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = []

    # Run tests
    results.append(("Database Connection", test_database_connection()))
    results.append(("Player Queries", test_player_queries()))
    results.append(("Game Data", test_game_data()))
    results.append(("Feature Engineering", test_feature_engineering()))
    results.append(("Data Collectors", test_data_collectors()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} | {test_name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ All tests passed! System is operational.")
        print("\nNext steps:")
        print("  1. Collect historical data (2022-2023)")
        print("  2. Train XGBoost models")
        print("  3. Generate parlay recommendations")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Test suite crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
