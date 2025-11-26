#!/usr/bin/env python3
"""Test with real The Odds API integration."""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_real_odds_api():
    """Test with real The Odds API if key is available."""
    api_key = os.getenv('ODDS_API_KEY')
    
    if not api_key or api_key == 'your_odds_api_key_here':
        print("⚠️  No Odds API key found")
        print("To test with real data:")
        print("1. Get API key from: https://the-odds-api.com/")
        print("2. Set ODDS_API_KEY in .env file")
        print("3. Run this script again")
        return False
    
    try:
        from sports_betting.data.collectors.odds_api import OddsAPICollector
        
        print("🔗 Testing The Odds API connection...")
        collector = OddsAPICollector()
        
        # Test getting available sports
        print("📊 Getting available sports...")
        sports = collector.get_available_sports()
        
        nfl_sport = None
        for sport in sports:
            if 'nfl' in sport.get('key', '').lower():
                nfl_sport = sport
                break
        
        if nfl_sport:
            print(f"✅ Found NFL: {nfl_sport['title']}")
            
            # Try to get current NFL odds
            print("💰 Getting current NFL player props...")
            try:
                props = collector.get_player_props()
                
                if props:
                    print(f"✅ Retrieved {len(props)} games with props")
                    
                    # Show sample of what we got
                    for i, game in enumerate(props[:2]):  # Just first 2 games
                        home = game.get('home_team', 'Unknown')
                        away = game.get('away_team', 'Unknown')
                        print(f"\n🏈 Game {i+1}: {away} @ {home}")
                        
                        bookmakers = game.get('bookmakers', [])
                        if bookmakers:
                            print(f"   📚 Found {len(bookmakers)} bookmakers")
                            
                            for book in bookmakers[:1]:  # Just first bookmaker
                                print(f"   📖 {book.get('title', 'Unknown Book')}")
                                
                                markets = book.get('markets', [])
                                for market in markets[:3]:  # First 3 markets
                                    market_key = market.get('key', '')
                                    print(f"      📊 {market_key}")
                                    
                                    outcomes = market.get('outcomes', [])
                                    for outcome in outcomes[:2]:  # First 2 outcomes
                                        desc = outcome.get('description', 'Unknown')
                                        name = outcome.get('name', 'Unknown')
                                        price = outcome.get('price', 'N/A')
                                        point = outcome.get('point', '')
                                        
                                        if point:
                                            print(f"         • {desc} {name} {point}: {price}")
                                        else:
                                            print(f"         • {desc} {name}: {price}")
                    
                    return True
                else:
                    print("⚠️  No props data retrieved")
                    return False
                    
            except Exception as e:
                print(f"❌ Error getting props: {e}")
                return False
        else:
            print("❌ NFL not found in available sports")
            return False
            
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def test_nfl_data_collection():
    """Test NFLverse data collection."""
    try:
        from sports_betting.data.collectors.nfl_data import NFLDataCollector
        
        print("\n📊 Testing NFLverse data collection...")
        collector = NFLDataCollector()
        
        # Get current season
        current_season = collector.current_season
        print(f"🗓️  Current NFL season: {current_season}")
        
        # Try to get some basic schedule data
        print("📅 Getting recent schedule data...")
        schedule_df = None
        
        try:
            import nfl_data_py as nfl
            schedule_df = nfl.import_schedules(years=[current_season])
            
            if not schedule_df.empty:
                print(f"✅ Retrieved {len(schedule_df)} games")
                
                # Show recent/upcoming games
                recent_games = schedule_df.head(5)
                print("\n🏈 Sample games:")
                for _, game in recent_games.iterrows():
                    week = game.get('week', 'Unknown')
                    home = game.get('home_team', 'Unknown')
                    away = game.get('away_team', 'Unknown')
                    date = game.get('gameday', 'Unknown')
                    print(f"   Week {week}: {away} @ {home} ({date})")
                
                return True
            else:
                print("⚠️  No schedule data retrieved")
                return False
                
        except Exception as e:
            print(f"❌ NFLverse error: {e}")
            return False
            
    except Exception as e:
        print(f"❌ NFL data test failed: {e}")
        return False

def test_database_integration():
    """Test database operations."""
    try:
        from sports_betting.database import init_db, get_session
        from sports_betting.database.models import Team, Player
        
        print("\n🗄️  Testing database operations...")
        
        # Initialize database
        init_db()
        print("✅ Database initialized")
        
        # Test querying teams
        with get_session() as session:
            teams = session.query(Team).limit(5).all()
            print(f"✅ Found {len(teams)} teams in database")
            
            for team in teams[:3]:
                print(f"   🏈 {team.name} ({team.abbreviation})")
            
            # Count players
            player_count = session.query(Player).count()
            print(f"✅ Found {player_count} players in database")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def run_integration_workflow():
    """Run a complete integration test workflow."""
    print("\n🚀 RUNNING INTEGRATION WORKFLOW")
    print("=" * 50)
    
    results = {
        "odds_api": test_real_odds_api(),
        "nfl_data": test_nfl_data_collection(), 
        "database": test_database_integration()
    }
    
    print("\n📊 INTEGRATION TEST RESULTS:")
    print("-" * 30)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All systems operational!")
        print("🚀 Ready for live trading!")
    elif total_passed >= 2:
        print("⚡ Mostly operational")
        print("🔧 Some features need API keys or setup")
    else:
        print("⚠️  Need to complete setup")
        print("📋 Check API keys and dependencies")

def main():
    """Run live odds integration test."""
    print("🏈 LIVE ODDS API INTEGRATION TEST")
    print("=" * 50)
    
    # Check environment
    print("🔧 Checking environment...")
    
    # Load environment variables
    from pathlib import Path
    env_file = Path('.env')
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv()
        print("✅ Environment loaded from .env")
    else:
        print("⚠️  No .env file found")
    
    run_integration_workflow()
    
    print("\n💡 NEXT STEPS:")
    if not os.getenv('ODDS_API_KEY') or os.getenv('ODDS_API_KEY') == 'your_odds_api_key_here':
        print("1. Get The Odds API key: https://the-odds-api.com/")
        print("2. Update ODDS_API_KEY in .env file")
    else:
        print("1. ✅ Odds API configured")
    
    print("3. Run full analysis: python tonight_game_test.py")
    print("4. Build ML models for real predictions")

if __name__ == "__main__":
    main()