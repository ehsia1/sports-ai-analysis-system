#!/usr/bin/env python3
"""Test The Odds API connection."""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.data.odds_api import OddsAPIClient


def main():
    client = OddsAPIClient()

    print("Testing The Odds API connection...")
    print(f"API Key: {client.api_key[:10]}..." if client.api_key else "No API key")
    print()

    # Test 1: Get sports (free endpoint)
    print("1. Testing /sports endpoint (free)...")
    sports = client.get_sports()
    if sports:
        print(f"✓ Success! Found {len(sports)} sports")
        nfl = [s for s in sports if 'nfl' in s.get('key', '').lower()]
        if nfl:
            print(f"   NFL sport: {nfl[0]}")
    else:
        print("❌ Failed to get sports")
    print()

    # Test 2: Get NFL events (free endpoint)
    print("2. Testing /events endpoint (free)...")
    events = client.get_nfl_games()
    if events:
        print(f"✓ Success! Found {len(events)} upcoming NFL games")
        if events:
            print(f"   Example: {events[0].get('home_team')} vs {events[0].get('away_team')}")
    else:
        print("❌ Failed to get events (might be offseason)")
    print()

    # Test 3: Get odds with different endpoint
    print("3. Testing player props endpoint...")
    print("   Note: This costs credits!")

    # Try the correct endpoint format
    import requests
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
    params = {"apiKey": client.api_key}

    response = requests.get(url, params=params)
    print(f"   Events response: {response.status_code}")
    if response.status_code == 200:
        events = response.json()
        if events:
            # Try getting odds for the first event
            event_id = events[0]['id']
            print(f"   Trying to get odds for event: {event_id}")

            odds_url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
            odds_params = {
                "apiKey": client.api_key,
                "regions": "us",
                "markets": "player_reception_yds",
                "oddsFormat": "american"
            }
            odds_response = requests.get(odds_url, params=odds_params)
            print(f"   Odds response: {odds_response.status_code}")
            if odds_response.status_code != 200:
                print(f"   Error: {odds_response.text}")
    else:
        print(f"   Error: {response.text}")


if __name__ == "__main__":
    main()
