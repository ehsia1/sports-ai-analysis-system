#!/usr/bin/env python3
"""Test player props fetching."""
import sys
from pathlib import Path
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.data.odds_api import OddsAPIClient


def main():
    client = OddsAPIClient()

    print("Testing player props API (costs credits!)...")
    print()

    # Get just ONE event to test (costs 1 credit)
    events = client.get_nfl_games()
    if not events:
        print("No upcoming games found")
        return

    test_event = events[0]
    print(f"Testing with: {test_event['home_team']} vs {test_event['away_team']}")
    print(f"Event ID: {test_event['id']}")
    print()

    # Fetch props for just this one event
    props = client.get_player_props(
        markets=['player_reception_yds'],
        event_ids=[test_event['id']]
    )

    if props:
        print(f"✓ Success!")
        print(f"  Events fetched: {props['events_count']}")
        print(f"  Credits used: {props['total_cost']}")
        print()

        # Show structure
        if props['data']:
            event_data = props['data'][0]
            print("Response structure:")
            print(f"  Keys: {list(event_data.keys())}")

            if 'bookmakers' in event_data:
                print(f"  Bookmakers: {len(event_data['bookmakers'])}")

                if event_data['bookmakers']:
                    first_book = event_data['bookmakers'][0]
                    print(f"  Example bookmaker: {first_book.get('title')}")

                    if 'markets' in first_book:
                        print(f"  Markets: {len(first_book['markets'])}")

                        if first_book['markets']:
                            first_market = first_book['markets'][0]
                            print(f"  Market type: {first_market.get('key')}")
                            print(f"  Outcomes: {len(first_market.get('outcomes', []))}")

                            # Show first few players
                            print()
                            print("Sample players:")
                            for outcome in first_market.get('outcomes', [])[:5]:
                                print(f"    - {outcome.get('description')}: {outcome.get('name')} {outcome.get('point')} @ {outcome.get('price')}")

        # Save to file for inspection
        with open('/tmp/test_props.json', 'w') as f:
            json.dump(props, f, indent=2)
        print()
        print("Full response saved to: /tmp/test_props.json")

    else:
        print("❌ Failed to fetch props")


if __name__ == "__main__":
    main()
