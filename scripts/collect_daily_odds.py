#!/usr/bin/env python3
"""
Daily odds collection script.

Run this once per day to fetch odds and calculate edges.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.data.odds_api import OddsAPIClient
from src.sports_betting.analysis.edge_calculator import EdgeCalculator


def main():
    """Collect daily odds and calculate edges."""
    print("=" * 60)
    print("DAILY ODDS COLLECTION")
    print("=" * 60)
    print()

    # Initialize API client
    client = OddsAPIClient()

    if not client.api_key:
        print("❌ No API key configured!")
        print("\nTo set up:")
        print("1. Sign up for free at: https://the-odds-api.com/")
        print("2. Get your API key")
        print("3. Add to .env file: ODDS_API_KEY=your_key_here")
        print("\nAlternatively, use manual entry:")
        print("  python -m src.sports_betting.data.manual_odds_entry")
        return

    # Check if we need to fetch today
    if not client.should_fetch_new_odds():
        print("✓ Already fetched odds today. Using cached data.")
        print(f"  Cache location: {client.cache_dir}")
        print("\nTo force refresh, delete today's cache file or use --force flag")
        return

    print("Fetching odds from The Odds API...")
    print("(This will use API credits)")
    print()

    # Get upcoming games
    events = client.get_nfl_games()
    print(f"Found {len(events)} upcoming NFL games")
    print()

    # Limit to first 5 games to conserve credits
    # Cost: 1 market × 1 region × 5 events = 5 credits
    max_events = 5
    event_ids = [e['id'] for e in events[:max_events]]

    print(f"Fetching odds for {len(event_ids)} games (saves API credits)")
    for i, event in enumerate(events[:max_events], 1):
        print(f"  {i}. {event['away_team']} @ {event['home_team']}")
    print()

    # Fetch odds for receiving yards only (most accurate model)
    markets = ['player_reception_yds']

    odds_data = client.fetch_and_cache_daily_odds(
        markets=markets,
        event_ids=event_ids
    )

    if not odds_data:
        print("❌ Failed to fetch odds")
        return

    print(f"✓ Successfully fetched odds")
    print(f"  Events fetched: {odds_data.get('events_count', 0)}")
    print(f"  Credits used: {odds_data.get('total_cost', 0)}")
    print(f"  Credits remaining: {client.credits_remaining}")
    print()

    # Store in database
    print("Storing odds in database...")
    stored = client.store_odds_in_database(odds_data)
    print(f"✓ Stored {stored} props")
    print()

    # Calculate edges
    print("Calculating betting edges...")
    calculator = EdgeCalculator()

    # For current week (you'd need to determine current week dynamically)
    current_week = 13  # Update this
    current_season = 2024

    edges = calculator.find_edges_for_week(
        week=current_week,
        season=current_season,
        min_edge=0.03  # 3% minimum edge
    )

    if edges:
        print(f"✓ Found {len(edges)} betting opportunities!")
        print()

        # Display report
        report = calculator.format_edge_report(edges)
        print(report)

        # Store edges in database
        calculator.store_edges_in_database(edges)
        print("\n✓ Edges stored in database")

    else:
        print("No edges found for this week.")
        print("This means:")
        print("  - Market lines align well with model predictions")
        print("  - OR not enough predictions generated yet")
        print("  - OR minimum edge threshold too high")

    print()
    print("=" * 60)
    print("Usage Summary:")
    print(f"  Today's credits used: {odds_data.get('total_cost', 0)}")
    print(f"  Remaining this month: {client.credits_remaining}")
    print(f"  Free tier limit: 500 credits/month")
    print()
    print("Next steps:")
    print("  1. Review edges above")
    print("  2. Record paper trades: python scripts/record_paper_trades.py")
    print("  3. After games, evaluate: python scripts/evaluate_paper_trades.py")
    print("  4. View ROI report: python scripts/view_paper_trading_report.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
