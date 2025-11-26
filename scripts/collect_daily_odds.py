#!/usr/bin/env python3
"""
Daily odds collection script.

Run this once per day to fetch odds and calculate edges.
"""
import sys
from pathlib import Path

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

    # Fetch odds for receiving yards only (most accurate model)
    # Cost: 1 market × 1 region = 1 credit
    markets = ['player_reception_yds']

    odds_data = client.fetch_and_cache_daily_odds(markets=markets)

    if not odds_data:
        print("❌ Failed to fetch odds")
        return

    print(f"✓ Successfully fetched odds")
    print(f"  Credits used: {client.last_request_cost}")
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
    print(f"  Today's credits used: {client.last_request_cost}")
    print(f"  Remaining this month: {client.credits_remaining}")
    print(f"  Free tier limit: 500 credits/month")
    print()
    print("Next steps:")
    print("  1. Review edges above")
    print("  2. Manually verify lines on sportsbook")
    print("  3. Place bets on highest +EV opportunities")
    print("  4. Track results for model improvement")
    print("=" * 60)


if __name__ == "__main__":
    main()
