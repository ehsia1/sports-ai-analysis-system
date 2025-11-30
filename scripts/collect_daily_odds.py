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

from src.sports_betting.config import get_settings
from src.sports_betting.utils import get_logger
from src.sports_betting.data.odds_api import OddsAPIClient
from src.sports_betting.analysis.edge_calculator import EdgeCalculator

logger = get_logger(__name__)


def get_current_week() -> tuple[int, int]:
    """Get current NFL season and week.

    TODO: Move to utils/nfl_schedule.py in Phase 2
    """
    # For now, hardcode but log a warning
    # Phase 2 will implement dynamic detection
    season = 2024
    week = 13
    logger.warning(f"Using hardcoded week {week} - Phase 2 will add dynamic detection")
    return season, week


def main():
    """Collect daily odds and calculate edges."""
    settings = get_settings()

    logger.info("=" * 50)
    logger.info("DAILY ODDS COLLECTION")
    logger.info("=" * 50)

    # Initialize API client
    client = OddsAPIClient()

    if not client.api_key:
        logger.error("No API key configured!")
        logger.info("To set up:")
        logger.info("  1. Sign up for free at: https://the-odds-api.com/")
        logger.info("  2. Get your API key")
        logger.info("  3. Add to .env file: ODDS_API_KEY=your_key_here")
        return 1

    # Check if we need to fetch today
    if not client.should_fetch_new_odds():
        logger.info("Already fetched odds today - using cached data")
        logger.debug(f"Cache location: {client.cache_dir}")
        return 0

    logger.info("Fetching odds from The Odds API...")

    # Get upcoming games
    events = client.get_nfl_games()
    logger.info(f"Found {len(events)} upcoming NFL games")

    # Limit to conserve credits (1 market × 1 region × N events = N credits)
    max_events = min(5, len(events))
    event_ids = [e['id'] for e in events[:max_events]]

    logger.info(f"Fetching odds for {len(event_ids)} games")
    for i, event in enumerate(events[:max_events], 1):
        logger.debug(f"  {i}. {event['away_team']} @ {event['home_team']}")

    # Fetch odds for receiving yards (most accurate model)
    markets = ['player_reception_yds']

    odds_data = client.fetch_and_cache_daily_odds(
        markets=markets,
        event_ids=event_ids
    )

    if not odds_data:
        logger.error("Failed to fetch odds")
        return 1

    logger.info(f"Successfully fetched odds")
    logger.info(f"  Events: {odds_data.get('events_count', 0)}")
    logger.info(f"  Credits used: {odds_data.get('total_cost', 0)}")
    logger.info(f"  Credits remaining: {client.credits_remaining}")

    # Store in database
    logger.info("Storing odds in database...")
    stored = client.store_odds_in_database(odds_data)
    logger.info(f"Stored {stored} props")

    # Calculate edges
    logger.info("Calculating betting edges...")
    calculator = EdgeCalculator()

    # Use settings for thresholds
    calculator.min_edge = settings.min_edge
    calculator.min_confidence = settings.min_confidence

    current_season, current_week = get_current_week()

    edges = calculator.find_edges_for_week(
        week=current_week,
        season=current_season,
        min_edge=settings.min_edge
    )

    if edges:
        logger.info(f"Found {len(edges)} betting opportunities!")

        # Display report
        report = calculator.format_edge_report(edges)
        print(report)  # Keep print for formatted report display

        # Store edges in database
        calculator.store_edges_in_database(edges)
        logger.info("Edges stored in database")

    else:
        logger.warning("No edges found for this week")
        logger.info("Possible reasons: market aligned with model, no predictions, or threshold too high")

    # Summary
    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info(f"  Credits used today: {odds_data.get('total_cost', 0)}")
    logger.info(f"  Remaining this month: {client.credits_remaining}")
    logger.info(f"  Monthly limit: {settings.odds_api_monthly_limit}")
    logger.info("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
