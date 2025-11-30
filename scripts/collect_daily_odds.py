#!/usr/bin/env python3
"""
Daily odds collection script.

Thin wrapper around the orchestrator's collect_odds stage.
Run this once per day to fetch odds from The Odds API.

Usage:
    python scripts/collect_daily_odds.py
    python scripts/collect_daily_odds.py --force  # Force fetch even if cached
"""
import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.workflow import Orchestrator, StageStatus
from src.sports_betting.utils import get_logger, setup_logging

# Set up logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Collect daily odds using the orchestrator."""
    parser = argparse.ArgumentParser(description="Collect daily odds from The Odds API")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force fetch even if cached",
    )
    parser.add_argument(
        "--week",
        type=int,
        help="Override week number",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Override season year",
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("DAILY ODDS COLLECTION")
    logger.info("=" * 50)

    orchestrator = Orchestrator()
    result = orchestrator.run_stage(
        "collect_odds",
        season=args.season,
        week=args.week,
        force=args.force,
    )

    # Display result
    logger.info(f"Status: {result.status.value}")
    logger.info(f"Message: {result.message}")

    if result.data:
        if "props_stored" in result.data:
            logger.info(f"Props stored: {result.data['props_stored']}")
        if "credits_used" in result.data:
            logger.info(f"Credits used: {result.data['credits_used']}")
        if "credits_remaining" in result.data:
            logger.info(f"Credits remaining: {result.data['credits_remaining']}")

    logger.info("=" * 50)

    return 0 if result.status in (StageStatus.SUCCESS, StageStatus.SKIPPED) else 1


if __name__ == "__main__":
    sys.exit(main())
