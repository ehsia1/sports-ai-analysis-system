#!/usr/bin/env python3
"""
Generate predictions for upcoming week using enhanced model.

Thin wrapper around the orchestrator's generate_predictions stage.

Usage:
    python scripts/generate_predictions.py
    python scripts/generate_predictions.py --week 14 --season 2025
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.workflow import Orchestrator, StageStatus
from src.sports_betting.utils import get_logger, setup_logging, get_current_week

# Set up logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Generate predictions using the orchestrator."""
    # Get default week/season
    default_season, default_week = get_current_week()

    parser = argparse.ArgumentParser(description="Generate predictions for upcoming week")
    parser.add_argument(
        "--season",
        type=int,
        default=default_season,
        help=f"Season year (default: {default_season})",
    )
    parser.add_argument(
        "--week",
        type=int,
        default=default_week,
        help=f"Week number (default: {default_week})",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save predictions to database",
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("PREDICTION GENERATION")
    logger.info("=" * 50)
    logger.info(f"Target: {args.season} Week {args.week}")

    orchestrator = Orchestrator()
    result = orchestrator.run_stage(
        "generate_predictions",
        season=args.season,
        week=args.week,
        save_to_db=not args.no_save,
    )

    # Display result
    logger.info(f"Status: {result.status.value}")
    logger.info(f"Message: {result.message}")

    if result.data:
        if "predictions_generated" in result.data:
            logger.info(f"Predictions generated: {result.data['predictions_generated']}")
        if "games_processed" in result.data:
            logger.info(f"Games processed: {result.data['games_processed']}")

    logger.info("=" * 50)

    # Suggest next steps
    if result.status == StageStatus.SUCCESS:
        logger.info("Next steps:")
        logger.info("  1. Fetch odds: python scripts/collect_daily_odds.py")
        logger.info("  2. Or run full workflow: python scripts/orchestrate.py pre-game")

    return 0 if result.status in (StageStatus.SUCCESS, StageStatus.SKIPPED) else 1


if __name__ == "__main__":
    sys.exit(main())
