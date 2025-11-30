#!/usr/bin/env python3
"""
Collect actual results and score predictions.

Thin wrapper around the orchestrator's score_results stage.
Run this after games complete (typically Tuesday after Monday Night Football).

Usage:
    python scripts/collect_results.py
    python scripts/collect_results.py --week 12  # Score specific week
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
    """Collect results and score predictions using the orchestrator."""
    parser = argparse.ArgumentParser(description="Score predictions against actual results")
    parser.add_argument(
        "--week",
        type=int,
        help="Week number to score (default: current week)",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Season year (default: current season)",
    )
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("RESULTS COLLECTION & SCORING")
    logger.info("=" * 50)

    orchestrator = Orchestrator()
    result = orchestrator.run_stage(
        "score_results",
        season=args.season,
        week=args.week,
    )

    # Display result
    logger.info(f"Status: {result.status.value}")
    logger.info(f"Message: {result.message}")

    if result.data:
        if "completed_games" in result.data:
            logger.info(f"Completed games: {result.data['completed_games']}/{result.data['total_games']}")
        if "predictions_count" in result.data:
            logger.info(f"Predictions scored: {result.data['predictions_count']}")
        if "paper_trades_count" in result.data:
            logger.info(f"Paper trades evaluated: {result.data['paper_trades_count']}")

    logger.info("=" * 50)

    return 0 if result.status in (StageStatus.SUCCESS, StageStatus.SKIPPED) else 1


if __name__ == "__main__":
    sys.exit(main())
