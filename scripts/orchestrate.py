#!/usr/bin/env python3
"""
Central orchestration CLI for sports betting workflows.

Usage:
    # Check system status
    python scripts/orchestrate.py status

    # Run pre-game workflow
    python scripts/orchestrate.py pre-game

    # Run post-game workflow
    python scripts/orchestrate.py post-game

    # Run individual stage
    python scripts/orchestrate.py stage collect_odds

    # Override week/season
    python scripts/orchestrate.py pre-game --week 14 --season 2025
"""
import argparse
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.workflow import (
    Orchestrator,
    WorkflowType,
    list_stages,
)
from src.sports_betting.utils import get_logger, setup_logging

# Set up logging
setup_logging()
logger = get_logger(__name__)


def cmd_status(orchestrator: Orchestrator, args: argparse.Namespace) -> int:
    """Show current system status."""
    status = orchestrator.get_status()

    print()
    print("=" * 50)
    print("SPORTS BETTING SYSTEM STATUS")
    print("=" * 50)
    print()
    print(f"Season:     {status['season']}")
    print(f"Week:       {status['week']}")
    print(f"Phase:      {status['phase']}")
    print(f"Games:      {status['games_completed']}/{status['total_games']} completed")
    print()

    if status["first_game"]:
        print(f"First game: {status['first_game']}")
    if status["last_game"]:
        print(f"Last game:  {status['last_game']}")
    print()

    print("Runnable workflows:")
    for wf in status["runnable_workflows"]:
        print(f"  - {wf}")
    print()

    print("Runnable stages:")
    for stage in status["runnable_stages"]:
        print(f"  - {stage}")
    print()

    return 0


def cmd_pre_game(orchestrator: Orchestrator, args: argparse.Namespace) -> int:
    """Run pre-game workflow."""
    print()
    print("=" * 50)
    print("PRE-GAME WORKFLOW")
    print("=" * 50)
    print()

    result = orchestrator.run_pre_game(
        season=args.season,
        week=args.week,
        force=args.force,
    )

    print(result.summary())
    print()

    # Print edge report if available
    for stage_result in result.stage_results:
        if stage_result.stage_name == "calculate_edges" and "report" in stage_result.data:
            print()
            print("=" * 50)
            print("BETTING EDGES")
            print("=" * 50)
            print(stage_result.data["report"])

    return 0 if result.success else 1


def cmd_post_game(orchestrator: Orchestrator, args: argparse.Namespace) -> int:
    """Run post-game workflow."""
    print()
    print("=" * 50)
    print("POST-GAME WORKFLOW")
    print("=" * 50)
    print()

    result = orchestrator.run_post_game(
        season=args.season,
        week=args.week,
    )

    print(result.summary())
    print()

    return 0 if result.success else 1


def cmd_full(orchestrator: Orchestrator, args: argparse.Namespace) -> int:
    """Run full workflow."""
    print()
    print("=" * 50)
    print("FULL WORKFLOW")
    print("=" * 50)
    print()

    result = orchestrator.run_full(
        season=args.season,
        week=args.week,
        force=args.force,
    )

    print(result.summary())
    print()

    return 0 if result.success else 1


def cmd_health(orchestrator: Orchestrator, args: argparse.Namespace) -> int:
    """Run health checks."""
    from src.sports_betting.monitoring import run_health_check

    print()
    print("=" * 50)
    print("SYSTEM HEALTH CHECK")
    print("=" * 50)
    print()

    report = run_health_check(notify=args.notify)
    print(report.summary())
    print()

    return 0 if report.status.value == "healthy" else 1


def cmd_stage(orchestrator: Orchestrator, args: argparse.Namespace) -> int:
    """Run a single stage."""
    stage_name = args.stage_name

    print()
    print("=" * 50)
    print(f"RUNNING STAGE: {stage_name}")
    print("=" * 50)
    print()

    result = orchestrator.run_stage(
        stage_name,
        season=args.season,
        week=args.week,
        force=getattr(args, "force", False),
    )

    print(f"Status:  {result.status.value}")
    print(f"Message: {result.message}")

    if result.data:
        print()
        print("Data:")
        for key, value in result.data.items():
            if key != "report":  # Skip long reports
                print(f"  {key}: {value}")

    if result.duration_seconds:
        print()
        print(f"Duration: {result.duration_seconds:.1f}s")

    print()

    return 0 if result.status.value in ("success", "skipped") else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sports Betting Workflow Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    Show current week and phase
  %(prog)s pre-game                  Run odds/predictions/edges workflow
  %(prog)s post-game                 Score results after games complete
  %(prog)s stage collect_odds        Run a single stage
  %(prog)s pre-game --week 14        Override week number
        """,
    )

    # Global options
    parser.add_argument(
        "--week",
        type=int,
        help="Override week number (default: auto-detect)",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Override season year (default: auto-detect)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Status command
    subparsers.add_parser("status", help="Show current system status")

    # Pre-game workflow
    pre_game = subparsers.add_parser("pre-game", help="Run pre-game workflow")
    pre_game.add_argument(
        "--force",
        action="store_true",
        help="Force fetch even if cached",
    )

    # Post-game workflow
    subparsers.add_parser("post-game", help="Run post-game workflow")

    # Full workflow
    full = subparsers.add_parser("full", help="Run full workflow")
    full.add_argument(
        "--force",
        action="store_true",
        help="Force fetch even if cached",
    )

    # Health check
    health = subparsers.add_parser("health", help="Run system health checks")
    health.add_argument(
        "--notify",
        action="store_true",
        help="Send Discord notification if unhealthy",
    )

    # Stage command
    stage = subparsers.add_parser("stage", help="Run a single stage")
    stage.add_argument(
        "stage_name",
        choices=list_stages(),
        help="Stage to run",
    )
    stage.add_argument(
        "--force",
        action="store_true",
        help="Force execution even if cached",
    )

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = Orchestrator()

    # Route to command
    if args.command == "status":
        return cmd_status(orchestrator, args)
    elif args.command == "pre-game":
        return cmd_pre_game(orchestrator, args)
    elif args.command == "post-game":
        return cmd_post_game(orchestrator, args)
    elif args.command == "full":
        return cmd_full(orchestrator, args)
    elif args.command == "health":
        return cmd_health(orchestrator, args)
    elif args.command == "stage":
        return cmd_stage(orchestrator, args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
