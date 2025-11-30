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
        notify=not getattr(args, "no_notify", False),
    )

    print(result.summary())
    print()

    # Print edge report if available
    for stage_result in result.stage_results:
        if stage_result.stage_name == "calculate_edges" and "edges" in stage_result.data:
            edges = stage_result.data["edges"]
            if edges:
                print()
                print("=" * 50)
                print("BETTING EDGES")
                print("=" * 50)
                # Re-generate report with user's --top value
                from src.sports_betting.analysis.edge_calculator import EdgeCalculator
                calculator = EdgeCalculator()
                print(calculator.format_edge_report(edges, top_n=args.top))

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


def cmd_notify(orchestrator: Orchestrator, args: argparse.Namespace) -> int:
    """Send Discord notifications for edges."""
    from src.sports_betting.notifications.discord import send_edge_alert, DiscordNotifier
    from src.sports_betting.analysis.edge_calculator import EdgeCalculator

    notifier = DiscordNotifier()

    if not notifier.enabled:
        print("Error: Discord webhook URL not configured")
        print("Set DISCORD_WEBHOOK_URL in your .env file")
        return 1

    # Test mode - send sample notifications
    if args.test:
        print("Sending test notifications to Discord...")
        print()

        calc = EdgeCalculator()

        # Test 1: Strong over edge (good weather)
        print("1. Sending strong OVER edge (dome game)...")
        reasoning1 = calc.generate_reasoning(
            player="Ja'Marr Chase",
            position="WR",
            market="player_reception_yds",
            prediction=95.2,
            line=72.5,
            side="over",
            edge_pct=12.1,
            ev_pct=15.3,
            confidence=0.88,
            weather="Dome",
            weather_warning=None,
        )
        send_edge_alert(
            player_name="Ja'Marr Chase",
            stat_type="player_reception_yds",
            line=72.5,
            prediction=95.2,
            edge_pct=12.1,
            direction="over",
            confidence=0.88,
            game="CIN @ DAL",
            ev_pct=15.3,
            reasoning=reasoning1['short'],
            reasoning_detailed=reasoning1['detailed'],
            weather="Dome",
        )

        # Test 2: Under edge with weather warning
        print("2. Sending UNDER edge (bad weather)...")
        reasoning2 = calc.generate_reasoning(
            player="Brock Purdy",
            position="QB",
            market="player_pass_yds",
            prediction=218.5,
            line=245.5,
            side="under",
            edge_pct=6.8,
            ev_pct=9.2,
            confidence=0.72,
            weather="Snow, 28°F, 15mph wind",
            weather_warning="Snow, 28°F, 15mph wind",
        )
        send_edge_alert(
            player_name="Brock Purdy",
            stat_type="player_pass_yds",
            line=245.5,
            prediction=218.5,
            edge_pct=6.8,
            direction="under",
            confidence=0.72,
            game="SF @ CLE",
            ev_pct=9.2,
            reasoning=reasoning2['short'],
            reasoning_detailed=reasoning2['detailed'],
            weather="Snow, 28°F, 15mph wind",
            weather_warning="Snow, 28°F, 15mph wind",
        )

        print()
        print("✓ Test notifications sent! Check your Discord channel.")
        return 0

    # Send real edges from predictions file
    status = orchestrator.get_status()
    season = args.season or status['season']
    week = args.week or status['week']

    predictions_file = Path(f"data/predictions/predictions_{season}_week{week}.json")
    if not predictions_file.exists():
        print(f"No predictions found for {season} Week {week}")
        print("Run 'orchestrate.py pre-game' first")
        return 1

    # For now, show what would be sent (edges need to be calculated with props)
    print(f"Discord notifications ready for {season} Week {week}")
    print()
    print("To send edge alerts, edges must be calculated with live odds data.")
    print("Run: orchestrate.py stage collect_odds")
    print("Then: orchestrate.py stage calculate_edges")
    print()
    print("Or send a test notification with: orchestrate.py notify --test")

    return 0


def cmd_weather(orchestrator: Orchestrator, args: argparse.Namespace) -> int:
    """Set or fetch weather for games."""
    from src.sports_betting.data.weather import set_game_weather, get_weather_service

    ws = get_weather_service()

    # If fetching all weather
    if args.fetch_all:
        status = orchestrator.get_status()
        season = args.season or status['season']
        week = args.week or status['week']

        print()
        print(f"Fetching weather for {season} Week {week}...")
        print()

        results = ws.fetch_all_outdoor_games(season, week)

        print(f"Weather for {len(results)} games:")
        print()
        for key, w in sorted(results.items()):
            status = ""
            if w.is_dome:
                status = " (dome)"
            elif w.is_bad_weather:
                status = f" ⚠️ BAD WEATHER (-{(1 - w.weather_impact)*100:.0f}% confidence)"
            print(f"  {key}: {w.summary}{status}")
        print()
        return 0

    # Manual weather set
    if not args.away_team or not args.home_team or not args.conditions:
        print("Error: Must provide away_team, home_team, and conditions")
        print("  Or use --fetch to fetch weather for all games")
        return 1

    set_game_weather(
        home_team=args.home_team.upper(),
        away_team=args.away_team.upper(),
        conditions=args.conditions,
        temp_f=args.temp,
        wind_mph=args.wind,
    )

    weather = ws._manual_weather
    print()
    print("Weather set successfully!")
    print()
    for key, w in weather.items():
        print(f"  {key}: {w.summary}")
        if w.is_bad_weather:
            print(f"    → Bad weather: confidence will be reduced by {(1 - w.weather_impact)*100:.0f}%")
    print()
    print("Run 'orchestrate.py stage calculate_edges' to recalculate with weather adjustments")
    print()

    return 0


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


def cmd_parlay(orchestrator: Orchestrator, args: argparse.Namespace) -> int:
    """Generate parlay recommendations."""
    from src.sports_betting.analysis.parlay_generator import ParlayGenerator
    from src.sports_betting.analysis.edge_calculator import EdgeCalculator

    status = orchestrator.get_status()
    season = args.season or status['season']
    week = args.week or status['week']

    print()
    print("=" * 60)
    print(f"PARLAY GENERATOR - {season} Week {week}")
    print("=" * 60)
    print()

    # Get edges
    print("Fetching edges...")
    calculator = EdgeCalculator()
    edges = calculator.find_edges_for_week(week=week, season=season, min_edge=0.03)

    if not edges:
        print("No edges found. Run 'orchestrate.py pre-game' first.")
        return 1

    print(f"Found {len(edges)} edges")
    print()

    # Generate parlays
    generator = ParlayGenerator(
        min_leg_probability=args.min_prob,
        min_leg_ev=args.min_leg_ev / 100,  # Convert from percentage
        min_parlay_ev=args.min_parlay_ev / 100,
        max_legs=args.max_legs,
        max_candidates=args.max_candidates,
    )

    parlays_by_size = generator.generate_all_parlays(edges, max_per_size=args.top)

    # Print report
    report = generator.format_parlay_report(parlays_by_size)
    print(report)

    # Store in database if requested
    if args.save:
        all_parlays = []
        for parlays in parlays_by_size.values():
            all_parlays.extend(parlays)
        stored = generator.store_parlays(all_parlays)
        print(f"Stored {stored} parlays in database")

    # Send Discord notifications if requested
    if args.notify:
        print("Sending parlay notifications to Discord...")
        sent = generator.send_discord_notifications(parlays_by_size, max_per_size=2)
        print(f"Sent {sent} parlay notifications")

    return 0


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
        notify=not getattr(args, "no_notify", False),
    )

    print(f"Status:  {result.status.value}")
    print(f"Message: {result.message}")

    if result.data:
        print()
        print("Data:")
        # Keys to skip (large data structures)
        skip_keys = {"report", "edges", "predictions", "all_predictions"}
        for key, value in result.data.items():
            if key in skip_keys:
                continue
            # Format lists/dicts nicely
            if isinstance(value, (list, dict)) and len(str(value)) > 100:
                if isinstance(value, list):
                    print(f"  {key}: [{len(value)} items]")
                else:
                    print(f"  {key}: {{{len(value)} keys}}")
            else:
                print(f"  {key}: {value}")

        # Print edge report if edges are available
        if "edges" in result.data and result.data["edges"]:
            print()
            from src.sports_betting.analysis.edge_calculator import EdgeCalculator
            calculator = EdgeCalculator()
            top_n = getattr(args, "top", 20)
            print(calculator.format_edge_report(result.data["edges"], top_n=top_n))

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
    pre_game.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top edges to display (default: 20)",
    )
    pre_game.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip Discord notifications",
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
    stage.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top edges to display (default: 20)",
    )
    stage.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip Discord notifications",
    )

    # Notify command
    notify = subparsers.add_parser("notify", help="Send Discord notifications")
    notify.add_argument(
        "--test",
        action="store_true",
        help="Send a test notification to verify Discord setup",
    )

    # Weather command
    weather = subparsers.add_parser("weather", help="Set or fetch weather for games")
    weather.add_argument(
        "away_team",
        nargs="?",
        help="Away team abbreviation (e.g., SF)",
    )
    weather.add_argument(
        "home_team",
        nargs="?",
        help="Home team abbreviation (e.g., CLE)",
    )
    weather.add_argument(
        "conditions",
        nargs="?",
        choices=["clear", "rain", "snow", "wind"],
        help="Weather conditions",
    )
    weather.add_argument(
        "--temp",
        type=float,
        help="Temperature in Fahrenheit",
    )
    weather.add_argument(
        "--wind",
        type=float,
        help="Wind speed in mph",
    )
    weather.add_argument(
        "--fetch",
        dest="fetch_all",
        action="store_true",
        help="Fetch live weather from Weather.gov for all outdoor games",
    )

    # Parlay command
    parlay = subparsers.add_parser("parlay", help="Generate parlay recommendations")
    parlay.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of parlays per size to show (default: 5)",
    )
    parlay.add_argument(
        "--min-prob",
        type=float,
        default=0.55,
        help="Minimum probability per leg (default: 0.55)",
    )
    parlay.add_argument(
        "--min-leg-ev",
        type=float,
        default=5.0,
        help="Minimum EV%% per leg (default: 5)",
    )
    parlay.add_argument(
        "--min-parlay-ev",
        type=float,
        default=15.0,
        help="Minimum combined EV%% for parlay (default: 15)",
    )
    parlay.add_argument(
        "--save",
        action="store_true",
        help="Save parlays to database",
    )
    parlay.add_argument(
        "--notify",
        action="store_true",
        help="Send top parlays to Discord",
    )
    parlay.add_argument(
        "--max-legs",
        type=int,
        default=5,
        help="Maximum legs per parlay (default: 5, max: 20). Uses greedy search for 5+ legs.",
    )
    parlay.add_argument(
        "--max-candidates",
        type=int,
        default=30,
        help="Maximum candidate legs to consider (default: 30, max: 50)",
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
    elif args.command == "weather":
        return cmd_weather(orchestrator, args)
    elif args.command == "notify":
        return cmd_notify(orchestrator, args)
    elif args.command == "parlay":
        return cmd_parlay(orchestrator, args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
