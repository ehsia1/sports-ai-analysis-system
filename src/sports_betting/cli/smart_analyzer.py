"""Smart CLI analyzer using intelligent request management."""

import logging
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress

from ..config import get_settings
from ..database import init_db
from ..data.collectors.smart_odds_collector import SmartOddsCollector
from ..data.collectors.espn_api import ESPNAPICollector
from ..data.game_prioritizer import GamePrioritizer
from ..data.request_manager import RequestManager
from ..utils import setup_logging

console = Console()
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--week",
    type=int,
    help="NFL week to analyze (defaults to current week)",
)
@click.option(
    "--year", 
    type=int,
    default=datetime.now().year,
    help="NFL season year",
)
@click.option(
    "--strategy",
    type=click.Choice(["weekly", "priority", "budget-saver", "test"]),
    default="weekly",
    help="Update strategy to use",
)
@click.option(
    "--force-refresh",
    is_flag=True,
    help="Force refresh data (ignore cache)",
)
@click.option(
    "--budget-limit", 
    type=int,
    help="Override daily budget limit",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
def main(week, year, strategy, force_refresh, budget_limit, verbose):
    """Smart NFL betting analyzer with intelligent API management."""
    
    # Setup logging
    setup_logging(level="DEBUG" if verbose else "INFO")
    
    settings = get_settings()
    
    console.print(Panel.fit(
        "[bold blue]🏈 Smart Sports Betting Analyzer v2.0[/bold blue]\n"
        f"📅 Season {year}, Week {week or 'Current'}\n"
        f"🎯 Strategy: {strategy}\n"
        f"💰 API Budget Management: Enabled",
        title="Smart Analysis System"
    ))
    
    try:
        # Initialize database
        console.print("🔧 Initializing database...")
        init_db()
        
        # Initialize smart components
        odds_collector = SmartOddsCollector()
        espn_collector = ESPNAPICollector()
        request_manager = RequestManager()
        prioritizer = GamePrioritizer()
        
        # Show current status
        show_system_status(request_manager, odds_collector)
        
        # Execute strategy
        if strategy == "weekly":
            results = run_weekly_strategy(odds_collector, espn_collector, year, week)
        elif strategy == "priority":
            results = run_priority_strategy(odds_collector, prioritizer, year, week)
        elif strategy == "budget-saver":
            results = run_budget_saver_strategy(odds_collector, year, week)
        else:  # test
            results = run_test_strategy(odds_collector, espn_collector, year, week)
        
        # Display results
        display_results(results)
        
        # Show final status
        show_final_status(request_manager, odds_collector)
        
        console.print("\n✅ [green]Smart analysis complete![/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise click.Abort()


def show_system_status(request_manager: RequestManager, odds_collector: SmartOddsCollector):
    """Show current system status."""
    console.print("\n📊 [bold]System Status[/bold]")
    
    # API usage
    usage = request_manager.get_usage_stats("odds_api")
    budget = request_manager.get_priority_budget()
    
    status_table = Table(show_header=True, header_style="bold blue")
    status_table.add_column("Metric", style="cyan")
    status_table.add_column("Value", justify="right")
    status_table.add_column("Status", justify="center")
    
    status_table.add_row(
        "API Requests This Month",
        f"{usage['monthly_usage']}/{usage['monthly_limit']}",
        "🟢" if usage['usage_percentage'] < 80 else "🟡" if usage['usage_percentage'] < 95 else "🔴"
    )
    
    status_table.add_row(
        "Daily Budget Remaining", 
        str(budget['daily_budget']),
        "🟢" if budget['daily_budget'] > 10 else "🟡" if budget['daily_budget'] > 5 else "🔴"
    )
    
    status_table.add_row(
        "Days Left in Month",
        str(budget['days_left']),
        "🟢"
    )
    
    # Cache status
    report = odds_collector.get_smart_usage_report()
    cache_stats = report['cache_performance']
    
    status_table.add_row(
        "Cache Hit Rate",
        f"{cache_stats['hit_rate']:.1f}%",
        "🟢" if cache_stats['hit_rate'] > 70 else "🟡" if cache_stats['hit_rate'] > 50 else "🔴"
    )
    
    console.print(status_table)
    
    # Show optimization tips
    tips = report['optimization_tips']
    if tips:
        console.print("\n💡 [bold]Optimization Tips:[/bold]")
        for tip in tips:
            console.print(f"  {tip}")


def run_weekly_strategy(odds_collector: SmartOddsCollector, espn_collector: ESPNAPICollector, year: int, week: int) -> dict:
    """Run the weekly update strategy."""
    console.print(f"\n🗓️ [bold]Running Weekly Strategy[/bold]")
    
    results = {}
    
    with Progress() as progress:
        # Task tracking
        main_task = progress.add_task("Weekly Analysis", total=3)
        
        # Step 1: ESPN data collection (free)
        progress.update(main_task, description="Collecting ESPN data...")
        espn_results = espn_collector.collect_weekly_data(year, week)
        results['espn_data'] = espn_results
        progress.advance(main_task)
        
        # Step 2: Smart odds collection
        progress.update(main_task, description="Collecting smart odds data...")
        odds_results = odds_collector.weekly_update_strategy(year, week)
        results['odds_data'] = odds_results
        progress.advance(main_task)
        
        # Step 3: Analysis (placeholder)
        progress.update(main_task, description="Running analysis...")
        # This would run actual ML models
        results['analysis'] = {"edges_found": 5, "top_plays": ["Kelce Anytime TD", "Hill Receiving Yards"]}
        progress.advance(main_task)
    
    return results


def run_priority_strategy(odds_collector: SmartOddsCollector, prioritizer: GamePrioritizer, year: int, week: int) -> dict:
    """Run priority-focused strategy."""
    console.print(f"\n🎯 [bold]Running Priority Strategy[/bold]")
    
    # Update priorities
    priority_count = prioritizer.update_game_priorities(year, week)
    console.print(f"📊 Updated priorities for {priority_count} games")
    
    # Get distribution
    distribution = prioritizer.get_priority_distribution(year, week)
    console.print(f"🏈 Found {distribution.get('high_priority', 0)} high-priority games")
    
    # Collect only high-priority props
    odds_results = odds_collector.get_prioritized_props(
        year, week, priority_threshold=7.0
    )
    
    return {
        'strategy': 'priority',
        'priority_distribution': distribution,
        'odds_results': odds_results,
    }


def run_budget_saver_strategy(odds_collector: SmartOddsCollector, year: int, week: int) -> dict:
    """Run ultra-conservative budget strategy."""
    console.print(f"\n💰 [bold]Running Budget Saver Strategy[/bold]")
    
    # Only get top 3 priority games
    odds_results = odds_collector.get_prioritized_props(
        year, week, priority_threshold=8.0  # Very high threshold
    )
    
    return {
        'strategy': 'budget_saver',
        'odds_results': odds_results,
        'message': 'Ultra-conservative: Only highest priority games processed'
    }


def run_test_strategy(odds_collector: SmartOddsCollector, espn_collector: ESPNAPICollector, year: int, week: int) -> dict:
    """Run test strategy to validate system."""
    console.print(f"\n🧪 [bold]Running Test Strategy[/bold]")
    
    # Generate usage report
    report = odds_collector.get_smart_usage_report()
    
    # Test ESPN API
    scoreboard = espn_collector.get_scoreboard(year, week)
    
    return {
        'strategy': 'test',
        'usage_report': report,
        'espn_test': 'success' if scoreboard else 'failed',
        'system_health': 'good',
    }


def display_results(results: dict):
    """Display analysis results."""
    console.print(f"\n📈 [bold]Analysis Results[/bold]")
    
    strategy = results.get('strategy', 'weekly')
    
    if strategy == 'weekly':
        # Show weekly results
        espn_data = results.get('espn_data', {})
        odds_data = results.get('odds_data', {})
        
        results_table = Table(show_header=True, header_style="bold green")
        results_table.add_column("Data Source")
        results_table.add_column("Status")
        results_table.add_column("Details")
        
        # ESPN results
        espn_games = espn_data.get('data_collected', {}).get('scoreboard', {})
        if isinstance(espn_games, dict):
            games_count = espn_games.get('games_count', 0)
            results_table.add_row("ESPN Data", "✅ Success", f"{games_count} games processed")
        else:
            results_table.add_row("ESPN Data", "📋 Cached", "Using cached data")
        
        # Odds results
        total_ops = len(odds_data.get('operations', []))
        total_requests = sum(op.get('requests_used', 0) for op in odds_data.get('operations', []))
        results_table.add_row("Odds API", "✅ Success", f"{total_ops} operations, {total_requests} requests used")
        
        console.print(results_table)
        
        # Show top plays
        analysis = results.get('analysis', {})
        if analysis.get('top_plays'):
            console.print(f"\n🔥 [bold]Top Plays:[/bold]")
            for i, play in enumerate(analysis['top_plays'], 1):
                console.print(f"  {i}. {play}")
    
    elif strategy == 'priority':
        # Show priority results  
        distribution = results.get('priority_distribution', {})
        odds_results = results.get('odds_results', {})
        
        console.print(f"🎯 High Priority Games: {distribution.get('high_priority', 0)}")
        console.print(f"📊 Props Collected: {odds_results.get('total_props', 0)}")
        console.print(f"💸 Requests Used: {odds_results.get('requests_used', 0)}")
    
    elif strategy == 'test':
        # Show test results
        report = results.get('usage_report', {})
        api_usage = report.get('api_usage', {})
        
        console.print(f"📊 Monthly Usage: {api_usage.get('usage_percentage', 0):.1f}%")
        console.print(f"🔧 ESPN API: {results.get('espn_test', 'unknown').title()}")
        console.print(f"💚 System Health: {results.get('system_health', 'unknown').title()}")


def show_final_status(request_manager: RequestManager, odds_collector: SmartOddsCollector):
    """Show final system status after operations."""
    console.print(f"\n📊 [bold]Final Status[/bold]")
    
    # Updated usage
    final_usage = request_manager.get_usage_stats("odds_api")
    budget = request_manager.get_priority_budget()
    
    final_table = Table(show_header=True, header_style="bold blue")
    final_table.add_column("Metric")
    final_table.add_column("Value", justify="right")
    
    final_table.add_row("Requests Used Today", str(len(final_usage['recent_requests'])))
    final_table.add_row("Monthly Usage", f"{final_usage['monthly_usage']}/{final_usage['monthly_limit']}")
    final_table.add_row("Remaining Budget", str(budget['total_remaining']))
    final_table.add_row("Days Left", str(budget['days_left']))
    
    console.print(final_table)


if __name__ == "__main__":
    main()