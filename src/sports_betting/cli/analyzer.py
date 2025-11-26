"""Main CLI for weekly betting analysis."""

import logging
from datetime import datetime
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ..config import get_settings
from ..database import init_db
from ..data import NFLDataCollector, OddsAPICollector
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
    "--top",
    type=int,
    default=20,
    help="Number of top opportunities to show",
)
@click.option(
    "--min-edge",
    type=float,
    default=0.02,
    help="Minimum edge threshold (default: 2%)",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Output directory for reports",
)
@click.option(
    "--update-data",
    is_flag=True,
    help="Update NFL and odds data before analysis",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Verbose output",
)
def main(week, year, top, min_edge, output_dir, update_data, verbose):
    """Analyze NFL betting opportunities for the specified week."""
    # Setup logging
    setup_logging(level="DEBUG" if verbose else "INFO")
    
    settings = get_settings()
    
    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = settings.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[bold blue]Sports Betting Analyzer v0.1.0[/bold blue]")
    console.print(f"Analyzing Week {week or 'Current'}, {year}")
    console.print(f"Minimum edge threshold: {min_edge:.1%}")
    console.print()
    
    try:
        # Initialize database
        console.print("🔧 Initializing database...")
        init_db()
        
        if update_data:
            console.print("📊 Updating data...")
            update_data_sources(week, year)
        
        # Run analysis
        console.print("🤖 Running analysis...")
        results = run_analysis(week, year, min_edge)
        
        # Display results
        display_results(results, top)
        
        # Save results
        output_file = output_dir / f"analysis_week_{week or 'current'}_{year}.json"
        save_results(results, output_file)
        
        console.print(f"\n✅ Analysis complete! Results saved to {output_file}")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise click.Abort()


def update_data_sources(week: int, year: int):
    """Update NFL and odds data."""
    with console.status("Updating NFL data..."):
        nfl_collector = NFLDataCollector()
        
        # Update schedule and rosters
        games_count = nfl_collector.collect_schedule([year])
        players_count = nfl_collector.collect_rosters([year])
        
        console.print(f"  📅 Updated {games_count} games")
        console.print(f"  👥 Updated {players_count} players")
    
    with console.status("Updating odds data..."):
        odds_collector = OddsAPICollector()
        
        try:
            props_count = odds_collector.collect_and_store_props(year, week or 1)
            console.print(f"  💰 Updated {props_count} props")
        except Exception as e:
            console.print(f"  [yellow]⚠️ Could not update odds: {e}[/yellow]")


def run_analysis(week: int, year: int, min_edge: float) -> dict:
    """Run the betting analysis."""
    # This is a placeholder for the actual analysis
    # In the full implementation, this would:
    # 1. Load features for all players/games
    # 2. Run ML models to generate predictions
    # 3. Compare to market lines to find edges
    # 4. Calculate expected values and bet sizes
    
    # For now, return sample results
    sample_edges = [
        {
            "player": "Ja'Marr Chase",
            "team": "CIN", 
            "opponent": "PIT",
            "market": "receiving_yards",
            "line": 67.5,
            "side": "over",
            "fair_line": 78.2,
            "edge": 0.156,
            "ev": 0.089,
            "kelly_size": 0.045,
            "confidence": 0.82,
            "reasoning": "Favorable matchup vs PIT secondary, positive game script expected",
        },
        {
            "player": "Austin Ekeler",
            "team": "LAC",
            "opponent": "KC", 
            "market": "receptions",
            "line": 4.5,
            "side": "over",
            "fair_line": 5.8,
            "edge": 0.124,
            "ev": 0.067,
            "kelly_size": 0.033,
            "confidence": 0.78,
            "reasoning": "High volume passing game expected, Ekeler's receiving role trending up",
        },
        {
            "player": "Travis Kelce",
            "team": "KC",
            "opponent": "LAC",
            "market": "anytime_td", 
            "line": "+110",
            "side": "yes",
            "fair_probability": 0.58,
            "market_probability": 0.476,
            "edge": 0.104,
            "ev": 0.071,
            "kelly_size": 0.025,
            "confidence": 0.75,
            "reasoning": "Red zone target leader, strong historical performance vs LAC",
        }
    ]
    
    return {
        "week": week,
        "year": year,
        "min_edge": min_edge,
        "edges": [edge for edge in sample_edges if edge["edge"] >= min_edge],
        "total_opportunities": len(sample_edges),
        "analysis_time": datetime.now().isoformat(),
    }


def display_results(results: dict, top: int):
    """Display analysis results in a nice table."""
    edges = results["edges"][:top]
    
    if not edges:
        console.print("[yellow]No betting opportunities found above threshold.[/yellow]")
        return
    
    # Create table
    table = Table(title=f"Top {len(edges)} Betting Opportunities")
    table.add_column("Player", style="cyan")
    table.add_column("Matchup", style="magenta")
    table.add_column("Market", style="green")
    table.add_column("Line", justify="right")
    table.add_column("Side", justify="center")
    table.add_column("Edge", justify="right", style="bold")
    table.add_column("EV", justify="right")
    table.add_column("Kelly %", justify="right")
    table.add_column("Confidence", justify="right")
    
    for edge in edges:
        table.add_row(
            edge["player"],
            f"{edge['team']} vs {edge['opponent']}",
            edge["market"].replace("_", " ").title(),
            str(edge["line"]),
            edge["side"].upper(),
            f"{edge['edge']:.1%}",
            f"{edge['ev']:.1%}",
            f"{edge['kelly_size']:.1%}",
            f"{edge['confidence']:.0%}",
        )
    
    console.print(table)
    
    # Show reasoning for top opportunities
    console.print("\n[bold]Key Insights:[/bold]")
    for i, edge in enumerate(edges[:3], 1):
        console.print(f"{i}. [cyan]{edge['player']}[/cyan] - {edge['reasoning']}")


def save_results(results: dict, output_file: Path):
    """Save results to JSON file."""
    import json
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()