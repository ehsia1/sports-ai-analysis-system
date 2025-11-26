#!/usr/bin/env python3
"""Demo script to test the sports betting system with sample data."""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_sample_results():
    """Create sample betting analysis results."""
    return {
        "week": 5,
        "year": 2024,
        "min_edge": 0.02,
        "analysis_time": datetime.now().isoformat(),
        "total_opportunities": 15,
        "edges": [
            {
                "player": "Ja'Marr Chase",
                "team": "CIN",
                "opponent": "PIT",
                "market": "receiving_yards",
                "line": 67.5,
                "side": "over",
                "fair_line": 78.2,
                "offered_odds": -110,
                "fair_probability": 0.621,
                "market_probability": 0.524,
                "edge": 0.156,
                "ev": 0.089,
                "kelly_size": 0.045,
                "confidence": 0.82,
                "reasoning": "Favorable matchup vs PIT secondary (#28 vs WRs), positive game script with CIN -3.5 spread. Chase averages 8.2 targets in dome games.",
                "key_factors": [
                    "Pittsburgh allows 8.1 Y/target to slot receivers",
                    "Chase runs 72% of routes from slot",
                    "Dome game removes wind variable",
                    "Model agreement: 4/5 models predict over"
                ]
            },
            {
                "player": "Austin Ekeler", 
                "team": "LAC",
                "opponent": "KC",
                "market": "receptions",
                "line": 4.5,
                "side": "over", 
                "fair_line": 5.8,
                "offered_odds": -105,
                "fair_probability": 0.643,
                "market_probability": 0.512,
                "edge": 0.124,
                "ev": 0.067,
                "kelly_size": 0.033,
                "confidence": 0.78,
                "reasoning": "High-volume passing game expected with LAC as 7.5-point underdog. Ekeler's receiving role has expanded with 6.3 targets/game over last 3 weeks.",
                "key_factors": [
                    "LAC implied to be trailing (negative game script)",
                    "KC allows 5.8 receptions to RBs (4th worst)",
                    "Ekeler: 78% snap share in passing situations",
                    "Weather: Clear skies, no wind concerns"
                ]
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
                "reasoning": "Kelce leads KC in red zone targets (31%) and has scored in 3 of last 4 games. LAC allows 1.8 TDs/game to TEs (worst in NFL).",
                "key_factors": [
                    "31% red zone target share (team leader)",
                    "LAC allows 1.8 TDs/game to TEs (#32 defense)",
                    "Mahomes 78% completion rate to Kelce in RZ",
                    "Historical: 8 TDs in 6 games vs LAC"
                ]
            },
            {
                "player": "Tee Higgins",
                "team": "CIN",
                "opponent": "PIT", 
                "market": "receiving_yards",
                "line": 54.5,
                "side": "over",
                "fair_line": 63.2,
                "offered_odds": -110,
                "fair_probability": 0.589,
                "market_probability": 0.524,
                "edge": 0.098,
                "ev": 0.054,
                "kelly_size": 0.027,
                "confidence": 0.71,
                "reasoning": "Secondary option benefits from Chase coverage. Higgins averages 72 yards when Chase draws 8+ targets.",
                "key_factors": [
                    "Chase expected to draw top coverage",
                    "Higgins: 15.2 Y/reception vs man coverage",
                    "PIT plays man coverage 68% of snaps",
                    "Indoor game favors passing volume"
                ]
            },
            {
                "player": "Isiah Pacheco",
                "team": "KC",
                "opponent": "LAC",
                "market": "rushing_yards", 
                "line": 68.5,
                "side": "over",
                "fair_line": 76.8,
                "offered_odds": -115,
                "fair_probability": 0.567,
                "market_probability": 0.535,
                "edge": 0.087,
                "ev": 0.043,
                "kelly_size": 0.021,
                "confidence": 0.69,
                "reasoning": "LAC run defense allows 4.9 YPC (25th). KC expected to control game flow with 7.5-point spread.",
                "key_factors": [
                    "LAC allows 4.9 YPC to RBs (#25 defense)", 
                    "KC expected positive game script (+7.5)",
                    "Pacheco: 18.2 carries in games KC leads",
                    "Weather: No precipitation expected"
                ]
            }
        ],
        "market_summary": {
            "receiving_yards": {"count": 2, "avg_edge": 0.127},
            "receptions": {"count": 1, "avg_edge": 0.124}, 
            "anytime_td": {"count": 1, "avg_edge": 0.104},
            "rushing_yards": {"count": 1, "avg_edge": 0.087}
        },
        "team_summary": {
            "CIN": {"opportunities": 2, "best_edge": 0.156},
            "KC": {"opportunities": 2, "best_edge": 0.104},
            "LAC": {"opportunities": 1, "best_edge": 0.124}
        },
        "risk_metrics": {
            "total_kelly": 0.151,
            "max_single_bet": 0.045,
            "portfolio_correlation": 0.23,
            "bankroll_risk": "Low"
        }
    }

def display_fancy_table():
    """Display a fancy formatted table of results."""
    print("🏈 Sports Betting AI/ML Analysis System")
    print("=" * 60)
    print("📊 Week 5, 2024 NFL Analysis")
    print("⚡ Minimum Edge: 2.0%")
    print("🎯 Total Opportunities Found: 15")
    print("✅ High-Confidence Plays: 5")
    print()

    # Header
    print("🔥 TOP BETTING OPPORTUNITIES")
    print("-" * 60)
    
    results = create_sample_results()
    
    for i, edge in enumerate(results["edges"], 1):
        print(f"\n#{i}. {edge['player']} ({edge['team']}) vs {edge['opponent']}")
        print(f"    📈 Market: {edge['market'].replace('_', ' ').title()}")
        print(f"    📊 Line: {edge['side'].title()} {edge['line']}")
        print(f"    💰 Edge: {edge['edge']:.1%} | EV: {edge['ev']:.1%} | Kelly: {edge['kelly_size']:.1%}")
        print(f"    🎯 Confidence: {edge['confidence']:.0%}")
        print(f"    💡 {edge['reasoning']}")
        
        # Show key factors for top 3
        if i <= 3:
            print("    🔍 Key Factors:")
            for factor in edge['key_factors']:
                print(f"       • {factor}")

def display_summary_stats():
    """Display summary statistics."""
    results = create_sample_results()
    
    print("\n" + "=" * 60)
    print("📈 PORTFOLIO ANALYSIS")
    print("-" * 60)
    
    print(f"💵 Total Kelly Allocation: {results['risk_metrics']['total_kelly']:.1%}")
    print(f"🎲 Max Single Bet: {results['risk_metrics']['max_single_bet']:.1%}")
    print(f"🔗 Portfolio Correlation: {results['risk_metrics']['portfolio_correlation']:.1%}")
    print(f"⚠️  Risk Level: {results['risk_metrics']['bankroll_risk']}")
    
    print("\n📊 Market Breakdown:")
    for market, stats in results['market_summary'].items():
        print(f"   {market.replace('_', ' ').title()}: {stats['count']} plays, {stats['avg_edge']:.1%} avg edge")
    
    print("\n🏈 Team Focus:")
    for team, stats in results['team_summary'].items():
        print(f"   {team}: {stats['opportunities']} opportunities, {stats['best_edge']:.1%} best edge")

def save_sample_output():
    """Save sample output to file."""
    results = create_sample_results()
    
    # Create output directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed JSON
    with open(output_dir / "sample_analysis_week5.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Save simplified CSV
    import csv
    with open(output_dir / "sample_edges_week5.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'player', 'team', 'opponent', 'market', 'line', 'side',
            'edge', 'ev', 'kelly_size', 'confidence'
        ])
        writer.writeheader()
        for edge in results['edges']:
            writer.writerow({
                'player': edge['player'],
                'team': edge['team'],
                'opponent': edge['opponent'],
                'market': edge['market'],
                'line': edge['line'],
                'side': edge['side'],
                'edge': f"{edge['edge']:.3f}",
                'ev': f"{edge['ev']:.3f}",
                'kelly_size': f"{edge['kelly_size']:.3f}",
                'confidence': f"{edge['confidence']:.2f}"
            })
    
    print(f"\n💾 Results saved to:")
    print(f"   📄 {output_dir / 'sample_analysis_week5.json'}")
    print(f"   📊 {output_dir / 'sample_edges_week5.csv'}")

def main():
    """Run the demo."""
    try:
        display_fancy_table()
        display_summary_stats()
        save_sample_output()
        
        print("\n" + "=" * 60)
        print("🎉 DEMO COMPLETE!")
        print("\nThis demonstrates what the full system will output:")
        print("✅ Real-time odds collection")
        print("✅ ML-powered predictions") 
        print("✅ Edge detection with EV calculations")
        print("✅ Kelly criterion position sizing")
        print("✅ Risk management and portfolio analysis")
        print("✅ Detailed reasoning for each play")
        
        print("\n🚀 Ready to build the real ML models!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()