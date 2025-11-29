#!/usr/bin/env python3
"""
Record paper trades from identified edges.

This script helps you validate the system WITHOUT risking real money.
"""
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.analysis.edge_calculator import EdgeCalculator
from src.sports_betting.tracking.paper_trader import PaperTrader


def main():
    print("=" * 60)
    print("PAPER TRADING - RECORD HYPOTHETICAL BETS")
    print("=" * 60)
    print()

    # Find edges for current week
    calculator = EdgeCalculator()
    current_week = 13  # Week 13 2025
    current_season = 2025

    print(f"Finding betting opportunities for Week {current_week}...")
    edges = calculator.find_edges_for_week(
        week=current_week,
        season=current_season,
        min_edge=0.03
    )

    if not edges:
        print("No edges found for this week.")
        print("\nThis means:")
        print("  - Market lines are efficient")
        print("  - No +EV opportunities detected")
        print("  - Or predictions haven't been generated yet")
        return

    print(f"\n✓ Found {len(edges)} betting opportunities!")
    print()

    # Display edges with numbering
    print("AVAILABLE BETS:")
    print("-" * 60)
    for i, edge in enumerate(edges, 1):
        side = "OVER" if edge['over']['should_bet'] else "UNDER"
        ev = max(edge['over']['ev_pct'], edge['under']['ev_pct'])

        print(f"{i}. {edge['player']} - {edge['market'].replace('_', ' ').title()}")
        print(f"   {side} {edge['line']} ({ev:+.1f}% EV)")
        print()

    # Ask user which bets to record
    print("-" * 60)
    print("\nWhich bets do you want to record?")
    print("Enter numbers separated by commas (e.g., 1,3,5)")
    print("Or 'all' to record all bets")
    print("Or 'q' to quit")
    print()

    selection = input("Selection: ").strip().lower()

    if selection == 'q':
        print("No bets recorded.")
        return

    # Parse selection
    if selection == 'all':
        selected_indices = list(range(len(edges)))
    else:
        try:
            selected_indices = [int(x.strip()) - 1 for x in selection.split(',')]
        except ValueError:
            print("Invalid selection format.")
            return

    # Validate indices
    selected_indices = [i for i in selected_indices if 0 <= i < len(edges)]

    if not selected_indices:
        print("No valid selections.")
        return

    # Record paper trades
    trader = PaperTrader()
    recorded = []

    print()
    print("Recording paper trades...")
    print()

    for idx in selected_indices:
        edge = edges[idx]

        try:
            trade = trader.record_paper_trade(edge, stake=100.0)
            recorded.append(trade)

            # Display summary
            print(trader.format_trade_summary(trade))
            print()

        except Exception as e:
            print(f"❌ Error recording bet for {edge['player']}: {e}")
            continue

    # Summary
    print("=" * 60)
    print(f"✓ Recorded {len(recorded)} paper trades")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Wait for games to complete")
    print("  2. Run: python scripts/evaluate_paper_trades.py")
    print("  3. Review results and ROI")
    print("  4. Refine system based on learnings")
    print()
    print("These are HYPOTHETICAL bets - no money at risk!")
    print("Use this to validate the system before considering real bets.")


if __name__ == "__main__":
    main()
