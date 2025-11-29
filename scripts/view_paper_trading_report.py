#!/usr/bin/env python3
"""
View paper trading ROI and performance reports.
"""
import sys
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.tracking.paper_trader import ROICalculator


def main():
    parser = argparse.ArgumentParser(
        description='View paper trading performance reports'
    )
    parser.add_argument(
        '--week',
        type=int,
        help='Filter to specific week'
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2024,
        help='Season (default: 2024)'
    )

    args = parser.parse_args()

    calculator = ROICalculator()

    print()
    report = calculator.generate_report(week=args.week, season=args.season)
    print(report)
    print()

    # Additional analysis
    print("=" * 60)
    print("INSIGHTS & RECOMMENDATIONS")
    print("=" * 60)
    print()
    print("What to look for:")
    print("  ✓ Win rate > 53% (break-even is 52.4% at -110 odds)")
    print("  ✓ Positive ROI (target: 3-5%)")
    print("  ✓ Model errors within expected range")
    print("  ✓ Edge% correlates with win rate")
    print()
    print("If system is profitable:")
    print("  → Track 20-30 more bets to confirm")
    print("  → Consider small real money bets")
    print("  → Continue refining models")
    print()
    print("If system is losing:")
    print("  → Review reasoning for losses")
    print("  → Check if model errors are systematic")
    print("  → Adjust edge threshold")
    print("  → Continue improving features")
    print()
    print("Remember: Even with an edge, variance means losing streaks happen!")


if __name__ == "__main__":
    main()
