#!/usr/bin/env python3
"""
Evaluate paper trades after games complete.

Enter actual results to track system performance.
"""
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.tracking.paper_trader import PaperTrader, BetEvaluator, ROICalculator
from src.sports_betting.database import get_session
from src.sports_betting.database.models import PaperTrade


def main():
    print("=" * 60)
    print("PAPER TRADING - EVALUATE RESULTS")
    print("=" * 60)
    print()

    trader = PaperTrader()
    evaluator = BetEvaluator()

    # Get pending trades
    pending = trader.get_pending_trades()

    if not pending:
        print("No pending paper trades to evaluate.")
        print("\nRecord some paper trades first:")
        print("  python scripts/record_paper_trades.py")
        return

    print(f"Found {len(pending)} pending paper trades")
    print()

    # Group by game
    with get_session() as session:
        for trade in pending:
            # Re-attach to session
            trade = session.merge(trade)
            player = trade.player
            game = trade.game

            print("-" * 60)
            print(f"{player.name} ({player.position}) - {trade.market.replace('_', ' ').title()}")
            print(f"Game: {game.away_team.abbreviation} @ {game.home_team.abbreviation}")
            print(f"Bet: {trade.bet_side.upper()} {trade.line} @ {trade.odds:+d}")
            print(f"Prediction: {trade.model_prediction:.1f}")
            print()

            # Ask for actual result
            while True:
                result_input = input(
                    f"Enter actual {trade.market.replace('_', ' ')} (or 'skip'): "
                ).strip()

                if result_input.lower() == 'skip':
                    print("Skipping...")
                    break

                try:
                    actual_result = float(result_input)

                    # Evaluate
                    evaluation = evaluator.evaluate_trade(trade, actual_result)

                    # Update trade
                    trade.actual_result = evaluation['actual_result']
                    trade.won = evaluation['won']
                    trade.profit_loss = evaluation['profit_loss']
                    trade.evaluated_at = datetime.now()

                    # Display result
                    result_icon = "✓" if evaluation['won'] else "✗"
                    print()
                    print(f"{result_icon} Actual: {actual_result:.1f}")
                    print(f"   Result: {'WON' if evaluation['won'] else 'LOST'}")
                    print(f"   P/L: ${evaluation['profit_loss']:+.2f}")
                    print(f"   Model Error: {evaluation['accuracy']:.1f} yards")
                    print()

                    break

                except ValueError:
                    print("Invalid input. Please enter a number.")

        session.commit()

    # Generate ROI report
    print()
    print("=" * 60)
    print("CALCULATING ROI...")
    print("=" * 60)
    print()

    calculator = ROICalculator()
    report = calculator.generate_report()
    print(report)


if __name__ == "__main__":
    main()
