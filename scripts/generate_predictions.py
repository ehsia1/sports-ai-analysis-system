#!/usr/bin/env python3
"""
Generate predictions for upcoming week using enhanced model.

This script:
1. Loads the enhanced XGBoost model
2. Generates predictions for specified week
3. Saves predictions to database
4. Optionally fetches odds and calculates edges
"""
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.ml.predictor import ReceivingYardsPredictor
from src.sports_betting.database import get_session
from src.sports_betting.database.models import Game, Player, Prediction


def save_predictions_to_db(predictions_df, season: int, week: int):
    """Save predictions to database."""
    print("\nSaving predictions to database...")

    with get_session() as session:
        saved_count = 0
        skipped_count = 0

        for idx, row in predictions_df.iterrows():
            # Find the player
            player = session.query(Player).filter_by(
                external_id=row['player_id']
            ).first()

            if not player:
                print(f"⚠️  Player not found in DB: {row['player_name']} ({row['player_id']})")
                skipped_count += 1
                continue

            # Find the game for this player's team in this week
            # Note: Player could be on away or home team
            game = session.query(Game).filter(
                Game.season == season,
                Game.week == week
            ).filter(
                (Game.away_team_id == player.team_id) |
                (Game.home_team_id == player.team_id)
            ).first()

            if not game:
                print(f"⚠️  No game found for {row['player_name']}'s team in Week {week}")
                skipped_count += 1
                continue

            # Check if prediction already exists
            existing = session.query(Prediction).filter_by(
                player_id=player.id,
                game_id=game.id,
                market='receiving_yards'
            ).first()

            if existing:
                # Update existing
                existing.prediction = row['predicted_receiving_yards']
                existing.model_name = 'xgboost_enhanced'
                existing.model_version = 'v2.0'
                existing.confidence = 0.5  # Placeholder - could calculate from model
                existing.p50 = row['predicted_receiving_yards']
                existing.created_at = datetime.now()
            else:
                # Create new
                prediction = Prediction(
                    player_id=player.id,
                    game_id=game.id,
                    market='receiving_yards',
                    model_name='xgboost_enhanced',
                    model_version='v2.0',
                    prediction=row['predicted_receiving_yards'],
                    confidence=0.5,  # Placeholder - could calculate from model
                    p50=row['predicted_receiving_yards'],  # Median prediction
                    created_at=datetime.now()
                )
                session.add(prediction)

            saved_count += 1

        session.commit()

    print(f"✓ Saved {saved_count} predictions to database")
    if skipped_count > 0:
        print(f"⚠️  Skipped {skipped_count} predictions (player/game not found)")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate predictions for upcoming week')
    parser.add_argument('--season', type=int, default=2025, help='Season year')
    parser.add_argument('--week', type=int, default=14, help='Week number')
    parser.add_argument('--save-db', action='store_true', help='Save predictions to database')
    parser.add_argument('--output', type=str, help='Save predictions to CSV file')

    args = parser.parse_args()

    print("=" * 80)
    print("ENHANCED MODEL PREDICTION GENERATION")
    print("=" * 80)
    print(f"\nTarget: {args.season} Week {args.week}")

    # Load predictor
    predictor = ReceivingYardsPredictor()

    # Generate predictions
    predictions = predictor.predict_for_week(season=args.season, week=args.week)

    if len(predictions) == 0:
        print("\n❌ No predictions generated")
        return

    # Display top predictions
    print("\n" + "=" * 80)
    print(f"TOP 30 PREDICTED RECEIVING YARDS ({args.season} Week {args.week})")
    print("=" * 80)
    print()

    top_30 = predictions.head(30)
    print(f"{'Rank':<5} {'Player':<25} {'Pos':<4} {'Team':<4} {'Predicted':<10} {'Last 5':<10} {'Tgt Share':<10}")
    print("-" * 80)

    for i, (idx, row) in enumerate(top_30.iterrows(), 1):
        print(
            f"{i:<5} "
            f"{row['player_name']:<25} "
            f"{row['position']:<4} "
            f"{row['recent_team']:<4} "
            f"{row['predicted_receiving_yards']:>6.1f}    "
            f"{row['rec_yards_last_5']:>6.1f}    "
            f"{row['target_share_last_5']:>6.1%}"
        )

    # Summary stats
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"\nBy position:")
    print(predictions['position'].value_counts().to_string())

    print(f"\nPrediction range:")
    print(f"  Max: {predictions['predicted_receiving_yards'].max():.1f} yards")
    print(f"  Min: {predictions['predicted_receiving_yards'].min():.1f} yards")
    print(f"  Avg: {predictions['predicted_receiving_yards'].mean():.1f} yards")
    print(f"  Median: {predictions['predicted_receiving_yards'].median():.1f} yards")

    # Save to database if requested
    if args.save_db:
        save_predictions_to_db(predictions, args.season, args.week)

    # Save to CSV if requested
    if args.output:
        predictions.to_csv(args.output, index=False)
        print(f"\n✓ Saved predictions to {args.output}")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
To find betting edges:
  1. Fetch odds: python scripts/fetch_odds.py --week 14
  2. Calculate edges: python scripts/calculate_edges.py --week 14
  3. Record paper trades: python scripts/record_paper_trades.py

Or run all at once:
  python scripts/generate_weekly_report.py --week 14
""")


if __name__ == "__main__":
    main()
