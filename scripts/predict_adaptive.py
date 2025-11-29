#!/usr/bin/env python3
"""
Adaptive Prediction Script

Generates predictions using the adaptive prediction system that:
1. Automatically finds the best available data source
2. Adjusts confidence based on data quality
3. Applies situational context (injuries, QB changes, etc.)
4. Saves predictions with confidence bands to database
"""
import sys
from pathlib import Path
from datetime import datetime
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.ml import ReceivingYardsPredictor
from src.sports_betting.ml.context import get_context, set_qb_change, set_injury
from src.sports_betting.database import get_session
from src.sports_betting.database.models import Game, Player, Prediction


def get_current_week() -> tuple[int, int]:
    """Determine current NFL season and week."""
    today = datetime.now()

    # NFL season typically starts first week of September
    # and has 18 weeks (17 games + bye)
    if today.month >= 9:
        season = today.year
    elif today.month <= 2:
        season = today.year  # Playoffs
    else:
        season = today.year - 1  # Offseason, reference last season

    # Approximate week calculation
    # Week 1 starts around Sept 5-10
    if today.month >= 9:
        week_start = datetime(today.year, 9, 5)
        days_since_start = (today - week_start).days
        week = min(18, max(1, (days_since_start // 7) + 1))
    elif today.month <= 2:
        week = 18  # Playoffs/Super Bowl
    else:
        week = 1  # Offseason default

    return season, week


def setup_context():
    """
    Set up situational context for current week.

    This is where you'd add known QB changes, injuries, etc.
    In production, this would be pulled from database or API.
    """
    context = get_context()

    # Example: Add known situational factors
    # set_qb_change('SF', 'Brock Purdy', 'injury', games_started=0)
    # set_injury('Ja\'Marr Chase', 'questionable', 'WR')

    return context


def save_predictions_to_db(predictions_df, metadata, season: int, week: int):
    """Save predictions to database."""
    if len(predictions_df) == 0:
        print("No predictions to save")
        return 0

    saved = 0

    with get_session() as session:
        for _, row in predictions_df.iterrows():
            player_name = row['player_name']
            team = row['recent_team']

            # Find player by name
            player = session.query(Player).filter(
                Player.name.ilike(f"%{player_name.split()[0]}%")
            ).first()

            if not player:
                continue

            # Find game for this week
            game = session.query(Game).filter(
                Game.season == season,
                Game.week == week
            ).join(Game.home_team).filter_by(abbreviation=team).first()

            if not game:
                game = session.query(Game).filter(
                    Game.season == season,
                    Game.week == week
                ).join(Game.away_team).filter_by(abbreviation=team).first()

            if not game:
                continue

            # Create or update prediction
            pred = session.query(Prediction).filter_by(
                game_id=game.id,
                player_id=player.id,
                market='receiving_yards'
            ).first()

            pred_value = row['predicted_yards']
            confidence = row['confidence']

            # Calculate percentile bands if available
            p10 = row.get('p10', pred_value - 15)
            p90 = row.get('p90', pred_value + 15)

            if pred:
                pred.prediction = pred_value
                pred.confidence = confidence
                pred.p10 = p10
                pred.p50 = pred_value
                pred.p90 = p90
                pred.model_name = 'xgboost_adaptive'
                pred.model_version = 'v2.0'
            else:
                pred = Prediction(
                    game_id=game.id,
                    player_id=player.id,
                    market='receiving_yards',
                    prediction=pred_value,
                    confidence=confidence,
                    p10=p10,
                    p50=pred_value,
                    p90=p90,
                    model_name='xgboost_adaptive',
                    model_version='v2.0'
                )
                session.add(pred)

            saved += 1

        session.commit()

    return saved


def main():
    parser = argparse.ArgumentParser(description='Generate adaptive predictions')
    parser.add_argument('--season', type=int, help='NFL season year')
    parser.add_argument('--week', type=int, help='Week number')
    parser.add_argument('--player', type=str, help='Specific player to predict')
    parser.add_argument('--save', action='store_true', help='Save to database')
    parser.add_argument('--top', type=int, default=25, help='Show top N predictions')

    args = parser.parse_args()

    # Determine season/week
    if args.season and args.week:
        season, week = args.season, args.week
    else:
        season, week = get_current_week()
        print(f"Auto-detected: {season} Week {week}")

    print(f"\n{'=' * 80}")
    print(f"ADAPTIVE PREDICTION SYSTEM")
    print(f"Season: {season}, Week: {week}")
    print(f"{'=' * 80}")

    # Set up situational context
    context = setup_context()
    ctx_summary = context.get_context_summary()
    if ctx_summary['qb_changes'] or ctx_summary['injuries_tracked']:
        print(f"\nContext: {ctx_summary['qb_changes']} QB changes, {ctx_summary['injuries_tracked']} injuries tracked")

    # Initialize predictor
    try:
        predictor = ReceivingYardsPredictor()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\nRun: python scripts/train_enhanced_model.py first")
        return

    # Generate predictions
    if args.player:
        # Single player
        result = predictor.get_prediction_for_player(args.player, season, week)
        if result:
            print(f"\n{result['player_name']} ({result['team']})")
            print(f"  Predicted: {result['predicted_yards']:.1f} yards")
            print(f"  Confidence: {result['confidence']:.0%}")
            print(f"  Data Quality: {result['data_quality']}")
            print(f"  Last 5 avg: {result['rec_yards_last_5']:.1f} yards")
        else:
            print(f"Player '{args.player}' not found in predictions")
    else:
        # All players with confidence bands
        predictions = predictor.predict_with_confidence_bands(season, week)

        if len(predictions) == 0:
            print("❌ No predictions generated")
            return

        # Apply situational context
        predictions = context.apply_context_to_predictions(predictions)

        # Display results
        print(f"\n{'=' * 80}")
        print(f"TOP {args.top} PREDICTIONS - {season} Week {week}")
        print(f"{'=' * 80}")

        print(f"\n{'#':<3} {'Player':<22} {'Team':<4} {'Pred':>6} {'Conf':>5} {'P10':>5} {'P90':>5} {'Context':<20}")
        print("-" * 85)

        for i, row in predictions.head(args.top).iterrows():
            adj_pred = row.get('adjusted_prediction', row['predicted_yards'])
            ctx = row.get('situation_reasons', 'None')
            if len(ctx) > 18:
                ctx = ctx[:18] + '..'

            print(
                f"{predictions.head(args.top).index.get_loc(i) + 1:<3} "
                f"{row['player_name']:<22} "
                f"{row['recent_team']:<4} "
                f"{adj_pred:>6.1f} "
                f"{row['confidence']:>4.0%} "
                f"{row['p10']:>5.1f} "
                f"{row['p90']:>5.1f} "
                f"{ctx:<20}"
            )

        # Summary
        print(f"\nTotal predictions: {len(predictions)}")
        print(f"Average confidence: {predictions['confidence'].mean():.0%}")

        high_conf = predictions[predictions['confidence'] > 0.7]
        print(f"High confidence (>70%): {len(high_conf)}")

        # Save to database if requested
        if args.save:
            print(f"\nSaving to database...")
            # Get metadata from prediction process (approximate)
            metadata = {'season': season, 'week': week}
            saved = save_predictions_to_db(predictions, metadata, season, week)
            print(f"✓ Saved {saved} predictions")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
