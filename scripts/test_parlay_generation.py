#!/usr/bin/env python3
"""
Test parlay generation with trained models.
"""
import sys
from pathlib import Path
import pandas as pd

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import nfl_data_py as nfl
except ImportError:
    print("❌ nfl_data_py not installed")
    sys.exit(1)

from src.sports_betting.models.nfl.xgboost_model import XGBoostPropsModel
from src.sports_betting.database import get_session
from src.sports_betting.database.models import Game, Player


def load_upcoming_games():
    """Load upcoming NFL games."""
    print("Loading upcoming games from database...")

    with get_session() as session:
        # Get upcoming games (week 13 of 2024)
        games = session.query(Game).filter(
            Game.season == 2024,
            Game.week == 13,
            Game.season_type == 'REG'
        ).limit(5).all()

        if not games:
            print("❌ No upcoming games found in database")
            return []

        print(f"✓ Found {len(games)} games for week 13")
        return games


def get_player_recent_stats(player_name, position, years=[2024]):
    """Get recent stats for a player."""
    try:
        # Load recent weekly data
        weekly_df = nfl.import_weekly_data(years=years)

        # Filter to specific player
        player_df = weekly_df[
            (weekly_df['player_display_name'] == player_name) |
            (weekly_df['player_name'] == player_name)
        ].copy()

        if player_df.empty:
            return None

        # Get most recent records
        player_df = player_df.sort_values(['season', 'week'], ascending=False)

        return player_df.head(10)  # Last 10 games
    except Exception as e:
        print(f"Error getting stats for {player_name}: {e}")
        return None


def generate_test_predictions():
    """Generate test predictions for upcoming games."""
    print("\n" + "=" * 60)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 60)

    # Load trained models
    model_dir = Path(__file__).parent.parent / "models"

    # Check if models exist
    rec_model_path = model_dir / "receiving_yards_xgboost.pkl"
    rush_model_path = model_dir / "rushing_yards_xgboost.pkl"

    if not rec_model_path.exists() or not rush_model_path.exists():
        print("❌ Trained models not found. Please run train_models.py first")
        return []

    # Load models
    print("Loading trained models...")
    rec_model = XGBoostPropsModel(prop_type="receiving_yards")
    rec_model.load_model(rec_model_path)
    print("✓ Receiving yards model loaded")

    rush_model = XGBoostPropsModel(prop_type="rushing_yards")
    rush_model.load_model(rush_model_path)
    print("✓ Rushing yards model loaded")

    # Get some example players (top receivers/rushers)
    print("\nGenerating predictions for top players...")

    # Example players
    test_players = [
        {'name': 'Tyreek Hill', 'position': 'WR', 'team': 'MIA'},
        {'name': 'CeeDee Lamb', 'position': 'WR', 'team': 'DAL'},
        {'name': 'Justin Jefferson', 'position': 'WR', 'team': 'MIN'},
        {'name': 'Christian McCaffrey', 'position': 'RB', 'team': 'SF'},
        {'name': 'Josh Allen', 'position': 'QB', 'team': 'BUF'},
    ]

    predictions = []

    for player in test_players:
        print(f"\n{player['name']} ({player['position']}, {player['team']})")

        # Get recent stats
        recent_stats = get_player_recent_stats(player['name'], player['position'])

        if recent_stats is None or len(recent_stats) == 0:
            print("  ⚠ No recent stats found")
            continue

        # Prepare data for prediction
        recent_stats['team'] = recent_stats['recent_team']
        recent_stats['opponent'] = 'OPP'  # Placeholder
        recent_stats['home_team'] = recent_stats['recent_team']
        recent_stats['game_date'] = pd.to_datetime('2024-12-01')

        try:
            # Make predictions
            if player['position'] in ['WR', 'TE']:
                pred = rec_model.predict(recent_stats.head(1))
                pred_value = pred[f'receiving_yards_prediction'].iloc[0]
                confidence = pred[f'receiving_yards_confidence'].iloc[0]
                prop_type = 'Receiving Yards'
            elif player['position'] in ['RB', 'QB']:
                pred = rush_model.predict(recent_stats.head(1))
                pred_value = pred[f'rushing_yards_prediction'].iloc[0]
                confidence = pred[f'rushing_yards_confidence'].iloc[0]
                prop_type = 'Rushing Yards'
            else:
                continue

            print(f"  Predicted {prop_type}: {pred_value:.1f} (confidence: {confidence:.2f})")

            # Get actual recent averages for comparison
            if prop_type == 'Receiving Yards' and 'receiving_yards' in recent_stats.columns:
                recent_avg = recent_stats['receiving_yards'].head(5).mean()
                print(f"  Recent 5-game avg: {recent_avg:.1f}")
            elif prop_type == 'Rushing Yards' and 'rushing_yards' in recent_stats.columns:
                recent_avg = recent_stats['rushing_yards'].head(5).mean()
                print(f"  Recent 5-game avg: {recent_avg:.1f}")

            predictions.append({
                'player': player['name'],
                'team': player['team'],
                'position': player['position'],
                'prop_type': prop_type,
                'prediction': pred_value,
                'confidence': confidence
            })

        except Exception as e:
            print(f"  ❌ Error making prediction: {e}")

    return predictions


def build_sample_parlay(predictions):
    """Build a sample parlay from predictions."""
    print("\n" + "=" * 60)
    print("BUILDING SAMPLE PARLAY")
    print("=" * 60)

    if len(predictions) < 3:
        print("❌ Need at least 3 predictions to build a parlay")
        return

    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x['confidence'], reverse=True)

    # Take top 4 predictions
    parlay_legs = predictions[:4]

    print("\n📋 Proposed 4-Leg Parlay:")
    print("-" * 60)

    for i, leg in enumerate(parlay_legs, 1):
        print(f"{i}. {leg['player']} ({leg['team']}) - {leg['prop_type']}")
        print(f"   Prediction: {leg['prediction']:.1f} yards")
        print(f"   Confidence: {leg['confidence']:.2%}")
        print()

    # Calculate combined probability (assuming independence for now)
    combined_prob = 1.0
    for leg in parlay_legs:
        combined_prob *= leg['confidence']

    # Estimate parlay odds (simplified)
    # Assuming each leg is -110 odds (probability ~0.524)
    individual_odds = -110
    parlay_odds_decimal = (1.91 ** len(parlay_legs))  # Convert -110 to decimal 1.91
    parlay_odds_american = int((parlay_odds_decimal - 1) * 100)

    print("=" * 60)
    print(f"Parlay Odds (estimated): +{parlay_odds_american}")
    print(f"Combined Confidence: {combined_prob:.2%}")
    print(f"Potential Payout: ${100 * parlay_odds_decimal:.2f} on $100 bet")
    print("=" * 60)

    # Risk assessment
    print(f"\n⚠️  Risk Assessment:")
    if combined_prob > 0.40:
        print("  ✅ HIGH CONFIDENCE - This parlay has strong model support")
    elif combined_prob > 0.25:
        print("  ⚠️  MODERATE CONFIDENCE - Proceed with caution")
    else:
        print("  ❌ LOW CONFIDENCE - High risk parlay")


def main():
    """Main test function."""
    print("=" * 60)
    print("PARLAY GENERATION TEST")
    print("=" * 60)

    # Generate predictions
    predictions = generate_test_predictions()

    if len(predictions) == 0:
        print("\n❌ No predictions generated. Cannot build parlay.")
        return

    print(f"\n✓ Generated {len(predictions)} predictions")

    # Build sample parlay
    build_sample_parlay(predictions)

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    print("\nNote: This is a demo using trained models.")
    print("In production, you would:")
    print("  1. Get real market lines from sportsbooks")
    print("  2. Calculate true edge vs market")
    print("  3. Consider correlation between props")
    print("  4. Apply Kelly criterion for bet sizing")
    print("  5. Track actual results for model improvement")


if __name__ == "__main__":
    main()
