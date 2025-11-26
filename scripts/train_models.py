#!/usr/bin/env python3
"""
Train ML models on historical NFL data.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import nfl_data_py as nfl
except ImportError:
    print("❌ nfl_data_py not installed. Install with: pip install nfl_data_py")
    sys.exit(1)

from src.sports_betting.models.nfl.xgboost_model import XGBoostPropsModel
from src.sports_betting.database import get_session
from src.sports_betting.database.models import Player, InjuryReport


def load_training_data(years=[2022, 2023, 2024]):
    """Load and prepare training data from nfl_data_py."""
    print(f"Loading weekly player data for {years}...")

    # Load weekly data
    weekly_df = nfl.import_weekly_data(years=years)

    print(f"✓ Loaded {len(weekly_df)} player-week records")
    print(f"  Columns: {weekly_df.columns.tolist()[:10]}...")

    # Filter to only include players with significant playing time
    # Focus on skill positions: QB, RB, WR, TE
    weekly_df = weekly_df[weekly_df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()

    print(f"✓ Filtered to {len(weekly_df)} skill position records")

    # Handle missing values
    numeric_cols = weekly_df.select_dtypes(include=[np.number]).columns
    weekly_df[numeric_cols] = weekly_df[numeric_cols].fillna(0)

    return weekly_df


def train_receiving_yards_model(weekly_df, model_dir):
    """Train model for receiving yards predictions."""
    print("\n" + "=" * 60)
    print("TRAINING: Receiving Yards Model")
    print("=" * 60)

    # Filter to players with receiving stats
    receivers_df = weekly_df[
        (weekly_df['position'].isin(['WR', 'TE', 'RB'])) &
        (weekly_df['targets'] > 0)
    ].copy()

    print(f"Training on {len(receivers_df)} receiver records")

    # Ensure we have the target column
    if 'receiving_yards' not in receivers_df.columns:
        print("❌ No receiving_yards column found")
        return None

    # Prepare the dataframe for training
    # The model expects certain columns, so let's add them
    receivers_df['team'] = receivers_df['recent_team']
    receivers_df['opponent'] = receivers_df.get('opponent_team', 'UNK')
    receivers_df['home_team'] = receivers_df.get('recent_team', 'UNK')
    receivers_df['game_date'] = pd.to_datetime(receivers_df['season'].astype(str) + receivers_df['week'].astype(str) + '1', format='%Y%W%w', errors='coerce')

    # Create model
    model = XGBoostPropsModel(prop_type="receiving_yards")

    # Train
    try:
        metrics = model.train(receivers_df, target_column="receiving_yards")

        print("\n📊 Model Performance:")
        print(f"  MAE (Mean Absolute Error): {metrics['mae']:.2f} yards")
        print(f"  RMSE (Root Mean Squared Error): {metrics['rmse']:.2f} yards")
        print(f"  R² Score: {metrics['r2']:.3f}")
        print(f"  Cross-validation MAE: {metrics['cv_mae']:.2f} ± {metrics['cv_mae_std']:.2f}")
        print(f"  Training samples: {metrics['training_samples']}")
        print(f"  Test samples: {metrics['test_samples']}")

        # Save model
        model_path = model_dir / "receiving_yards_xgboost.pkl"
        model.save_model(model_path)
        print(f"\n✓ Model saved to {model_path}")

        # Show feature importance
        importance = model.get_feature_importance()
        if importance:
            print("\n🔍 Top 10 Most Important Features:")
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
            for feature, score in sorted_features:
                print(f"  {feature}: {score:.4f}")

        return metrics

    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_rushing_yards_model(weekly_df, model_dir):
    """Train model for rushing yards predictions."""
    print("\n" + "=" * 60)
    print("TRAINING: Rushing Yards Model")
    print("=" * 60)

    # Filter to players with rushing stats
    rushers_df = weekly_df[
        (weekly_df['position'].isin(['RB', 'QB'])) &
        (weekly_df['carries'] > 0)
    ].copy()

    print(f"Training on {len(rushers_df)} rusher records")

    # Ensure we have the target column
    if 'rushing_yards' not in rushers_df.columns:
        print("❌ No rushing_yards column found")
        return None

    # Prepare the dataframe
    rushers_df['team'] = rushers_df['recent_team']
    rushers_df['opponent'] = rushers_df.get('opponent_team', 'UNK')
    rushers_df['home_team'] = rushers_df.get('recent_team', 'UNK')
    rushers_df['game_date'] = pd.to_datetime(rushers_df['season'].astype(str) + rushers_df['week'].astype(str) + '1', format='%Y%W%w', errors='coerce')

    # Create model
    model = XGBoostPropsModel(prop_type="rushing_yards")

    # Train
    try:
        metrics = model.train(rushers_df, target_column="rushing_yards")

        print("\n📊 Model Performance:")
        print(f"  MAE: {metrics['mae']:.2f} yards")
        print(f"  RMSE: {metrics['rmse']:.2f} yards")
        print(f"  R² Score: {metrics['r2']:.3f}")
        print(f"  Cross-validation MAE: {metrics['cv_mae']:.2f} ± {metrics['cv_mae_std']:.2f}")

        # Save model
        model_path = model_dir / "rushing_yards_xgboost.pkl"
        model.save_model(model_path)
        print(f"\n✓ Model saved to {model_path}")

        return metrics

    except Exception as e:
        print(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Train all models."""
    print("=" * 60)
    print("NFL ML MODEL TRAINING")
    print("=" * 60)
    print()

    # Create models directory
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)

    # Load training data
    try:
        weekly_df = load_training_data(years=[2022, 2023, 2024])
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Train models
    results = {}

    # 1. Receiving Yards
    rec_metrics = train_receiving_yards_model(weekly_df, model_dir)
    if rec_metrics:
        results['receiving_yards'] = rec_metrics

    # 2. Rushing Yards
    rush_metrics = train_rushing_yards_model(weekly_df, model_dir)
    if rush_metrics:
        results['rushing_yards'] = rush_metrics

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModels trained: {len(results)}")
    for prop_type, metrics in results.items():
        print(f"\n{prop_type}:")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  R²: {metrics['r2']:.3f}")
        print(f"  Samples: {metrics['training_samples'] + metrics['test_samples']}")

    print(f"\nModels saved to: {model_dir}/")


if __name__ == "__main__":
    main()
