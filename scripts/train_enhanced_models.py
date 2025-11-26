#!/usr/bin/env python3
"""
Train enhanced ML models with improved features and position-specific models.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import nfl_data_py as nfl
except ImportError:
    print("❌ nfl_data_py not installed. Install with: pip install nfl_data_py")
    sys.exit(1)

from src.sports_betting.models.nfl.xgboost_model import XGBoostPropsModel
from src.sports_betting.features.enhanced_features import EnhancedFeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_enhanced_training_data(years=[2022, 2023, 2024]):
    """Load and prepare enhanced training data."""
    print(f"Loading data for {years}...")

    # Load weekly player data
    weekly_df = nfl.import_weekly_data(years=years)
    print(f"✓ Loaded {len(weekly_df)} player-week records")

    # Load schedule data for game context
    schedule_df = nfl.import_schedules(years=years)
    print(f"✓ Loaded {len(schedule_df)} games")

    # Filter to skill positions
    weekly_df = weekly_df[weekly_df['position'].isin(['QB', 'RB', 'WR', 'TE'])].copy()
    print(f"✓ Filtered to {len(weekly_df)} skill position records")

    # Apply enhanced feature engineering
    print("\nApplying enhanced feature engineering...")
    engineer = EnhancedFeatureEngineer()

    weekly_df = engineer.engineer_all_features(
        player_df=weekly_df,
        all_weekly_data=weekly_df,
        schedule_df=schedule_df
    )

    print("✓ Feature engineering complete")

    # Handle missing values
    numeric_cols = weekly_df.select_dtypes(include=[np.number]).columns
    weekly_df[numeric_cols] = weekly_df[numeric_cols].fillna(0)

    return weekly_df


def train_position_specific_model(
    weekly_df,
    position_group,
    prop_type,
    model_name,
    model_dir
):
    """Train a position-specific model."""
    print("\n" + "=" * 60)
    print(f"TRAINING: {model_name}")
    print(f"Position: {position_group}, Prop: {prop_type}")
    print("=" * 60)

    # Filter data
    if prop_type == 'receiving_yards':
        model_df = weekly_df[
            (weekly_df['position'].isin(position_group)) &
            (weekly_df['targets'] > 0)
        ].copy()
    elif prop_type == 'rushing_yards':
        model_df = weekly_df[
            (weekly_df['position'].isin(position_group)) &
            (weekly_df['carries'] > 0)
        ].copy()
    else:
        model_df = weekly_df[weekly_df['position'].isin(position_group)].copy()

    if len(model_df) < 100:
        print(f"❌ Insufficient data: only {len(model_df)} records")
        return None

    print(f"Training on {len(model_df)} records")

    # Check for target column
    if prop_type not in model_df.columns:
        print(f"❌ Target column {prop_type} not found")
        return None

    # Create custom feature list based on prop type
    from src.sports_betting.features.enhanced_features import get_enhanced_feature_list

    base_features = get_enhanced_feature_list()

    # Filter to only features that exist
    available_features = [f for f in base_features if f in model_df.columns]

    print(f"Using {len(available_features)} features")

    # Create and train model
    model = XGBoostPropsModel(prop_type=prop_type)

    # Override feature preparation to use our enhanced features
    original_prepare = model.prepare_features

    def enhanced_prepare(df):
        """Use our enhanced features directly."""
        df = df.reset_index(drop=True)
        model.feature_columns = available_features

        # Ensure all features exist
        for feature in available_features:
            if feature not in df.columns:
                df[feature] = 0

        return df

    model.prepare_features = enhanced_prepare

    try:
        # Train
        metrics = model.train(model_df, target_column=prop_type)

        print("\n📊 Model Performance:")
        print(f"  MAE: {metrics['mae']:.2f}")
        print(f"  RMSE: {metrics['rmse']:.2f}")
        print(f"  R² Score: {metrics['r2']:.3f}")
        print(f"  CV MAE: {metrics['cv_mae']:.2f} ± {metrics['cv_mae_std']:.2f}")
        print(f"  Training samples: {metrics['training_samples']}")
        print(f"  Test samples: {metrics['test_samples']}")

        # Save model
        model_path = model_dir / f"{model_name}.pkl"
        model.save_model(model_path)
        print(f"\n✓ Model saved to {model_path}")

        # Show top features
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


def main():
    """Train all enhanced models."""
    print("=" * 60)
    print("ENHANCED NFL ML MODEL TRAINING")
    print("=" * 60)
    print()

    # Create models directory
    model_dir = Path(__file__).parent.parent / "models" / "enhanced"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load enhanced training data
    try:
        weekly_df = load_enhanced_training_data(years=[2022, 2023, 2024])
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Track all results
    results = {}

    # Position-specific models for receiving
    print("\n" + "=" * 60)
    print("POSITION-SPECIFIC RECEIVING MODELS")
    print("=" * 60)

    # 1. Wide Receivers
    wr_metrics = train_position_specific_model(
        weekly_df, ['WR'], 'receiving_yards',
        'wr_receiving_xgboost', model_dir
    )
    if wr_metrics:
        results['WR_receiving'] = wr_metrics

    # 2. Tight Ends
    te_metrics = train_position_specific_model(
        weekly_df, ['TE'], 'receiving_yards',
        'te_receiving_xgboost', model_dir
    )
    if te_metrics:
        results['TE_receiving'] = te_metrics

    # 3. Running Backs (receiving)
    rb_rec_metrics = train_position_specific_model(
        weekly_df, ['RB'], 'receiving_yards',
        'rb_receiving_xgboost', model_dir
    )
    if rb_rec_metrics:
        results['RB_receiving'] = rb_rec_metrics

    # Position-specific models for rushing
    print("\n" + "=" * 60)
    print("POSITION-SPECIFIC RUSHING MODELS")
    print("=" * 60)

    # 4. Running Backs (rushing)
    rb_rush_metrics = train_position_specific_model(
        weekly_df, ['RB'], 'rushing_yards',
        'rb_rushing_xgboost', model_dir
    )
    if rb_rush_metrics:
        results['RB_rushing'] = rb_rush_metrics

    # 5. Quarterbacks (rushing)
    qb_rush_metrics = train_position_specific_model(
        weekly_df, ['QB'], 'rushing_yards',
        'qb_rushing_xgboost', model_dir
    )
    if qb_rush_metrics:
        results['QB_rushing'] = qb_rush_metrics

    # Summary
    print("\n" + "=" * 60)
    print("ENHANCED TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nModels trained: {len(results)}")

    print("\n📊 Performance Summary:")
    print("-" * 60)
    print(f"{'Model':<20} {'MAE':<10} {'R²':<10} {'Samples':<10}")
    print("-" * 60)

    for model_name, metrics in results.items():
        samples = metrics['training_samples'] + metrics['test_samples']
        print(f"{model_name:<20} {metrics['mae']:<10.2f} {metrics['r2']:<10.3f} {samples:<10}")

    print(f"\n✓ Models saved to: {model_dir}/")

    # Compare with baseline
    print("\n" + "=" * 60)
    print("COMPARISON WITH BASELINE")
    print("=" * 60)
    print("\nBaseline models (from original training):")
    print("  Receiving Yards: MAE 14.48, R² 0.539")
    print("  Rushing Yards: MAE 14.33, R² 0.578")
    print("\nEnhanced models:")

    # Calculate average improvement
    if results:
        avg_mae = np.mean([m['mae'] for m in results.values()])
        avg_r2 = np.mean([m['r2'] for m in results.values()])
        print(f"  Average MAE: {avg_mae:.2f}")
        print(f"  Average R²: {avg_r2:.3f}")

        baseline_mae = 14.4
        baseline_r2 = 0.56

        mae_improvement = ((baseline_mae - avg_mae) / baseline_mae) * 100
        r2_improvement = ((avg_r2 - baseline_r2) / baseline_r2) * 100

        print(f"\n📈 Improvements:")
        print(f"  MAE: {mae_improvement:+.1f}%")
        print(f"  R²: {r2_improvement:+.1f}%")


if __name__ == "__main__":
    main()
