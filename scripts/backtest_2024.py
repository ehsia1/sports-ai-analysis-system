#!/usr/bin/env python3
"""
Backtest enhanced model on 2024 season.

This simulates what would have happened if we used the enhanced model
throughout the 2024 season.
"""
import sys
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.ml.feature_engineering import ReceivingYardsFeatureEngineer


def backtest_2024():
    print("=" * 80)
    print("2024 SEASON BACKTEST")
    print("=" * 80)

    # 1. Load the enhanced model
    print("\nLoading enhanced model...")
    model_path = Path(__file__).parent.parent / 'models' / 'receiving_yards_enhanced.pkl'

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("   Run: python scripts/train_enhanced_model.py")
        return

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_cols = model_data['features']
    engineer = model_data['engineer']

    print(f"✓ Model loaded")
    print(f"   Training R²: {model_data['metrics']['val_r2']:.4f}")
    print(f"   Features: {len(feature_cols)}")

    # 2. Build features for 2024
    print("\nBuilding features for 2024...")
    # We need historical data to calculate rolling averages
    df = engineer.build_features(seasons=[2020, 2021, 2022, 2023, 2024])

    # Filter to 2024 only
    df_2024 = df[df['season'] == 2024].copy()
    print(f"✓ {len(df_2024)} games in 2024")

    # 3. Remove rows with missing features
    # Avoid duplicates - week is already in feature_cols
    extra_cols = ['receiving_yards', 'player_name', 'position']
    all_cols = feature_cols + [col for col in extra_cols if col not in feature_cols]
    df_clean = df_2024[all_cols].dropna()
    print(f"✓ {len(df_clean)} games with complete feature data")

    if len(df_clean) == 0:
        print("\n❌ No 2024 data available with complete features")
        print("   2024 season may not have finished or data not collected yet")
        return

    # 4. Generate predictions
    print("\nGenerating predictions...")
    X_2024 = df_clean[feature_cols]
    y_2024 = df_clean['receiving_yards']

    # Convert to numpy to avoid xgboost dtype error
    predictions = model.predict(X_2024.values)

    # 5. Evaluate
    print("\n" + "=" * 80)
    print("2024 SEASON PERFORMANCE")
    print("=" * 80)

    rmse = np.sqrt(mean_squared_error(y_2024, predictions))
    mae = mean_absolute_error(y_2024, predictions)
    r2 = r2_score(y_2024, predictions)

    print(f"\nOverall Metrics:")
    print(f"  RMSE: {rmse:.2f} yards")
    print(f"  MAE: {mae:.2f} yards")
    print(f"  R²: {r2:.4f}")

    # 6. Analyze by position
    print("\n" + "-" * 80)
    print("PERFORMANCE BY POSITION")
    print("-" * 80)

    df_results = df_clean.copy()
    df_results['predicted'] = predictions
    df_results['error'] = df_results['receiving_yards'] - predictions

    for position in ['WR', 'TE', 'RB']:
        pos_df = df_results[df_results['position'] == position]
        if len(pos_df) > 0:
            pos_mae = pos_df['error'].abs().mean()
            pos_r2 = r2_score(pos_df['receiving_yards'], pos_df['predicted'])
            print(f"\n{position}:")
            print(f"  Games: {len(pos_df)}")
            print(f"  MAE: {pos_mae:.2f} yards")
            print(f"  R²: {pos_r2:.4f}")

    # 7. Show biggest errors
    print("\n" + "-" * 80)
    print("BIGGEST OVER-PREDICTIONS (Model too high)")
    print("-" * 80)

    df_results['error_abs'] = df_results['error'].abs()
    worst_over = df_results.nsmallest(10, 'error')[['player_name', 'week', 'predicted', 'receiving_yards', 'error']]

    for idx, row in worst_over.iterrows():
        print(f"Week {row['week']:2.0f} - {row['player_name']:20s}: Predicted {row['predicted']:5.1f}, Actual {row['receiving_yards']:5.1f} (off by {row['error']:+5.1f})")

    print("\n" + "-" * 80)
    print("BIGGEST UNDER-PREDICTIONS (Model too low)")
    print("-" * 80)

    worst_under = df_results.nlargest(10, 'error')[['player_name', 'week', 'predicted', 'receiving_yards', 'error']]

    for idx, row in worst_under.iterrows():
        print(f"Week {row['week']:2.0f} - {row['player_name']:20s}: Predicted {row['predicted']:5.1f}, Actual {row['receiving_yards']:5.1f} (off by {row['error']:+5.1f})")

    # 8. Best predictions
    print("\n" + "-" * 80)
    print("MOST ACCURATE PREDICTIONS")
    print("-" * 80)

    best = df_results.nsmallest(10, 'error_abs')[['player_name', 'week', 'predicted', 'receiving_yards', 'error']]

    for idx, row in best.iterrows():
        print(f"Week {row['week']:2.0f} - {row['player_name']:20s}: Predicted {row['predicted']:5.1f}, Actual {row['receiving_yards']:5.1f} (error {row['error']:+5.1f})")

    # 9. Summary
    print("\n" + "=" * 80)
    print("BACKTEST SUMMARY")
    print("=" * 80)

    print(f"""
Model Performance on 2024:
  Games Analyzed: {len(df_results)}
  RMSE: {rmse:.2f} yards
  MAE: {mae:.2f} yards
  R²: {r2:.4f}

Comparison to Training:
  Training R²: {model_data['metrics']['val_r2']:.4f}
  2024 R²: {r2:.4f}
  Difference: {r2 - model_data['metrics']['val_r2']:+.4f}

""")

    if r2 > 0.25:
        print("✅ Model maintains good performance on 2024 data")
        print("   R² > 0.25 indicates meaningful predictive power")
    else:
        print("⚠️  Model performance degraded on 2024")
        print("   May need retraining with 2024 data")

    print("\nNext steps:")
    print("  1. Analyze systematic errors (over/under predictions)")
    print("  2. Check if certain players/teams are harder to predict")
    print("  3. Consider retraining with 2024 data included")


if __name__ == "__main__":
    backtest_2024()
