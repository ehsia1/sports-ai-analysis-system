#!/usr/bin/env python3
"""
Train enhanced receiving yards prediction model.

Improvements over previous version:
- More training data (2020-2023)
- Enhanced features (target share, opponent defense, recent form)
- Proper train/validation/test split
- Feature importance analysis
- Better hyperparameters
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.sports_betting.ml.feature_engineering import ReceivingYardsFeatureEngineer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np


def train_model():
    print("=" * 80)
    print("ENHANCED RECEIVING YARDS MODEL TRAINING")
    print("=" * 80)

    # 1. Build features
    engineer = ReceivingYardsFeatureEngineer()

    # Use 2020-2023 for training (more data!)
    print("\nBuilding features for 2020-2023...")
    df = engineer.build_features(seasons=[2020, 2021, 2022, 2023])

    # 2. Get feature list
    feature_cols = engineer.get_feature_list()
    target_col = 'receiving_yards'

    # 3. Remove rows with missing data
    print(f"\nPreparing data...")
    print(f"Initial records: {len(df)}")

    # Keep only rows with all features available
    df_clean = df[feature_cols + [target_col, 'season', 'player_name']].dropna()
    print(f"After removing missing data: {len(df_clean)}")

    # 4. Create train/validation/test split
    # Strategy: Use 2020-2022 for training, 2023 Week 1-12 for validation, 2023 Week 13+ for test
    print(f"\nCreating train/validation/test split...")

    train_df = df_clean[df_clean['season'].isin([2020, 2021, 2022])]
    val_df = df_clean[(df_clean['season'] == 2023) & (df_clean['week'] <= 12)]
    test_df = df_clean[(df_clean['season'] == 2023) & (df_clean['week'] > 12)]

    print(f"Training set: {len(train_df)} records (2020-2022)")
    print(f"Validation set: {len(val_df)} records (2023 weeks 1-12)")
    print(f"Test set: {len(test_df)} records (2023 weeks 13+)")

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # 5. Train XGBoost model
    print(f"\nTraining XGBoost model...")
    print(f"Features: {len(feature_cols)}")

    model = xgb.XGBRegressor(
        n_estimators=200,  # More trees
        max_depth=6,  # Deeper trees
        learning_rate=0.05,  # Lower learning rate
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=20
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # 6. Evaluate on validation set
    print(f"\n" + "=" * 80)
    print("VALIDATION SET PERFORMANCE (2023 Weeks 1-12)")
    print("=" * 80)

    y_val_pred = model.predict(X_val)

    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\nRMSE: {val_rmse:.2f} yards")
    print(f"MAE: {val_mae:.2f} yards")
    print(f"R²: {val_r2:.4f}")

    # 7. Evaluate on test set
    print(f"\n" + "=" * 80)
    print("TEST SET PERFORMANCE (2023 Weeks 13+)")
    print("=" * 80)

    y_test_pred = model.predict(X_test)

    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print(f"\nRMSE: {test_rmse:.2f} yards")
    print(f"MAE: {test_mae:.2f} yards")
    print(f"R²: {test_r2:.4f}")

    # 8. Show sample predictions
    print(f"\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (Test Set)")
    print("=" * 80)

    test_df_results = test_df.copy()
    test_df_results['predicted'] = y_test_pred

    print("\nTop 10 highest predicted:")
    top_pred = test_df_results.nlargest(10, 'predicted')[['player_name', 'predicted', target_col]]
    for idx, row in top_pred.iterrows():
        print(f"  {row['player_name']}: Predicted {row['predicted']:.1f}, Actual {row[target_col]:.1f}")

    # 9. Feature importance
    print(f"\n" + "=" * 80)
    print("TOP 15 MOST IMPORTANT FEATURES")
    print("=" * 80)

    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + importance_df.head(15).to_string(index=False))

    # 10. Save model
    print(f"\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_dir = Path(__file__).parent.parent / 'models'
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / 'receiving_yards_enhanced.pkl'

    # Save model and feature list
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'features': feature_cols,
            'engineer': engineer,
            'metrics': {
                'val_rmse': val_rmse,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
            }
        }, f)

    print(f"\n✓ Model saved to: {model_path}")

    # 11. Summary
    print(f"\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"""
Summary:
  Training Data: 2020-2022 ({len(train_df)} records)
  Validation Data: 2023 Weeks 1-12 ({len(val_df)} records)
  Test Data: 2023 Weeks 13+ ({len(test_df)} records)

  Features: {len(feature_cols)}

  Validation Performance:
    - RMSE: {val_rmse:.2f} yards
    - MAE: {val_mae:.2f} yards
    - R²: {val_r2:.4f}

  Test Performance:
    - RMSE: {test_rmse:.2f} yards
    - MAE: {test_mae:.2f} yards
    - R²: {test_r2:.4f}

Next Steps:
  1. Review feature importance
  2. Backtest on 2024 season (if data available)
  3. Compare to previous model (R² = -0.012)
  4. Generate predictions for upcoming games
""")

    # Compare to previous model
    if val_r2 > 0:
        print(f"✅ SUCCESS! R² improved from -0.012 to {val_r2:.4f}")
        print(f"   Model now explains {val_r2*100:.1f}% of variance")
    else:
        print(f"⚠️  WARNING: R² is still negative ({val_r2:.4f})")
        print(f"   Model needs more work")


if __name__ == "__main__":
    train_model()
