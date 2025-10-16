"""
Train Daily Bitcoin Prediction Models
======================================
Trains XGBoost models for daily predictions (1d, 3d, 7d)
Uses Yahoo Finance 2-year daily data

Output:
- models/saved_models/daily/xgboost_1d.json
- models/saved_models/daily/xgboost_3d.json
- models/saved_models/daily/xgboost_7d.json
- models/saved_models/daily/scaler_daily.pkl
- models/saved_models/daily/feature_cols_daily.pkl
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from pathlib import Path
import joblib
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.feature_engineering import engineer_technical_features, add_sentiment_features


def calculate_metrics(y_true, y_pred):
    """
    Calculate metrics for return predictions
    Note: MAPE removed - doesn't work for values near 0 (returns)
    """
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_acc = np.mean(direction_true == direction_pred) * 100
    
    return {
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_acc
    }


def prepare_data_for_horizon(df, horizon_days=1):
    """
    Prepare data for a specific prediction horizon
    
    Args:
        df: DataFrame with features and price
        horizon_days: Days ahead to predict (1, 3, or 7)
    
    Returns:
        X, y, feature_cols, scaler, current_prices
    """
    # Create return target
    df[f'future_return_{horizon_days}d'] = df['close'].pct_change(horizon_days).shift(-horizon_days)
    
    # Store current prices for later conversion
    df['current_price'] = df['close']
    
    # Drop rows with NaN targets
    df_clean = df.dropna(subset=[f'future_return_{horizon_days}d']).copy()
    
    # Select feature columns (exclude price, volume, and targets)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'current_price'] + \
                   [col for col in df_clean.columns if 'future_return' in col or 'future_price' in col]
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols].values
    y = df_clean[f'future_return_{horizon_days}d'].values
    current_prices = df_clean['current_price'].values
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, scaler, current_prices


def train_model_for_horizon(X, y, horizon_days, current_prices):
    """
    Train XGBoost model for specific horizon
    
    Args:
        X: Features
        y: Return targets
        horizon_days: Prediction horizon
        current_prices: Current prices for evaluation
    
    Returns:
        model, metrics
    """
    # Train/val/test split (70/15/15, chronological)
    # More training data helps with Bitcoin's high noise
    split_1 = int(len(X) * 0.7)
    split_2 = int(len(X) * 0.85)

    X_train = X[:split_1]
    X_val = X[split_1:split_2]
    X_test = X[split_2:]

    y_train = y[:split_1]
    y_val = y[split_1:split_2]
    y_test = y[split_2:]

    prices_val = current_prices[split_1:split_2]
    prices_test = current_prices[split_2:]
    
    # XGBoost parameters (with stronger regularization to prevent overfitting)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,              # Reduced from 5 (less complex trees)
        'learning_rate': 0.05,
        'n_estimators': 200,          # Reduced from 300 (fewer trees)
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.3,                 # Increased from 0.1 (stronger pruning)
        'reg_alpha': 0.5,             # Increased from 0.1 (stronger L1)
        'reg_lambda': 2.0,            # Increased from 1.0 (stronger L2)
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    print(f"\n{'='*60}")
    print(f"Training XGBoost for {horizon_days}-day prediction")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # XGBoost 3.0+ uses early_stopping_rounds in params, not fit()
    params['early_stopping_rounds'] = 20

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Calculate metrics on returns
    train_metrics = calculate_metrics(y_train, y_pred_train)
    val_metrics = calculate_metrics(y_val, y_pred_val)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Convert returns to prices for price-based metrics
    pred_prices_test = prices_test * (1 + y_pred_test)
    actual_prices_test = prices_test * (1 + y_test)
    
    price_mae = mean_absolute_error(actual_prices_test, pred_prices_test)
    price_mape = mean_absolute_percentage_error(actual_prices_test, pred_prices_test) * 100
    
    # Print results with proper interpretation
    print(f"\n{'='*60}")
    print(f"📊 MODEL PERFORMANCE SUMMARY")
    print(f"{'='*60}")

    print(f"\n🎯 Price Prediction Accuracy (Primary Metrics):")
    print(f"  Test MAPE:  {price_mape:.2f}%")
    print(f"  Test MAE:   ${price_mae:,.2f}")

    # Quality assessment
    if price_mape < 2:
        quality = "⭐ EXCELLENT"
        desc = "(<2% error - industry-leading)"
    elif price_mape < 5:
        quality = "✅ GOOD"
        desc = "(<5% error - professional-grade)"
    elif price_mape < 10:
        quality = "⚠️  ACCEPTABLE"
        desc = "(5-10% error - usable)"
    else:
        quality = "❌ POOR"
        desc = "(>10% error - needs improvement)"

    print(f"  Quality: {quality} {desc}")

    print(f"\n📈 Trend Prediction:")
    print(f"  Directional Accuracy: {test_metrics['Directional_Accuracy']:.1f}%")

    if test_metrics['Directional_Accuracy'] > 52:
        trend_quality = "✅ Slight edge"
        trend_desc = f"({test_metrics['Directional_Accuracy']:.1f}% > 50% random)"
    elif test_metrics['Directional_Accuracy'] >= 48:
        trend_quality = "⚠️  No clear signal"
        trend_desc = f"({test_metrics['Directional_Accuracy']:.1f}% ≈ 50% random)"
    else:
        trend_quality = "❌ Unreliable"
        trend_desc = f"({test_metrics['Directional_Accuracy']:.1f}% < 50% random)"

    print(f"  Assessment: {trend_quality} {trend_desc}")

    print(f"\n🔍 Model Diagnostics:")
    print(f"  R² (returns): {test_metrics['R2']:.4f}")
    print(f"  Return MAE: {test_metrics['MAE']:.6f}")
    print(f"  Note: R² near 0 is typical for crypto (high volatility & random walk)")
    print(f"  Val R²: {val_metrics['R2']:.4f} | Test R²: {test_metrics['R2']:.4f}")

    print(f"\n⚖️  Generalization Check:")
    print(f"  No overfitting detected ✅") if abs(test_metrics['R2'] - val_metrics['R2']) < 0.05 else print(f"  Potential overfitting ⚠️")

    print(f"\n📋 Detailed Stats:")
    print(f"  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
    print(f"  Test samples:  {len(X_test)}")
    
    # Store comprehensive metrics
    metrics = {
        'horizon': f'{horizon_days}d',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'return_train_mae': train_metrics['MAE'],
        'return_test_mae': test_metrics['MAE'],
        'return_test_r2': test_metrics['R2'],
        'directional_accuracy': test_metrics['Directional_Accuracy'],
        'price_mae': price_mae,
        'price_mape': price_mape
    }
    
    return model, metrics


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("  DAILY MODEL TRAINING PIPELINE")
    print("="*70)

    # Create output directories
    models_dir = Path('models/saved_models/daily')
    models_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load cached data (training should NOT fetch fresh data)
    print("\n[STEP 1] Loading cached Yahoo Finance data...")

    # Try 5-year cache first, fallback to 2-year
    data_path_5y = Path('data/raw/btc_yahoo_5y_daily.csv')
    data_path_2y = Path('data/raw/btc_yahoo_2y_daily.csv')

    if data_path_5y.exists():
        data_path = data_path_5y
        print(f"✓ Using 5-year data cache")
    elif data_path_2y.exists():
        data_path = data_path_2y
        print(f"⚠️  Using 2-year data cache (consider fetching 5-year data)")
    else:
        print(f"❌ Error: No cached data found")
        print(f"   Expected: {data_path_5y} or {data_path_2y}")
        print("   Please run: python run_full_pipeline.py")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} daily bars from {data_path.name}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Step 2: Engineer features
    print("\n[STEP 2] Engineering features...")
    df = engineer_technical_features(df)
    print(f"✓ Created {len(df.columns)} technical features")
    
    # Step 2b: Add sentiment features
    print("\n[STEP 2b] Adding sentiment features...")
    df = add_sentiment_features(df)
    print(f"✓ Total features (technical + sentiment): {len(df.columns)}")
    
    # Step 3: Train models for each horizon
    horizons = [1, 3, 7]
    all_metrics = []
    
    for horizon in horizons:
        print(f"\n[STEP 3.{horizon}] Training {horizon}-day model...")
        
        # Prepare data
        X, y, feature_cols, scaler, current_prices = prepare_data_for_horizon(df, horizon_days=horizon)
        
        # Train model
        model, metrics = train_model_for_horizon(X, y, horizon, current_prices)
        all_metrics.append(metrics)
        
        # Save model
        model_path = models_dir / f'xgboost_{horizon}d.json'
        model.save_model(str(model_path))
        print(f"✓ Saved model to {model_path}")
    
    # Step 4: Save scaler and feature columns (same for all horizons)
    print("\n[STEP 4] Saving preprocessing objects...")
    
    # Use the last prepared data's scaler and features
    _, _, feature_cols, scaler, _ = prepare_data_for_horizon(df, horizon_days=1)
    
    scaler_path = models_dir / 'scaler_daily.pkl'
    feature_cols_path = models_dir / 'feature_cols_daily.pkl'
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, feature_cols_path)
    
    print(f"✓ Saved scaler to {scaler_path}")
    print(f"✓ Saved feature columns to {feature_cols_path}")
    
    # Step 5: Summary
    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)

    metrics_df = pd.DataFrame(all_metrics)

    # Display compact summary
    print("\n📊 Performance Overview:")
    print(f"{'Horizon':<10} {'Price MAPE':<12} {'Directional':<15} {'Quality':<20}")
    print("-" * 70)
    for _, row in metrics_df.iterrows():
        # Quality assessment
        if row['price_mape'] < 2:
            quality = "⭐ EXCELLENT"
        elif row['price_mape'] < 5:
            quality = "✅ GOOD"
        else:
            quality = "⚠️  ACCEPTABLE"

        # Directional assessment
        dir_acc = row['directional_accuracy']
        if dir_acc > 52:
            dir_symbol = "✅"
        elif dir_acc >= 48:
            dir_symbol = "⚠️"
        else:
            dir_symbol = "❌"

        print(f"{row['horizon']:<10} {row['price_mape']:>6.2f}%      {dir_symbol} {dir_acc:>5.1f}%      {quality}")

    print("\n💡 Interpretation Guide:")
    print("  • Price MAPE: Lower is better (<2% excellent, <5% good)")
    print("  • Directional: >52% has predictive edge, ≈50% is random")
    print("  • R² near 0 is NORMAL for crypto (high noise, random walk)")

    print("\n📋 Recommendations:")
    best_price = metrics_df.loc[metrics_df['price_mape'].idxmin()]
    best_dir = metrics_df.loc[metrics_df['directional_accuracy'].idxmax()]

    print(f"  • Best price accuracy: {best_price['horizon']} ({best_price['price_mape']:.2f}% MAPE)")
    print(f"  • Best trend prediction: {best_dir['horizon']} ({best_dir['directional_accuracy']:.1f}% accuracy)")

    # Use case recommendations
    print("\n🎯 Suggested Use Cases:")
    for _, row in metrics_df.iterrows():
        if row['price_mape'] < 3 and row['directional_accuracy'] > 52:
            print(f"  • {row['horizon']}: Price targeting + Trend following ✅")
        elif row['price_mape'] < 5:
            print(f"  • {row['horizon']}: Price targeting, Risk management ⚠️")
        else:
            print(f"  • {row['horizon']}: Reference only (limited reliability) ❌")

    # Save metrics
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    metrics_df.to_csv(results_dir / 'daily_models_metrics.csv', index=False)

    print(f"\n✓ Training complete!")
    print(f"✓ Models saved to: {models_dir}")
    print(f"✓ Metrics saved to: results/daily_models_metrics.csv")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
