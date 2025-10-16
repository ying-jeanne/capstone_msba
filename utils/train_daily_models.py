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
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_acc = np.mean(direction_true == direction_pred) * 100
    
    return {
        'MAE': mae,
        'MAPE': mape,
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
    # Train/test split (80/20, chronological)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    prices_test = current_prices[split_idx:]
    
    # XGBoost parameters (optimized for return prediction)
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 300,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model
    print(f"\n{'='*60}")
    print(f"Training XGBoost for {horizon_days}-day prediction")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics on returns
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Convert returns to prices for price-based metrics
    pred_prices_test = prices_test * (1 + y_pred_test)
    actual_prices_test = prices_test * (1 + y_test)
    
    price_mae = mean_absolute_error(actual_prices_test, pred_prices_test)
    price_mape = mean_absolute_percentage_error(actual_prices_test, pred_prices_test) * 100
    
    # Print results
    print(f"\nReturn-Based Metrics:")
    print(f"  Train MAPE: {train_metrics['MAPE']:.2f}%")
    print(f"  Test MAPE:  {test_metrics['MAPE']:.2f}%")
    print(f"  Test R²:    {test_metrics['R2']:.4f}")
    print(f"  Test Dir:   {test_metrics['Directional_Accuracy']:.1f}%")
    
    print(f"\nPrice-Based Metrics:")
    print(f"  Test MAE:   ${price_mae:,.2f}")
    print(f"  Test MAPE:  {price_mape:.2f}%")
    
    # Store comprehensive metrics
    metrics = {
        'horizon': f'{horizon_days}d',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'return_train_mape': train_metrics['MAPE'],
        'return_test_mape': test_metrics['MAPE'],
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
    data_path = Path('data/raw/btc_yahoo_2y_daily.csv')

    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("   Please run data fetching pipeline first to create cached data.")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} daily bars from {data_path}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
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
    print(metrics_df.to_string(index=False))
    
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
