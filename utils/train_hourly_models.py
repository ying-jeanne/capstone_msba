"""
Train Hourly Bitcoin Prediction Models
=======================================
Trains XGBoost models for hourly predictions (1h, 6h, 24h)
Uses Binance 1-hour data (60-90 days, ~1440-2160 samples)

Output:
- models/saved_models/hourly/xgboost_1h.json
- models/saved_models/hourly/xgboost_6h.json
- models/saved_models/hourly/xgboost_24h.json
- models/saved_models/hourly/scaler_hourly.pkl
- models/saved_models/hourly/feature_cols_hourly.pkl
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.feature_engineering import engineer_technical_features


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_acc = np.mean(direction_true == direction_pred) * 100
    
    return {'MAE': mae, 'MAPE': mape, 'R2': r2, 'Directional_Accuracy': directional_acc}


def prepare_data_for_horizon(df, horizon_hours=1):
    """Prepare data for specific prediction horizon (in hours)"""
    df[f'future_return_{horizon_hours}h'] = df['close'].pct_change(horizon_hours).shift(-horizon_hours)
    df['current_price'] = df['close']
    
    df_clean = df.dropna(subset=[f'future_return_{horizon_hours}h']).copy()
    
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'current_price'] + \
                   [col for col in df_clean.columns if 'future_return' in col or 'future_price' in col]
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols].values
    y = df_clean[f'future_return_{horizon_hours}h'].values
    current_prices = df_clean['current_price'].values
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, scaler, current_prices


def train_model_for_horizon(X, y, horizon_hours, current_prices):
    """Train XGBoost model for specific horizon"""
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    prices_test = current_prices[split_idx:]
    
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    print(f"\n{'='*60}")
    print(f"Training XGBoost for {horizon_hours}-hour prediction")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    pred_prices_test = prices_test * (1 + y_pred_test)
    actual_prices_test = prices_test * (1 + y_test)
    
    price_mae = mean_absolute_error(actual_prices_test, pred_prices_test)
    price_mape = mean_absolute_percentage_error(actual_prices_test, pred_prices_test) * 100
    
    print(f"\nReturn-Based Metrics:")
    print(f"  Train MAPE: {train_metrics['MAPE']:.2f}%")
    print(f"  Test MAPE:  {test_metrics['MAPE']:.2f}%")
    print(f"  Test R²:    {test_metrics['R2']:.4f}")
    print(f"  Test Dir:   {test_metrics['Directional_Accuracy']:.1f}%")
    
    print(f"\nPrice-Based Metrics:")
    print(f"  Test MAE:   ${price_mae:,.2f}")
    print(f"  Test MAPE:  {price_mape:.2f}%")
    
    metrics = {
        'horizon': f'{horizon_hours}h',
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
    print("  HOURLY MODEL TRAINING PIPELINE")
    print("="*70)
    
    models_dir = Path('models/saved_models/hourly')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[STEP 1] Loading cached Binance 1-hour data...")
    print("  Note: Using TRUE 1-hour candles from Binance")
    print("        Expected: ~8760 hourly candles (365 days × 24 hours)")

    data_path = Path('data/raw/btc_binance_365d_1hour.csv')

    if not data_path.exists():
        print(f"❌ Error: Data file not found at {data_path}")
        print("   Please run data fetching pipeline first to create cached data.")
        sys.exit(1)

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    print(f"✓ Loaded {len(df)} hourly candles from {data_path}")
    print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    print(f"  Latest price: ${df['close'].iloc[-1]:,.2f}")
    
    print("\n[STEP 2] Engineering features...")
    df = engineer_technical_features(df)
    print(f"✓ Created {len(df.columns)} features")
    
    # True hourly predictions with 1-hour candles
    horizons = [1, 6, 24]  # 1 hour, 6 hours, 24 hours ahead
    all_metrics = []

    print(f"\n  Prediction horizons:")
    print(f"    - 1 hour ahead")
    print(f"    - 6 hours ahead")
    print(f"    - 24 hours (1 day) ahead")

    for horizon in horizons:
        print(f"\n[STEP 3.{horizon}] Training {horizon}-hour model...")
        X, y, feature_cols, scaler, current_prices = prepare_data_for_horizon(df, horizon)
        model, metrics = train_model_for_horizon(X, y, horizon, current_prices)
        all_metrics.append(metrics)

        model_path = models_dir / f'xgboost_{horizon}h.json'
        model.save_model(str(model_path))
        print(f"✓ Saved model to {model_path}")
    
    print("\n[STEP 4] Saving preprocessing objects...")
    _, _, feature_cols, scaler, _ = prepare_data_for_horizon(df, 1)
    
    scaler_path = models_dir / 'scaler_hourly.pkl'
    feature_cols_path = models_dir / 'feature_cols_hourly.pkl'
    
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, feature_cols_path)
    
    print(f"✓ Saved scaler to {scaler_path}")
    print(f"✓ Saved feature columns to {feature_cols_path}")
    
    print("\n" + "="*70)
    print("  TRAINING SUMMARY")
    print("="*70)
    
    metrics_df = pd.DataFrame(all_metrics)
    print(metrics_df.to_string(index=False))
    
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    metrics_df.to_csv(results_dir / 'hourly_models_metrics.csv', index=False)
    
    print(f"\n✓ Training complete!")
    print(f"✓ Models saved to: {models_dir}")
    print(f"✓ Metrics saved to: results/hourly_models_metrics.csv")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
