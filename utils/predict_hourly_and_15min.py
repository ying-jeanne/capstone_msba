"""
Generate Hourly and 15-Minute Bitcoin Price Predictions
========================================================
Loads pre-trained hourly and 15-min models and generates predictions

Hourly predictions: 1h, 6h, 24h
15-min predictions: 15m, 1h, 4h

This script runs fast (~40 seconds) - NO TRAINING HAPPENS HERE

Output: 
- data/predictions/hourly_predictions.csv
- data/predictions/15min_predictions.csv
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetcher import get_bitcoin_data, get_bitcoin_data_incremental
from utils.feature_engineering import engineer_technical_features


def load_models_for_timeframe(models_dir, horizons):
    """Load pre-trained models for specific timeframe"""
    models = {}
    
    for horizon in horizons:
        model_path = models_dir / f'xgboost_{horizon}.json'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        models[horizon] = model
    
    # Load scaler and feature columns
    scaler_path = list(models_dir.glob('scaler_*.pkl'))[0]
    feature_cols_path = list(models_dir.glob('feature_cols_*.pkl'))[0]
    
    scaler = joblib.load(scaler_path)
    feature_cols = joblib.load(feature_cols_path)
    
    return models, scaler, feature_cols


def prepare_latest_data(df, feature_cols, scaler):
    """Prepare the latest data point for prediction"""
    latest_features = df[feature_cols].iloc[-1:].values
    latest_features_scaled = scaler.transform(latest_features)
    current_price = df['close'].iloc[-1]
    current_timestamp = df.index[-1]
    
    return latest_features_scaled, current_price, current_timestamp


def generate_predictions(models, latest_features_scaled, current_price):
    """Generate predictions for all horizons"""
    predictions = {}
    
    for horizon, model in models.items():
        predicted_return = model.predict(latest_features_scaled)[0]
        predicted_price = current_price * (1 + predicted_return)
        
        predictions[horizon] = {
            'predicted_return': predicted_return,
            'predicted_price': predicted_price,
            'change_percent': predicted_return * 100
        }
    
    return predictions


def save_predictions(predictions, current_price, current_timestamp, output_path, 
                     data_source, timeframe):
    """Save predictions to CSV"""
    record = {
        'timestamp': current_timestamp,
        'current_price': current_price,
        'data_source': data_source,
        'timeframe': timeframe,
        'generated_at': datetime.now().isoformat()
    }
    
    # Add predictions dynamically
    for horizon, pred in predictions.items():
        record[f'pred_{horizon}_price'] = pred['predicted_price']
        record[f'pred_{horizon}_return'] = pred['predicted_return']
        record[f'pred_{horizon}_change_pct'] = pred['change_percent']
    
    # Save to CSV
    df = pd.DataFrame([record])
    df.to_csv(output_path, index=False)
    
    # Also append to history (avoid duplicates)
    history_path = output_path.parent / f'{timeframe}_predictions_history.csv'
    if history_path.exists():
        # Load existing history to check for duplicates
        history_df = pd.read_csv(history_path)
        # Check if this timestamp already exists
        if current_timestamp not in history_df['timestamp'].values:
            df.to_csv(history_path, mode='a', header=False, index=False)
        else:
            print(f"  ⚠ Prediction for {current_timestamp} already exists in history, skipping append")
    else:
        df.to_csv(history_path, index=False)
    
    return record


def generate_hourly_predictions():
    """Generate hourly predictions (1h, 6h, 24h)"""
    print("\n" + "="*70)
    print("  HOURLY PREDICTIONS GENERATOR")
    print("="*70)
    
    models_dir = Path('models/saved_models/hourly')
    output_dir = Path('data/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'hourly_predictions.csv'
    
    # Load models
    print("\n[1/5] Loading hourly models...")
    models, scaler, feature_cols = load_models_for_timeframe(models_dir, ['1h', '6h', '24h'])
    print(f"✓ Loaded 3 models (1h, 6h, 24h)")
    print(f"  Note: Models predict 1h/6h/24h ahead using TRUE 1-hour candles from Binance")

    # Fetch data (365 days to match training data and prevent overfitting)
    # Use incremental fetch to only fetch new data if cache exists
    print("\n[2/5] Fetching Binance 1-hour data (incremental update)...")
    df = get_bitcoin_data_incremental(source='binance_1h', days=365, verbose=True)

    if df is None or df.empty:
        raise Exception("Failed to fetch data")

    print(f"✓ Loaded {len(df)} hourly candles")
    print(f"  Current price: ${df['close'].iloc[-1]:,.2f}")

    # Engineer features
    print("\n[3/5] Engineering features...")
    df = engineer_technical_features(df)
    print(f"✓ Created features")

    # Prepare data
    print("\n[4/5] Preparing latest data...")
    latest_features_scaled, current_price, current_timestamp = prepare_latest_data(
        df, feature_cols, scaler
    )

    # Generate predictions
    print("\n[5/5] Generating predictions...")
    predictions = generate_predictions(models, latest_features_scaled, current_price)

    print(f"\nHOURLY PREDICTIONS:")
    print(f"  Current: ${current_price:,.2f}")
    print(f"  1h:  ${predictions['1h']['predicted_price']:,.2f} ({predictions['1h']['change_percent']:+.2f}%)")
    print(f"  6h:  ${predictions['6h']['predicted_price']:,.2f} ({predictions['6h']['change_percent']:+.2f}%)")
    print(f"  24h: ${predictions['24h']['predicted_price']:,.2f} ({predictions['24h']['change_percent']:+.2f}%)")

    # Save
    save_predictions(predictions, current_price, current_timestamp, output_path,
                    'binance_1h', 'hourly')
    print(f"\n✓ Saved to {output_path}")
    
    return predictions


def generate_15min_predictions():
    """Generate 15-minute predictions (15m, 1h, 4h)"""
    print("\n" + "="*70)
    print("  15-MINUTE PREDICTIONS GENERATOR")
    print("="*70)
    
    models_dir = Path('models/saved_models/15min')
    output_dir = Path('data/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / '15min_predictions.csv'
    
    # Load models
    print("\n[1/5] Loading 15-min models...")
    models, scaler, feature_cols = load_models_for_timeframe(models_dir, ['15m', '1h', '4h'])
    print(f"✓ Loaded 3 models (15m, 1h, 4h)")
    
    # Fetch data (use incremental fetch to only fetch new data)
    print("\n[2/5] Fetching Binance 15-min data (incremental update)...")
    df = get_bitcoin_data_incremental(source='binance', days=60, verbose=True)

    if df is None or df.empty:
        raise Exception("Failed to fetch data")

    print(f"✓ Loaded {len(df)} 15-min bars")
    print(f"  Current price: ${df['close'].iloc[-1]:,.2f}")
    
    # Engineer features
    print("\n[3/5] Engineering features...")
    df = engineer_technical_features(df)
    print(f"✓ Created features")
    
    # Prepare data
    print("\n[4/5] Preparing latest data...")
    latest_features_scaled, current_price, current_timestamp = prepare_latest_data(
        df, feature_cols, scaler
    )
    
    # Generate predictions
    print("\n[5/5] Generating predictions...")
    predictions = generate_predictions(models, latest_features_scaled, current_price)
    
    print(f"\n15-MIN PREDICTIONS:")
    print(f"  Current: ${current_price:,.2f}")
    print(f"  15m: ${predictions['15m']['predicted_price']:,.2f} ({predictions['15m']['change_percent']:+.2f}%)")
    print(f"  1h:  ${predictions['1h']['predicted_price']:,.2f} ({predictions['1h']['change_percent']:+.2f}%)")
    print(f"  4h:  ${predictions['4h']['predicted_price']:,.2f} ({predictions['4h']['change_percent']:+.2f}%)")
    
    # Save
    save_predictions(predictions, current_price, current_timestamp, output_path, 
                    'binance', '15min')
    print(f"\n✓ Saved to {output_path}")
    
    return predictions


def main():
    """Main prediction pipeline"""
    print("\n" + "="*70)
    print("  HOURLY + 15-MIN PREDICTIONS")
    print("="*70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    try:
        # Generate hourly predictions
        hourly_preds = generate_hourly_predictions()
        
        # Generate 15-min predictions
        min15_preds = generate_15min_predictions()
        
        print("\n" + "="*70)
        print("  ✓ ALL PREDICTIONS COMPLETE")
        print("="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
