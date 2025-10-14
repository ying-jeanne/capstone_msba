"""
Generate Daily Bitcoin Price Predictions
=========================================
Loads pre-trained daily models and generates predictions for 1d, 3d, 7d

This script is designed to run quickly (~20 seconds) as part of GitHub Actions
Uses pre-trained models - NO TRAINING HAPPENS HERE

Output: data/predictions/daily_predictions.csv
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib
import sys
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetcher import get_bitcoin_data
from utils.feature_engineering import engineer_technical_features


def load_models(models_dir):
    """Load all pre-trained daily models"""
    models = {}
    horizons = ['1d', '3d', '7d']
    
    for horizon in horizons:
        model_path = models_dir / f'xgboost_{horizon}.json'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        models[horizon] = model
    
    # Load scaler and feature columns
    scaler = joblib.load(models_dir / 'scaler_daily.pkl')
    feature_cols = joblib.load(models_dir / 'feature_cols_daily.pkl')
    
    return models, scaler, feature_cols


def prepare_latest_data(df, feature_cols, scaler):
    """Prepare the latest data point for prediction"""
    # Get features for the latest timestamp
    latest_features = df[feature_cols].iloc[-1:].values
    
    # Scale features
    latest_features_scaled = scaler.transform(latest_features)
    
    # Get current price
    current_price = df['close'].iloc[-1]
    current_timestamp = df.index[-1]
    
    return latest_features_scaled, current_price, current_timestamp


def generate_predictions(models, latest_features_scaled, current_price):
    """Generate predictions for all horizons"""
    predictions = {}
    
    for horizon, model in models.items():
        # Predict return
        predicted_return = model.predict(latest_features_scaled)[0]
        
        # Convert return to price
        predicted_price = current_price * (1 + predicted_return)
        
        predictions[horizon] = {
            'predicted_return': predicted_return,
            'predicted_price': predicted_price,
            'change_percent': predicted_return * 100
        }
    
    return predictions


def save_predictions(predictions, current_price, current_timestamp, output_path):
    """Save predictions to CSV"""
    # Create prediction record
    record = {
        'timestamp': current_timestamp,
        'current_price': current_price,
        'pred_1d_price': predictions['1d']['predicted_price'],
        'pred_1d_return': predictions['1d']['predicted_return'],
        'pred_1d_change_pct': predictions['1d']['change_percent'],
        'pred_3d_price': predictions['3d']['predicted_price'],
        'pred_3d_return': predictions['3d']['predicted_return'],
        'pred_3d_change_pct': predictions['3d']['change_percent'],
        'pred_7d_price': predictions['7d']['predicted_price'],
        'pred_7d_return': predictions['7d']['predicted_return'],
        'pred_7d_change_pct': predictions['7d']['change_percent'],
        'data_source': 'yahoo',
        'generated_at': datetime.now().isoformat()
    }
    
    # Save to CSV
    df = pd.DataFrame([record])
    df.to_csv(output_path, index=False)
    
    # Also append to history file (avoid duplicates)
    history_path = output_path.parent / 'daily_predictions_history.csv'
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


def main():
    """Main prediction pipeline"""
    print("\n" + "="*70)
    print("  DAILY PREDICTIONS GENERATOR")
    print("="*70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Paths
    models_dir = Path('models/saved_models/daily')
    output_dir = Path('data/predictions')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'daily_predictions.csv'
    
    try:
        # Step 1: Load models
        print("\n[STEP 1] Loading pre-trained models...")
        models, scaler, feature_cols = load_models(models_dir)
        print(f"✓ Loaded 3 models (1d, 3d, 7d)")
        print(f"✓ Loaded scaler and {len(feature_cols)} features")
        
        # Step 2: Fetch latest data
        print("\n[STEP 2] Fetching latest Yahoo Finance data...")
        result = get_bitcoin_data(source='yahoo', ticker='BTC-USD', days=200, return_dict=True)
        
        if result['status'] != 'success':
            raise Exception(f"Failed to fetch data: {result.get('message')}")
        
        df = result['data']
        print(f"✓ Fetched {len(df)} days of data")
        print(f"  Latest timestamp: {df.index[-1]}")
        print(f"  Current price: ${df['close'].iloc[-1]:,.2f}")
        
        # Step 3: Engineer features
        print("\n[STEP 3] Engineering features...")
        df = engineer_technical_features(df)
        print(f"✓ Created features")
        
        # Step 4: Prepare latest data point
        print("\n[STEP 4] Preparing latest data point...")
        latest_features_scaled, current_price, current_timestamp = prepare_latest_data(
            df, feature_cols, scaler
        )
        print(f"✓ Prepared features for {current_timestamp}")
        
        # Step 5: Generate predictions
        print("\n[STEP 5] Generating predictions...")
        predictions = generate_predictions(models, latest_features_scaled, current_price)
        
        print(f"\n{'='*70}")
        print("  PREDICTIONS")
        print(f"{'='*70}")
        print(f"Current Price: ${current_price:,.2f}")
        print(f"\n1-Day Prediction:")
        print(f"  Price: ${predictions['1d']['predicted_price']:,.2f}")
        print(f"  Change: {predictions['1d']['change_percent']:+.2f}%")
        print(f"\n3-Day Prediction:")
        print(f"  Price: ${predictions['3d']['predicted_price']:,.2f}")
        print(f"  Change: {predictions['3d']['change_percent']:+.2f}%")
        print(f"\n7-Day Prediction:")
        print(f"  Price: ${predictions['7d']['predicted_price']:,.2f}")
        print(f"  Change: {predictions['7d']['change_percent']:+.2f}%")
        
        # Step 6: Save predictions
        print(f"\n[STEP 6] Saving predictions...")
        record = save_predictions(predictions, current_price, current_timestamp, output_path)
        print(f"✓ Saved to {output_path}")
        print(f"✓ Appended to history")
        
        print("\n" + "="*70)
        print("  ✓ DAILY PREDICTIONS COMPLETE")
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
