"""
Generate Prediction for a Specific Historical Date
===================================================
This script generates predictions for a specific past date using historical data.
This is useful for filling gaps in the prediction history.

Usage:
    python3 generate_historical_prediction.py YYYY-MM-DD
    
Example:
    python3 generate_historical_prediction.py 2025-10-16
"""

import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import joblib
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

from utils.data_fetcher import get_bitcoin_data
from utils.feature_engineering import engineer_technical_features, add_sentiment_features
from utils.blockchain_integration import store_prediction_onchain


def generate_prediction_for_date(target_date_str: str):
    """Generate prediction for a specific historical date"""
    
    print("\n" + "="*70)
    print("  GENERATE HISTORICAL PREDICTION")
    print("="*70)
    print(f"  Target Date: {target_date_str}")
    print("="*70)
    
    try:
        target_date = pd.to_datetime(target_date_str)
        
        # Check if date is in the future
        if target_date > pd.Timestamp.now():
            raise ValueError(f"Cannot generate predictions for future date: {target_date_str}")
        
        print(f"\n[STEP 1] Loading pre-trained models...")
        models_dir = Path('models/saved_models/daily')
        
        # Load models
        models = {}
        horizons = ['1d', '3d', '7d']
        
        for horizon in horizons:
            model_path = models_dir / f'xgboost_{horizon}.json'
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            models[horizon] = model
        
        scaler = joblib.load(models_dir / 'scaler_daily.pkl')
        feature_cols = joblib.load(models_dir / 'feature_cols_daily.pkl')
        print(f"✓ Loaded 3 models (1d, 3d, 7d)")
        
        print(f"\n[STEP 2] Fetching historical data up to {target_date_str}...")
        # Fetch enough historical data (we need ~200 days for features)
        result = get_bitcoin_data(source='yahoo', ticker='BTC-USD', days=400, return_dict=True)
        
        if result['status'] != 'success':
            raise Exception(f"Failed to fetch data: {result.get('message')}")
        
        df = result['data']
        print(f"✓ Fetched {len(df)} days of data")
        
        # Filter data up to target date (simulate "we're on that date")
        df = df[df.index <= target_date]
        
        if len(df) == 0:
            raise ValueError(f"No data available for {target_date_str}")
        
        print(f"✓ Filtered to {len(df)} days up to {target_date_str}")
        print(f"  Latest timestamp: {df.index[-1]}")
        print(f"  Price on {target_date_str}: ${df['close'].iloc[-1]:,.2f}")
        
        print(f"\n[STEP 3] Engineering features...")
        df = engineer_technical_features(df)
        print(f"✓ Created technical features")
        
        print(f"\n[STEP 4] Adding sentiment features...")
        df = add_sentiment_features(df)
        print(f"✓ Added sentiment features")
        
        print(f"\n[STEP 5] Preparing data point for {target_date_str}...")
        latest_features = df[feature_cols].iloc[-1:].values
        latest_features_scaled = scaler.transform(latest_features)
        current_price = df['close'].iloc[-1]
        current_timestamp = df.index[-1]
        print(f"✓ Prepared features")
        
        print(f"\n[STEP 6] Generating predictions...")
        predictions = {}
        
        for horizon, model in models.items():
            predicted_return = model.predict(latest_features_scaled)[0]
            predicted_price = current_price * (1 + predicted_return)
            
            predictions[horizon] = {
                'predicted_return': predicted_return,
                'predicted_price': predicted_price,
                'change_percent': predicted_return * 100
            }
        
        print(f"\n{'='*70}")
        print(f"  PREDICTIONS FOR {target_date_str}")
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
        
        # Ask if user wants to push to blockchain
        print(f"\n{'='*70}")
        response = input("Push this prediction to blockchain? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            print(f"\n[STEP 7] Pushing to blockchain...")
            blockchain_result = store_prediction_onchain(
                predicted_1d=predictions['1d']['predicted_price'],
                predicted_3d=predictions['3d']['predicted_price'],
                predicted_7d=predictions['7d']['predicted_price'],
                model_name="xgboost_v1"
            )
            
            print(f"✓ Successfully stored on blockchain!")
            print(f"   TX Hash: {blockchain_result['tx_hash']}")
            print(f"   Block: {blockchain_result['block_number']}")
            
            # Add to CSV
            print(f"\n[STEP 8] Adding to prediction tracking CSV...")
            csv_path = Path('data/blockchain/prediction_tracking_demo.csv')
            tracking_df = pd.read_csv(csv_path)
            next_id = tracking_df['prediction_id'].max() + 1
            
            new_row = {
                'date': target_date.strftime('%Y-%m-%d'),
                'prediction_id': next_id,
                'tx_hash': blockchain_result['tx_hash'],
                'block_number': blockchain_result['block_number'],
                'blockchain_timestamp': blockchain_result['timestamp'],
                'current_price': current_price,
                'pred_1d': predictions['1d']['predicted_price'],
                'pred_3d': predictions['3d']['predicted_price'],
                'pred_7d': predictions['7d']['predicted_price'],
                'actual_1d': None,
                'actual_3d': None,
                'actual_7d': None,
                'error_1d': None,
                'error_3d': None,
                'error_7d': None,
                'mape_1d': None,
                'mape_3d': None,
                'mape_7d': None,
                'direction_correct_1d': None,
                'direction_correct_3d': None,
                'direction_correct_7d': None,
                'gas_used': blockchain_result['gas_used'],
                'outcomes_updated_at': datetime.now().isoformat()
            }
            
            tracking_df = pd.concat([tracking_df, pd.DataFrame([new_row])], ignore_index=True)
            tracking_df = tracking_df.sort_values('date').reset_index(drop=True)
            tracking_df.to_csv(csv_path, index=False)
            
            print(f"✓ Added prediction #{next_id} to CSV (sorted by date)")
            print(f"\n{'='*70}")
            print(f"  ✓ SUCCESS!")
            print(f"{'='*70}")
            print(f"\nView on Moonscan:")
            print(f"  https://moonbase.moonscan.io/tx/{blockchain_result['tx_hash']}")
        else:
            print("\n⚠️  Prediction generated but NOT pushed to blockchain")
            print("   Run again with 'yes' to push it")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 generate_historical_prediction.py YYYY-MM-DD")
        print("Example: python3 generate_historical_prediction.py 2025-10-16")
        sys.exit(1)
    
    target_date = sys.argv[1]
    exit_code = generate_prediction_for_date(target_date)
    sys.exit(exit_code)
