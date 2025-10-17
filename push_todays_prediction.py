"""
Generate Today's Prediction and Push to Blockchain
===================================================
This script:
1. Generates predictions for today using predict_daily.py
2. Pushes the predictions to Moonbase Alpha blockchain
3. Adds them to the prediction tracking CSV

Usage:
    python3 push_todays_prediction.py
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from utils.predict_daily import main as generate_predictions
from utils.blockchain_integration import store_prediction_onchain


def main():
    print("\n" + "="*70)
    print("  GENERATE TODAY'S PREDICTION AND PUSH TO BLOCKCHAIN")
    print("="*70)
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d')}")
    print("="*70)
    
    try:
        # Step 1: Generate predictions
        print("\n[STEP 1] Generating predictions for today...")
        result = generate_predictions()
        if result != 0:
            raise Exception("Failed to generate predictions")
        
        # Step 2: Read the generated predictions
        print("\n[STEP 2] Reading generated predictions...")
        predictions_path = Path('data/predictions/daily_predictions.csv')
        if not predictions_path.exists():
            raise Exception(f"Predictions file not found: {predictions_path}")
        
        df = pd.read_csv(predictions_path)
        latest = df.iloc[-1]
        
        pred_1d = latest['pred_1d_price']
        pred_3d = latest['pred_3d_price']
        pred_7d = latest['pred_7d_price']
        current_price = latest['current_price']
        timestamp = latest['timestamp']
        
        print(f"✓ Loaded predictions:")
        print(f"   Timestamp: {timestamp}")
        print(f"   Current Price: ${current_price:,.2f}")
        print(f"   1-day: ${pred_1d:,.2f}")
        print(f"   3-day: ${pred_3d:,.2f}")
        print(f"   7-day: ${pred_7d:,.2f}")
        
        # Step 3: Push to blockchain
        print("\n[STEP 3] Pushing to blockchain...")
        blockchain_result = store_prediction_onchain(
            predicted_1d=pred_1d,
            predicted_3d=pred_3d,
            predicted_7d=pred_7d,
            model_name="xgboost_v1"
        )
        
        print(f"✓ Successfully stored on blockchain!")
        print(f"   TX Hash: {blockchain_result['tx_hash']}")
        print(f"   Block: {blockchain_result['block_number']}")
        print(f"   Prediction ID: {blockchain_result['prediction_id']}")
        print(f"   Gas Used: {blockchain_result['gas_used']:,}")
        
        # Step 4: Add to tracking CSV
        print("\n[STEP 4] Adding to prediction tracking CSV...")
        csv_path = Path('data/blockchain/prediction_tracking_demo.csv')
        
        # Read existing CSV
        tracking_df = pd.read_csv(csv_path)
        next_id = tracking_df['prediction_id'].max() + 1
        
        # Create new row
        new_row = {
            'date': pd.to_datetime(timestamp).strftime('%Y-%m-%d'),
            'prediction_id': next_id,
            'tx_hash': blockchain_result['tx_hash'],
            'block_number': blockchain_result['block_number'],
            'blockchain_timestamp': blockchain_result['timestamp'],
            'current_price': current_price,
            'pred_1d': pred_1d,
            'pred_3d': pred_3d,
            'pred_7d': pred_7d,
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
        
        # Append to DataFrame
        tracking_df = pd.concat([tracking_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save back to CSV
        tracking_df.to_csv(csv_path, index=False)
        print(f"✓ Added prediction #{next_id} to {csv_path}")
        
        # Summary
        print("\n" + "="*70)
        print("  ✓ SUCCESS!")
        print("="*70)
        print(f"\nPrediction Details:")
        print(f"  Date: {new_row['date']}")
        print(f"  ID: {next_id}")
        print(f"  Current: ${current_price:,.2f}")
        print(f"  Predicted 1d: ${pred_1d:,.2f}")
        print(f"  Predicted 3d: ${pred_3d:,.2f}")
        print(f"  Predicted 7d: ${pred_7d:,.2f}")
        print(f"\nBlockchain:")
        print(f"  TX Hash: {blockchain_result['tx_hash']}")
        print(f"  Block: {blockchain_result['block_number']}")
        print(f"  View on Moonscan:")
        print(f"    https://moonbase.moonscan.io/tx/{blockchain_result['tx_hash']}")
        print("\n" + "="*70 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
