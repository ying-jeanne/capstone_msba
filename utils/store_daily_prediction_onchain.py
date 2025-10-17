"""
Store Daily Bitcoin Predictions On-Chain
=========================================
Reads latest predictions from daily_predictions.csv and stores them on Moonbase Alpha blockchain

This script is designed to run via GitHub Actions daily at 6:30 PM UTC
(30 minutes after predict_daily.py runs at 6:00 PM)

Author: Bitcoin Price Prediction System
Date: October 2025
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.blockchain_integration import store_prediction_onchain


def main():
    """
    Main function to store daily prediction on blockchain

    Steps:
    1. Load latest prediction from daily_predictions.csv
    2. Extract predicted prices (1d, 3d, 7d)
    3. Store on Moonbase Alpha blockchain
    4. Update CSV with blockchain transaction info
    5. Save tracking file
    """

    print("\n" + "="*70)
    print("  STORE DAILY PREDICTION ON BLOCKCHAIN")
    print("="*70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    # ========================================================================
    # STEP 1: Load Latest Prediction
    # ========================================================================

    predictions_file = Path(__file__).parent.parent / 'data' / 'predictions' / 'daily_predictions.csv'

    if not predictions_file.exists():
        print(f"\n❌ Error: Predictions file not found: {predictions_file}")
        print("   Please run predict_daily.py first to generate predictions.")
        sys.exit(1)

    print(f"\n📂 Loading predictions from: {predictions_file}")
    df = pd.read_csv(predictions_file)

    if len(df) == 0:
        print(f"\n❌ Error: No predictions found in file")
        sys.exit(1)

    # Get the latest prediction (last row)
    latest = df.iloc[-1]

    print(f"\n📊 Latest Prediction:")
    print(f"   Timestamp: {latest['timestamp']}")
    print(f"   Current Price: ${latest['current_price']:,.2f}")
    print(f"   Predicted 1d: ${latest['pred_1d_price']:,.2f}")
    print(f"   Predicted 3d: ${latest['pred_3d_price']:,.2f}")
    print(f"   Predicted 7d: ${latest['pred_7d_price']:,.2f}")

    # ========================================================================
    # STEP 2: Check if Already Stored
    # ========================================================================

    # Check if this prediction already has blockchain data
    if 'tx_hash' in df.columns and pd.notna(latest.get('tx_hash')) and latest.get('tx_hash') != 'N/A':
        print(f"\n⚠️  Warning: This prediction already has blockchain data:")
        print(f"   TX Hash: {latest['tx_hash']}")
        print(f"   Skipping storage to avoid duplicate.")
        return

    # ========================================================================
    # STEP 3: Store on Blockchain
    # ========================================================================

    try:
        result = store_prediction_onchain(
            predicted_1d=float(latest['pred_1d_price']),
            predicted_3d=float(latest['pred_3d_price']),
            predicted_7d=float(latest['pred_7d_price']),
            model_name="xgboost_v1"
        )

        print(f"\n✅ Successfully stored on blockchain!")

    except Exception as e:
        print(f"\n❌ Error storing on blockchain: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================================================
    # STEP 4: Update CSV with Blockchain Data
    # ========================================================================

    print(f"\n📝 Updating CSV with blockchain data...")

    # Add blockchain columns if they don't exist
    if 'tx_hash' not in df.columns:
        df['tx_hash'] = 'N/A'
    if 'block_number' not in df.columns:
        df['block_number'] = 0
    if 'prediction_id' not in df.columns:
        df['prediction_id'] = -1
    if 'blockchain_stored' not in df.columns:
        df['blockchain_stored'] = False
    if 'blockchain_timestamp' not in df.columns:
        df['blockchain_timestamp'] = ''

    # Update the latest row
    df.loc[df.index[-1], 'tx_hash'] = result['tx_hash']
    df.loc[df.index[-1], 'block_number'] = result['block_number']
    df.loc[df.index[-1], 'prediction_id'] = result['prediction_id']
    df.loc[df.index[-1], 'blockchain_stored'] = True
    df.loc[df.index[-1], 'blockchain_timestamp'] = result['timestamp']

    # Save updated CSV
    df.to_csv(predictions_file, index=False)
    print(f"   ✅ Updated: {predictions_file}")

    # ========================================================================
    # STEP 5: Save to Tracking File
    # ========================================================================

    tracking_dir = Path(__file__).parent.parent / 'data' / 'blockchain'
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_file = tracking_dir / 'prediction_tracking.csv'

    print(f"\n📋 Saving to tracking file...")

    # Create tracking record
    tracking_record = {
        'date': latest['timestamp'],
        'prediction_id': result['prediction_id'],
        'tx_hash': result['tx_hash'],
        'block_number': result['block_number'],
        'blockchain_timestamp': result['timestamp'],
        'current_price': latest['current_price'],
        'pred_1d': latest['pred_1d_price'],
        'pred_3d': latest['pred_3d_price'],
        'pred_7d': latest['pred_7d_price'],
        'gas_used': result['gas_used']
    }

    # Append to tracking file or create if doesn't exist
    if tracking_file.exists():
        tracking_df = pd.read_csv(tracking_file)

        # Check for duplicates (same date)
        if latest['timestamp'] in tracking_df['date'].values:
            print(f"   ⚠️  Warning: Prediction for {latest['timestamp']} already in tracking file")
        else:
            tracking_df = pd.concat([tracking_df, pd.DataFrame([tracking_record])], ignore_index=True)
            tracking_df.to_csv(tracking_file, index=False)
            print(f"   ✅ Appended to: {tracking_file}")
    else:
        tracking_df = pd.DataFrame([tracking_record])
        tracking_df.to_csv(tracking_file, index=False)
        print(f"   ✅ Created: {tracking_file}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"\n✅ Prediction stored successfully on Moonbase Alpha!")
    print(f"\n🔗 Blockchain Details:")
    print(f"   Transaction Hash: {result['tx_hash']}")
    print(f"   Block Number: {result['block_number']}")
    print(f"   Prediction ID: {result['prediction_id']}")
    print(f"   Timestamp: {result['timestamp']}")
    print(f"   Gas Used: {result['gas_used']}")
    print(f"\n🌐 View on Moonscan:")
    print(f"   https://moonbase.moonscan.io/tx/{result['tx_hash']}")
    print(f"\n📁 Files Updated:")
    print(f"   - {predictions_file}")
    print(f"   - {tracking_file}")
    print("\n" + "="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
