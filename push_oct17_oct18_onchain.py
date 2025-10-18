"""
Push October 17 and 18 Predictions to Blockchain
================================================
This script pushes the real predictions for Oct 17 and Oct 18 to Moonbase Alpha
"""

import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils.blockchain_integration import store_prediction_onchain


def push_prediction_for_date(date_str, pred_1d, pred_3d, pred_7d, current_price):
    """Push a single prediction to blockchain"""

    print(f"\n{'='*70}")
    print(f"  PUSHING PREDICTION FOR {date_str}")
    print(f"{'='*70}")
    print(f"  Current Price: ${current_price:,.2f}")
    print(f"  Predicted 1d: ${pred_1d:,.2f}")
    print(f"  Predicted 3d: ${pred_3d:,.2f}")
    print(f"  Predicted 7d: ${pred_7d:,.2f}")
    print(f"{'='*70}")

    try:
        result = store_prediction_onchain(
            predicted_1d=float(pred_1d),
            predicted_3d=float(pred_3d),
            predicted_7d=float(pred_7d),
            model_name="xgboost_v1"
        )

        print(f"\n✅ Successfully stored on blockchain!")
        print(f"   TX Hash: {result['tx_hash']}")
        print(f"   Block: {result['block_number']}")
        print(f"   Prediction ID: {result['prediction_id']}")
        print(f"   Gas Used: {result['gas_used']}")
        print(f"   View: https://moonbase.moonscan.io/tx/{result['tx_hash']}")

        return result

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("\n" + "="*70)
    print("  PUSH OCTOBER 17 & 18 PREDICTIONS TO BLOCKCHAIN")
    print("="*70)

    # Load predictions history
    predictions_file = Path(__file__).parent / 'data' / 'predictions' / 'daily_predictions_history.csv'

    if not predictions_file.exists():
        print(f"\n❌ Error: File not found: {predictions_file}")
        sys.exit(1)

    df = pd.read_csv(predictions_file)

    # Get Oct 17 and Oct 18 predictions
    oct17 = df[df['timestamp'] == '2025-10-17']
    oct18 = df[df['timestamp'] == '2025-10-18']

    if len(oct17) == 0:
        print("\n❌ Error: No prediction found for 2025-10-17")
        sys.exit(1)

    if len(oct18) == 0:
        print("\n❌ Error: No prediction found for 2025-10-18")
        sys.exit(1)

    # Extract data
    oct17_row = oct17.iloc[0]
    oct18_row = oct18.iloc[0]

    results = []

    # Push Oct 17
    print("\n\n" + "="*70)
    print("  STEP 1: PUSH OCTOBER 17 PREDICTION")
    print("="*70)
    result_oct17 = push_prediction_for_date(
        date_str='2025-10-17',
        pred_1d=oct17_row['pred_1d_price'],
        pred_3d=oct17_row['pred_3d_price'],
        pred_7d=oct17_row['pred_7d_price'],
        current_price=oct17_row['current_price']
    )

    if result_oct17:
        results.append({
            'date': '2025-10-17',
            'prediction_id': result_oct17['prediction_id'],
            'tx_hash': result_oct17['tx_hash'],
            'block_number': result_oct17['block_number'],
            'blockchain_timestamp': result_oct17['timestamp'],
            'current_price': oct17_row['current_price'],
            'pred_1d': oct17_row['pred_1d_price'],
            'pred_3d': oct17_row['pred_3d_price'],
            'pred_7d': oct17_row['pred_7d_price'],
            'gas_used': result_oct17['gas_used']
        })

    # Wait a bit before next transaction
    print("\n⏳ Waiting 5 seconds before next transaction...")
    import time
    time.sleep(5)

    # Push Oct 18
    print("\n\n" + "="*70)
    print("  STEP 2: PUSH OCTOBER 18 PREDICTION")
    print("="*70)
    result_oct18 = push_prediction_for_date(
        date_str='2025-10-18',
        pred_1d=oct18_row['pred_1d_price'],
        pred_3d=oct18_row['pred_3d_price'],
        pred_7d=oct18_row['pred_7d_price'],
        current_price=oct18_row['current_price']
    )

    if result_oct18:
        results.append({
            'date': '2025-10-18',
            'prediction_id': result_oct18['prediction_id'],
            'tx_hash': result_oct18['tx_hash'],
            'block_number': result_oct18['block_number'],
            'blockchain_timestamp': result_oct18['timestamp'],
            'current_price': oct18_row['current_price'],
            'pred_1d': oct18_row['pred_1d_price'],
            'pred_3d': oct18_row['pred_3d_price'],
            'pred_7d': oct18_row['pred_7d_price'],
            'gas_used': result_oct18['gas_used']
        })

    # Save to tracking files
    if len(results) > 0:
        print("\n\n" + "="*70)
        print("  STEP 3: UPDATE TRACKING FILES")
        print("="*70)

        tracking_dir = Path(__file__).parent / 'data' / 'blockchain'
        tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_file = tracking_dir / 'prediction_tracking.csv'

        # Load existing tracking data
        if tracking_file.exists():
            tracking_df = pd.read_csv(tracking_file)
        else:
            tracking_df = pd.DataFrame()

        # Append new results
        for result in results:
            # Check if already exists
            if len(tracking_df) > 0 and result['date'] in tracking_df['date'].values:
                print(f"   ⚠️  {result['date']} already in tracking file, skipping")
            else:
                tracking_df = pd.concat([tracking_df, pd.DataFrame([result])], ignore_index=True)
                print(f"   ✅ Added {result['date']} to tracking file")

        # Save
        tracking_df.to_csv(tracking_file, index=False)
        print(f"\n   📁 Saved to: {tracking_file}")

    # Final summary
    print("\n\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    print(f"\n✅ Successfully pushed {len(results)} predictions to blockchain!")

    for r in results:
        print(f"\n📅 {r['date']}:")
        print(f"   TX: https://moonbase.moonscan.io/tx/{r['tx_hash']}")
        print(f"   Prediction ID: {r['prediction_id']}")
        print(f"   Block: {r['block_number']}")

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
