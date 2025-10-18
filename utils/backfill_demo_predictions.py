"""
Backfill Missing Predictions to Demo File
==========================================
Adds Oct 17 and Oct 18 predictions from daily_predictions_history.csv
to prediction_tracking_demo.csv with mock blockchain data

This is a one-time script to sync historical predictions.
Going forward, store_daily_prediction_onchain.py will handle both files.

Author: Bitcoin Price Prediction System
Date: October 2025
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

def backfill_predictions():
    """Backfill missing predictions to demo file"""

    print("\n" + "="*70)
    print("  BACKFILL MISSING PREDICTIONS TO DEMO FILE")
    print("="*70)

    # Load daily predictions history
    history_file = Path('data/predictions/daily_predictions_history.csv')
    demo_file = Path('data/blockchain/prediction_tracking_demo.csv')

    if not history_file.exists():
        print(f"\n❌ Error: {history_file} not found")
        return 1

    print(f"\n📂 Loading predictions from: {history_file}")
    history_df = pd.read_csv(history_file)

    print(f"📂 Loading demo file from: {demo_file}")
    demo_df = pd.read_csv(demo_file)

    # Find predictions that are in history but not in demo
    missing_dates = set(history_df['timestamp']) - set(demo_df['date'])

    if not missing_dates:
        print(f"\n✅ No missing predictions found. Demo file is up to date!")
        return 0

    print(f"\n📊 Found {len(missing_dates)} missing predictions:")
    for date in sorted(missing_dates):
        print(f"   - {date}")

    # Get the next prediction_id
    next_id = demo_df['prediction_id'].max() + 1 if len(demo_df) > 0 else 1

    # Add missing predictions
    added_count = 0
    for date in sorted(missing_dates):
        # Get prediction from history
        pred_row = history_df[history_df['timestamp'] == date].iloc[0]

        # Create demo record with mock blockchain data
        demo_record = {
            'date': date,
            'prediction_id': next_id,
            'tx_hash': f'0x{next_id:064x}',  # Mock transaction hash
            'block_number': 14028000 + next_id,  # Mock block number
            'blockchain_timestamp': datetime.now().isoformat(),
            'current_price': pred_row['current_price'],
            'pred_1d': pred_row['pred_1d_price'],
            'pred_3d': pred_row['pred_3d_price'],
            'pred_7d': pred_row['pred_7d_price'],
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
            'gas_used': 212280,  # Mock gas used
            'outcomes_updated_at': None
        }

        # Append to demo_df
        demo_df = pd.concat([demo_df, pd.DataFrame([demo_record])], ignore_index=True)
        next_id += 1
        added_count += 1

        print(f"   ✅ Added {date}: ${pred_row['current_price']:,.2f}")

    # Save updated demo file
    demo_df.to_csv(demo_file, index=False)

    print(f"\n{'='*70}")
    print(f"  ✓ COMPLETE")
    print(f"{'='*70}")
    print(f"  Added {added_count} predictions")
    print(f"  Saved to {demo_file}")
    print(f"{'='*70}\n")

    return 0


if __name__ == '__main__':
    try:
        exit_code = backfill_predictions()
        exit(exit_code)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
