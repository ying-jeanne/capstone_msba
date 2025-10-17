"""
Update Actual Outcomes for Blockchain Predictions
==================================================
This script fetches actual Bitcoin prices for past predictions and calculates:
- Actual prices (1d, 3d, 7d after prediction date)
- Errors (predicted - actual)
- MAPE (Mean Absolute Percentage Error)
- Direction correctness (did price go up/down as predicted?)

Usage:
    python3 update_outcomes.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

sys.path.append(str(Path(__file__).parent))

from utils.data_fetcher import get_bitcoin_data


def update_outcomes():
    """Update actual outcomes for all predictions"""
    
    print("\n" + "="*70)
    print("  UPDATE ACTUAL OUTCOMES")
    print("="*70)
    
    # Load prediction tracking CSV
    csv_path = Path('data/blockchain/prediction_tracking_demo.csv')
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 Loaded {len(df)} predictions from CSV")
    
    # Fetch historical Bitcoin prices
    print(f"\n📥 Fetching historical Bitcoin prices...")
    result = get_bitcoin_data(source='yahoo', ticker='BTC-USD', days=400, return_dict=True)
    
    if result['status'] != 'success':
        raise Exception(f"Failed to fetch data: {result.get('message')}")
    
    prices_df = result['data']
    print(f"✓ Fetched {len(prices_df)} days of price data")
    
    # Convert to dict for easy lookup by date
    price_dict = {pd.to_datetime(idx).strftime('%Y-%m-%d'): row['close'] 
                  for idx, row in prices_df.iterrows()}
    
    # Track updates
    updates_count = 0
    
    # Process each prediction
    for idx, row in df.iterrows():
        pred_date = pd.to_datetime(row['date'])
        pred_date_str = pred_date.strftime('%Y-%m-%d')
        
        # Calculate target dates
        date_1d = (pred_date + timedelta(days=1)).strftime('%Y-%m-%d')
        date_3d = (pred_date + timedelta(days=3)).strftime('%Y-%m-%d')
        date_7d = (pred_date + timedelta(days=7)).strftime('%Y-%m-%d')
        
        updated = False
        
        # Update 1-day actual
        if pd.isna(row['actual_1d']) and date_1d in price_dict:
            actual_1d = price_dict[date_1d]
            df.at[idx, 'actual_1d'] = actual_1d
            
            # Calculate error and MAPE
            error_1d = actual_1d - row['pred_1d']
            mape_1d = abs(error_1d) / actual_1d * 100
            
            df.at[idx, 'error_1d'] = error_1d
            df.at[idx, 'mape_1d'] = mape_1d
            
            # Direction correctness
            pred_direction = 1 if row['pred_1d'] > row['current_price'] else 0
            actual_direction = 1 if actual_1d > row['current_price'] else 0
            df.at[idx, 'direction_correct_1d'] = (pred_direction == actual_direction)
            
            updated = True
            print(f"  ✓ {pred_date_str} 1d: ${actual_1d:,.2f} (MAPE: {mape_1d:.2f}%)")
        
        # Update 3-day actual
        if pd.isna(row['actual_3d']) and date_3d in price_dict:
            actual_3d = price_dict[date_3d]
            df.at[idx, 'actual_3d'] = actual_3d
            
            error_3d = actual_3d - row['pred_3d']
            mape_3d = abs(error_3d) / actual_3d * 100
            
            df.at[idx, 'error_3d'] = error_3d
            df.at[idx, 'mape_3d'] = mape_3d
            
            pred_direction = 1 if row['pred_3d'] > row['current_price'] else 0
            actual_direction = 1 if actual_3d > row['current_price'] else 0
            df.at[idx, 'direction_correct_3d'] = (pred_direction == actual_direction)
            
            updated = True
            print(f"  ✓ {pred_date_str} 3d: ${actual_3d:,.2f} (MAPE: {mape_3d:.2f}%)")
        
        # Update 7-day actual
        if pd.isna(row['actual_7d']) and date_7d in price_dict:
            actual_7d = price_dict[date_7d]
            df.at[idx, 'actual_7d'] = actual_7d
            
            error_7d = actual_7d - row['pred_7d']
            mape_7d = abs(error_7d) / actual_7d * 100
            
            df.at[idx, 'error_7d'] = error_7d
            df.at[idx, 'mape_7d'] = mape_7d
            
            pred_direction = 1 if row['pred_7d'] > row['current_price'] else 0
            actual_direction = 1 if actual_7d > row['current_price'] else 0
            df.at[idx, 'direction_correct_7d'] = (pred_direction == actual_direction)
            
            updated = True
            print(f"  ✓ {pred_date_str} 7d: ${actual_7d:,.2f} (MAPE: {mape_7d:.2f}%)")
        
        if updated:
            updates_count += 1
            df.at[idx, 'outcomes_updated_at'] = datetime.now().isoformat()
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    
    # Count pending outcomes
    pending_1d = df['actual_1d'].isna().sum()
    pending_3d = df['actual_3d'].isna().sum()
    pending_7d = df['actual_7d'].isna().sum()
    
    print(f"\n{'='*70}")
    print(f"  ✓ COMPLETE")
    print(f"{'='*70}")
    print(f"  Updated {updates_count} predictions")
    print(f"  Saved to {csv_path}")
    print(f"\n📊 Summary:")
    print(f"  Total predictions: {len(df)}")
    print(f"  Pending 1-day outcomes: {pending_1d}")
    print(f"  Pending 3-day outcomes: {pending_3d}")
    print(f"  Pending 7-day outcomes: {pending_7d}")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = update_outcomes()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
