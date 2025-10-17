"""
Backfill Historical Predictions to Blockchain
==============================================
Pushes the 30 demo predictions to the blockchain to generate real transaction hashes

This script will:
1. Read the demo predictions CSV
2. Push each prediction to Moonbase Alpha blockchain
3. Update the CSV with real transaction hashes
4. Generate real Moonscan links

Author: Bitcoin Price Prediction System  
Date: October 17, 2025
"""

import pandas as pd
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.blockchain_integration import store_prediction_onchain


def main():
    """
    Backfill historical predictions to blockchain
    """
    
    print("\n" + "="*70)
    print("  BACKFILL PREDICTIONS TO BLOCKCHAIN")
    print("="*70)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  Network: Moonbase Alpha Testnet")
    print("="*70)
    
    # Load demo predictions
    demo_file = Path(__file__).parent.parent / 'data' / 'blockchain' / 'prediction_tracking_demo.csv'
    
    if not demo_file.exists():
        print(f"\n❌ Error: Demo file not found: {demo_file}")
        sys.exit(1)
    
    print(f"\n📂 Loading demo predictions from: {demo_file.name}")
    df = pd.read_csv(demo_file)
    
    print(f"   Total predictions to push: {len(df)}")
    
    # Ask for confirmation
    print(f"\n⚠️  WARNING: This will push {len(df)} transactions to the blockchain.")
    print(f"   Each transaction will cost gas fees (testnet DEV tokens).")
    print(f"   Estimated time: ~{len(df) * 15} seconds ({len(df)} tx × 15s)")
    
    response = input(f"\nContinue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("\n❌ Cancelled by user")
        return
    
    # ========================================================================
    # Push Each Prediction to Blockchain
    # ========================================================================
    
    print(f"\n🚀 Starting blockchain storage...\n")
    
    success_count = 0
    failed_count = 0
    real_hashes = []
    
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] Processing {row['date']}...")
        
        try:
            # Store on blockchain
            result = store_prediction_onchain(
                predicted_1d=float(row['pred_1d']),
                predicted_3d=float(row['pred_3d']),
                predicted_7d=float(row['pred_7d']),
                model_name="xgboost_v1"
            )
            
            # Update dataframe with real blockchain data
            df.at[idx, 'tx_hash'] = result['tx_hash']
            df.at[idx, 'block_number'] = result['block_number']
            df.at[idx, 'prediction_id'] = result['prediction_id']
            df.at[idx, 'blockchain_timestamp'] = result['timestamp']
            df.at[idx, 'gas_used'] = result['gas_used']
            
            print(f"   ✅ Success! TX: {result['tx_hash'][:16]}...")
            print(f"      Block: {result['block_number']}, Gas: {result['gas_used']}")
            
            success_count += 1
            real_hashes.append(result['tx_hash'])
            
            # Wait a bit between transactions to avoid rate limiting
            if idx < len(df) - 1:  # Don't wait after last one
                print(f"   ⏳ Waiting 10 seconds before next transaction...")
                time.sleep(10)
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            failed_count += 1
            
            # Keep the original mock hash
            real_hashes.append(row['tx_hash'])
            
            # Continue with next prediction
            continue
    
    # ========================================================================
    # Save Updated CSV
    # ========================================================================
    
    print(f"\n💾 Saving updated predictions...")
    df.to_csv(demo_file, index=False)
    print(f"   ✅ Saved: {demo_file}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    print(f"\n✅ Successfully pushed: {success_count} predictions")
    print(f"❌ Failed: {failed_count} predictions")
    print(f"\n📊 Results:")
    print(f"   Total predictions: {len(df)}")
    print(f"   Real blockchain TXs: {success_count}")
    print(f"   Updated file: {demo_file.name}")
    
    if success_count > 0:
        print(f"\n🔗 View on Moonscan:")
        print(f"   First TX: https://moonbase.moonscan.io/tx/{real_hashes[0]}")
        if len(real_hashes) > 1:
            print(f"   Last TX:  https://moonbase.moonscan.io/tx/{real_hashes[-1]}")
    
    print(f"\n🌐 Refresh your webapp to see real blockchain links!")
    print(f"   http://localhost:5002/live")
    
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
