"""
Decode Bitcoin Prediction Smart Contract Transaction Data
==========================================================
This script uses Web3.py with ABI to automatically decode transaction data
from the blockchain.

Usage:
    1. Edit the TRANSACTION_HASH variable below
    2. Run: python3 decode_simple.py

Author: Bitcoin Price Prediction System
Date: October 17, 2025
"""

import sys
import os
from web3 import Web3

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our blockchain integration module which has the ABI
from utils.blockchain_integration import get_contract

# ============================================================================
# 📝 MANUALLY SET YOUR TRANSACTION HASH HERE:
# ============================================================================
TRANSACTION_HASH = '0x6a04d2c82924ae4cf55bd10b7e5a5256746d9cf9bea15ba777635425cb80c999'
# ============================================================================


def decode_transaction(tx_hash: str):
    """
    Decode a transaction using Web3.py and contract ABI
    
    Args:
        tx_hash: Transaction hash (with or without 0x prefix)
        
    Returns:
        dict: Decoded transaction data
    """
    
    # Ensure tx_hash has 0x prefix
    if not tx_hash.startswith('0x'):
        tx_hash = '0x' + tx_hash
    
    # Connect to blockchain
    w3 = Web3(Web3.HTTPProvider('https://rpc.api.moonbase.moonbeam.network'))
    
    print("="*70)
    print("  DECODE TRANSACTION WITH WEB3.PY + ABI")
    print("="*70)
    print(f"\n🔗 Transaction: {tx_hash}")
    
    # Get contract with ABI
    contract = get_contract()
    
    try:
        # Get transaction from blockchain
        tx = w3.eth.get_transaction(tx_hash)
        
        print(f"✅ Transaction found!")
        print(f"   Block: {tx['blockNumber']}")
        print(f"   From: {tx['from']}")
        print(f"   To: {tx['to']}")
        print(f"   Gas Used: {tx['gas']:,}")
        
        # Get block timestamp
        block = w3.eth.get_block(tx['blockNumber'])
        block_timestamp = block['timestamp']
        
        # Convert timestamp to readable date
        from datetime import datetime
        block_date = datetime.fromtimestamp(block_timestamp)
        
        print(f"   Timestamp: {block_date.strftime('%Y-%m-%d %H:%M:%S')} (Block time)")
        
        # Decode the function input using ABI
        func, params = contract.decode_function_input(tx['input'])
        
        print(f"\n📋 Function Called: {func.fn_name}")
        print(f"\n📦 Decoded Parameters:")
        
        # Check if this is a storePrediction call
        if func.fn_name == 'storePrediction':
            # Extract parameters with their names (note: ABI uses underscore prefix)
            pred_1d_cents = params['_predictedPrice1d']
            pred_3d_cents = params['_predictedPrice3d']
            pred_7d_cents = params['_predictedPrice7d']
            model_hash = params['_modelHash']
            
            # Convert from cents to dollars
            pred_1d = pred_1d_cents / 100
            pred_3d = pred_3d_cents / 100
            pred_7d = pred_7d_cents / 100
            
            print(f"   _predictedPrice1d: {pred_1d_cents:,} cents = ${pred_1d:,.2f}")
            print(f"   _predictedPrice3d: {pred_3d_cents:,} cents = ${pred_3d:,.2f}")
            print(f"   _predictedPrice7d: {pred_7d_cents:,} cents = ${pred_7d:,.2f}")
            print(f"   _modelHash: {model_hash.hex()}")
            
            print(f"\n💰 Bitcoin Price Predictions:")
            print(f"   1-day:  ${pred_1d:,.2f}")
            print(f"   3-day:  ${pred_3d:,.2f}")
            print(f"   7-day:  ${pred_7d:,.2f}")
            
            # Return structured data
            return {
                'tx_hash': tx_hash,
                'block_number': tx['blockNumber'],
                'function': func.fn_name,
                'from': tx['from'],
                'to': tx['to'],
                'gas_used': tx['gas'],
                'parameters': {
                    'predicted_1d': pred_1d,
                    'predicted_3d': pred_3d,
                    'predicted_7d': pred_7d,
                    'predicted_1d_cents': pred_1d_cents,
                    'predicted_3d_cents': pred_3d_cents,
                    'predicted_7d_cents': pred_7d_cents,
                    'model_hash': model_hash.hex()
                }
            }
        else:
            # Other functions
            print(f"   Function: {func.fn_name}")
            for key, value in params.items():
                print(f"   {key}: {value}")
            
            return {
                'tx_hash': tx_hash,
                'block_number': tx['blockNumber'],
                'function': func.fn_name,
                'from': tx['from'],
                'to': tx['to'],
                'gas_used': tx['gas'],
                'parameters': params
            }
            
    except Exception as e:
        print(f"\n❌ Error decoding transaction: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        result = decode_transaction(TRANSACTION_HASH)
        print(f"\n{'='*70}")
        print("✅ Decoding complete!")
        print(f"{'='*70}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
