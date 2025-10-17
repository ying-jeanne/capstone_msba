"""
Blockchain Integration for Bitcoin Price Predictions
====================================================
Interfaces with the BitcoinPredictionSimplified smart contract on Moonbase Alpha

SIMPLIFIED VERSION - Only stores predictions (not current/actual prices)

Functions:
- store_prediction_onchain() - Store daily predictions (1d, 3d, 7d)
- get_recent_predictions() - Fetch predictions from blockchain
- get_contract_stats() - Get contract statistics

Author: Bitcoin Price Prediction System
Date: October 2025
"""

import os
from web3 import Web3
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Moonbase Alpha Configuration
MOONBASE_RPC_URL = os.getenv('MOONBASE_RPC_URL', 'https://rpc.api.moonbase.moonbeam.network')
CONTRACT_ADDRESS = os.getenv('CONTRACT_ADDRESS', None)
PRIVATE_KEY = os.getenv('MOONBASE_PRIVATE_KEY', None)

# Connect to Moonbase Alpha
w3 = Web3(Web3.HTTPProvider(MOONBASE_RPC_URL))

# Smart Contract ABI (from Remix deployment - verified)
CONTRACT_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "internalType": "uint256", "name": "predictionId", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "predictedPrice1d", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "predictedPrice3d", "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "predictedPrice7d", "type": "uint256"},
            {"indexed": False, "internalType": "bytes32", "name": "modelHash", "type": "bytes32"}
        ],
        "name": "PredictionStored",
        "type": "event"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "_predictedPrice1d", "type": "uint256"},
            {"internalType": "uint256", "name": "_predictedPrice3d", "type": "uint256"},
            {"internalType": "uint256", "name": "_predictedPrice7d", "type": "uint256"},
            {"internalType": "bytes32", "name": "_modelHash", "type": "bytes32"}
        ],
        "name": "storePrediction",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "_newOwner", "type": "address"}],
        "name": "transferOwnership",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getAllPredictions",
        "outputs": [
            {
                "components": [
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedPrice1d", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedPrice3d", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedPrice7d", "type": "uint256"},
                    {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}
                ],
                "internalType": "struct BitcoinPredictionSimplified.Prediction[]",
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getContractInfo",
        "outputs": [
            {"internalType": "address", "name": "_owner", "type": "address"},
            {"internalType": "uint256", "name": "_totalPredictions", "type": "uint256"},
            {"internalType": "uint256", "name": "_contractBalance", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "_predictionId", "type": "uint256"}],
        "name": "getPrediction",
        "outputs": [
            {
                "components": [
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedPrice1d", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedPrice3d", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedPrice7d", "type": "uint256"},
                    {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}
                ],
                "internalType": "struct BitcoinPredictionSimplified.Prediction",
                "name": "",
                "type": "tuple"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getPredictionCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "_predictionId", "type": "uint256"}],
        "name": "getPredictionDetails",
        "outputs": [
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "predictedPrice1d", "type": "uint256"},
            {"internalType": "uint256", "name": "predictedPrice3d", "type": "uint256"},
            {"internalType": "uint256", "name": "predictedPrice7d", "type": "uint256"},
            {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "_count", "type": "uint256"}],
        "name": "getRecentPredictions",
        "outputs": [
            {
                "components": [
                    {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedPrice1d", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedPrice3d", "type": "uint256"},
                    {"internalType": "uint256", "name": "predictedPrice7d", "type": "uint256"},
                    {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}
                ],
                "internalType": "struct BitcoinPredictionSimplified.Prediction[]",
                "name": "",
                "type": "tuple[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "owner",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "name": "predictions",
        "outputs": [
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "uint256", "name": "predictedPrice1d", "type": "uint256"},
            {"internalType": "uint256", "name": "predictedPrice3d", "type": "uint256"},
            {"internalType": "uint256", "name": "predictedPrice7d", "type": "uint256"},
            {"internalType": "bytes32", "name": "modelHash", "type": "bytes32"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "totalPredictions",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_connection():
    """Check if Web3 is connected to Moonbase Alpha"""
    if not w3.is_connected():
        raise ConnectionError("Failed to connect to Moonbase Alpha RPC")
    print("✅ Connected to Moonbase Alpha")
    return True


def get_contract():
    """Get contract instance"""
    if not CONTRACT_ADDRESS:
        raise ValueError("CONTRACT_ADDRESS not set in environment variables (.env file)")

    check_connection()
    contract = w3.eth.contract(address=Web3.to_checksum_address(CONTRACT_ADDRESS), abi=CONTRACT_ABI)
    return contract


def get_account():
    """Get account from private key"""
    if not PRIVATE_KEY:
        raise ValueError("MOONBASE_PRIVATE_KEY not set in environment variables (.env file)")

    account = w3.eth.account.from_key(PRIVATE_KEY)
    return account


def price_to_cents(price: float) -> int:
    """Convert price in dollars to cents (for Solidity uint256)"""
    return int(price * 100)


def cents_to_price(cents: int) -> float:
    """Convert cents back to dollars"""
    return cents / 100


def create_model_hash(model_name: str = "xgboost_v1") -> bytes:
    """Create a 32-byte hash for model identification"""
    return w3.keccak(text=model_name)


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def store_prediction_onchain(
    predicted_1d: float,
    predicted_3d: float,
    predicted_7d: float,
    model_name: str = "xgboost_v1"
) -> dict:
    """
    Store prediction on Moonbase Alpha blockchain

    SIMPLIFIED VERSION - Only stores the 3 predicted prices
    (No current price or actual prices needed)

    Args:
        predicted_1d: Predicted price 1 day later in dollars
        predicted_3d: Predicted price 3 days later in dollars
        predicted_7d: Predicted price 7 days later in dollars
        model_name: Name of the model (default: "xgboost_v1")

    Returns:
        dict: {
            'tx_hash': Transaction hash,
            'prediction_id': On-chain prediction ID,
            'block_number': Block number,
            'timestamp': Block timestamp (ISO format),
            'gas_used': Gas used for transaction
        }
    """
    print(f"\n📤 Storing prediction on Moonbase Alpha...")
    print(f"   Predicted 1d: ${predicted_1d:,.2f}")
    print(f"   Predicted 3d: ${predicted_3d:,.2f}")
    print(f"   Predicted 7d: ${predicted_7d:,.2f}")
    print(f"   Model: {model_name}")

    # Get contract and account
    contract = get_contract()
    account = get_account()

    # Convert prices to cents
    pred_1d_cents = price_to_cents(predicted_1d)
    pred_3d_cents = price_to_cents(predicted_3d)
    pred_7d_cents = price_to_cents(predicted_7d)
    model_hash = create_model_hash(model_name)

    print(f"   Contract: {CONTRACT_ADDRESS}")
    print(f"   From: {account.address}")

    # Build transaction
    nonce = w3.eth.get_transaction_count(account.address)

    txn = contract.functions.storePrediction(
        pred_1d_cents,
        pred_3d_cents,
        pred_7d_cents,
        model_hash
    ).build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gas': 500000,  # Gas limit (actual usage ~297k)
        'gasPrice': w3.eth.gas_price
    })

    # Sign and send transaction
    signed_txn = account.sign_transaction(txn)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

    print(f"   Transaction sent: {tx_hash.hex()}")
    print(f"   Waiting for confirmation...")

    # Wait for receipt
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

    # Extract prediction ID from logs
    prediction_id = None
    for log in receipt['logs']:
        try:
            decoded = contract.events.PredictionStored().process_log(log)
            prediction_id = decoded['args']['predictionId']
            break
        except:
            continue

    # Get block timestamp
    block = w3.eth.get_block(receipt['blockNumber'])
    timestamp = datetime.fromtimestamp(block['timestamp'])

    result = {
        'tx_hash': tx_hash.hex(),
        'prediction_id': prediction_id,
        'block_number': receipt['blockNumber'],
        'timestamp': timestamp.isoformat(),
        'gas_used': receipt['gasUsed']
    }

    print(f"   ✅ Success!")
    print(f"   Prediction ID: {prediction_id}")
    print(f"   Block: {receipt['blockNumber']}")
    print(f"   Timestamp: {timestamp}")
    print(f"   Gas used: {receipt['gasUsed']}")
    print(f"   View on Moonscan: https://moonbase.moonscan.io/tx/{tx_hash.hex()}")

    return result


def get_recent_predictions(count: int = 30) -> list:
    """
    Fetch recent predictions from blockchain

    Args:
        count: Number of recent predictions to fetch

    Returns:
        list: List of prediction dicts with fields:
            - prediction_id
            - timestamp (ISO format)
            - predicted_price_1d
            - predicted_price_3d
            - predicted_price_7d
            - model_hash
    """
    print(f"\n📥 Fetching last {count} predictions from blockchain...")

    contract = get_contract()

    # Fetch predictions from contract
    predictions = contract.functions.getRecentPredictions(count).call()

    results = []
    total_count = contract.functions.getPredictionCount().call()

    for i, pred in enumerate(predictions):
        # Calculate actual prediction ID
        pred_id = total_count - count + i

        results.append({
            'prediction_id': pred_id,
            'timestamp': datetime.fromtimestamp(pred[0]).isoformat(),
            'predicted_price_1d': cents_to_price(pred[1]),
            'predicted_price_3d': cents_to_price(pred[2]),
            'predicted_price_7d': cents_to_price(pred[3]),
            'model_hash': pred[4].hex()
        })

    print(f"   ✅ Fetched {len(results)} predictions")
    return results


def get_contract_stats() -> dict:
    """
    Get contract statistics

    Returns:
        dict: Contract stats (owner, total predictions, etc.)
    """
    contract = get_contract()

    owner, total_predictions, balance = contract.functions.getContractInfo().call()

    return {
        'owner': owner,
        'total_predictions': total_predictions,
        'contract_balance': w3.from_wei(balance, 'ether'),
        'contract_address': CONTRACT_ADDRESS
    }


# ============================================================================
# DEMO MODE FUNCTIONS
# ============================================================================

def get_predictions_with_outcomes(count=30, use_demo=True) -> list:
    """
    Load predictions with outcomes from CSV

    Args:
        count: Number of recent predictions to fetch
        use_demo: If True, load from demo CSV. If False, load from real blockchain CSV

    Returns:
        list: List of prediction dicts with outcomes
    """
    import pandas as pd
    from pathlib import Path

    # Determine which CSV to load
    if use_demo:
        csv_path = Path(__file__).parent.parent / 'data' / 'blockchain' / 'prediction_tracking_demo.csv'
    else:
        csv_path = Path(__file__).parent.parent / 'data' / 'blockchain' / 'prediction_tracking.csv'

    # Check if file exists
    if not csv_path.exists():
        print(f"⚠️  CSV file not found: {csv_path}")
        return []

    # Load CSV
    df = pd.read_csv(csv_path)

    # Add Moonscan URLs
    df['moonscan_url'] = df['tx_hash'].apply(
        lambda tx: f'https://moonbase.moonscan.io/tx/{tx}'
    )

    # Get last N predictions
    recent = df.tail(count)

    # Convert to list of dicts
    return recent.to_dict('records')


def get_performance_summary(use_demo=True) -> dict:
    """
    Calculate aggregate performance metrics

    Args:
        use_demo: If True, use demo data. If False, use real blockchain data

    Returns:
        dict: Performance summary statistics
    """
    import numpy as np

    # Get all predictions
    preds = get_predictions_with_outcomes(count=1000, use_demo=use_demo)

    if not preds:
        return {
            'total_predictions': 0,
            'predictions_with_outcomes_1d': 0,
            'predictions_with_outcomes_3d': 0,
            'predictions_with_outcomes_7d': 0,
            'avg_mape_1d': 0,
            'avg_mape_3d': 0,
            'avg_mape_7d': 0,
            'directional_accuracy_1d': 0,
            'directional_accuracy_3d': 0,
            'directional_accuracy_7d': 0,
            'best_prediction_1d': None,
            'worst_prediction_1d': None
        }

    # Filter predictions with outcomes
    with_outcome_1d = [p for p in preds if p.get('actual_1d') and not pd.isna(p.get('actual_1d'))]
    with_outcome_3d = [p for p in preds if p.get('actual_3d') and not pd.isna(p.get('actual_3d'))]
    with_outcome_7d = [p for p in preds if p.get('actual_7d') and not pd.isna(p.get('actual_7d'))]

    # Calculate 1-day metrics
    avg_mape_1d = np.mean([p['mape_1d'] for p in with_outcome_1d if not pd.isna(p.get('mape_1d'))]) if with_outcome_1d else 0
    dir_acc_1d = np.mean([p['direction_correct_1d'] for p in with_outcome_1d if not pd.isna(p.get('direction_correct_1d'))]) * 100 if with_outcome_1d else 0

    # Calculate 3-day metrics
    avg_mape_3d = np.mean([p['mape_3d'] for p in with_outcome_3d if not pd.isna(p.get('mape_3d'))]) if with_outcome_3d else 0
    dir_acc_3d = np.mean([p['direction_correct_3d'] for p in with_outcome_3d if not pd.isna(p.get('direction_correct_3d'))]) * 100 if with_outcome_3d else 0

    # Calculate 7-day metrics
    avg_mape_7d = np.mean([p['mape_7d'] for p in with_outcome_7d if not pd.isna(p.get('mape_7d'))]) if with_outcome_7d else 0
    dir_acc_7d = np.mean([p['direction_correct_7d'] for p in with_outcome_7d if not pd.isna(p.get('direction_correct_7d'))]) * 100 if with_outcome_7d else 0

    # Find best and worst predictions (based on 1-day MAPE)
    valid_preds_1d = [p for p in with_outcome_1d if not pd.isna(p.get('mape_1d'))]
    best_pred = min(valid_preds_1d, key=lambda p: p['mape_1d']) if valid_preds_1d else None
    worst_pred = max(valid_preds_1d, key=lambda p: p['mape_1d']) if valid_preds_1d else None

    return {
        'total_predictions': len(preds),
        'predictions_with_outcomes_1d': len(with_outcome_1d),
        'predictions_with_outcomes_3d': len(with_outcome_3d),
        'predictions_with_outcomes_7d': len(with_outcome_7d),
        'avg_mape_1d': float(avg_mape_1d),
        'avg_mape_3d': float(avg_mape_3d),
        'avg_mape_7d': float(avg_mape_7d),
        'directional_accuracy_1d': float(dir_acc_1d),
        'directional_accuracy_3d': float(dir_acc_3d),
        'directional_accuracy_7d': float(dir_acc_7d),
        'best_prediction_1d': best_pred,
        'worst_prediction_1d': worst_pred
    }


# ============================================================================
# TESTING/DEBUG FUNCTIONS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  BLOCKCHAIN INTEGRATION TEST - SIMPLIFIED CONTRACT")
    print("="*70)

    try:
        # Test connection
        print("\n1. Testing connection to Moonbase Alpha...")
        check_connection()

        # Get contract stats
        print("\n2. Getting contract stats...")
        stats = get_contract_stats()
        print(f"   Contract: {stats['contract_address']}")
        print(f"   Owner: {stats['owner']}")
        print(f"   Total Predictions: {stats['total_predictions']}")

        # Fetch recent predictions
        if stats['total_predictions'] > 0:
            print("\n3. Fetching recent predictions...")
            predictions = get_recent_predictions(count=5)
            for pred in predictions:
                print(f"\n   Prediction #{pred['prediction_id']}:")
                print(f"      Date: {pred['timestamp']}")
                print(f"      Predicted 1d: ${pred['predicted_price_1d']:,.2f}")
                print(f"      Predicted 3d: ${pred['predicted_price_3d']:,.2f}")
                print(f"      Predicted 7d: ${pred['predicted_price_7d']:,.2f}")
        else:
            print("\n3. No predictions stored yet.")
            print("   Run the test below to store a sample prediction:")
            print("   >>> from utils.blockchain_integration import store_prediction_onchain")
            print("   >>> store_prediction_onchain(106500, 107800, 109200)")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
