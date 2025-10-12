# Smart Contract Integration Plan
## Bitcoin Price Prediction - Immutable Performance Tracking

---

## Executive Summary

This document outlines the strategy for storing Bitcoin price predictions on a blockchain using smart contracts. The goal is to provide **immutable, transparent, and verifiable proof** of our model's real-world performance over time.

### Key Benefits
- **Immutability**: Once stored, predictions cannot be altered or deleted
- **Transparency**: Anyone can verify our predictions and outcomes
- **Timestamping**: Blockchain timestamp proves predictions were made before outcomes
- **Credibility**: Cryptographic proof builds trust in model performance
- **Performance Tracking**: Historical data enables long-term analysis

---

## Recommendation: Daily Predictions

### Why Daily Over 15-Minute Intervals?

| Factor | Daily | 15-Minute |
|--------|-------|-----------|
| **Gas Costs** | ~$1-5 per day | ~$96-480 per day (96 intervals) |
| **Data Points/Month** | ~30 | ~2,880 |
| **Aligns with Model** | ✅ Best model is 1-day horizon | ⚠️ Would need new model training |
| **Interpretability** | ✅ Clear daily trends | ❌ Noisy, hard to interpret |
| **Test Comparison** | ✅ Easy to compare with test results | ❌ Different timeframe than test |
| **Maintenance** | ✅ Simple automation | ⚠️ Complex, requires monitoring |

**Decision: Start with daily predictions (1-day, 3-day, 7-day horizons)**

We can always expand to 15-minute predictions later if needed, but daily provides the best balance of cost, clarity, and comparability with test results.

---

## Smart Contract Design

### Contract Structure

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract BitcoinPricePrediction {

    // Struct to store each prediction
    struct Prediction {
        uint256 timestamp;          // Unix timestamp when prediction made
        uint256 currentPrice;        // BTC price at prediction time (in cents to avoid decimals)
        uint256 predictedPrice1d;    // Predicted price 1 day later
        uint256 predictedPrice3d;    // Predicted price 3 days later
        uint256 predictedPrice7d;    // Predicted price 7 days later
        uint256 actualPrice1d;       // Actual price 1 day later (filled later)
        uint256 actualPrice3d;       // Actual price 3 days later (filled later)
        uint256 actualPrice7d;       // Actual price 7 days later (filled later)
        bytes32 modelHash;           // Hash of model version used
        string signalType;           // "BUY", "SELL", or "HOLD"
        bool verified1d;             // True once 1-day outcome is recorded
        bool verified3d;             // True once 3-day outcome is recorded
        bool verified7d;             // True once 7-day outcome is recorded
    }

    // Storage
    Prediction[] public predictions;
    address public owner;
    uint256 public totalPredictions;

    // Events
    event PredictionStored(
        uint256 indexed predictionId,
        uint256 timestamp,
        uint256 currentPrice,
        uint256 predictedPrice1d
    );

    event OutcomeRecorded(
        uint256 indexed predictionId,
        uint8 horizon,  // 1, 3, or 7 days
        uint256 actualPrice,
        uint256 errorInCents
    );

    // Modifiers
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }

    // Constructor
    constructor() {
        owner = msg.sender;
    }

    // Store new prediction
    function storePrediction(
        uint256 _currentPrice,
        uint256 _predictedPrice1d,
        uint256 _predictedPrice3d,
        uint256 _predictedPrice7d,
        bytes32 _modelHash,
        string memory _signalType
    ) external onlyOwner returns (uint256) {

        Prediction memory newPrediction = Prediction({
            timestamp: block.timestamp,
            currentPrice: _currentPrice,
            predictedPrice1d: _predictedPrice1d,
            predictedPrice3d: _predictedPrice3d,
            predictedPrice7d: _predictedPrice7d,
            actualPrice1d: 0,
            actualPrice3d: 0,
            actualPrice7d: 0,
            modelHash: _modelHash,
            signalType: _signalType,
            verified1d: false,
            verified3d: false,
            verified7d: false
        });

        predictions.push(newPrediction);
        totalPredictions++;

        uint256 predictionId = predictions.length - 1;

        emit PredictionStored(
            predictionId,
            block.timestamp,
            _currentPrice,
            _predictedPrice1d
        );

        return predictionId;
    }

    // Record actual outcome after 1, 3, or 7 days
    function recordOutcome(
        uint256 _predictionId,
        uint8 _horizon,  // 1, 3, or 7
        uint256 _actualPrice
    ) external onlyOwner {
        require(_predictionId < predictions.length, "Invalid prediction ID");
        require(_horizon == 1 || _horizon == 3 || _horizon == 7, "Invalid horizon");

        Prediction storage pred = predictions[_predictionId];

        // Calculate error
        uint256 predictedPrice;
        uint256 error;

        if (_horizon == 1) {
            require(!pred.verified1d, "Already verified");
            pred.actualPrice1d = _actualPrice;
            pred.verified1d = true;
            predictedPrice = pred.predictedPrice1d;
        } else if (_horizon == 3) {
            require(!pred.verified3d, "Already verified");
            pred.actualPrice3d = _actualPrice;
            pred.verified3d = true;
            predictedPrice = pred.predictedPrice3d;
        } else {
            require(!pred.verified7d, "Already verified");
            pred.actualPrice7d = _actualPrice;
            pred.verified7d = true;
            predictedPrice = pred.predictedPrice7d;
        }

        // Calculate absolute error
        if (_actualPrice > predictedPrice) {
            error = _actualPrice - predictedPrice;
        } else {
            error = predictedPrice - _actualPrice;
        }

        emit OutcomeRecorded(_predictionId, _horizon, _actualPrice, error);
    }

    // Get prediction by ID
    function getPrediction(uint256 _predictionId) external view returns (Prediction memory) {
        require(_predictionId < predictions.length, "Invalid prediction ID");
        return predictions[_predictionId];
    }

    // Get recent predictions
    function getRecentPredictions(uint256 _count) external view returns (Prediction[] memory) {
        uint256 count = _count > predictions.length ? predictions.length : _count;
        Prediction[] memory recent = new Prediction[](count);

        for (uint256 i = 0; i < count; i++) {
            recent[i] = predictions[predictions.length - count + i];
        }

        return recent;
    }

    // Calculate MAPE for verified predictions
    function calculateMAPE(uint8 _horizon, uint256 _lastN) external view returns (uint256) {
        require(_horizon == 1 || _horizon == 3 || _horizon == 7, "Invalid horizon");

        uint256 count = _lastN > predictions.length ? predictions.length : _lastN;
        uint256 totalPercentError = 0;
        uint256 verifiedCount = 0;

        for (uint256 i = predictions.length - count; i < predictions.length; i++) {
            Prediction memory pred = predictions[i];

            bool verified;
            uint256 actual;
            uint256 predicted;

            if (_horizon == 1 && pred.verified1d) {
                verified = true;
                actual = pred.actualPrice1d;
                predicted = pred.predictedPrice1d;
            } else if (_horizon == 3 && pred.verified3d) {
                verified = true;
                actual = pred.actualPrice3d;
                predicted = pred.predictedPrice3d;
            } else if (_horizon == 7 && pred.verified7d) {
                verified = true;
                actual = pred.actualPrice7d;
                predicted = pred.predictedPrice7d;
            }

            if (verified && actual > 0) {
                uint256 error = actual > predicted ? actual - predicted : predicted - actual;
                uint256 percentError = (error * 10000) / actual;  // multiply by 10000 for 2 decimal places
                totalPercentError += percentError;
                verifiedCount++;
            }
        }

        if (verifiedCount == 0) return 0;
        return totalPercentError / verifiedCount;  // returns MAPE * 100 (e.g., 150 = 1.50%)
    }
}
```

---

## Data Storage Strategy

### What to Store

Each daily prediction includes:

1. **Timestamp**: Unix timestamp when prediction was made
2. **Current Price**: BTC/USD price at prediction time (in cents to avoid decimals)
3. **Predictions**:
   - 1-day predicted price
   - 3-day predicted price
   - 7-day predicted price
4. **Model Info**: Hash of the model version used
5. **Trading Signal**: BUY, SELL, or HOLD
6. **Actual Outcomes** (filled later):
   - Actual price after 1 day
   - Actual price after 3 days
   - Actual price after 7 days

### Price Representation

Store prices in **cents** (multiply by 100) to avoid decimals in Solidity:
- $67,850.23 → 6,785,023 cents
- Allows exact integer arithmetic
- Convert back to dollars in frontend: `price_cents / 100`

---

## Blockchain Selection

### Recommended: Polygon (MATIC)

| Chain | Gas Cost/Tx | Speed | Pros | Cons |
|-------|-------------|-------|------|------|
| **Ethereum Mainnet** | $5-50 | 12s | Most secure, widely used | Expensive |
| **Polygon** | $0.01-0.10 | 2s | Very cheap, EVM compatible | Less decentralized |
| **Arbitrum** | $0.10-1 | 2s | L2 Ethereum, secure | Slightly more complex |
| **BSC** | $0.10-0.50 | 3s | Fast, cheap | Centralized |

**Recommendation: Start with Polygon for cost-effectiveness. Can bridge to Ethereum mainnet later if needed.**

### Estimated Costs (Polygon)

- **Deploy contract**: ~$0.50-1.00 one-time
- **Store prediction**: ~$0.02-0.05 per day
- **Record outcome**: ~$0.02-0.05 per outcome (3 times per prediction)
- **Total per prediction**: ~$0.08-0.20
- **Monthly cost**: ~$2.40-6.00 (30 days)
- **Annual cost**: ~$29-73

Very affordable for demonstrating the concept!

---

## Implementation Steps

### Phase 1: Development & Testing (Week 1-2)

1. **Write Smart Contract**
   - Copy contract code above
   - Add additional helper functions if needed
   - Write unit tests using Hardhat/Truffle

2. **Deploy to Testnet**
   - Deploy to Polygon Mumbai testnet
   - Get free test MATIC from faucet
   - Test all functions with mock data

3. **Build Python Integration**
   - Install web3.py: `pip install web3`
   - Create `utils/blockchain_integration.py`
   - Implement functions:
     - `store_prediction()` - Store new prediction
     - `record_outcome()` - Record actual price
     - `get_predictions()` - Fetch historical predictions
     - `calculate_live_mape()` - Get on-chain MAPE

### Phase 2: Automation (Week 3)

1. **Daily Prediction Job**
   ```python
   # scheduled_prediction.py

   from utils.blockchain_integration import store_prediction
   from utils.data_fetcher import get_latest_price
   import joblib

   def daily_prediction_job():
       # 1. Get current price
       current_price = get_latest_price()['price']

       # 2. Load model and make prediction
       model = joblib.load('models/saved_models/xgboost_returns_1d.json')
       predicted_return = model.predict(features)
       predicted_price_1d = current_price * (1 + predicted_return)

       # ... same for 3d and 7d

       # 3. Store on blockchain
       tx_hash = store_prediction(
           current_price=int(current_price * 100),  # convert to cents
           predicted_1d=int(predicted_price_1d * 100),
           predicted_3d=int(predicted_price_3d * 100),
           predicted_7d=int(predicted_price_7d * 100),
           model_hash=calculate_model_hash(),
           signal="BUY"  # or SELL/HOLD based on threshold
       )

       print(f"Prediction stored: {tx_hash}")

   # Run with cron: 0 0 * * * python scheduled_prediction.py
   ```

2. **Outcome Recording Job**
   ```python
   # scheduled_outcome.py

   from utils.blockchain_integration import record_outcome, get_predictions
   from datetime import datetime, timedelta

   def daily_outcome_job():
       predictions = get_predictions(last_n=10)
       current_price = get_latest_price()['price']

       for pred in predictions:
           pred_date = datetime.fromtimestamp(pred['timestamp'])
           days_elapsed = (datetime.now() - pred_date).days

           # Record 1-day outcome
           if days_elapsed >= 1 and not pred['verified1d']:
               record_outcome(pred['id'], horizon=1, actual_price=int(current_price * 100))

           # Record 3-day outcome
           if days_elapsed >= 3 and not pred['verified3d']:
               record_outcome(pred['id'], horizon=3, actual_price=int(current_price * 100))

           # Record 7-day outcome
           if days_elapsed >= 7 and not pred['verified7d']:
               record_outcome(pred['id'], horizon=7, actual_price=int(current_price * 100))

   # Run with cron: 0 1 * * * python scheduled_outcome.py
   ```

### Phase 3: Web Integration (Week 4)

1. **Update Flask App**
   - Modify `webapp/app.py` to use real blockchain data
   - Replace mock `get_blockchain_predictions()` with actual contract calls
   - Add loading states for blockchain queries

2. **Add Blockchain Explorer Links**
   - Link to Polygonscan for transaction verification
   - Show block numbers and timestamps
   - Display gas costs

### Phase 4: Production Deployment (Week 5+)

1. **Deploy to Polygon Mainnet**
   - Buy small amount of MATIC (~$10-20 for several months)
   - Deploy contract to Polygon mainnet
   - Update Python scripts with mainnet contract address

2. **Setup Automation**
   - Configure cron jobs on server
   - Setup monitoring/alerts for failed transactions
   - Create backup private key storage

---

## Python Integration Code

### `utils/blockchain_integration.py`

```python
"""
Blockchain Integration for Bitcoin Price Predictions
Stores predictions and outcomes on Polygon blockchain
"""

from web3 import Web3
from pathlib import Path
import json
import os
from datetime import datetime

# Configuration
POLYGON_RPC = "https://polygon-rpc.com"  # Mainnet
# POLYGON_RPC = "https://rpc-mumbai.maticvigil.com"  # Testnet

CONTRACT_ADDRESS = "0x..."  # Fill after deployment
CONTRACT_ABI_PATH = Path(__file__).parent.parent / "contracts" / "abi.json"

# Initialize Web3
w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))

# Load private key from environment variable (NEVER commit this!)
PRIVATE_KEY = os.getenv("BLOCKCHAIN_PRIVATE_KEY")
ACCOUNT = w3.eth.account.from_key(PRIVATE_KEY)

# Load contract
with open(CONTRACT_ABI_PATH) as f:
    contract_abi = json.load(f)
contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)


def store_prediction(current_price, predicted_1d, predicted_3d, predicted_7d,
                     model_hash, signal):
    """
    Store a new prediction on the blockchain.

    Args:
        current_price (int): Current BTC price in cents
        predicted_1d (int): 1-day predicted price in cents
        predicted_3d (int): 3-day predicted price in cents
        predicted_7d (int): 7-day predicted price in cents
        model_hash (str): Hash of model version (bytes32)
        signal (str): "BUY", "SELL", or "HOLD"

    Returns:
        str: Transaction hash
    """
    # Build transaction
    txn = contract.functions.storePrediction(
        current_price,
        predicted_1d,
        predicted_3d,
        predicted_7d,
        Web3.toBytes(hexstr=model_hash),
        signal
    ).build_transaction({
        'from': ACCOUNT.address,
        'nonce': w3.eth.get_transaction_count(ACCOUNT.address),
        'gas': 200000,
        'gasPrice': w3.eth.gas_price
    })

    # Sign and send
    signed_txn = w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

    # Wait for confirmation
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"✅ Prediction stored: {tx_hash.hex()}")
    print(f"   Block: {tx_receipt['blockNumber']}")
    print(f"   Gas used: {tx_receipt['gasUsed']}")

    return tx_hash.hex()


def record_outcome(prediction_id, horizon, actual_price):
    """
    Record actual outcome for a prediction.

    Args:
        prediction_id (int): ID of prediction
        horizon (int): 1, 3, or 7 days
        actual_price (int): Actual BTC price in cents

    Returns:
        str: Transaction hash
    """
    txn = contract.functions.recordOutcome(
        prediction_id,
        horizon,
        actual_price
    ).build_transaction({
        'from': ACCOUNT.address,
        'nonce': w3.eth.get_transaction_count(ACCOUNT.address),
        'gas': 150000,
        'gasPrice': w3.eth.gas_price
    })

    signed_txn = w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"✅ Outcome recorded: {tx_hash.hex()}")

    return tx_hash.hex()


def get_recent_predictions(count=30):
    """
    Fetch recent predictions from blockchain.

    Args:
        count (int): Number of predictions to fetch

    Returns:
        list: List of prediction dicts
    """
    predictions = contract.functions.getRecentPredictions(count).call()

    # Convert to list of dicts
    result = []
    for pred in predictions:
        result.append({
            'timestamp': pred[0],
            'date': datetime.fromtimestamp(pred[0]).strftime('%Y-%m-%d'),
            'current_price': pred[1] / 100,  # Convert cents to dollars
            'predicted_price_1d': pred[2] / 100,
            'predicted_price_3d': pred[3] / 100,
            'predicted_price_7d': pred[4] / 100,
            'actual_price_1d': pred[5] / 100 if pred[5] > 0 else None,
            'actual_price_3d': pred[6] / 100 if pred[6] > 0 else None,
            'actual_price_7d': pred[7] / 100 if pred[7] > 0 else None,
            'model_hash': pred[8].hex(),
            'signal': pred[9],
            'verified1d': pred[10],
            'verified3d': pred[11],
            'verified7d': pred[12]
        })

    return result


def calculate_live_mape(horizon=1, last_n=7):
    """
    Calculate MAPE from blockchain data.

    Args:
        horizon (int): 1, 3, or 7 days
        last_n (int): Number of recent predictions to include

    Returns:
        float: MAPE as percentage
    """
    mape_raw = contract.functions.calculateMAPE(horizon, last_n).call()
    return mape_raw / 100  # Contract returns MAPE * 100


def calculate_model_hash():
    """
    Calculate hash of current model for versioning.

    Returns:
        str: Hex string of model hash
    """
    import hashlib

    model_file = Path('models/saved_models/xgboost_returns_1d.json')
    with open(model_file, 'rb') as f:
        model_bytes = f.read()

    hash_obj = hashlib.sha256(model_bytes)
    return hash_obj.hexdigest()
```

---

## Security Considerations

### Private Key Management

1. **NEVER commit private key to Git**
2. Use environment variables: `export BLOCKCHAIN_PRIVATE_KEY="0x..."`
3. For production, use AWS Secrets Manager or similar
4. Create a dedicated wallet for this project (not your personal wallet)
5. Only fund with minimal MATIC needed (~$20-50)

### Contract Security

1. **Owner-only functions**: Only deployer can store predictions
2. **Input validation**: Check all parameters before storing
3. **Overflow protection**: Use Solidity 0.8+ with built-in overflow checks
4. **Event logging**: Emit events for all state changes
5. **Pausable**: Consider adding emergency pause functionality

---

## Monitoring & Maintenance

### Daily Monitoring

- Check cron job logs for errors
- Verify transactions succeeded on Polygonscan
- Monitor MATIC balance (alert if below threshold)
- Check website displays latest predictions

### Weekly Review

- Compare on-chain MAPE with test set MAPE
- Investigate any performance degradation
- Review gas costs and optimize if needed

### Model Updates

When retraining model:
1. Deploy new model version
2. Update model hash in predictions
3. Consider creating new contract for major changes
4. Document model version changes on-chain

---

## Expansion Options

### Future Enhancements

1. **Multi-Model Ensemble**
   - Store predictions from multiple models
   - Compare performance on-chain
   - Use voting/averaging for final prediction

2. **15-Minute Predictions**
   - Train intraday model
   - Store high-frequency predictions
   - Useful for day trading strategies

3. **Automated Trading**
   - Smart contract executes trades based on signals
   - DeFi integration (Uniswap, etc.)
   - Fully autonomous trading system

4. **NFT Prediction Cards**
   - Mint NFT for each prediction
   - Tradeable performance tokens
   - Gamification of predictions

5. **DAO Governance**
   - Community votes on model parameters
   - Distributed ownership of prediction system
   - Revenue sharing from trading profits

---

## Timeline

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Write & test contract | Contract code, tests |
| 2 | Deploy testnet, Python integration | Working testnet demo |
| 3 | Build automation scripts | Cron jobs for predictions/outcomes |
| 4 | Web integration | Live page showing blockchain data |
| 5+ | Production deployment | Mainnet contract, daily predictions |

---

## Cost Summary

### One-Time Costs
- Contract deployment: $0.50-1.00
- Testing & development: ~10 hours

### Ongoing Costs (Monthly)
- Prediction storage: ~$1.50-3.00 (30 predictions)
- Outcome recording: ~$1.50-3.00 (90 outcomes)
- **Total: $3-6/month** on Polygon

Very affordable for the credibility and transparency benefits!

---

## Questions to Consider

1. **Privacy**: Do we want predictions public before outcomes? (Currently yes for transparency)
2. **Model updates**: Create new contract or update hash in existing contract?
3. **Backup**: Store predictions off-chain as well for redundancy?
4. **Analytics**: Build on-chain analytics dashboard or off-chain?
5. **Monetization**: Charge for prediction access? Subscription model?

---

## Conclusion

Storing predictions on blockchain provides **immutable proof** of model performance and builds **trust** through transparency. Starting with daily predictions on Polygon is cost-effective (~$3-6/month) and aligns perfectly with our test results (1-day horizon is our best model).

The system can be expanded later with 15-minute predictions, automated trading, or even NFT-based prediction markets. But for now, daily predictions provide the perfect balance of cost, clarity, and credibility.

**Next Steps:**
1. Review this plan with the team
2. Set up Polygon wallet and get testnet MATIC
3. Begin Phase 1 (contract development) this week
4. Target going live on mainnet within 4-5 weeks

---

*Document created: 2025-10-12*
*For: Bitcoin Price Prediction Capstone Project*
