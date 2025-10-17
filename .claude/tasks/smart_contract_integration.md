# Smart Contract Integration - Implementation Plan

**Task**: Integrate blockchain storage for daily predictions using Moonbase Alpha testnet
**Goal**: Store 1d, 3d, 7d predictions on-chain and display them on `/live` page with prediction vs reality comparison

---

## Overview

We need to:
1. Deploy a Solidity smart contract to **Moonbase Alpha** (Moonbeam testnet)
2. Create a Python script that runs daily via GitHub Actions to store predictions on-chain
3. Create a Python script to record actual outcomes (1 day later)
4. Update the `/live` webpage to fetch and display on-chain data (prediction vs reality for 1d)

---

## Architecture

```
Daily GitHub Action (6 PM UTC)
    ↓
1. Generate predictions (1d, 3d, 7d) → predict_daily.py
    ↓
2. Store predictions on Moonbase Alpha → store_prediction_onchain.py
    ↓
3. Record yesterday's actual outcome → record_outcome_onchain.py
    ↓
4. Commit predictions to CSV (backup)
    ↓
Flask /live page
    ↓
5. Fetch on-chain predictions + outcomes
    ↓
6. Display: Prediction vs Reality chart (1d simplified)
```

---

## Step 1: Smart Contract (Solidity)

**File**: `contracts/BitcoinPrediction.sol`

### Contract Features:
- Store predictions (timestamp, current_price, predicted_1d/3d/7d)
- Record actual outcomes (1 day, 3 days, 7 days later)
- Query recent predictions
- Calculate on-chain MAPE

### Why Moonbase Alpha?
- Free testnet (no real money)
- EVM-compatible (works with Web3.py)
- Similar to Ethereum (easy to port to mainnet later)
- Active faucet for test tokens

### Deployment Process (using Remix):
1. Go to https://remix.ethereum.org/
2. Create `BitcoinPrediction.sol` file
3. Compile with Solidity 0.8.0+
4. Connect MetaMask to Moonbase Alpha network
5. Get testnet DEV tokens from faucet
6. Deploy contract
7. Copy contract address

**Moonbase Alpha Network Config**:
```
Network Name: Moonbase Alpha
RPC URL: https://rpc.api.moonbase.moonbeam.network
Chain ID: 1287
Currency Symbol: DEV
Block Explorer: https://moonbase.moonscan.io/
```

---

## Step 2: Python Blockchain Integration

**File**: `utils/blockchain_integration.py`

### Functions:

#### 2.1 Store Prediction
```python
def store_prediction_onchain(
    current_price: float,
    predicted_1d: float,
    predicted_3d: float,
    predicted_7d: float,
    model_hash: str
) -> str:
    """
    Store prediction on Moonbase Alpha blockchain

    Args:
        current_price: Current BTC price
        predicted_1d/3d/7d: Predicted prices for horizons
        model_hash: Hash of model version (e.g., "xgboost_v1")

    Returns:
        Transaction hash
    """
    # Convert prices to cents (uint256)
    # Call smart contract storePrediction()
    # Return transaction hash
```

#### 2.2 Record Outcome
```python
def record_outcome_onchain(
    prediction_id: int,
    horizon: int,  # 1, 3, or 7
    actual_price: float
) -> str:
    """
    Record actual outcome for a prediction

    Called 1, 3, 7 days after prediction made

    Args:
        prediction_id: On-chain prediction ID
        horizon: 1, 3, or 7 days
        actual_price: Actual BTC price at horizon

    Returns:
        Transaction hash
    """
    # Convert price to cents
    # Call smart contract recordOutcome()
    # Return transaction hash
```

#### 2.3 Fetch Predictions
```python
def get_recent_predictions(count: int = 30) -> list:
    """
    Fetch recent predictions from blockchain

    Args:
        count: Number of recent predictions to fetch

    Returns:
        List of prediction dicts with all fields
    """
    # Call smart contract getRecentPredictions()
    # Convert uint256 prices back to floats
    # Return list of dicts
```

---

## Step 3: Daily Prediction Script

**File**: `utils/store_daily_prediction_onchain.py`

### What it does:
1. Load today's predictions from `data/predictions/daily_predictions.csv`
2. Get current BTC price
3. Store prediction on-chain using `store_prediction_onchain()`
4. Save transaction hash and prediction_id to tracking file
5. Record yesterday's actual outcome (if exists)

### Prediction Tracking File

**File**: `data/blockchain/prediction_tracking.csv`

Columns:
```csv
date,prediction_id,tx_hash,current_price,predicted_1d,predicted_3d,predicted_7d,outcome_recorded_1d,outcome_recorded_3d,outcome_recorded_7d
2025-10-17,0,0xabc123...,105000.50,106500.25,107800.00,109200.50,false,false,false
2025-10-18,1,0xdef456...,106200.75,107000.00,108500.25,110000.00,true,false,false
```

This file helps us:
- Track which predictions need outcome recording
- Map dates to on-chain prediction IDs
- Avoid duplicate storage

---

## Step 4: Outcome Recording Script

**File**: `utils/record_outcomes_onchain.py`

### What it does:
1. Read `prediction_tracking.csv`
2. For each prediction where outcome_recorded_1d == false:
   - Check if 1 day has passed
   - Get actual price from historical data
   - Call `record_outcome_onchain(prediction_id, 1, actual_price)`
   - Update outcome_recorded_1d = true
3. Same for 3d and 7d horizons

---

## Step 5: GitHub Action Workflow

**File**: `.github/workflows/store_predictions_onchain.yml`

```yaml
name: Store Daily Predictions On-Chain

on:
  schedule:
    # Run daily at 6 PM UTC (after predict_daily.yml)
    - cron: '30 18 * * *'  # 30 min after predictions generated
  workflow_dispatch:

permissions:
  contents: write

jobs:
  store-onchain:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install web3 python-dotenv

    - name: Store prediction on Moonbase Alpha
      env:
        MOONBASE_PRIVATE_KEY: ${{ secrets.MOONBASE_PRIVATE_KEY }}
        CONTRACT_ADDRESS: ${{ secrets.CONTRACT_ADDRESS }}
      run: |
        python utils/store_daily_prediction_onchain.py

    - name: Record outcomes for past predictions
      env:
        MOONBASE_PRIVATE_KEY: ${{ secrets.MOONBASE_PRIVATE_KEY }}
        CONTRACT_ADDRESS: ${{ secrets.CONTRACT_ADDRESS }}
      run: |
        python utils/record_outcomes_onchain.py

    - name: Commit tracking file
      run: |
        git config --local user.email "github-actions[bot]@users.noreply.github.com"
        git config --local user.name "github-actions[bot]"
        git add data/blockchain/
        git diff --quiet && git diff --staged --quiet || (git commit -m "⛓️ Blockchain predictions - $(date +'%Y-%m-%d')" && git push)
```

---

## Step 6: Update /live Page

**File**: `webapp/templates/live.html`

### New Section: Blockchain Predictions (1-Day Simplified)

```html
<div class="section">
    <h2>🔗 Blockchain Predictions (Moonbase Alpha)</h2>
    <p>Immutable, on-chain predictions vs reality (1-day horizon)</p>

    <!-- Table: Last 30 Predictions -->
    <table class="blockchain-table">
        <thead>
            <tr>
                <th>Date</th>
                <th>Current Price</th>
                <th>Predicted (1d)</th>
                <th>Actual (1d)</th>
                <th>Error</th>
                <th>MAPE</th>
                <th>Verified</th>
            </tr>
        </thead>
        <tbody id="blockchain-predictions">
            <!-- Populated by JavaScript -->
        </tbody>
    </table>

    <!-- Chart: Prediction vs Reality -->
    <canvas id="blockchain-chart"></canvas>

    <!-- On-Chain Stats -->
    <div class="blockchain-stats">
        <div class="stat">
            <h3>Total Predictions</h3>
            <p id="total-predictions">-</p>
        </div>
        <div class="stat">
            <h3>Verified (1d)</h3>
            <p id="verified-1d">-</p>
        </div>
        <div class="stat">
            <h3>On-Chain MAPE (1d)</h3>
            <p id="onchain-mape">-</p>
        </div>
        <div class="stat">
            <h3>Contract Address</h3>
            <p><a href="https://moonbase.moonscan.io/address/CONTRACT_ADDR" target="_blank">View on Explorer</a></p>
        </div>
    </div>
</div>
```

### New API Endpoint

**File**: `webapp/app.py`

```python
@app.route('/api/blockchain-predictions')
def blockchain_predictions():
    """
    Fetch predictions from Moonbase Alpha blockchain

    Returns:
        JSON with recent predictions and stats
    """
    from utils.blockchain_integration import get_recent_predictions

    try:
        predictions = get_recent_predictions(count=30)

        # Calculate stats
        total = len(predictions)
        verified_1d = sum(1 for p in predictions if p['verified1d'])

        # Calculate MAPE for verified predictions
        errors = []
        for p in predictions:
            if p['verified1d'] and p['actualPrice1d'] > 0:
                predicted = p['predictedPrice1d']
                actual = p['actualPrice1d']
                mape = abs(predicted - actual) / actual * 100
                errors.append(mape)

        avg_mape = sum(errors) / len(errors) if errors else 0

        return jsonify({
            'success': True,
            'predictions': predictions,
            'stats': {
                'total': total,
                'verified_1d': verified_1d,
                'mape_1d': round(avg_mape, 2)
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
```

### JavaScript to Populate Table

**File**: `webapp/static/js/blockchain.js`

```javascript
async function loadBlockchainPredictions() {
    try {
        const response = await fetch('/api/blockchain-predictions');
        const data = await response.json();

        if (!data.success) {
            console.error('Failed to load blockchain data:', data.error);
            return;
        }

        // Populate stats
        document.getElementById('total-predictions').textContent = data.stats.total;
        document.getElementById('verified-1d').textContent = data.stats.verified_1d;
        document.getElementById('onchain-mape').textContent = data.stats.mape_1d + '%';

        // Populate table
        const tbody = document.getElementById('blockchain-predictions');
        tbody.innerHTML = '';

        data.predictions.forEach(pred => {
            const row = document.createElement('tr');

            const date = new Date(pred.timestamp * 1000).toLocaleDateString();
            const currentPrice = formatPrice(pred.currentPrice);
            const predicted1d = formatPrice(pred.predictedPrice1d);
            const actual1d = pred.verified1d ? formatPrice(pred.actualPrice1d) : 'Pending';
            const error = pred.verified1d ? calculateError(pred.predictedPrice1d, pred.actualPrice1d) : '-';
            const mape = pred.verified1d ? calculateMAPE(pred.predictedPrice1d, pred.actualPrice1d) : '-';
            const verified = pred.verified1d ? '✅' : '⏳';

            row.innerHTML = `
                <td>${date}</td>
                <td>$${currentPrice}</td>
                <td>$${predicted1d}</td>
                <td>${actual1d}</td>
                <td>${error}</td>
                <td>${mape}</td>
                <td>${verified}</td>
            `;

            tbody.appendChild(row);
        });

        // Draw chart
        drawBlockchainChart(data.predictions);

    } catch (error) {
        console.error('Error loading blockchain predictions:', error);
    }
}

function formatPrice(cents) {
    return (cents / 100).toFixed(2);
}

function calculateError(predicted, actual) {
    const error = Math.abs(predicted - actual) / 100;
    return '$' + error.toFixed(2);
}

function calculateMAPE(predicted, actual) {
    const mape = Math.abs(predicted - actual) / actual * 100;
    return mape.toFixed(2) + '%';
}

function drawBlockchainChart(predictions) {
    // Filter verified predictions
    const verified = predictions.filter(p => p.verified1d);

    const labels = verified.map(p => new Date(p.timestamp * 1000).toLocaleDateString());
    const predicted = verified.map(p => p.predictedPrice1d / 100);
    const actual = verified.map(p => p.actualPrice1d / 100);

    const ctx = document.getElementById('blockchain-chart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Predicted (1d)',
                    data: predicted,
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    tension: 0.1
                },
                {
                    label: 'Actual (1d)',
                    data: actual,
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'On-Chain Predictions vs Reality (1-Day Horizon)'
                },
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    title: {
                        display: true,
                        text: 'BTC Price ($)'
                    }
                }
            }
        }
    });
}

// Load on page load
document.addEventListener('DOMContentLoaded', loadBlockchainPredictions);
```

---

## Step 7: Environment Setup

### Required GitHub Secrets

Add these to your GitHub repository settings (Settings → Secrets → Actions):

1. **MOONBASE_PRIVATE_KEY**: Private key of wallet with DEV tokens
   - Create a new MetaMask account just for this project
   - Export the private key
   - Get testnet DEV tokens from faucet: https://faucet.moonbeam.network/

2. **CONTRACT_ADDRESS**: Deployed smart contract address on Moonbase Alpha
   - Deploy contract via Remix
   - Copy the contract address

### Local .env File

**File**: `.env` (add to .gitignore!)

```
MOONBASE_PRIVATE_KEY=your_private_key_here
CONTRACT_ADDRESS=0x...
MOONBASE_RPC_URL=https://rpc.api.moonbase.moonbeam.network
```

---

## Implementation Order

### Phase 1: Smart Contract (Day 1)
1. ✅ Write `BitcoinPrediction.sol` based on SMART_CONTRACT_PLAN.md
2. ✅ Deploy to Moonbase Alpha via Remix
3. ✅ Test contract functions in Remix
4. ✅ Save contract address

### Phase 2: Python Integration (Day 2)
1. ✅ Create `utils/blockchain_integration.py` with Web3.py
2. ✅ Test storing prediction manually
3. ✅ Test recording outcome manually
4. ✅ Test fetching predictions

### Phase 3: Daily Automation (Day 3)
1. ✅ Create `utils/store_daily_prediction_onchain.py`
2. ✅ Create `utils/record_outcomes_onchain.py`
3. ✅ Create prediction tracking CSV structure
4. ✅ Test end-to-end flow locally

### Phase 4: GitHub Actions (Day 4)
1. ✅ Create `.github/workflows/store_predictions_onchain.yml`
2. ✅ Add GitHub secrets
3. ✅ Test workflow manually
4. ✅ Monitor for 3 days

### Phase 5: Web Interface (Day 5)
1. ✅ Add `/api/blockchain-predictions` endpoint
2. ✅ Update `live.html` template
3. ✅ Create `blockchain.js` for frontend
4. ✅ Test display with real on-chain data

---

## Data Flow Example

### Day 1 (Oct 17, 6:00 PM UTC):
```
1. predict_daily.py runs → generates predictions
   - Current: $105,000
   - Predicted 1d: $106,500
   - Predicted 3d: $107,800
   - Predicted 7d: $109,200

2. store_daily_prediction_onchain.py runs
   - Calls contract.storePrediction(105000_00, 106500_00, 107800_00, 109200_00, ...)
   - Gets prediction_id = 0, tx_hash = 0xabc123...
   - Saves to prediction_tracking.csv

3. Blockchain state:
   predictions[0] = {
       timestamp: Oct 17 6:00 PM,
       currentPrice: 105000_00,
       predictedPrice1d: 106500_00,
       actualPrice1d: 0,  // Not yet known
       verified1d: false
   }
```

### Day 2 (Oct 18, 6:00 PM UTC):
```
1. predict_daily.py runs → generates new predictions

2. record_outcomes_onchain.py runs
   - Checks prediction_tracking.csv
   - Finds prediction_id=0 from Oct 17
   - Gets actual price from today: $106,200
   - Calls contract.recordOutcome(0, 1, 106200_00)
   - Updates outcome_recorded_1d = true

3. Blockchain state:
   predictions[0] = {
       timestamp: Oct 17 6:00 PM,
       currentPrice: 105000_00,
       predictedPrice1d: 106500_00,
       actualPrice1d: 106200_00,  // ✅ Recorded!
       verified1d: true
   }

4. /live page shows:
   Oct 17: Predicted $106,500 | Actual $106,200 | Error $300 | MAPE 0.28% ✅
```

---

## Cost Estimation

### Moonbase Alpha (Testnet):
- **FREE** - No real money required
- Get testnet DEV tokens from faucet

### Gas Costs (if on mainnet):
- Store prediction: ~0.0001 DEV (~$0.01)
- Record outcome: ~0.00005 DEV (~$0.005)
- **Daily total**: ~$0.025 per day = ~$9/year

---

## Testing Strategy

### Local Testing:
1. Use Ganache or Hardhat local blockchain
2. Test all functions without spending gas
3. Verify data structures

### Moonbase Alpha Testing:
1. Deploy contract
2. Store 1 prediction manually
3. Wait 1 day, record outcome manually
4. Verify on Moonscan explorer
5. Test fetching from Web3.py

### Production Testing:
1. Run for 7 days to get full cycle (1d, 3d, 7d outcomes)
2. Verify MAPE calculations match off-chain results
3. Check gas costs and optimize if needed

---

## Fallback Strategy

If blockchain integration has issues:
1. All predictions still saved to CSV files (backup)
2. /live page can show CSV data if blockchain fetch fails
3. Can manually upload predictions later
4. No data loss

---

## Security Considerations

1. **Private Key Protection**:
   - Never commit private keys to Git
   - Use GitHub Secrets for CI/CD
   - Use environment variables locally
   - Create dedicated wallet for this project only

2. **Contract Ownership**:
   - Only owner can store predictions (prevents spam)
   - Consider multi-sig for production
   - Can transfer ownership if needed

3. **Data Validation**:
   - Validate prices are reasonable (e.g., > 0, < $1M)
   - Validate horizons are 1, 3, or 7
   - Check timestamp is recent

---

## Future Enhancements

1. **Multi-coin Support**: Add ETH, other cryptocurrencies
2. **NFT Prediction Cards**: Mint NFT for each prediction
3. **On-chain MAPE Calculation**: Gas-optimized MAPE in smart contract
4. **Mainnet Deployment**: Move to Moonbeam mainnet
5. **Frontend Web3**: Let users query contract directly from browser
6. **Prediction Marketplace**: Users can stake on predictions

---

## Success Criteria

✅ Smart contract deployed to Moonbase Alpha
✅ Daily predictions stored on-chain automatically
✅ Outcomes recorded 1, 3, 7 days later
✅ /live page displays blockchain data
✅ Prediction vs Reality chart working
✅ On-chain MAPE matches off-chain MAPE
✅ Zero manual intervention required
✅ All data verifiable on Moonscan

---

## Resources

- **Moonbase Alpha Docs**: https://docs.moonbeam.network/builders/get-started/networks/moonbase/
- **Moonbase Faucet**: https://faucet.moonbeam.network/
- **Moonscan Explorer**: https://moonbase.moonscan.io/
- **Remix IDE**: https://remix.ethereum.org/
- **Web3.py Docs**: https://web3py.readthedocs.io/
- **MetaMask**: https://metamask.io/

---

## Next Steps

1. **Approve this plan** - Review and confirm approach
2. **Start Phase 1** - Deploy smart contract via Remix
3. **Implement Phase 2** - Build Python integration
4. **Test locally** - Verify everything works
5. **Deploy to production** - Set up GitHub Actions
6. **Monitor** - Watch for 7 days to get full cycle

Let me know when you're ready to proceed with Phase 1!
