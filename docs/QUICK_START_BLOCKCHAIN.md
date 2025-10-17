# Quick Start - Blockchain Integration

Your contract is already deployed! Here's how to start using it.

---

## ✅ What's Already Done

- Smart contract deployed to Moonbase Alpha
- Contract address: `0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E`
- All Python scripts created
- GitHub Actions workflow ready
- Documentation complete

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Add Your Private Key

1. Open the file: `.env`
2. Replace `YOUR_PRIVATE_KEY_HERE` with your MetaMask private key
3. Save the file

**To get your private key from MetaMask:**
- Click MetaMask → 3 dots → Account Details
- Click "Export Private Key"
- Enter password → Copy the key
- Paste into `.env` file

**Your `.env` file should look like:**
```bash
MOONBASE_RPC_URL=https://rpc.api.moonbase.moonbeam.network
CONTRACT_ADDRESS=0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E
MOONBASE_PRIVATE_KEY=0xYOUR_ACTUAL_KEY_HERE
```

---

### Step 2: Install Python Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install Web3.py
pip install web3 python-dotenv
```

---

### Step 3: Test Connection

```bash
# Test blockchain connection
python utils/blockchain_integration.py
```

**Expected output:**
```
======================================================================
  BLOCKCHAIN INTEGRATION TEST - SIMPLIFIED CONTRACT
======================================================================

1. Testing connection to Moonbase Alpha...
✅ Connected to Moonbase Alpha

2. Getting contract stats...
   Contract: 0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E
   Owner: 0xYourAddress...
   Total Predictions: X

3. Fetching recent predictions...
   (Shows any existing predictions)
```

✅ **If you see this, you're connected!**

---

### Step 4: Store Your First Prediction

```bash
# Make sure you have predictions in CSV
python run_full_pipeline.py  # Generate predictions if needed

# Store on blockchain
python utils/store_daily_prediction_onchain.py
```

**Expected output:**
```
📤 Storing prediction on Moonbase Alpha...
   Predicted 1d: $XXX,XXX.XX
   Predicted 3d: $XXX,XXX.XX
   Predicted 7d: $XXX,XXX.XX
   Transaction sent: 0x...
   ✅ Success!
   View on Moonscan: https://moonbase.moonscan.io/tx/0x...
```

---

### Step 5: Verify on Blockchain

1. Copy the transaction hash from the output
2. Go to: https://moonbase.moonscan.io/
3. Paste the transaction hash
4. ✅ See your prediction stored on blockchain!

---

### Step 6: Configure GitHub Secrets (for automation)

1. Go to your GitHub repository
2. Settings → Secrets and variables → Actions
3. Click "New repository secret"

Add these two secrets:

**Secret 1:**
- Name: `MOONBASE_PRIVATE_KEY`
- Value: (paste your private key from `.env`)

**Secret 2:**
- Name: `CONTRACT_ADDRESS`
- Value: `0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E`

---

### Step 7: Test GitHub Actions (Optional)

```bash
# Manually trigger the workflow
gh workflow run store_predictions_onchain.yml

# Or via GitHub UI:
# Actions tab → "Store Daily Predictions On-Chain" → "Run workflow"
```

---

## 🎯 What Happens Now?

### Automatic Daily Flow:

```
6:00 PM UTC: predict_daily.yml runs
              ↓
              Generates predictions
              ↓
6:30 PM UTC: store_predictions_onchain.yml runs
              ↓
              Stores on blockchain automatically
              ↓
              Updates CSV files
              ↓
              Commits to GitHub
```

### What Users See:

Visit `/live` page → See predictions with blockchain proof → Click transaction hash → Verify on Moonscan ✅

---

## 🔍 Quick Commands

```bash
# Test connection
python utils/blockchain_integration.py

# Store prediction manually
python utils/store_daily_prediction_onchain.py

# View on Moonscan
open https://moonbase.moonscan.io/address/0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E

# Check GitHub Actions
gh run list

# Generate predictions
python run_full_pipeline.py

# Start webapp
python webapp/app.py
```

---

## 📊 Your Contract Info

- **Contract Address**: `0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E`
- **Network**: Moonbase Alpha (testnet)
- **Explorer**: https://moonbase.moonscan.io/address/0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E
- **Type**: BitcoinPrediction (your deployed contract)

---

## ✅ Success Checklist

After completing quick start:

- [ ] `.env` file has your private key
- [ ] Python successfully connects to contract
- [ ] First prediction stored on blockchain
- [ ] Transaction visible on Moonscan
- [ ] GitHub secrets configured
- [ ] GitHub Actions workflow tested
- [ ] Daily automation running

---

## 🐛 Troubleshooting

### "CONTRACT_ADDRESS not set"
→ Check `.env` file exists and has the contract address

### "MOONBASE_PRIVATE_KEY not set"
→ Add your MetaMask private key to `.env` file

### "Failed to connect"
→ Check internet connection, verify RPC URL

### "Insufficient funds"
→ Get more DEV tokens from https://faucet.moonbeam.network/

---

## 📚 Full Documentation

- **Deployment Guide**: `docs/BLOCKCHAIN_DEPLOYMENT_GUIDE.md`
- **Integration Summary**: `docs/BLOCKCHAIN_INTEGRATION_SUMMARY.md`
- **Smart Contract Plan**: `SMART_CONTRACT_PLAN.md`

---

## 🎉 You're Ready!

Your blockchain integration is set up and ready to use.

**Next steps:**
1. Add private key to `.env`
2. Run `python utils/blockchain_integration.py` to test
3. Store first prediction with `python utils/store_daily_prediction_onchain.py`
4. Configure GitHub secrets for automation
5. Watch it run automatically every day at 6:30 PM UTC!

Enjoy your immutable, blockchain-verified Bitcoin predictions! 🚀
