# Blockchain Deployment Guide
## Bitcoin Price Prediction - Moonbase Alpha Integration

This guide walks you through deploying the **BitcoinPredictionSimplified** smart contract to Moonbase Alpha testnet using Remix IDE.

---

## 📋 Prerequisites

- MetaMask browser extension installed
- Basic understanding of blockchain/smart contracts
- Contract file: `contracts/BitcoinPredictionSimplified.sol`

---

## Step 1: Set Up MetaMask for Moonbase Alpha

### 1.1 Open MetaMask
- Click the MetaMask extension in your browser
- If you don't have an account, create one

### 1.2 Add Moonbase Alpha Network

1. Click the network dropdown (usually shows "Ethereum Mainnet")
2. Click "Add Network" → "Add a network manually"
3. Fill in these exact details:

```
Network Name: Moonbase Alpha
RPC URL: https://rpc.api.moonbase.moonbeam.network
Chain ID: 1287
Currency Symbol: DEV
Block Explorer: https://moonbase.moonscan.io/
```

4. Click "Save"
5. Switch to "Moonbase Alpha" network

### 1.3 Get Testnet Tokens (DEV)

1. Copy your MetaMask wallet address (click to copy)
2. Go to the faucet: https://faucet.moonbeam.network/
3. Paste your wallet address
4. Click "Get DEV tokens"
5. Wait ~30 seconds for tokens to arrive
6. Check MetaMask - you should see "1 DEV"

✅ You now have free testnet tokens to deploy and interact with the contract!

---

## Step 2: Deploy Contract via Remix IDE

### 2.1 Open Remix

1. Go to: https://remix.ethereum.org/
2. You'll see the Remix IDE interface

### 2.2 Create Contract File

1. In the left sidebar, click "File Explorer" icon
2. Click the "+" icon to create a new file
3. Name it: `BitcoinPredictionSimplified.sol`
4. Copy the entire contents from `/Users/ying-jeanne/Workspace/capstone_bitcoin/contracts/BitcoinPredictionSimplified.sol`
5. Paste into the Remix editor

### 2.3 Compile the Contract

1. Click the "Solidity Compiler" icon (2nd icon in left sidebar)
2. Select compiler version: `0.8.0` or higher (e.g., `0.8.20`)
3. Click "Compile BitcoinPredictionSimplified.sol"
4. ✅ You should see a green checkmark

If you see errors:
- Make sure you copied the entire file
- Check the compiler version is 0.8.0+
- Look for syntax errors (red underlines)

### 2.4 Deploy the Contract

1. Click "Deploy & Run Transactions" icon (3rd icon in left sidebar)
2. Under "Environment", select: **Injected Provider - MetaMask**
3. MetaMask will pop up:
   - Click "Connect"
   - Select your account
   - Click "Next" → "Connect"
4. Confirm MetaMask shows "Moonbase Alpha" network in the popup
5. Under "Contract", select: `BitcoinPredictionSimplified`
6. Click the orange "Deploy" button
7. MetaMask pops up with transaction:
   - Review gas fees (~0.001 DEV)
   - Click "Confirm"
8. Wait for deployment (~15 seconds)
9. ✅ You'll see the contract under "Deployed Contracts"

### 2.5 Save Your Contract Address

1. Under "Deployed Contracts", you'll see something like:
   ```
   BITCOINPREDICTION AT 0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E
   ```

2. **Your deployed contract address**: `0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E`
3. This address is already configured in the example files
   - `.env` file
   - Python scripts
   - GitHub secrets

---

## Step 3: Test Contract Functions

Now let's test the contract to make sure it works!

### 3.1 Store a Test Prediction

1. Under "Deployed Contracts", expand your contract
2. Find the **orange** `storePrediction` button
3. Click the dropdown arrow to expand parameters
4. Fill in these test values:

```
_predictedPrice1d: 10650000
_predictedPrice3d: 10780000
_predictedPrice7d: 10920000
_modelHash: 0x1234567890123456789012345678901234567890123456789012345678901234
```

**What these mean:**
- Predicted 1 day: $106,500.00
- Predicted 3 days: $107,800.00
- Predicted 7 days: $109,200.00
- Model hash: Test identifier

5. Click "transact"
6. Confirm in MetaMask
7. Wait ~15 seconds
8. ✅ Check Remix console for `status: true`

### 3.2 Verify the Prediction

1. Find the **blue** `getPrediction` button (blue = read, no gas)
2. Enter: `0` (first prediction ID)
3. Click "call"
4. ✅ You should see:

```
0: uint256: timestamp 1729535842       (when stored)
1: uint256: predictedPrice1d 10650000  ($106,500)
2: uint256: predictedPrice3d 10780000  ($107,800)
3: uint256: predictedPrice7d 10920000  ($109,200)
4: bytes32: modelHash 0x1234...
```

### 3.3 Get Contract Stats

1. Find the **blue** `getContractInfo` button
2. Click "call"
3. ✅ You should see:

```
_owner: [your wallet address]
_totalPredictions: 1
_contractBalance: 0
```

### 3.4 Verify on Blockchain Explorer

1. Copy a transaction hash from Remix console (at bottom)
2. Go to: https://moonbase.moonscan.io/
3. Paste the transaction hash
4. ✅ You can see:
   - Transaction details
   - Block number
   - Timestamp (proves WHEN prediction was made!)
   - Input data (your prediction values)
   - Status: Success ✅

---

## Step 4: Set Up Environment Variables

Now configure your local environment to use the deployed contract.

### 4.1 Create `.env` File

In your project root (`/Users/ying-jeanne/Workspace/capstone_bitcoin/`), create a file named `.env`:

```bash
# Moonbase Alpha Configuration
MOONBASE_RPC_URL=https://rpc.api.moonbase.moonbeam.network
CONTRACT_ADDRESS=0xYOUR_CONTRACT_ADDRESS_HERE
MOONBASE_PRIVATE_KEY=your_metamask_private_key_here
```

### 4.2 Get Your MetaMask Private Key

⚠️ **WARNING**: Never share your private key! This is for testnet only.

1. Open MetaMask
2. Click the 3 dots menu (top right)
3. Click "Account Details"
4. Click "Export Private Key"
5. Enter your MetaMask password
6. Copy the private key (starts with `0x`)
7. Paste it into `.env` file

### 4.3 Add to `.gitignore`

**CRITICAL**: Make sure `.env` is in your `.gitignore` file!

Check `/Users/ying-jeanne/Workspace/capstone_bitcoin/.gitignore` includes:
```
.env
*.env
```

### 4.4 Example `.env` File

```bash
# Moonbase Alpha Blockchain Configuration
# WARNING: Never commit this file to Git!

MOONBASE_RPC_URL=https://rpc.api.moonbase.moonbeam.network
CONTRACT_ADDRESS=0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E
MOONBASE_PRIVATE_KEY=your_actual_private_key_here
```

---

## Step 5: Test Python Integration

Now test that Python can interact with your deployed contract.

### 5.1 Install Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install Web3.py
pip install web3 python-dotenv
```

### 5.2 Test Connection

```bash
# Run the blockchain integration test
python utils/blockchain_integration.py
```

Expected output:
```
======================================================================
  BLOCKCHAIN INTEGRATION TEST - SIMPLIFIED CONTRACT
======================================================================

1. Testing connection to Moonbase Alpha...
✅ Connected to Moonbase Alpha

2. Getting contract stats...
   Contract: 0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E
   Owner: 0xYourAddress...
   Total Predictions: 1

3. Fetching recent predictions...

   Prediction #0:
      Date: 2025-10-17T18:00:03
      Predicted 1d: $106,500.00
      Predicted 3d: $107,800.00
      Predicted 7d: $109,200.00
```

✅ If you see this, Python is successfully connecting to your contract!

### 5.3 Store a Prediction via Python

```bash
# Run the daily storage script (uses latest prediction from CSV)
python utils/store_daily_prediction_onchain.py
```

This will:
1. Read `data/predictions/daily_predictions.csv`
2. Store prediction on blockchain
3. Update CSV with tx_hash
4. Save to tracking file

---

## Step 6: Configure GitHub Secrets

For GitHub Actions to work, you need to add secrets to your repository.

### 6.1 Go to GitHub Repository Settings

1. Go to your GitHub repository: `https://github.com/your-username/capstone_bitcoin`
2. Click "Settings" tab
3. Click "Secrets and variables" → "Actions"

### 6.2 Add Secrets

Click "New repository secret" and add these two secrets:

**Secret 1:**
- Name: `MOONBASE_PRIVATE_KEY`
- Value: Your MetaMask private key (from `.env` file)

**Secret 2:**
- Name: `CONTRACT_ADDRESS`
- Value: Your contract address (e.g., `0xc297...`)

⚠️ **Important**: Use the SAME private key and contract address from your `.env` file!

---

## 🎯 Quick Reference

### Price Conversion

When testing, remember prices are stored in cents:

| Price | Cents (Solidity uint256) |
|-------|--------------------------|
| $100,000 | 10000000 |
| $105,000 | 10500000 |
| $106,500 | 10650000 |
| $1,234.56 | 123456 |

**Formula**: `price_in_cents = price_in_dollars × 100`

### Useful Links

- **Moonbase Faucet**: https://faucet.moonbeam.network/
- **Moonscan Explorer**: https://moonbase.moonscan.io/
- **Remix IDE**: https://remix.ethereum.org/
- **Moonbeam Docs**: https://docs.moonbeam.network/builders/get-started/networks/moonbase/

---

## ✅ Success Checklist

After completing this guide, you should have:

- ✅ MetaMask connected to Moonbase Alpha
- ✅ Testnet DEV tokens in wallet
- ✅ Smart contract deployed to Moonbase Alpha
- ✅ Contract address saved
- ✅ Test prediction stored via Remix
- ✅ Verified transaction on Moonscan
- ✅ `.env` file configured
- ✅ Python successfully connecting to contract
- ✅ GitHub secrets configured

---

## 🐛 Troubleshooting

### "Failed to connect to Moonbase Alpha"
- Check your internet connection
- Verify RPC URL is correct: `https://rpc.api.moonbase.moonbeam.network`
- Try again - sometimes RPC nodes are temporarily down

### "Insufficient funds for gas"
- Go to faucet and get more DEV tokens
- Wait a few minutes between faucet requests

### "Contract not found"
- Double-check CONTRACT_ADDRESS in `.env`
- Make sure contract was deployed successfully
- Verify you're on Moonbase Alpha network

### "Transaction failed"
- Check gas price isn't too low
- Verify input values are valid (all > 0)
- Make sure you're the contract owner

### Python import errors
- Make sure you're in the virtual environment: `source venv/bin/activate`
- Install dependencies: `pip install web3 python-dotenv`
- Check you're running from project root directory

---

## 🚀 Next Steps

Once deployment is complete:

1. ✅ Deploy simplified contract
2. ✅ Test manually via Remix
3. ✅ Test Python integration
4. Set up GitHub Actions (see workflow file)
5. Update webapp to show blockchain data
6. Monitor daily automation

You're now ready to store predictions on-chain! 🎉
