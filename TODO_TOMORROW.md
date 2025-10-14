### Phase 4: Update Flask Webapp (20 minutes)

#### Task 4.1: Update app.py to Use Prediction Loader ⏱️ 10 min

**Current status:** Flask app probably has hardcoded prediction loading  
**Goal:** Use `prediction_loader.py` to fetch from GitHub with caching

Open `webapp/app.py` and replace the prediction loading logic with:

```python
import sys
import os

# Add parent directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify
from utils.prediction_loader import PredictionLoader

app = Flask(__name__)

# Initialize prediction loader (will cache predictions)
loader = PredictionLoader()

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/live')
def live():
    # Load predictions with smart caching
    daily = loader.load_predictions('daily')
    hourly = loader.load_predictions('hourly')
    intraday = loader.load_predictions('15min')
    
    return render_template('live.html', 
                         daily_predictions=daily,
                         hourly_predictions=hourly,
                         intraday_predictions=intraday)

@app.route('/api/predictions/<timeframe>')
def api_predictions(timeframe):
    """API endpoint for JSON predictions"""
    if timeframe not in ['daily', 'hourly', '15min']:
        return jsonify({'error': 'Invalid timeframe'}), 400
    
    predictions = loader.load_predictions(timeframe)
    return jsonify(predictions)

@app.route('/methodology')
def methodology():
    return render_template('methodology.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
```

---

#### Task 4.2: Update Templates to Use New Data Structure ⏱️ 5 min

Check `webapp/templates/live.html` - update to match the data structure from `prediction_loader.py`:

The loader returns data in this format:
```python
{
    'metadata': {
        'generated_at': '2025-10-14 12:34:56',
        'timeframe': 'daily',
        'horizons': ['1d', '3d', '7d']
    },
    'predictions': [
        {
            'horizon': '1d',
            'predicted_return': 0.0234,
            'predicted_price': 67890.12,
            'current_price': 67123.45,
            'direction': 'UP',
            'confidence': 'high'
        },
        ...
    ]
}
```

Make sure your template can display this data.

---

#### Task 4.3: Test Flask App Locally ⏱️ 5 min
```bash
cd /Users/ying-jeanne/Workspace/capstone_bitcoin
python webapp/app.py
```

**Open in browser:** http://localhost:5000/live

**Expected:**
- Should see predictions from GitHub (if cache expired) or cached data
- No errors in terminal
- Predictions display correctly

**Stop server:** Press Ctrl+C

---

#### Task 4.4: Commit Webapp Changes ⏱️ 2 min
```bash
git add webapp/app.py webapp/templates/
git commit -m "Update webapp to use prediction_loader with GitHub caching"
git push origin main
```

---

### Phase 5: Deploy to PythonAnywhere (30 minutes)

#### Task 5.1: Create PythonAnywhere Account ⏱️ 5 min
1. Go to: https://www.pythonanywhere.com/
2. Click "Pricing & signup"
3. Select "Create a Beginner account" (FREE)
4. Sign up with email/password
5. Verify email

---

#### Task 5.2: Clone Repository ⏱️ 3 min
1. In PythonAnywhere, go to **"Consoles"** tab
2. Click **"Bash"** to start a bash console
3. Run:
```bash
git clone https://github.com/ying-jeanne/capstone_msba.git
cd capstone_msba
```

---

#### Task 5.3: Install Dependencies ⏱️ 5 min
```bash
pip3.11 install --user -r requirements.txt
```

**Wait for installation to complete** (might take 3-5 minutes)

---

#### Task 5.4: Create Web App ⏱️ 5 min
1. Go to **"Web"** tab
2. Click **"Add a new web app"**
3. Click "Next" (domain will be `YOUR_USERNAME.pythonanywhere.com`)
4. Select **"Manual configuration"**
5. Select **"Python 3.11"**
6. Click "Next"

---

#### Task 5.5: Configure WSGI File ⏱️ 3 min
1. In "Web" tab, find "Code" section
2. Click on WSGI configuration file link (something like `/var/www/YOUR_USERNAME_pythonanywhere_com_wsgi.py`)
3. **Delete all contents**
4. Replace with (change YOUR_USERNAME):

```python
import sys
import os

# Add your project to the path
project_home = '/home/YOUR_USERNAME/capstone_msba'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Change working directory
os.chdir(project_home)

# Import Flask app
from webapp.app import app as application
```

5. **Save file** (Ctrl+S or Cmd+S)

---

#### Task 5.6: Configure Static Files ⏱️ 2 min
1. In "Web" tab, scroll to **"Static files"** section
2. Click "Enter URL" under a new entry
3. Add:
   - **URL:** `/static/`
   - **Directory:** `/home/YOUR_USERNAME/capstone_msba/webapp/static/`
4. Click the checkmark to save

---

#### Task 5.7: Reload & Test ⏱️ 5 min
1. Scroll to top of "Web" tab
2. Click big green **"Reload YOUR_USERNAME.pythonanywhere.com"** button
3. Wait ~10 seconds
4. Click the link to open your site

**Expected:**
- Website loads ✅
- Can navigate to /live page ✅
- Predictions display correctly ✅
- Images and CSS load ✅

**If errors:**
- Click "Error log" link in Web tab
- Read error messages
- Fix and reload

---

#### Task 5.8: Test Prediction Updates ⏱️ 2 min
1. Open your site: `https://YOUR_USERNAME.pythonanywhere.com/live`
2. Note the timestamp on predictions
3. Wait 30 seconds (15-min prediction cache duration)
4. Refresh page
5. Check if timestamp updates (if GitHub Actions ran in meantime)

---

### Phase 6: Smart Contract Setup (40 minutes)

#### Task 6.1: Install Blockchain Dependencies ⏱️ 5 min
```bash
pip install web3 eth-account
```

---

#### Task 6.2: Create Polygon Wallet ⏱️ 5 min
1. Install MetaMask browser extension: https://metamask.io/
2. Create new wallet (save seed phrase securely!)
3. Get your wallet address (0x...)
4. Add Polygon network to MetaMask:
   - Network Name: Polygon Mumbai Testnet
   - RPC URL: https://rpc-mumbai.maticvigil.com
   - Chain ID: 80001
   - Currency: MATIC

---

#### Task 6.3: Get Test MATIC ⏱️ 5 min
1. Go to Polygon faucet: https://faucet.polygon.technology/
2. Select "Mumbai" network
3. Paste your wallet address
4. Click "Submit" to get free test MATIC
5. Wait ~30 seconds, check MetaMask balance

---

#### Task 6.4: Create Smart Contract Files ⏱️ 10 min

Create contract directory and files:
```bash
mkdir -p contracts
```

**Create `contracts/BitcoinPrediction.sol`** - Copy the Solidity code from SMART_CONTRACT_PLAN.md (the complete contract starting with `pragma solidity ^0.8.0`)

**Create `contracts/deploy.py`:**
```python
"""Deploy Bitcoin Prediction smart contract to Polygon testnet"""
from web3 import Web3
from solcx import compile_source, install_solc
import json
import os

# Install Solidity compiler
install_solc('0.8.0')

# Read contract
with open('contracts/BitcoinPrediction.sol', 'r') as f:
    contract_source = f.read()

# Compile
compiled = compile_source(contract_source, output_values=['abi', 'bin'])
contract_id, contract_interface = compiled.popitem()

# Connect to Polygon Mumbai testnet
w3 = Web3(Web3.HTTPProvider('https://rpc-mumbai.maticvigil.com'))

# Your wallet (from MetaMask)
private_key = os.getenv('BLOCKCHAIN_PRIVATE_KEY')  # Set this!
account = w3.eth.account.from_key(private_key)

print(f"Deploying from: {account.address}")
print(f"Balance: {w3.eth.get_balance(account.address) / 10**18} MATIC")

# Deploy contract
Contract = w3.eth.contract(
    abi=contract_interface['abi'],
    bytecode=contract_interface['bin']
)

# Build transaction
txn = Contract.constructor().build_transaction({
    'from': account.address,
    'nonce': w3.eth.get_transaction_count(account.address),
    'gas': 3000000,
    'gasPrice': w3.eth.gas_price
})

# Sign and send
signed_txn = w3.eth.account.sign_transaction(txn, private_key)
tx_hash = w3.eth.send_raw_transaction(signed_txn.rawTransaction)

print(f"Transaction hash: {tx_hash.hex()}")
print("Waiting for confirmation...")

tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
contract_address = tx_receipt['contractAddress']

print(f"\n✅ Contract deployed at: {contract_address}")
print(f"   View on explorer: https://mumbai.polygonscan.com/address/{contract_address}")

# Save ABI and address
with open('contracts/abi.json', 'w') as f:
    json.dump(contract_interface['abi'], f, indent=2)

with open('contracts/contract_address.txt', 'w') as f:
    f.write(contract_address)

print("\n✅ Saved ABI to contracts/abi.json")
print("✅ Saved address to contracts/contract_address.txt")
```

---

#### Task 6.5: Deploy Contract to Testnet ⏱️ 10 min
```bash
# Set your private key (get from MetaMask)
export BLOCKCHAIN_PRIVATE_KEY="0x..."  # Your private key from MetaMask

# Install solidity compiler
pip install py-solc-x

# Deploy contract
python contracts/deploy.py
```

**Expected output:**
- Contract address (0x...)
- Link to Polygonscan
- Creates `contracts/abi.json` and `contracts/contract_address.txt`

---

#### Task 6.6: Test Contract ⏱️ 5 min

**Create `contracts/test_contract.py`:**
```python
"""Test the deployed contract"""
from web3 import Web3
import json
import os

# Load contract info
with open('contracts/abi.json', 'r') as f:
    abi = json.load(f)

with open('contracts/contract_address.txt', 'r') as f:
    contract_address = f.read().strip()

# Connect
w3 = Web3(Web3.HTTPProvider('https://rpc-mumbai.maticvigil.com'))
contract = w3.eth.contract(address=contract_address, abi=abi)

# Test: Store a prediction
private_key = os.getenv('BLOCKCHAIN_PRIVATE_KEY')
account = w3.eth.account.from_key(private_key)

print("Testing contract...")
print(f"Contract address: {contract_address}")
print(f"Total predictions: {contract.functions.totalPredictions().call()}")

# Store test prediction
txn = contract.functions.storePrediction(
    6785000,  # $67,850 in cents
    6800000,  # 1d prediction
    6820000,  # 3d prediction
    6850000,  # 7d prediction
    bytes(32),  # model hash (zeros for test)
    "BUY"
).build_transaction({
    'from': account.address,
    'nonce': w3.eth.get_transaction_count(account.address),
    'gas': 200000,
    'gasPrice': w3.eth.gas_price
})

signed = w3.eth.account.sign_transaction(txn, private_key)
tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)

print(f"Storing test prediction...")
print(f"Tx hash: {tx_hash.hex()}")

receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print(f"✅ Success! Gas used: {receipt['gasUsed']}")
print(f"Total predictions: {contract.functions.totalPredictions().call()}")
```

Run the test:
```bash
python contracts/test_contract.py
```

Should show successful prediction storage on testnet!

---

### Phase 7: Final Verification (10 minutes)

#### Task 7.1: Verify Automated System ⏱️ 5 min
1. **Check GitHub Actions status:**
   - Go to https://github.com/ying-jeanne/capstone_msba/actions
   - Verify workflows are enabled ✅
   - Check last run times

2. **Check prediction files:**
   - Go to https://github.com/ying-jeanne/capstone_msba/tree/main/data/predictions
   - Verify timestamps are recent ✅

3. **Check website:**
   - Visit your PythonAnywhere site
   - Verify predictions display ✅
   - Check all pages work ✅

---

#### Task 7.2: Document Your URLs ⏱️ 2 min
Create a file `DEPLOYMENT_INFO.md` with:

```markdown
# Deployment Information

## Live Website
- **URL:** https://YOUR_USERNAME.pythonanywhere.com
- **Platform:** PythonAnywhere Free Tier
- **Status:** ✅ Live

## GitHub Repository
- **URL:** https://github.com/ying-jeanne/capstone_msba
- **Actions:** https://github.com/ying-jeanne/capstone_msba/actions
- **Predictions:** https://github.com/ying-jeanne/capstone_msba/tree/main/data/predictions

## GitHub Actions Schedules
- **Weekly Training:** Sundays at 2 AM UTC
- **Daily Predictions:** Every day at 6 PM UTC
- **Intraday Predictions:** Every 15 minutes

## Prediction Raw URLs
- Daily: https://raw.githubusercontent.com/ying-jeanne/capstone_msba/main/data/predictions/daily_predictions.csv
- Hourly: https://raw.githubusercontent.com/ying-jeanne/capstone_msba/main/data/predictions/hourly_predictions.csv
- 15-min: https://raw.githubusercontent.com/ying-jeanne/capstone_msba/main/data/predictions/15min_predictions.csv

## Smart Contract (Polygon Mumbai Testnet)
- Contract Address: [Your contract address from deployment]
- Explorer: https://mumbai.polygonscan.com/address/[your_address]
- Network: Polygon Mumbai Testnet
- Status: ✅ Deployed and tested

## Cache Durations
- 15-min predictions: 30 seconds
- Hourly predictions: 2 minutes
- Daily predictions: 5 minutes

## Cost
- **Total:** $0/month (completely free!)
```

Commit this file:
```bash
git add DEPLOYMENT_INFO.md
git commit -m "Add deployment information"
git push origin main
```

---

#### Task 7.3: Screenshot for Presentation ⏱️ 3 min
Take screenshots of:
1. Your live website showing predictions
2. GitHub Actions page showing successful runs
3. PythonAnywhere dashboard showing your app
4. Polygonscan showing your contract transactions
5. MetaMask showing your testnet wallet

Save these for your presentation!

---

## 📊 Summary Checklist

By end of tomorrow, you should have:

- ✅ All dependencies installed
- ✅ All 9 models trained locally
- ✅ Predictions generated and tested
- ✅ Everything pushed to GitHub
- ✅ GitHub Actions enabled and tested
- ✅ Flask webapp updated with prediction loader
- ✅ Deployed to PythonAnywhere
- ✅ Smart contract deployed to Polygon testnet
- ✅ Contract tested with sample prediction
- ✅ Fully automated system running

---

## ⏰ Time Estimate

| Phase | Time | Difficulty |
|-------|------|------------|
| Phase 1: Setup & Training | 30 min | Easy |
| Phase 2: GitHub Push | 10 min | Easy |
| Phase 3: GitHub Actions | 15 min | Medium |
| Phase 4: Update Webapp | 20 min | Medium |
| Phase 5: Deploy to PythonAnywhere | 30 min | Medium |
| Phase 6: Smart Contract Setup | 40 min | Medium |
| Phase 7: Final Verification | 10 min | Easy |
| **Total** | **~2.5 hours** | **Medium** |

**Note:** Training models takes 15-20 minutes but is mostly waiting. You can do other things while it runs!

---

## 🆘 If Something Goes Wrong

**Problem:** Training script fails  
**Fix:** Check error message, verify data sources are accessible, try running with smaller data

**Problem:** GitHub push fails (file too large)  
**Fix:** Models might be too big. Consider using Git LFS or compressing model files

**Problem:** GitHub Actions don't trigger  
**Fix:** Make sure repo has activity in last 60 days, manually trigger once to wake up

**Problem:** PythonAnywhere shows 502 error  
**Fix:** Check error log in Web tab, verify WSGI file paths are correct

**Problem:** Predictions not updating on website  
**Fix:** Check config.py has correct repo URL, verify GitHub Actions are running, check cache duration

**Problem:** Smart contract deployment fails  
**Fix:** Make sure you have test MATIC in wallet, check RPC URL is correct, verify private key is set

**Problem:** Transaction fails with "out of gas"  
**Fix:** Increase gas limit in transaction (try 300000 instead of 200000)

**Problem:** Can't see contract on Polygonscan  
**Fix:** Wait 30 seconds after deployment, make sure using Mumbai testnet explorer (not mainnet)

---

## 📝 Notes

- **Models take ~15-20 minutes to train** - be patient!
- **GitHub push might be slow** - model files are ~50-100 MB total
- **PythonAnywhere can be slow** - free tier has limited resources
- **Cache is important** - prevents hitting GitHub API limits
- **GitHub Actions have 6-hour job limit** - plenty for your workflows

---

## 🎉 After Tomorrow

You'll have:
- ✅ Fully automated Bitcoin prediction system
- ✅ Live website updating every 15 minutes
- ✅ Smart contract on blockchain for immutable predictions
- ✅ Zero monthly costs (testnet is free!)
- ✅ Production-ready deployment
- ✅ Impressive capstone project with Web3 integration!

**Next steps after deployment:**
1. Monitor system for a few days
2. Store daily predictions on blockchain (optional automation)
3. Deploy to Polygon mainnet when ready (~$3-6/month)
4. Prepare presentation slides (highlight blockchain feature!)
5. Practice demo (show Polygonscan transactions)
6. Add team page (optional)
7. Add blockchain data visualization (optional)

---

**Good luck! You've got this! 🚀**
