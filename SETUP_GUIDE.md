# Complete Setup Guide: Automated Bitcoin Prediction System

## Table of Contents
1. [Quick Start (15 Minutes)](#quick-start-15-minutes)
2. [System Overview](#system-overview)
3. [GitHub Actions Setup](#github-actions-setup)
4. [Deployment to PythonAnywhere](#deployment-to-pythonanywhere)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start (15 Minutes)

### Prerequisites
- Python 3.11+ installed
- Git configured
- GitHub account with academic/student license (unlimited Actions minutes)
- A GitHub repository (yours: `ying-jeanne/capstone_msba`)

### Step 1: Install Dependencies (2 minutes)
```bash
cd /Users/ying-jeanne/Workspace/capstone_bitcoin
pip install -r requirements.txt
```

### Step 2: Verify Configuration (1 minute)
Your config.py is already set to:
```python
GITHUB_REPO = "ying-jeanne/capstone_msba"
```

Run the system check:
```bash
python check_system.py
```

### Step 3: Train Models Locally (15-20 minutes)
**Important:** You must train models locally first before GitHub Actions can generate predictions.

```bash
# Train daily models (1d, 3d, 7d horizons)
python utils/train_daily_models.py

# Train hourly models (1h, 6h, 24h horizons)
python utils/train_hourly_models.py

# Train 15-minute models (15m, 1h, 4h horizons)
python utils/train_15min_models.py
```

This creates 9 model files in `models/saved_models/`:
- `daily/` - xgboost_1d.json, xgboost_3d.json, xgboost_7d.json + preprocessors
- `hourly/` - xgboost_1h.json, xgboost_6h.json, xgboost_24h.json + preprocessors
- `15min/` - xgboost_15m.json, xgboost_1h.json, xgboost_4h.json + preprocessors

### Step 4: Test Predictions Locally (2 minutes)
```bash
# Test daily predictions
python utils/predict_daily.py

# Test intraday predictions
python utils/predict_hourly_and_15min.py
```

Should create files in `data/predictions/`:
- `daily_predictions.csv`
- `hourly_predictions.csv`
- `15min_predictions.csv`

### Step 5: Commit Everything to GitHub (3 minutes)
```bash
# Stage all new files
git add .

# Commit with descriptive message
git commit -m "Add automated prediction system with GitHub Actions"

# Push to GitHub
git push origin main
```

### Step 6: Enable GitHub Actions (2 minutes)
1. Go to: https://github.com/ying-jeanne/capstone_msba
2. Click **"Actions"** tab at the top
3. If you see a green button "I understand my workflows, go ahead and enable them" → Click it
4. You should see 3 workflows:
   - 🏋️ Weekly Model Training
   - 📊 Daily Predictions
   - ⚡ Intraday Predictions (15-min)

### Step 7: Test First Workflow (5 minutes)
1. Click on **"Daily Predictions"** workflow
2. Click **"Run workflow"** dropdown (right side)
3. Select "main" branch
4. Click **"Run workflow"** button
5. Wait ~30 seconds, then refresh page
6. Click on the running workflow to see logs
7. ✅ Should complete in ~20 seconds and commit `daily_predictions.csv`

---

## System Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Repository                        │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Models (trained weekly)                               │ │
│  │  - models/saved_models/daily/*.json                   │ │
│  │  - models/saved_models/hourly/*.json                  │ │
│  │  - models/saved_models/15min/*.json                   │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Predictions (updated frequently)                      │ │
│  │  - data/predictions/daily_predictions.csv             │ │
│  │  - data/predictions/hourly_predictions.csv            │ │
│  │  - data/predictions/15min_predictions.csv             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          ↑
                          │ GitHub Actions (automated)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   Weekly Model      Daily Pred        Intraday Pred
   Training          Generation        (Every 15 min)
   (Sundays 2AM)     (Daily 6PM)      
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │
                    Commits updates
                          ↓
                  GitHub Repository
                          │
                          │ Raw URLs (no git pulls!)
                          ↓
                  ┌───────────────┐
                  │ PythonAnywhere│
                  │  Flask Webapp │
                  │               │
                  │  - Fetches    │
                  │    predictions│
                  │  - Smart cache│
                  │  - No git ops │
                  └───────────────┘
                          │
                          ↓
                    Users see latest
                    predictions
```

### Data Flow

**Training (Weekly - Sundays 2AM UTC):**
1. GitHub Actions triggers `train_models_weekly.yml`
2. Fetches latest data from Yahoo Finance, CoinGecko, Binance
3. Trains 9 XGBoost models (~15-20 minutes)
4. Commits models to `models/saved_models/`

**Predictions (Frequent):**
1. **Daily (6PM UTC):** `predict_daily.yml` runs
   - Loads pre-trained daily models
   - Fetches latest data
   - Generates 1d, 3d, 7d predictions (~20 seconds)
   - Commits `daily_predictions.csv`

2. **Intraday (Every 15 minutes):** `predict_intraday.yml` runs
   - Loads pre-trained hourly and 15-min models
   - Generates all intraday predictions (~40 seconds)
   - Commits both CSVs

**Flask Webapp (PythonAnywhere):**
1. User visits website
2. Webapp checks cache (30s for 15-min, 2min for hourly, 5min for daily)
3. If cache expired → Fetches from GitHub raw URL
4. Parses CSV and displays predictions
5. No git operations needed!

### Cost Analysis

| Component | Cost | Notes |
|-----------|------|-------|
| GitHub Actions | **$0/month** | Academic license = unlimited minutes |
| PythonAnywhere | **$0/month** | Free tier (500 MB storage, always-on webapp) |
| Data Sources | **$0/month** | All APIs are free (Yahoo, CoinGecko, Binance) |
| Domain | **$0/month** | Use PythonAnywhere subdomain (yourname.pythonanywhere.com) |
| **Total** | **$0/month** | 🎉 Completely free! |

**GitHub Actions Usage:**
- Weekly training: 20 min/week = 80 min/month
- Daily predictions: 20 sec/day × 30 = 10 min/month
- Intraday predictions: 40 sec × 96/day × 30 = 1,920 min/month
- **Total: ~2,010 minutes/month**
- With academic license: **Unlimited** ✅
- Without academic license: 2,000 free minutes (need to buy more)

### File Organization

```
capstone_bitcoin/
├── config.py                          # Central configuration
├── check_system.py                    # System health check
├── requirements.txt                   # Python dependencies
├── run_full_pipeline.py              # (Optional) Local pipeline runner
│
├── utils/                            # Training and prediction scripts
│   ├── train_daily_models.py        # Train 1d, 3d, 7d models
│   ├── train_hourly_models.py       # Train 1h, 6h, 24h models
│   ├── train_15min_models.py        # Train 15m, 1h, 4h models
│   ├── predict_daily.py             # Generate daily predictions
│   ├── predict_hourly_and_15min.py  # Generate intraday predictions
│   └── prediction_loader.py         # Load predictions with caching
│
├── models/
│   └── saved_models/                # Trained models (committed to GitHub)
│       ├── daily/                   # xgboost_*.json + preprocessors
│       ├── hourly/                  # xgboost_*.json + preprocessors
│       └── 15min/                   # xgboost_*.json + preprocessors
│
├── data/
│   ├── raw/                         # Raw data from APIs (not committed)
│   └── predictions/                 # Generated predictions (committed)
│       ├── daily_predictions.csv
│       ├── hourly_predictions.csv
│       └── 15min_predictions.csv
│
├── webapp/                          # Flask web application
│   ├── app.py                       # Main Flask app
│   ├── templates/                   # HTML templates
│   └── static/                      # CSS, JS, images
│
└── .github/
    └── workflows/                   # GitHub Actions workflows
        ├── train_models_weekly.yml  # Weekly training
        ├── predict_daily.yml        # Daily predictions
        └── predict_intraday.yml     # 15-minute predictions
```

---

## GitHub Actions Setup

### Workflow Details

#### 1. Weekly Model Training (`train_models_weekly.yml`)
**Schedule:** Every Sunday at 2 AM UTC  
**Duration:** ~15-20 minutes  
**Purpose:** Retrain all 9 models with latest historical data

```yaml
on:
  schedule:
    - cron: '0 2 * * 0'  # Sundays at 2 AM UTC
  workflow_dispatch:      # Manual trigger
```

**What it does:**
1. Checks out repository code
2. Sets up Python 3.11 environment
3. Installs dependencies from requirements.txt
4. Runs 3 training scripts sequentially
5. Commits all model files to repository
6. Pushes changes

**Files created/updated:**
- `models/saved_models/daily/` - 3 models + scaler.pkl + feature_cols.pkl
- `models/saved_models/hourly/` - 3 models + preprocessors
- `models/saved_models/15min/` - 3 models + preprocessors

#### 2. Daily Predictions (`predict_daily.yml`)
**Schedule:** Every day at 6 PM UTC  
**Duration:** ~20 seconds  
**Purpose:** Generate 1-day, 3-day, 7-day predictions

```yaml
on:
  schedule:
    - cron: '0 18 * * *'  # Daily at 6 PM UTC
  workflow_dispatch:       # Manual trigger
```

**What it does:**
1. Loads pre-trained daily models
2. Fetches latest Yahoo Finance data
3. Generates predictions for 3 horizons
4. Commits `daily_predictions.csv`

#### 3. Intraday Predictions (`predict_intraday.yml`)
**Schedule:** Every 15 minutes  
**Duration:** ~40 seconds  
**Purpose:** Generate hourly and 15-minute predictions

```yaml
on:
  schedule:
    - cron: '*/15 * * * *'  # Every 15 minutes
  workflow_dispatch:         # Manual trigger
```

**What it does:**
1. Loads pre-trained hourly and 15-min models
2. Fetches CoinGecko hourly data
3. Fetches Binance 15-minute data
4. Generates both prediction files
5. Commits both CSVs

### Manual Workflow Triggering

To manually trigger any workflow:
1. Go to: https://github.com/ying-jeanne/capstone_msba/actions
2. Click on the workflow name
3. Click "Run workflow" button (right side)
4. Select branch (usually "main")
5. Click "Run workflow"

Use manual triggers for:
- Testing workflows after setup
- Forcing predictions outside schedule
- Debugging workflow issues

### Monitoring Workflows

**View Workflow Runs:**
1. Go to Actions tab: https://github.com/ying-jeanne/capstone_msba/actions
2. See list of all workflow runs
3. Green ✅ = Success, Red ❌ = Failed, Yellow ⚠️ = Running

**View Logs:**
1. Click on any workflow run
2. Click on the job name (e.g., "predict-daily")
3. Expand steps to see detailed logs
4. Look for errors in red text

**Email Notifications:**
GitHub automatically sends emails when workflows fail. Check your GitHub notification settings.

### Common Issues

**Issue:** Workflow not appearing in Actions tab  
**Solution:** Make sure `.github/workflows/` directory exists and YAML files are committed

**Issue:** Permission denied when committing  
**Solution:** GitHub Actions has write permissions by default for public repos. For private repos, go to Settings → Actions → General → Workflow permissions → Select "Read and write permissions"

**Issue:** Training workflow takes too long  
**Solution:** 15-20 minutes is normal. GitHub free tier allows up to 6 hours per job.

**Issue:** Intraday workflow runs too frequently  
**Solution:** Cron `*/15` means every 15 minutes. With academic license, this is unlimited and free.

---

## Deployment to PythonAnywhere

### Step-by-Step Deployment

#### 1. Create PythonAnywhere Account
1. Go to: https://www.pythonanywhere.com/
2. Click "Pricing & signup"
3. Select "Create a Beginner account" (Free)
4. Complete registration

#### 2. Upload Code to PythonAnywhere

**Option A: Git Clone (Recommended)**
1. Open "Consoles" tab → Start a "Bash" console
2. Clone your repository:
   ```bash
   git clone https://github.com/ying-jeanne/capstone_msba.git
   cd capstone_msba
   ```

**Option B: Upload ZIP**
1. Compress your project locally
2. Go to "Files" tab
3. Upload ZIP file
4. Extract in bash console:
   ```bash
   unzip capstone_msba.zip
   cd capstone_msba
   ```

#### 3. Install Dependencies
```bash
# In the bash console
cd capstone_msba
pip3.11 install --user -r requirements.txt
```

#### 4. Configure Web App
1. Go to "Web" tab
2. Click "Add a new web app"
3. Select "Manual configuration"
4. Choose "Python 3.11"
5. Click through to create app

#### 5. Set Up WSGI File
1. In "Web" tab, click on WSGI configuration file link
2. Delete all contents
3. Replace with:
```python
import sys
import os

# Add your project directory to the sys.path
project_home = '/home/YOUR_USERNAME/capstone_msba'
if project_home not in sys.path:
    sys.path = [project_home] + sys.path

# Set up working directory
os.chdir(project_home)

# Import Flask app
from webapp.app import app as application
```
4. Replace `YOUR_USERNAME` with your PythonAnywhere username
5. Save file (Ctrl+S or Cmd+S)

#### 6. Set Static Files
1. In "Web" tab, scroll to "Static files" section
2. Add entries:
   - URL: `/static/` → Directory: `/home/YOUR_USERNAME/capstone_msba/webapp/static/`
3. Save

#### 7. Update Flask App to Use prediction_loader.py

Edit `webapp/app.py` to load predictions from GitHub:

```python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify
from utils.prediction_loader import PredictionLoader

app = Flask(__name__)
loader = PredictionLoader()

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/live')
def live():
    # Load predictions from GitHub (with caching)
    daily = loader.load_predictions('daily')
    hourly = loader.load_predictions('hourly')
    intraday = loader.load_predictions('15min')
    
    return render_template('live.html', 
                         daily_predictions=daily,
                         hourly_predictions=hourly,
                         intraday_predictions=intraday)

@app.route('/api/predictions/<timeframe>')
def api_predictions(timeframe):
    """API endpoint for fetching predictions"""
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
    app.run(debug=True)
```

#### 8. Reload Web App
1. Go to "Web" tab
2. Click big green "Reload" button
3. Wait ~10 seconds
4. Visit your site: `https://YOUR_USERNAME.pythonanywhere.com`

### How Predictions Stay Updated

**The Magic: No Git Operations Needed! 🎉**

1. **GitHub Actions** automatically:
   - Updates predictions every 15 minutes
   - Commits CSVs to your repository

2. **PythonAnywhere webapp** automatically:
   - Fetches predictions from GitHub raw URLs
   - Uses smart caching (30s–5min depending on timeframe)
   - Falls back to local files if GitHub unavailable
   - Never needs to do `git pull` or any git operations

3. **What you need to do:**
   - Nothing! It's fully automated once set up ✅

**Cache Durations (in config.py):**
- 15-minute predictions: 30 seconds cache
- Hourly predictions: 2 minutes cache
- Daily predictions: 5 minutes cache

This means:
- Website always shows recent data
- No need to manually update anything
- No git operations on PythonAnywhere
- Minimal API requests to GitHub

### PythonAnywhere Limitations (Free Tier)

| Feature | Free Tier Limit | Your Usage | Status |
|---------|----------------|------------|--------|
| Storage | 500 MB | ~100 MB (models + data) | ✅ OK |
| Always-on tasks | 0 | 0 (GitHub Actions does work) | ✅ OK |
| Outbound API calls | Whitelist only | GitHub.com (whitelisted) | ✅ OK |
| CPU time | 100 sec/day | ~1 sec per request | ✅ OK |
| Bandwidth | Unlimited | Low (just CSV fetches) | ✅ OK |

**Why this works perfectly:**
- No scheduled tasks needed on PythonAnywhere
- GitHub Actions does all heavy lifting
- Webapp only fetches small CSV files
- GitHub.com is on PythonAnywhere whitelist

---

## Troubleshooting

### Local Issues

**Problem:** `ModuleNotFoundError` when running scripts  
**Solution:**
```bash
pip install -r requirements.txt
```

**Problem:** `FileNotFoundError: models/saved_models/daily/...`  
**Solution:** Train models locally first:
```bash
python utils/train_daily_models.py
python utils/train_hourly_models.py
python utils/train_15min_models.py
```

**Problem:** Data fetching fails  
**Solution:** Check internet connection. APIs used:
- Yahoo Finance: Free, no API key needed
- CoinGecko: Free tier, rate limit 10-50 calls/min
- Binance: Free, no API key needed

### GitHub Actions Issues

**Problem:** Workflow doesn't trigger on schedule  
**Solution:** 
- Wait up to 15 minutes after push for first trigger
- Repository must have activity in last 60 days
- Manually trigger once to "wake up" scheduled workflows

**Problem:** Permission denied when committing  
**Solution:**
1. Go to repo Settings → Actions → General
2. Scroll to "Workflow permissions"
3. Select "Read and write permissions"
4. Click "Save"

**Problem:** Training workflow fails with "out of memory"  
**Solution:** 
- Reduce model complexity in training scripts
- GitHub Actions has 7 GB RAM, should be sufficient
- Check if data fetching is downloading too much

**Problem:** Prediction files not updating  
**Solution:**
1. Check workflow run logs in Actions tab
2. Look for errors in "Commit predictions" step
3. Verify models exist in repository
4. Manually trigger workflow to test

### PythonAnywhere Issues

**Problem:** 502 Bad Gateway  
**Solution:**
1. Check error log in "Web" tab → "Error log"
2. Common causes:
   - Wrong path in WSGI file
   - Import errors in Flask app
   - Missing dependencies
3. Fix and click "Reload"

**Problem:** Static files (CSS/images) not loading  
**Solution:**
1. Check "Static files" mapping in "Web" tab
2. Verify paths are absolute
3. Make sure files exist at those paths
4. Reload web app

**Problem:** Predictions not loading  
**Solution:**
1. Check that `config.py` has correct repo URL
2. Verify predictions exist in GitHub repository
3. Check webapp error logs
4. Test GitHub raw URL manually in browser:
   ```
   https://raw.githubusercontent.com/ying-jeanne/capstone_msba/main/data/predictions/daily_predictions.csv
   ```

**Problem:** "DisallowedHost" error  
**Solution:**
- PythonAnywhere handles this automatically
- If you see this, add to Flask app:
  ```python
  app.config['SERVER_NAME'] = None
  ```

### System Health Check

Run this anytime to check system status:
```bash
python check_system.py
```

This checks:
- ✅ Configuration (GitHub repo URL)
- ✅ Trained models existence
- ✅ Prediction files existence
- ✅ GitHub Actions workflows
- ✅ Python dependencies
- ✅ GitHub raw URL accessibility

---

## Presentation Tips

### Demo Script (5 minutes)

**1. Show the Problem (30 seconds)**
- "Bitcoin is volatile and hard to predict"
- "Traders need reliable forecasts across multiple timeframes"

**2. Show Your Solution (1 minute)**
- "Built ML system with 9 XGBoost models"
- "Predicts short-term (15 min) to long-term (7 days)"
- "Updates automatically every 15 minutes"

**3. Live Demo (2 minutes)**
- Open your website: `https://yourname.pythonanywhere.com/live`
- Show real-time predictions updating
- Explain the 3 timeframes
- Show confidence intervals

**4. Show the Architecture (1 minute)**
- "Fully automated with GitHub Actions"
- "Trains weekly, predicts every 15 minutes"
- "Deployed for free on PythonAnywhere"
- Show GitHub Actions tab running

**5. Show Results (30 seconds)**
- Show backtesting metrics
- Accuracy, MAPE, direction accuracy
- Compare to baseline

### Questions You Might Get

**Q: How accurate are your predictions?**  
A: "Our models achieve X% direction accuracy and Y% MAPE on test data. For short-term predictions (15-min to 1-hour), we focus on direction rather than exact price."

**Q: Why XGBoost?**  
A: "Tested multiple models (LSTM, Random Forest, XGBoost). XGBoost gave best balance of accuracy, speed, and interpretability. Can train all 9 models in under 20 minutes."

**Q: How do you handle Bitcoin's volatility?**  
A: "We use return-based predictions (% change) rather than absolute prices. Also include technical indicators, volatility measures, and rolling statistics as features."

**Q: Is this deployable in production?**  
A: "Yes! Already deployed. System runs 24/7 with automated updates. Fully serverless using GitHub Actions. Cost: $0/month."

**Q: What would you do differently?**  
A: "Consider adding:
- Sentiment analysis from news/Twitter
- Multi-asset predictions (ETH, altcoins)
- Trading strategy backtester
- More sophisticated ensemble methods"

### Presentation Slides Outline

**Slide 1:** Title + Team  
**Slide 2:** Problem Statement  
**Slide 3:** Solution Overview  
**Slide 4:** Data Sources & Features  
**Slide 5:** Model Architecture  
**Slide 6:** System Architecture (diagram)  
**Slide 7:** Live Demo (screenshot)  
**Slide 8:** Results & Metrics  
**Slide 9:** Deployment & Costs  
**Slide 10:** Future Work  
**Slide 11:** Q&A

---

## Next Steps

After completing setup:

1. ✅ **Monitor first week**
   - Check GitHub Actions runs daily
   - Verify predictions update correctly
   - Watch for any failures

2. ✅ **Update Flask webapp**
   - Integrate `prediction_loader.py`
   - Add nice visualizations
   - Make prediction tables interactive

3. ✅ **Deploy to PythonAnywhere**
   - Follow deployment section above
   - Test thoroughly

4. ✅ **Prepare presentation**
   - Create slides
   - Practice demo
   - Prepare for questions

5. ✅ **Optional enhancements**
   - Add team page (as discussed)
   - Add more visualizations
   - Implement backtesting UI
   - Add confidence intervals to charts

---

## Support Resources

- **GitHub Actions Docs:** https://docs.github.com/en/actions
- **PythonAnywhere Help:** https://help.pythonanywhere.com/
- **Flask Docs:** https://flask.palletsprojects.com/
- **XGBoost Docs:** https://xgboost.readthedocs.io/

---

**Good luck with your capstone project! 🚀**
