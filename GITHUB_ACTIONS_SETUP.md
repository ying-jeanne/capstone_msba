# GitHub Actions Setup Guide

## 🚀 Quick Setup (5 minutes)

### Step 1: Configure Your GitHub Repository

1. **Open `config.py`** in the root directory

2. **Update the GitHub repository information:**
   ```python
   # Change this line:
   GITHUB_REPO = "YOUR_USERNAME/YOUR_REPO_NAME"
   
   # To your actual repo, for example:
   GITHUB_REPO = "john-doe/capstone_bitcoin"
   ```

3. **Save the file**

### Step 2: Train Models Locally (First Time Only)

Before GitHub Actions can make predictions, you need trained models:

```bash
# Train all models (takes ~15-20 minutes)
python utils/train_daily_models.py
python utils/train_hourly_models.py
python utils/train_15min_models.py
```

This creates:
- `models/saved_models/daily/` - Daily models
- `models/saved_models/hourly/` - Hourly models
- `models/saved_models/15min/` - 15-minute models

### Step 3: Commit and Push Everything

```bash
git add .
git commit -m "Setup: Add models and prediction system"
git push origin main
```

### Step 4: Enable GitHub Actions

1. Go to your GitHub repository
2. Click on **"Actions"** tab
3. If prompted, click **"I understand my workflows, go ahead and enable them"**

That's it! GitHub Actions will now:
- ✅ Train models every Sunday at 2 AM
- ✅ Update daily predictions every day at 6 PM
- ✅ Update hourly + 15-min predictions every 15 minutes

---

## 📋 Workflows Explained

### 1. `train_models_weekly.yml`
- **Runs:** Every Sunday at 2 AM UTC
- **Duration:** ~20 minutes
- **What it does:**
  - Trains daily models (Yahoo 2-year data)
  - Trains hourly models (CoinGecko 60-day data)
  - Trains 15-min models (Binance 60-day data)
  - Commits updated models to repo

### 2. `predict_daily.yml`
- **Runs:** Every day at 6 PM UTC
- **Duration:** ~20 seconds
- **What it does:**
  - Fetches latest Yahoo Finance data
  - Generates 1d, 3d, 7d predictions
  - Commits to `data/predictions/daily_predictions.csv`

### 3. `predict_intraday.yml`
- **Runs:** Every 15 minutes
- **Duration:** ~40 seconds
- **What it does:**
  - Fetches latest CoinGecko hourly data
  - Fetches latest Binance 15-min data
  - Generates hourly predictions (1h, 6h, 24h)
  - Generates 15-min predictions (15m, 1h, 4h)
  - Commits to prediction CSVs

---

## 🧪 Testing Workflows Manually

You can trigger workflows manually to test them:

1. Go to **Actions** tab on GitHub
2. Select a workflow (e.g., "Update Hourly and 15-Min Predictions")
3. Click **"Run workflow"** button
4. Select branch (main)
5. Click **"Run workflow"**

---

## 📊 Monitoring

### Check if workflows are running:
1. Go to **Actions** tab
2. See recent workflow runs
3. Click on a run to see logs

### Check predictions:
Your predictions are automatically committed to:
- `data/predictions/daily_predictions.csv`
- `data/predictions/hourly_predictions.csv`
- `data/predictions/15min_predictions.csv`

You can view them directly on GitHub!

---

## 🔧 Troubleshooting

### "GitHub URL not configured" error
- Make sure you updated `GITHUB_REPO` in `config.py`
- Format should be: `"username/repository-name"`

### Workflow fails with "Model not found"
- Run training scripts locally first
- Commit and push the `models/saved_models/` directory

### "Failed to fetch data" errors
- API rate limits (temporary - will retry next run)
- Internet connectivity (GitHub Actions side - rare)

### Too many commits?
- This is normal! GitHub Actions commits predictions every 15 min
- These commits are small and won't cause issues
- You can squash them later if desired

---

## 💰 GitHub Actions Usage

With **Academic/Student license:**
- ✅ **Unlimited minutes** for public repos
- ✅ 3,000 minutes/month for private repos

**Estimated usage per month:**
- Weekly training: ~80 minutes (4 runs × 20 min)
- Daily predictions: ~10 minutes (30 runs × 0.3 min)
- 15-min predictions: ~1,200 minutes (2,880 runs × 0.4 min)
- **Total: ~1,290 minutes/month** (well within limits!)

---

## 🌐 Deploying Webapp

Your Flask webapp will automatically read predictions from GitHub raw URLs.

### Local Testing:
```bash
cd webapp
python app.py
# Visit http://localhost:5000
```

### Deploy to PythonAnywhere:
1. Upload your code to PythonAnywhere
2. Make sure `config.py` has correct GitHub repo
3. Your webapp will fetch predictions from GitHub automatically
4. No git pull needed - reads from raw GitHub URLs!

---

## 📝 Next Steps

1. ✅ Configure `config.py` with your GitHub repo
2. ✅ Train models locally
3. ✅ Commit and push everything
4. ✅ Enable GitHub Actions
5. ✅ Wait for first prediction run (or trigger manually)
6. ✅ Deploy your webapp to PythonAnywhere

---

## 🎓 Understanding the System

```
┌─────────────────────────────────────────┐
│  GitHub Actions (Cloud)                 │
│  - Trains models (weekly)               │
│  - Makes predictions (15min/hourly/daily)│
│  - Commits CSVs to repo                 │
└─────────────────────────────────────────┘
            ↓ (commits to repo)
┌─────────────────────────────────────────┐
│  GitHub Repository                      │
│  - models/saved_models/ (trained models)│
│  - data/predictions/*.csv (predictions) │
└─────────────────────────────────────────┘
            ↓ (raw URL fetch)
┌─────────────────────────────────────────┐
│  PythonAnywhere (Your Webapp)           │
│  - Reads CSVs from GitHub raw URLs      │
│  - Caches for 30-300 seconds            │
│  - Displays live predictions            │
└─────────────────────────────────────────┘
```

This architecture means:
- ✅ No heavy computation on PythonAnywhere
- ✅ No git operations needed
- ✅ Always fresh predictions
- ✅ Works with free tier perfectly

---

## 🆘 Need Help?

If something isn't working:
1. Check the **Actions** tab for error messages
2. Look at workflow logs for details
3. Make sure `config.py` is configured correctly
4. Verify models were trained and committed

Happy predicting! 🚀📈
