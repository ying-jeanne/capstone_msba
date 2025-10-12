# 🚀 QUICK START GUIDE

## Get Your Live Prediction System Running in 15 Minutes!

### Prerequisites
- Python 3.11+ installed
- GitHub account with academic license
- Git installed

---

## Step-by-Step Setup

### 1️⃣ Configure GitHub Repository (2 minutes)

Open `config.py` and update line 11:

```python
# Change this:
GITHUB_REPO = "YOUR_USERNAME/YOUR_REPO_NAME"

# To your actual repo (get it from your GitHub URL):
# Example: If your repo is https://github.com/john-doe/bitcoin-prediction
GITHUB_REPO = "john-doe/bitcoin-prediction"
```

**How to find your repo name:**
1. Go to your GitHub repository
2. Look at the URL: `https://github.com/USERNAME/REPO-NAME`
3. Copy `USERNAME/REPO-NAME`

---

### 2️⃣ Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train Models (10 minutes)

Run these three commands (they'll take about 10-15 minutes total):

```bash
# Train daily models (1d, 3d, 7d predictions)
python utils/train_daily_models.py

# Train hourly models (1h, 6h, 24h predictions)
python utils/train_hourly_models.py

# Train 15-min models (15m, 1h, 4h predictions)
python utils/train_15min_models.py
```

You should see output like:
```
==================================================================
  DAILY MODEL TRAINING PIPELINE
==================================================================
[STEP 1] Fetching Yahoo Finance data (2 years daily)...
✓ Fetched 730 daily bars
...
✓ Training complete!
```

---

### 4️⃣ Test the System (1 minute)

Run the health check:

```bash
python check_system.py
```

You should see mostly green checkmarks ✓. A few warnings are OK, but no red X's!

---

### 5️⃣ Generate First Predictions (Optional - 1 minute)

Test prediction generation locally:

```bash
# Generate daily predictions
python utils/predict_daily.py

# Generate hourly + 15-min predictions
python utils/predict_hourly_and_15min.py
```

---

### 6️⃣ Commit Everything to GitHub (2 minutes)

```bash
# Add all files
git add .

# Commit
git commit -m "Setup: Add live prediction system with trained models"

# Push to GitHub
git push origin main
```

---

### 7️⃣ Enable GitHub Actions (1 minute)

1. Go to your GitHub repository
2. Click the **"Actions"** tab at the top
3. If you see a message, click **"I understand my workflows, go ahead and enable them"**
4. Done! Your workflows are now active

---

### 8️⃣ Test GitHub Actions (2 minutes)

Trigger a workflow manually to test:

1. Go to **Actions** tab
2. Click **"Update Hourly and 15-Min Predictions"** on the left
3. Click **"Run workflow"** button (top right)
4. Select branch: `main`
5. Click green **"Run workflow"** button
6. Wait ~1 minute
7. Refresh page - you should see a green checkmark ✓

---

### 9️⃣ Verify Predictions on GitHub (1 minute)

After workflow runs:

1. Go to your repo main page
2. Navigate to: `data/predictions/`
3. You should see:
   - `hourly_predictions.csv`
   - `15min_predictions.csv`
4. Click on a file to view predictions!

---

### 🎉 Done! Your System is Live!

Your predictions will now update automatically:
- ⚡ **Every 15 minutes**: Hourly + 15-min predictions
- 📊 **Every day at 6 PM**: Daily predictions  
- 🤖 **Every Sunday at 2 AM**: Model retraining

---

## 📱 Access Your Predictions

### Option 1: Direct GitHub URLs

Your predictions are available at:
```
https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-REPO/main/data/predictions/daily_predictions.csv
https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-REPO/main/data/predictions/hourly_predictions.csv
https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-REPO/main/data/predictions/15min_predictions.csv
```

Replace `YOUR-USERNAME` and `YOUR-REPO` with your actual values!

### Option 2: Test Locally with Python

```python
from utils.prediction_loader import load_predictions

# Load daily predictions
daily = load_predictions('daily')
print(daily)

# Load hourly predictions
hourly = load_predictions('hourly')
print(hourly)

# Load 15-min predictions
min15 = load_predictions('15min')
print(min15)
```

---

## 🌐 Deploy Flask Webapp (Next Step)

Coming soon - we'll update your Flask app to display all predictions!

For now, you can test locally:
```bash
cd webapp
python app.py
```

Visit: http://localhost:5000

---

## 🆘 Troubleshooting

### "GitHub URL not configured" error
```bash
# Make sure you updated config.py with your actual GitHub repo
# Format: "username/repo-name" (no https://, no .git)
```

### Workflows not running automatically
```bash
# Make sure you pushed .github/workflows/ folder to GitHub
git add .github/workflows/
git commit -m "Add workflows"
git push origin main
```

### "Model not found" errors
```bash
# Make sure you ran all three training scripts
# And committed the models/ folder
git add models/
git commit -m "Add trained models"
git push origin main
```

### Python package errors
```bash
# Reinstall all requirements
pip install -r requirements.txt --upgrade
```

---

## 📊 Monitor Your System

### Check workflow status:
- Go to **Actions** tab on GitHub
- See all runs and their status
- Click on any run to see detailed logs

### Check predictions:
- Browse `data/predictions/` folder on GitHub
- See commit history for updates
- View CSV files directly in browser

---

## ✅ Success Checklist

After setup, you should have:

- [x] `config.py` updated with your GitHub repo
- [x] All dependencies installed
- [x] Models trained and saved in `models/saved_models/`
- [x] `check_system.py` shows all green ✓
- [x] Code pushed to GitHub
- [x] GitHub Actions enabled
- [x] First workflow ran successfully
- [x] Predictions visible in `data/predictions/`

---

## 🎓 What You Built

Congratulations! You now have:

✅ **9 trained XGBoost models** (daily, hourly, 15-min)
✅ **3 automated GitHub Actions workflows**
✅ **Predictions updating every 15 minutes**
✅ **Professional production ML system**
✅ **Zero-cost infrastructure** (GitHub Actions + PythonAnywhere free tier)
✅ **Perfect capstone demo!**

This is the same architecture used by fintech companies for production trading systems! 🚀

---

## 📚 Next Steps

1. ✅ System is live - predictions updating automatically
2. 📱 Update Flask webapp to display predictions (coming soon)
3. 🌐 Deploy to PythonAnywhere
4. 🎨 Polish UI for presentation
5. 📊 Add performance tracking dashboard
6. 🎓 Prepare capstone presentation

---

## 💡 Pro Tips

### View prediction files easily:
```bash
# Daily predictions
cat data/predictions/daily_predictions.csv

# Hourly predictions
cat data/predictions/hourly_predictions.csv

# 15-min predictions
cat data/predictions/15min_predictions.csv
```

### Check when workflows run:
- Daily: Every day at 6 PM UTC (1 PM EST / 10 AM PST)
- Intraday: Every 15 minutes (96 times per day!)
- Training: Every Sunday at 2 AM UTC (Saturday 9 PM EST / 6 PM PST)

### Monitor GitHub Actions usage:
- Go to Settings → Billing → Usage this month
- With academic license, you have unlimited minutes for public repos!

---

**Need help?** Check `SYSTEM_OVERVIEW.md` for detailed documentation!

**Ready to deploy?** See `GITHUB_ACTIONS_SETUP.md` for deployment guide!

🎉 **Happy Predicting!** 🚀📈
