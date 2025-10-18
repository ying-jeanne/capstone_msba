# GitHub Actions Workflow Timing - Optimized ⚡

## Summary of Changes

**Old Schedule**: All workflows ran at 2 PM UTC
**New Schedule**: All workflows run at ~1:30 AM UTC
**Time Saved**: **12.5 hours** - predictions available much earlier!

---

## New Daily Timeline

```
┌─────────────────────────────────────────────────────────────┐
│  00:00 UTC - Bitcoin Daily Close Recorded                   │
│              (Oct 17 close price now available)             │
└─────────────────────────────────────────────────────────────┘
                         ↓
                   [1 hour buffer]
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  01:00 UTC - Update Outcomes Workflow ✅                     │
│              - Fetches Oct 17 actual close from Yahoo       │
│              - Updates Oct 16 prediction with real outcome  │
│              - Calculates MAPE, errors, direction           │
│              - Commits to prediction_tracking_demo.csv      │
└─────────────────────────────────────────────────────────────┘
                         ↓
                   [30 min buffer]
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  01:30 UTC - Generate Predictions Workflow ✅                │
│              - Fetches Oct 17 close + Fear & Greed          │
│              - Generates Oct 18, 20, 24 predictions         │
│              - Saves to daily_predictions.csv               │
│              - Commits predictions to repo                  │
└─────────────────────────────────────────────────────────────┘
                         ↓
                   [10 min buffer]
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  01:40 UTC - Store Predictions On-Chain ✅                   │
│              - Reads latest prediction from CSV             │
│              - Stores on Moonbase Alpha blockchain          │
│              - Transaction hash saved to tracking files     │
│              - Commits blockchain data to repo              │
└─────────────────────────────────────────────────────────────┘
```

---

## Workflow Details

### 1. Update Outcomes (`update_outcomes.yml`)
**Schedule**: `0 1 * * *` (01:00 UTC daily)

**What it does**:
- Fetches historical Bitcoin prices from Yahoo Finance
- Updates predictions with actual outcomes (1d, 3d, 7d)
- Calculates MAPE and directional accuracy
- Updates `prediction_tracking_demo.csv`

**Why 1 AM UTC?**
- Bitcoin close recorded at 00:00 UTC
- 1-hour buffer ensures Yahoo Finance API is updated
- No dependency on Fear & Greed (uses historical price only)

---

### 2. Generate Predictions (`predict_daily.yml`)
**Schedule**: `30 1 * * *` (01:30 UTC daily)

**What it does**:
- Fetches latest Bitcoin OHLCV data (Yahoo Finance)
- Fetches Fear & Greed Index (historical data from API)
- Engineers 58 technical + 3 sentiment features
- Generates 1d, 3d, 7d predictions using XGBoost
- Saves to `daily_predictions.csv` and history file

**Why 1:30 AM UTC?**
- ✅ Bitcoin close available since 00:00 UTC (90 min ago)
- ✅ Fear & Greed API provides **historical data** (yesterday's sentiment)
- ✅ 30-min buffer after outcomes workflow prevents conflicts
- ❌ NO need to wait for 08:00 UTC publication (we use historical API)

**Key Insight**: Fear & Greed Index API returns historical data, so we don't need to wait for today's publication at 08:00 UTC. We fetch yesterday's sentiment from the API, which is already available.

---

### 3. Store On Blockchain (`store_predictions_onchain.yml`)
**Schedule**: `40 1 * * *` (01:40 UTC daily)

**What it does**:
- Reads latest prediction from `daily_predictions.csv`
- Stores on Moonbase Alpha blockchain via smart contract
- Updates CSV files with transaction hash and block number
- Commits tracking files to repository

**Why 1:40 AM UTC?**
- 10-min buffer ensures prediction file is committed by predict_daily workflow
- Avoids race conditions reading uncommitted files
- Still completes by 2 AM UTC (fast turnaround)

---

## Benefits of New Schedule

### 1. ⚡ **12.5 Hours Faster**
- **Old**: Predictions available at 2:10 PM UTC
- **New**: Predictions available at 1:40 AM UTC
- **Improvement**: Results ready **12.5 hours earlier**

### 2. 🎯 **Better User Experience**
- Users wake up to fresh predictions (instead of waiting until afternoon)
- Webapp shows updated data by morning
- Blockchain transactions completed overnight

### 3. 🔧 **No Race Conditions**
- Each workflow has buffer time before next one runs
- Sequential execution: Outcomes → Predictions → Blockchain
- No git conflicts from simultaneous commits

### 4. 📊 **Data Freshness**
- Outcomes updated just 1 hour after close
- Predictions use most recent close + sentiment
- Blockchain storage happens immediately after prediction

### 5. ✅ **No Timing Dependencies**
- Fear & Greed API provides historical data (no 08:00 UTC wait)
- Yahoo Finance reliable 1 hour after midnight
- All data sources available by 1 AM UTC

---

## FAQ

### Q: Why not run even earlier (like 00:30 UTC)?
**A**: Yahoo Finance API may have propagation delays. 1-hour buffer (01:00 UTC) is conservative and reliable.

### Q: Don't we need to wait for Fear & Greed at 08:00 UTC?
**A**: No! The Fear & Greed API returns **historical data**. When we run at 01:30 UTC on Oct 18, we fetch Oct 17's sentiment (which was published yesterday at 08:00 UTC on Oct 17).

### Q: Is there data leakage risk running earlier?
**A**: No. We use:
- Oct 17 close (available since Oct 18 00:00 UTC)
- Oct 17 sentiment (available from API history)
- To predict Oct 18 close (not available until Oct 19 00:00 UTC)

### Q: What if Yahoo Finance is delayed?
**A**: The 1-hour buffer handles typical API delays. If needed, can increase to 2 hours (02:00 UTC start).

### Q: Can I manually trigger workflows for testing?
**A**: Yes! All workflows have `workflow_dispatch` enabled. Go to Actions tab → Select workflow → "Run workflow"

---

## Testing the New Schedule

### Option 1: Wait for Automatic Run
- Next run: Tomorrow at 01:00 UTC
- Check GitHub Actions tab for results

### Option 2: Manual Trigger (Recommended)
1. Go to: https://github.com/ying-jeanne/capstone_msba/actions
2. Select workflow (e.g., "Update Daily Predictions")
3. Click "Run workflow" → "Run workflow"
4. Monitor execution in real-time

### Option 3: Local Simulation
```bash
# Simulate the full pipeline locally
python update_outcomes.py          # Step 1: Update outcomes
python utils/predict_daily.py      # Step 2: Generate predictions
python utils/store_daily_prediction_onchain.py  # Step 3: Store on blockchain
```

---

## Rollback Plan (If Needed)

If the new schedule causes issues, revert by changing cron schedules back:

```yaml
# update_outcomes.yml
- cron: '0 14 * * *'  # Back to 2 PM UTC

# predict_daily.yml
- cron: '0 14 * * *'  # Back to 2 PM UTC

# store_predictions_onchain.yml
- cron: '10 14 * * *'  # Back to 2:10 PM UTC
```

---

## Files Changed

| File | Old Schedule | New Schedule |
|------|--------------|--------------|
| `.github/workflows/update_outcomes.yml` | `0 14 * * *` | `0 1 * * *` |
| `.github/workflows/predict_daily.yml` | `0 14 * * *` | `30 1 * * *` |
| `.github/workflows/store_predictions_onchain.yml` | `10 14 * * *` | `40 1 * * *` |

---

## Monitoring

After deployment, monitor:
- ✅ All workflows complete successfully
- ✅ Predictions appear in `daily_predictions.csv`
- ✅ Blockchain transactions confirmed on Moonscan
- ✅ Webapp shows updated predictions
- ✅ No errors in GitHub Actions logs

---

**Status**: ✅ Deployed and Active
**Last Updated**: October 18, 2025
**Next Scheduled Run**: Tomorrow at 01:00 UTC
