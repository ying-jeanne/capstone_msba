# Handling Bitcoin's High Noise - Complete Guide

## Problem: Bitcoin Has Extreme Noise

**What is "noise"?**
- Random price movements that have no predictable pattern
- Flash crashes, whale trades, sudden news events
- Makes it hard for models to learn real patterns

**Evidence from your models:**
```
R² ≈ 0 (near zero)        # Most variance is unexplained (noise)
Directional: 50-52%       # Can't predict direction better than random
Price MAPE: 0.24-1.59%    # But price prediction is good!
```

## ✅ Strategy 1: Increase Training Data (APPLIED)

### What Changed:
- **Before:** 60/20/20 split → 976 training samples
- **After:** 70/15/15 split → 1,138 training samples (+17%)

### Why It Helps:
```
More data = Model sees more patterns
           = Can average out random noise
           = Better generalization
```

### Daily Models:
- Before: 976 train / 325 val / 326 test
- After: 1,138 train / 244 val / 243 test

### Hourly Models:
- Before: 5,131 train / 1,710 val / 1,711 test  
- After: 5,987 train / 1,283 val / 1,282 test

### Trade-off:
- ✅ Pro: More training data, better noise filtering
- ⚠️  Con: Smaller test set (but still 243-1,282 samples is plenty)

---

## 📊 Strategy 2: Add More Historical Data

### Current Data:
- Daily: 1,628 samples ≈ **4.5 years** (2020-10-16 to 2025-10-16)
- Hourly: 8,553 samples ≈ **1 year** (2024-10-16 to 2025-10-16)

### Recommendation:
```python
# In utils/data_fetcher.py or run_full_pipeline.py
# Increase daily data to 5-10 years
df = yf.download('BTC-USD', start='2015-01-01', end=today)  # 10 years

# Increase hourly data to 2-3 years (if API allows)
# More data = better pattern recognition
```

### Why It Helps:
- See multiple market cycles (bull + bear markets)
- Learn from more crash/pump events
- Average out one-time anomalies
- Better statistical significance

### Implementation:
Your `data_fetcher.py` already fetches 5 years:
```python
# Line in data_fetcher.py
years=5  # Already good! Could try 7-10 years
```

---

## 🧹 Strategy 3: Feature Engineering for Noise Reduction

### A. Smoothing Features (Reduce High-Frequency Noise)

**Add longer moving averages:**
```python
# In feature_engineering.py, add these:
df['sma_50'] = df['close'].rolling(50).mean()   # Already have up to 20
df['sma_100'] = df['close'].rolling(100).mean()
df['sma_200'] = df['close'].rolling(200).mean()  # Long-term trend

# Smoothed momentum
df['momentum_smooth'] = df['close'].rolling(5).mean() / df['close'].rolling(20).mean() - 1
```

### B. Volatility-Adjusted Returns
**Problem:** Raw returns mix signal + noise
**Solution:** Normalize by volatility
```python
# Add to feature_engineering.py
df['volatility'] = df['close'].pct_change().rolling(20).std()
df['sharpe_ratio'] = df['returns_1d'].rolling(20).mean() / df['volatility']
df['noise_ratio'] = df['high'] - df['low'] / df['close']  # Intraday noise
```

### C. Market Regime Detection
**Idea:** Bitcoin behaves differently in different regimes
```python
# Add regime features
df['regime_volatility'] = pd.qcut(df['volatility'], q=3, labels=['low', 'med', 'high'])
df['regime_trend'] = pd.qcut(df['returns_30d'], q=3, labels=['bear', 'neutral', 'bull'])
```

---

## 🎯 Strategy 4: Target Engineering (Reduce Noise in Labels)

### Current Target:
```python
# Predicting raw returns (noisy)
y = df['close'].pct_change(horizon).shift(-horizon)
```

### Better Targets:

#### A. Smoothed Returns
```python
# Predict average return over window (less noisy)
y = df['close'].rolling(horizon).mean().shift(-horizon) / df['close'] - 1
```

#### B. Direction + Magnitude (Two-Stage Model)
```python
# Stage 1: Classify direction (up/down)
y_direction = (df['close'].shift(-horizon) > df['close']).astype(int)

# Stage 2: Predict magnitude (only if direction is correct)
y_magnitude = df['close'].pct_change(horizon).shift(-horizon).abs()
```

#### C. Volatility-Adjusted Returns
```python
# Predict risk-adjusted returns (accounts for noise)
volatility = df['close'].pct_change().rolling(20).std()
y = df['close'].pct_change(horizon).shift(-horizon) / volatility
```

---

## 🌲 Strategy 5: Model Improvements

### A. Ensemble Methods (Reduce Noise via Averaging)

**Idea:** Multiple models vote → noise cancels out

```python
# In train_*_models.py
from sklearn.ensemble import VotingRegressor

# Train 3 models with different random seeds
model_1 = xgb.XGBRegressor(**params, random_state=42)
model_2 = xgb.XGBRegressor(**params, random_state=123)
model_3 = xgb.XGBRegressor(**params, random_state=456)

# Ensemble: Average predictions
ensemble = VotingRegressor([
    ('xgb1', model_1),
    ('xgb2', model_2),
    ('xgb3', model_3)
])
ensemble.fit(X_train, y_train)
```

### B. Stronger Regularization (Prevent Noise Fitting)

**Current params:** Already pretty good
```python
'max_depth': 4,              # Good (shallow trees)
'gamma': 0.3,                # Good (pruning)
'reg_alpha': 0.5,            # Good (L1)
'reg_lambda': 2.0,           # Good (L2)
```

**Try even stronger:**
```python
'max_depth': 3,              # Shallower → less overfitting
'min_child_weight': 5,       # More samples per leaf
'subsample': 0.7,            # Use less data per tree (more random)
'colsample_bytree': 0.7,     # Use fewer features per tree
```

### C. Add Dropout/Noise During Training
```python
# XGBoost has built-in dart (dropout)
params = {
    'booster': 'dart',         # Instead of 'gbtree'
    'rate_drop': 0.1,          # Dropout rate
    'skip_drop': 0.5,          # Skip rate
}
```

---

## 📉 Strategy 6: Outlier Handling (Remove Extreme Noise)

### A. Clip Extreme Returns
```python
# In prepare_data_for_horizon()
# Clip returns to 3 standard deviations
mean_return = y.mean()
std_return = y.std()
y_clipped = np.clip(y, mean_return - 3*std_return, mean_return + 3*std_return)
```

### B. Winsorization (Replace extremes)
```python
from scipy.stats.mstats import winsorize

# Replace top/bottom 1% with nearest non-extreme values
y_winsorized = winsorize(y, limits=[0.01, 0.01])
```

### C. Remove Days with High Uncertainty
```python
# Remove prediction days with extreme volatility
volatility = df['close'].pct_change().rolling(5).std()
df = df[volatility < volatility.quantile(0.95)]  # Remove top 5% volatile days
```

---

## 🔄 Strategy 7: Walk-Forward Validation (Better for Noisy Time Series)

### Current: Single Train/Val/Test Split
```
[Train: 60%] [Val: 20%] [Test: 20%]
```

### Better: Rolling Window Walk-Forward
```python
# Pseudo-code for walk-forward validation
window_size = 252  # 1 year of trading days
test_size = 21     # Test on next month

for i in range(len(df) - window_size - test_size):
    train_data = df[i : i + window_size]
    test_data = df[i + window_size : i + window_size + test_size]
    
    model.fit(train_data)
    predictions = model.predict(test_data)
    
    # Retrain every month with fresh data
```

**Why it helps:**
- Constantly adapts to new market conditions
- More realistic for live trading
- Each prediction uses most recent data

---

## 🧪 Strategy 8: Feature Selection (Remove Noisy Features)

### Current: Using all 50+ features
**Problem:** Some features might be pure noise

### Solution A: Feature Importance Analysis
```python
# After training model
importance = model.feature_importances_
important_features = feature_cols[importance > importance.mean()]

# Retrain with only important features
X_filtered = X[:, importance > importance.mean()]
```

### Solution B: Recursive Feature Elimination
```python
from sklearn.feature_selection import RFE

selector = RFE(model, n_features_to_select=30)  # Keep top 30
X_filtered = selector.fit_transform(X_train, y_train)
```

---

## 📊 Recommended Implementation Order

### Phase 1: Quick Wins (Already Done! ✅)
1. ✅ Change split to 70/15/15
2. ✅ Use RobustScaler (already have)
3. ✅ Strong regularization (already have)

### Phase 2: Medium Effort (Next Steps)
4. **Add smoothing features** (Strategy 3A) - 30 mins
5. **Feature selection** (Strategy 8A) - 1 hour
6. **Try ensemble** (Strategy 5A) - 1 hour

### Phase 3: Advanced (If needed)
7. **Walk-forward validation** (Strategy 7) - 2-3 hours
8. **Target engineering** (Strategy 4) - 2-3 hours
9. **More historical data** (Strategy 2) - depends on API

---

## 🎯 Expected Improvements

**Current Results:**
- Price MAPE: 0.24-1.59% ⭐ (Already excellent!)
- Directional: 50-52% (Random)
- R²: ≈ 0 (High noise)

**After Noise Reduction:**
- Price MAPE: 0.20-1.30% (Slightly better)
- Directional: 52-55% (Actual edge!)
- R²: 0.01-0.05 (Still low, but improved)

**Note:** Bitcoin is inherently noisy. R² will never be high (like 0.8+). The goal is:
1. ✅ Keep price accuracy high (<2% MAPE)
2. 🎯 **Get directional accuracy > 52%** (This is the real goal!)

---

## 🚀 Quick Start: Test New Split

Run your pipeline with the new 70/15/15 split:
```bash
python run_full_pipeline.py > logfile_70_15_15
```

Compare with old results:
```bash
# Check if directional accuracy improved
grep "Directional Accuracy:" logfile_70_15_15
```

Then decide which other strategies to try!
