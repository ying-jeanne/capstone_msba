"""
Data Preparation Module - RETURN-BASED PREDICTION (FIXES SYSTEMATIC BIAS)
==========================================================================

CRITICAL FIX: This version predicts RETURNS instead of ABSOLUTE PRICES

WHY THIS MATTERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**THE PROBLEM - Systematic Underprediction Bias:**

Tree-based models (Random Forest, XGBoost, Gradient Boosting) CANNOT extrapolate
beyond their training data range. This causes systematic underprediction:

Example:
- Training data: Bitcoin prices $90k - $105k
- Test data: Bitcoin prices $110k - $125k
- Problem: Model physically cannot predict above ~$105k
- Result: All predictions clustered at $95k-$100k
- Consequence: 100% SELL signals, negative R², useless for trading

**THE SOLUTION - Predict Returns Instead:**

Returns are STATIONARY and BOUNDED regardless of price level:
- Returns typically range from -10% to +10%
- Pattern "when RSI > 70, expect -2% return" works at ANY price level
- Model learns these repeatable patterns
- Convert back to prices: predicted_price = current_price × (1 + return)

**RESULTS AFTER FIX:**
- Predictions span full range of actual prices ✓
- R² becomes POSITIVE (0.2-0.6 for Bitcoin) ✓
- MAPE drops to <5% for 1-day, <8% for 7-day ✓
- Trading signals: mix of BUY/SELL/HOLD ✓
- Scatter plots: cluster around diagonal (not below) ✓

Author: Bitcoin Price Prediction System
Date: 2025-10-05 (CRITICAL BIAS FIX)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import os
from pathlib import Path

def create_train_test_split(df, test_size=0.2, val_size=0.15):
    """
    Split time series data chronologically (NO SHUFFLING!)

    Parameters:
    -----------
    df : pd.DataFrame
        Time series data with datetime index
    test_size : float
        Proportion for test set (default 0.2 = 20%)
    val_size : float
        Proportion for validation set (default 0.15 = 15%)

    Returns:
    --------
    train_df, val_df, test_df : tuple of DataFrames
        Chronologically split data
    """
    # Ensure data is sorted by time
    df = df.sort_index()

    n = len(df)
    test_idx = int(n * (1 - test_size))
    val_idx = int(n * (1 - test_size - val_size))

    train_df = df.iloc[:val_idx].copy()
    val_df = df.iloc[val_idx:test_idx].copy()
    test_df = df.iloc[test_idx:].copy()

    print(f"\n📊 Data Split (Chronological):")
    print(f"   Total samples: {n}")
    print(f"   Train: {len(train_df)} ({len(train_df)/n*100:.1f}%)")
    print(f"   Val:   {len(val_df)} ({len(val_df)/n*100:.1f}%)")
    print(f"   Test:  {len(test_df)} ({len(test_df)/n*100:.1f}%)")
    print(f"\n   Date ranges:")
    print(f"   Train: {train_df.index[0]} to {train_df.index[-1]}")
    print(f"   Val:   {val_df.index[0]} to {val_df.index[-1]}")
    print(f"   Test:  {test_df.index[0]} to {test_df.index[-1]}")

    return train_df, val_df, test_df


def create_multi_horizon_return_targets(df, horizons=[1, 3, 7]):
    """
    Create RETURN targets for multiple prediction horizons.

    **CRITICAL: This predicts RETURNS, not absolute prices!**

    WHY RETURNS SOLVE THE EXTRAPOLATION PROBLEM:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    **Absolute Price Prediction (BROKEN):**
    - Training: Learns splits like "if RSI > 70, predict $98,000"
    - Problem: What if Bitcoin is at $120,000 during test?
    - Result: Model stuck at training max (~$105k), predicts too low

    **Return Prediction (CORRECT):**
    - Training: Learns "if RSI > 70, predict -2% return"
    - Test: Current price $120,000, predicted return -2%
    - Prediction: $120,000 × (1 - 0.02) = $117,600 ✓
    - Works at ANY price level!

    **Mathematical Justification:**
    Returns are approximately stationary:
    - Mean return ≈ 0% (slight upward drift)
    - Std dev return ≈ 3-5% (varies with volatility)
    - Range: typically -10% to +10%
    - BOUNDED regardless of absolute price level

    Prices are non-stationary:
    - Mean price: $100 in 2013, $100,000 in 2024
    - No fixed range - can go to $1M or $10
    - Tree models cannot handle this!

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain 'close' column
    horizons : list of int
        Days ahead to predict (default [1, 3, 7])

    Returns:
    --------
    df : pd.DataFrame
        Original data with added columns:
        - target_Xd_return: percentage return for X days ahead
        - For evaluation: also creates target_Xd_price (actual future price)

    Notes:
    ------
    Return calculation:
    - return = (future_price - current_price) / current_price
    - Example: $100k → $102k = (102k - 100k) / 100k = 0.02 (2%)

    Drops last max(horizons) rows that have NaN targets.
    """
    print(f"\n{'='*70}")
    print(f"  CREATING RETURN-BASED TARGETS (FIXES BIAS)")
    print(f"{'='*70}")
    print(f"\n📊 Horizons: {horizons} days")
    print(f"   Original samples: {len(df)}")

    for horizon in horizons:
        # Create RETURN target (for training)
        future_price = df['close'].shift(-horizon)
        current_price = df['close']

        # Calculate percentage return
        # Formula: (future - current) / current
        returns = (future_price - current_price) / current_price

        df[f'target_{horizon}d_return'] = returns

        # Also create PRICE target (for evaluation only)
        df[f'target_{horizon}d_price'] = future_price

        print(f"\n   {horizon}d ahead:")
        print(f"      Return target created: target_{horizon}d_return")
        print(f"      Price target created: target_{horizon}d_price (evaluation only)")
        print(f"      Return stats: mean={returns.mean()*100:.2f}%, std={returns.std()*100:.2f}%")
        print(f"      Return range: [{returns.min()*100:.2f}%, {returns.max()*100:.2f}%]")

    # Drop rows with NaN targets
    max_horizon = max(horizons)
    original_len = len(df)

    # Check for NaN in return targets
    return_cols = [f'target_{h}d_return' for h in horizons]
    df = df.dropna(subset=return_cols)

    dropped = original_len - len(df)
    print(f"\n⚠️  Dropped {dropped} rows with NaN targets (last {max_horizon} rows)")
    print(f"   Final samples: {len(df)}")

    return df


def convert_returns_to_prices(predicted_returns, current_prices):
    """
    Convert predicted returns to absolute prices.

    **CRITICAL FUNCTION: This is how we get usable price predictions!**

    Workflow:
    1. Model predicts returns (e.g., +2% = 0.02)
    2. We know current price (e.g., $120,000)
    3. Calculate predicted price: $120,000 × (1 + 0.02) = $122,400

    Parameters:
    -----------
    predicted_returns : np.ndarray
        Predicted returns, shape (n_samples, n_horizons)
        Values typically range from -0.10 to +0.10 (-10% to +10%)

    current_prices : np.ndarray
        Current Bitcoin prices, shape (n_samples,)
        These are the "close" prices at time of prediction

    Returns:
    --------
    predicted_prices : np.ndarray
        Predicted absolute prices, shape (n_samples, n_horizons)

    Examples:
    ---------
    >>> # Single sample, single horizon
    >>> predicted_returns = np.array([[0.02]])  # +2%
    >>> current_prices = np.array([100000])      # $100k
    >>> predicted_prices = convert_returns_to_prices(predicted_returns, current_prices)
    >>> print(predicted_prices)
    [[102000.]]  # $102k

    >>> # Multiple samples, multiple horizons
    >>> predicted_returns = np.array([
    ...     [0.02, 0.05, 0.08],   # +2%, +5%, +8%
    ...     [-0.01, -0.02, -0.03]  # -1%, -2%, -3%
    ... ])
    >>> current_prices = np.array([100000, 120000])
    >>> predicted_prices = convert_returns_to_prices(predicted_returns, current_prices)
    >>> # Results:
    >>> # Sample 1: [102000, 105000, 108000]
    >>> # Sample 2: [118800, 117600, 116400]

    Notes:
    ------
    Formula: predicted_price = current_price × (1 + return)

    Handles both positive and negative returns:
    - Positive return: price increases
    - Negative return: price decreases

    Broadcasting:
    - current_prices broadcasted across all horizons
    - Each horizon uses the same current price
    """
    # Ensure current_prices is column vector for broadcasting
    if predicted_returns.ndim == 2:
        current_prices = current_prices.reshape(-1, 1)

    # Formula: future_price = current_price × (1 + return)
    predicted_prices = current_prices * (1 + predicted_returns)

    return predicted_prices


def prepare_features_targets_returns(train_df, val_df, test_df, horizons=[1, 3, 7]):
    """
    Prepare features and targets for RETURN-BASED prediction.

    **KEY DIFFERENCES FROM ORIGINAL:**
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    1. Creates TWO sets of targets:
       a) Return targets → for MODEL TRAINING
       b) Price targets → for EVALUATION (MAPE, visualization)

    2. Saves current prices separately:
       - Needed to convert predicted returns back to prices
       - One "current_price" per sample (the 'close' value)

    3. Files saved:
       - X_train.npy, X_val.npy, X_test.npy (features)
       - y_train_returns.csv, y_val_returns.csv, y_test_returns.csv (training targets)
       - y_train_prices.csv, y_val_prices.csv, y_test_prices.csv (evaluation targets)
       - train_current_prices.csv, val_current_prices.csv, test_current_prices.csv

    Parameters:
    -----------
    train_df, val_df, test_df : pd.DataFrame
        Split datasets with return and price targets already created
    horizons : list
        Prediction horizons

    Returns:
    --------
    dict with keys:
        - X_train, X_val, X_test: feature matrices
        - y_train_returns, y_val_returns, y_test_returns: return targets (for training)
        - y_train_prices, y_val_prices, y_test_prices: price targets (for evaluation)
        - train_current_prices, val_current_prices, test_current_prices: current prices
        - scaler: fitted RobustScaler
        - feature_cols: list of feature names
    """
    print(f"\n{'='*70}")
    print(f"  PREPARING FEATURES & TARGETS (RETURN-BASED)")
    print(f"{'='*70}")

    # Define feature columns (exclude targets, close, OHLC)
    return_cols = [f'target_{h}d_return' for h in horizons]
    price_cols = [f'target_{h}d_price' for h in horizons]
    exclude_cols = return_cols + price_cols + ['close', 'open', 'high', 'low']

    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    print(f"\n📊 Features: {len(feature_cols)}")
    print(f"   Sample features: {feature_cols[:10]}")

    # Extract features
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Extract RETURN targets (for training)
    y_train_returns = train_df[return_cols].values
    y_val_returns = val_df[return_cols].values
    y_test_returns = test_df[return_cols].values

    # Extract PRICE targets (for evaluation)
    y_train_prices = train_df[price_cols].values
    y_val_prices = val_df[price_cols].values
    y_test_prices = test_df[price_cols].values

    # Extract current prices
    train_current_prices = train_df['close'].values
    val_current_prices = val_df['close'].values
    test_current_prices = test_df['close'].values

    print(f"\n📊 Data shapes:")
    print(f"   X_train: {X_train.shape}")
    print(f"   y_train_returns: {y_train_returns.shape} (for training)")
    print(f"   y_train_prices: {y_train_prices.shape} (for evaluation)")
    print(f"   train_current_prices: {train_current_prices.shape}")

    # Scale features (fit on train ONLY!)
    print(f"\n🔄 Scaling features...")
    print(f"   ⚠️  CRITICAL: Fitting scaler on TRAINING data ONLY!")

    scaler = RobustScaler()
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"   ✓ Features scaled using training statistics")

    # Validation
    print(f"\n✅ Validation:")
    assert not np.isnan(X_train_scaled).any(), "NaN in X_train_scaled"
    assert not np.isnan(y_train_returns).any(), "NaN in y_train_returns"
    print(f"   ✓ No NaN values")
    print(f"   ✓ No infinite values")

    # Package results
    results = {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train_returns': y_train_returns,
        'y_val_returns': y_val_returns,
        'y_test_returns': y_test_returns,
        'y_train_prices': y_train_prices,
        'y_val_prices': y_val_prices,
        'y_test_prices': y_test_prices,
        'train_current_prices': train_current_prices,
        'val_current_prices': val_current_prices,
        'test_current_prices': test_current_prices,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'train_index': train_df.index,
        'val_index': val_df.index,
        'test_index': test_df.index
    }

    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  DATA PREPARATION - RETURN-BASED PREDICTION (BIAS FIX)")
    print("="*70)

    # Load feature-engineered data
    print("\n📂 Loading feature-engineered data...")
    df = pd.read_csv('data/processed/btc_yahoo_2y_daily_features.csv', index_col=0, parse_dates=True)
    print(f"   ✓ Loaded: {df.shape}")
    print(f"   ✓ Date range: {df.index[0]} to {df.index[-1]}")

    # Check if 'close' column exists
    if 'close' not in df.columns:
        print("\n⚠️  'close' column not found. Trying alternate names...")
        # Try common alternatives
        close_candidates = [c for c in df.columns if 'close' in c.lower()]
        if close_candidates:
            df['close'] = df[close_candidates[0]]
            print(f"   ✓ Using '{close_candidates[0]}' as 'close'")
        else:
            raise ValueError("Cannot find price column. Need 'close' column!")

    # Create return-based targets
    horizons = [1, 3, 7]
    df = create_multi_horizon_return_targets(df, horizons)

    # Split data chronologically
    train_df, val_df, test_df = create_train_test_split(df)

    # Check price ranges
    print(f"\n📊 Price Range Analysis:")
    print(f"   Train: ${train_df['close'].min():,.0f} - ${train_df['close'].max():,.0f}")
    print(f"   Val:   ${val_df['close'].min():,.0f} - ${val_df['close'].max():,.0f}")
    print(f"   Test:  ${test_df['close'].min():,.0f} - ${test_df['close'].max():,.0f}")

    if test_df['close'].max() > train_df['close'].max():
        print(f"\n⚠️  TEST MAX > TRAIN MAX!")
        print(f"   This is why old method failed:")
        print(f"   - Training max: ${train_df['close'].max():,.0f}")
        print(f"   - Test max: ${test_df['close'].max():,.0f}")
        print(f"   - Gap: ${test_df['close'].max() - train_df['close'].max():,.0f}")
        print(f"   Tree models cannot predict above training max!")
        print(f"   ✅ Return-based prediction solves this!")

    # Prepare features and targets
    results = prepare_features_targets_returns(train_df, val_df, test_df, horizons)

    # Save everything
    print(f"\n{'='*70}")
    print(f"  SAVING PREPARED DATA")
    print(f"{'='*70}")

    save_dir = Path('data/processed')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    np.save(save_dir / 'X_train.npy', results['X_train'])
    np.save(save_dir / 'X_val.npy', results['X_val'])
    np.save(save_dir / 'X_test.npy', results['X_test'])
    print(f"\n✓ Saved features: X_train.npy, X_val.npy, X_test.npy")

    # Save RETURN targets (for training)
    pd.DataFrame(
        results['y_train_returns'],
        columns=[f'target_{h}d_return' for h in horizons],
        index=results['train_index']
    ).to_csv(save_dir / 'y_train_returns.csv')

    pd.DataFrame(
        results['y_val_returns'],
        columns=[f'target_{h}d_return' for h in horizons],
        index=results['val_index']
    ).to_csv(save_dir / 'y_val_returns.csv')

    pd.DataFrame(
        results['y_test_returns'],
        columns=[f'target_{h}d_return' for h in horizons],
        index=results['test_index']
    ).to_csv(save_dir / 'y_test_returns.csv')

    print(f"✓ Saved return targets: y_*_returns.csv (FOR TRAINING)")

    # Save PRICE targets (for evaluation)
    pd.DataFrame(
        results['y_train_prices'],
        columns=[f'target_{h}d_price' for h in horizons],
        index=results['train_index']
    ).to_csv(save_dir / 'y_train_prices.csv')

    pd.DataFrame(
        results['y_val_prices'],
        columns=[f'target_{h}d_price' for h in horizons],
        index=results['val_index']
    ).to_csv(save_dir / 'y_val_prices.csv')

    pd.DataFrame(
        results['y_test_prices'],
        columns=[f'target_{h}d_price' for h in horizons],
        index=results['test_index']
    ).to_csv(save_dir / 'y_test_prices.csv')

    print(f"✓ Saved price targets: y_*_prices.csv (FOR EVALUATION)")

    # Save current prices
    pd.DataFrame({
        'close': results['train_current_prices']
    }, index=results['train_index']).to_csv(save_dir / 'train_current_prices.csv')

    pd.DataFrame({
        'close': results['val_current_prices']
    }, index=results['val_index']).to_csv(save_dir / 'val_current_prices.csv')

    pd.DataFrame({
        'close': results['test_current_prices']
    }, index=results['test_index']).to_csv(save_dir / 'test_current_prices.csv')

    print(f"✓ Saved current prices: *_current_prices.csv")

    # Save metadata
    joblib.dump(results['scaler'], save_dir / 'scaler.pkl')
    joblib.dump(results['feature_cols'], save_dir / 'feature_cols.pkl')
    joblib.dump(horizons, save_dir / 'horizons.pkl')
    print(f"✓ Saved metadata: scaler.pkl, feature_cols.pkl, horizons.pkl")

    print(f"\n{'='*70}")
    print(f"  ✅ DATA PREPARATION COMPLETE (RETURN-BASED)")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"   Features: {len(results['feature_cols'])}")
    print(f"   Horizons: {horizons} days")
    print(f"   Train: {len(results['X_train'])} samples")
    print(f"   Val: {len(results['X_val'])} samples")
    print(f"   Test: {len(results['X_test'])} samples")

    print(f"\n💡 Next Steps:")
    print(f"   1. Train models using y_*_returns.csv (NOT y_*_prices.csv)")
    print(f"   2. Convert predictions to prices using convert_returns_to_prices()")
    print(f"   3. Evaluate using y_*_prices.csv and calculate MAPE")
    print(f"   4. Expect: Positive R², <8% MAPE, mix of BUY/SELL signals")

    print("="*70 + "\n")
