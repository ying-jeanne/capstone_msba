"""
Feature Engineering Module
Transform raw OHLCV data into 50+ predictive features for Bitcoin price prediction

This module creates a rich feature space by combining:
1. Technical indicators (RSI, MACD, Bollinger Bands, etc.)
2. Lag features (historical values)
3. Rolling statistics (moving averages, volatility)
4. Returns and momentum (price changes)
5. Time-based features (cyclical encoding)
6. Interaction features (feature combinations)

Why feature engineering matters:
- Raw OHLCV data alone is limited (only 5 features)
- Technical indicators capture market patterns and psychology
- Multiple timeframes (short/medium/long) capture different trends
- Lag features give the model "memory" of recent prices
- More features = better model performance (if done right)
"""

import pandas as pd
import numpy as np
import ta
from datetime import datetime

def engineer_technical_features(df):
    """
    Engineer 50+ technical features from OHLCV data

    This function transforms raw price/volume data into a comprehensive
    feature set for machine learning models. Each feature captures different
    aspects of market behavior, trend, momentum, and volatility.

    Args:
        df (pd.DataFrame): DataFrame with columns:
            - timestamp (datetime index)
            - open, high, low, close, volume

    Returns:
        pd.DataFrame: Original data + 50+ engineered features

    Features Created:
        - 20+ technical indicators (RSI, MACD, BB, ATR, etc.)
        - 10 lag features (historical prices/volumes)
        - 7 rolling statistics (moving averages, std, min/max)
        - 3 return calculations
        - 2 momentum features
        - 4 time-based features (cyclical encoding)
        - 2 interaction features
        - 7+ volume-based features

    Note:
        - NaN values will be present in early rows (from rolling/lag)
        - Call dropna() after this function to remove incomplete rows
        - Lost rows = max(rolling_window, lag_period) at the start
    """

    print("\n" + "=" * 70)
    print("  FEATURE ENGINEERING - CREATING 50+ FEATURES")
    print("=" * 70)

    # Make a copy to avoid modifying original
    df = df.copy()

    print(f"\n📊 Input data shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")

    original_columns = len(df.columns)

    # =========================================================================
    # 1. TECHNICAL INDICATORS
    # =========================================================================

    print("\n1️⃣  Calculating Technical Indicators...")

    # --- RSI (Relative Strength Index) ---
    # Measures overbought/oversold conditions (0-100)
    # < 30 = oversold (potential buy), > 70 = overbought (potential sell)
    # Multiple periods capture short/medium/long-term momentum
    dataset_size = len(df)
    print(f"   • RSI (Relative Strength Index) - adapting to dataset size ({dataset_size} rows)")
    df['rsi_10'] = ta.momentum.rsi(df['close'], window=10)    # Short-term
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)    # Standard
    df['rsi_30'] = ta.momentum.rsi(df['close'], window=30)    # Medium-term

    # Only add RSI_200 if we have enough data
    if dataset_size >= 250:  # Need some margin above 200
        df['rsi_200'] = ta.momentum.rsi(df['close'], window=200)  # Long-term trend
    else:
        print(f"      ⚠️  Skipping rsi_200 (requires 250+ rows, have {dataset_size})")

    # --- MACD (Moving Average Convergence Divergence) ---
    # Shows relationship between two moving averages
    # macd > signal = bullish, macd < signal = bearish
    # Crossovers indicate trend changes
    print("   • MACD (Trend & Momentum) - 3 components")
    macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()  # Histogram (macd - signal)

    # --- Exponential Moving Averages (EMA) ---
    # Smoothed price trends, more weight on recent prices
    # Price > EMA = uptrend, Price < EMA = downtrend
    # Multiple periods capture different trend timescales
    print("   • EMA (Exponential Moving Averages)")
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)   # Short
    df['ema_30'] = ta.trend.ema_indicator(df['close'], window=30)   # Medium

    # Only add EMA_200 if we have enough data
    if dataset_size >= 250:
        df['ema_200'] = ta.trend.ema_indicator(df['close'], window=200) # Long
    else:
        print(f"      ⚠️  Skipping ema_200 (requires 250+ rows, have {dataset_size})")

    # --- Bollinger Bands ---
    # Price envelope based on standard deviation
    # Shows volatility and potential reversal points
    # Price near upper band = overbought, near lower band = oversold
    print("   • Bollinger Bands - 4 components")
    bollinger = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_width'] = bollinger.bollinger_wband()  # Width indicates volatility

    # --- Average True Range (ATR) ---
    # Measures volatility (average of true ranges)
    # High ATR = high volatility, Low ATR = low volatility
    # Helps with position sizing and stop-loss placement
    print("   • ATR (Average True Range) - Volatility measure")
    df['atr_14'] = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], window=14
    )

    # --- Stochastic Oscillator ---
    # Compares closing price to price range over period
    # K > 80 = overbought, K < 20 = oversold
    # K/D crossovers signal trend changes
    print("   • Stochastic Oscillator - 2 components")
    stoch = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'], window=14, smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    # --- Volume Indicators ---
    # Volume confirms price movements
    # Price up + volume up = strong trend
    # Price up + volume down = weak trend (potential reversal)

    # Check if volume data is available
    has_volume = 'volume' in df.columns and df['volume'].notna().sum() > 0

    if has_volume:
        print("   • Volume Indicators - 3 components")
        df['volume_ema_10'] = ta.trend.ema_indicator(df['volume'], window=10)
        df['volume_ema_30'] = ta.trend.ema_indicator(df['volume'], window=30)
        df['volume_ratio'] = df['volume'] / (df['volume_ema_30'] + 1e-10)  # Avoid div by 0
    else:
        print("   ⚠️  Skipping volume indicators (volume data not available)")
        # Drop volume column if it's all NaN
        if 'volume' in df.columns:
            df = df.drop('volume', axis=1)

    print(f"   ✓ Added {len(df.columns) - original_columns} technical indicators")

    # =========================================================================
    # 2. LAG FEATURES (Historical Values)
    # =========================================================================

    print("\n2️⃣  Creating Lag Features (Historical Values)...")

    # Lag features give the model "memory" of recent prices
    # LSTM models learn patterns from sequences, but traditional ML needs explicit lags
    # Multiple lags capture patterns at different time steps back

    print("   • Price lags (1-7 periods back)")
    for lag in [1, 2, 3, 5, 7]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)

    lag_features = 5  # 5 price lags

    if has_volume:
        print("   • Volume lags (1-3 periods back)")
        for lag in [1, 2, 3]:
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        lag_features += 3  # Add 3 volume lags

    print(f"   ✓ Added {lag_features} lag features")

    # =========================================================================
    # 3. ROLLING STATISTICS
    # =========================================================================

    print("\n3️⃣  Calculating Rolling Statistics...")

    # Rolling features capture trends and volatility over windows
    # Multiple windows (7, 30, 90) capture short/medium/long-term patterns

    print("   • Rolling means (moving averages)")
    df['close_rolling_mean_7'] = df['close'].rolling(window=7).mean()
    df['close_rolling_mean_30'] = df['close'].rolling(window=30).mean()
    df['close_rolling_mean_90'] = df['close'].rolling(window=90).mean()

    print("   • Rolling standard deviations (volatility)")
    df['close_rolling_std_7'] = df['close'].rolling(window=7).std()
    df['close_rolling_std_30'] = df['close'].rolling(window=30).std()

    print("   • Rolling min/max (support/resistance)")
    df['close_rolling_min_30'] = df['close'].rolling(window=30).min()
    df['close_rolling_max_30'] = df['close'].rolling(window=30).max()

    rolling_features = 7
    print(f"   ✓ Added {rolling_features} rolling statistics")

    # =========================================================================
    # 4. RETURNS & MOMENTUM
    # =========================================================================

    print("\n4️⃣  Calculating Returns and Momentum...")

    # --- Returns (Price Changes) ---
    # Returns normalize price changes (% change vs absolute change)
    # More meaningful for ML than raw prices
    # Log returns are symmetric and more suitable for modeling
    print("   • Return calculations")
    df['returns'] = df['close'].pct_change()  # (close_t - close_t-1) / close_t-1
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))  # log(P_t/P_t-1)
    df['returns_7d'] = df['close'].pct_change(periods=7)  # 7-period cumulative return

    # --- Momentum ---
    # Raw price change over N periods
    # Captures trend strength
    print("   • Momentum features")
    df['momentum_10'] = df['close'] - df['close'].shift(10)
    df['momentum_30'] = df['close'] - df['close'].shift(30)

    return_features = 3 + 2  # 3 returns + 2 momentum
    print(f"   ✓ Added {return_features} return/momentum features")

    # =========================================================================
    # 5. TIME-BASED FEATURES (Cyclical Encoding)
    # =========================================================================

    print("\n5️⃣  Creating Time-based Features...")

    # Cyclical encoding ensures continuity:
    # - Hour 23 should be close to Hour 0 (not far away)
    # - Monday should be close to Sunday
    # Using sin/cos creates smooth cyclical features

    # Note: Only applicable if timestamp has time component
    if hasattr(df.index, 'hour'):
        print("   • Hour features (cyclical encoding)")
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        hour_features = 2
    else:
        hour_features = 0

    # Day of week (0=Monday, 6=Sunday)
    if hasattr(df.index, 'dayofweek'):
        print("   • Day of week features (cyclical encoding)")
        df['day_of_week_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        day_features = 2
    else:
        day_features = 0

    time_features = hour_features + day_features
    if time_features > 0:
        print(f"   ✓ Added {time_features} time-based features")
    else:
        print("   ⚠️  No time features (data is daily frequency)")

    # =========================================================================
    # 6. INTERACTION FEATURES
    # =========================================================================

    print("\n6️⃣  Creating Interaction Features...")

    # Interaction features capture relationships between features
    # Often reveal patterns that single features miss

    interaction_features = 0

    if has_volume:
        print("   • Price-Volume interaction")
        df['price_volume_interaction'] = df['close'] * df['volume']
        interaction_features += 1

    print("   • RSI-MACD interaction")
    df['rsi_macd_interaction'] = df['rsi_14'] * df['macd_diff']
    interaction_features += 1

    print(f"   ✓ Added {interaction_features} interaction features")

    # =========================================================================
    # 7. ADDITIONAL USEFUL FEATURES
    # =========================================================================

    print("\n7️⃣  Creating Additional Features...")

    # High-Low range (intraday volatility)
    df['high_low_range'] = df['high'] - df['low']

    # Close position in High-Low range (0=at low, 1=at high)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

    # Price distance from moving averages (trend strength)
    df['price_to_ema_30_ratio'] = df['close'] / (df['ema_30'] + 1e-10)

    additional_features = 3

    # Only add price_to_ema_200_ratio if we have ema_200
    if 'ema_200' in df.columns:
        df['price_to_ema_200_ratio'] = df['close'] / (df['ema_200'] + 1e-10)
        additional_features += 1

    print(f"   ✓ Added {additional_features} additional features")

    # =========================================================================
    # 8. HANDLE NaN VALUES
    # =========================================================================

    print("\n8️⃣  Handling NaN Values...")

    print(f"\n   Before dropna: {df.shape[0]} rows")
    rows_before = len(df)

    # Count NaN per column (for reporting)
    nan_counts = df.isnull().sum()
    nan_columns = nan_counts[nan_counts > 0]

    if len(nan_columns) > 0:
        print(f"\n   Columns with NaN values:")
        for col, count in nan_columns.head(5).items():
            print(f"      {col}: {count} NaN values")

    # Drop rows with any NaN values
    # NaNs come from:
    # - Rolling windows (first N rows don't have full window)
    # - Lag features (first N rows don't have historical data)
    # - Indicator calculations (require minimum periods)
    df = df.dropna()

    rows_after = len(df)
    rows_dropped = rows_before - rows_after

    print(f"\n   After dropna: {df.shape[0]} rows")
    print(f"   Rows dropped: {rows_dropped} ({rows_dropped/rows_before*100:.1f}%)")

    if rows_dropped > 0:
        print(f"\n   ℹ️  Dropped rows are from the start of dataset")
        print(f"      (due to rolling windows and lag features)")

    # =========================================================================
    # 9. SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("  FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    total_features = len(df.columns)
    new_features = total_features - original_columns

    print(f"\n📊 Feature Summary:")
    print(f"   Original features: {original_columns}")
    print(f"   New features created: {new_features}")
    print(f"   Total features: {total_features}")

    print(f"\n📋 Feature Categories:")
    print(f"   • Technical Indicators: ~20")
    print(f"   • Lag Features: {lag_features}")
    print(f"   • Rolling Statistics: {rolling_features}")
    print(f"   • Returns & Momentum: {return_features}")
    print(f"   • Time-based Features: {time_features}")
    print(f"   • Interaction Features: {interaction_features}")
    print(f"   • Additional Features: {additional_features}")

    print(f"\n✅ Final dataset shape: {df.shape}")
    print(f"   Rows: {df.shape[0]:,}")
    print(f"   Columns: {df.shape[1]}")

    return df


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    """
    Test feature engineering on Bitcoin data

    This script:
    1. Loads raw OHLCV data
    2. Applies feature engineering
    3. Saves engineered features to CSV
    4. Shows sample output
    """

    print("=" * 70)
    print("  BITCOIN FEATURE ENGINEERING - TEST RUN")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. LOAD DATA
    # -------------------------------------------------------------------------

    print("\n📂 Loading data...")

    try:
        # Load Yahoo Finance data (best for daily features)
        df = pd.read_csv(
            'data/raw/btc_yahoo_2y_daily.csv',
            index_col='timestamp',
            parse_dates=True
        )
        print(f"   ✓ Loaded: data/raw/btc_yahoo_2y_daily.csv")
        data_source = "yahoo_2y_daily"

    except FileNotFoundError:
        try:
            # Fallback to Binance data
            df = pd.read_csv(
                'data/raw/btc_binance_15min.csv',
                index_col='timestamp',
                parse_dates=True
            )
            print(f"   ✓ Loaded: data/raw/btc_binance_15min.csv")
            data_source = "binance_15min"

        except FileNotFoundError:
            print("\n   ❌ Error: No data files found!")
            print("      Please run: python utils/data_fetcher.py")
            exit(1)

    print(f"\n   Original data shape: {df.shape}")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")
    print(f"   Columns: {list(df.columns)}")

    # -------------------------------------------------------------------------
    # 2. FEATURE ENGINEERING
    # -------------------------------------------------------------------------

    print("\n🔧 Applying feature engineering...")

    df_features = engineer_technical_features(df)

    # -------------------------------------------------------------------------
    # 3. SAVE RESULTS
    # -------------------------------------------------------------------------

    print("\n💾 Saving results...")

    output_file = f'data/processed/btc_{data_source}_features.csv'

    # Create processed directory if needed
    import os
    os.makedirs('data/processed', exist_ok=True)

    # Save to CSV
    df_features.to_csv(output_file)

    file_size = os.path.getsize(output_file) / 1024  # KB

    print(f"   ✓ Saved to: {output_file}")
    print(f"   File size: {file_size:.2f} KB")

    # -------------------------------------------------------------------------
    # 4. DISPLAY SAMPLE
    # -------------------------------------------------------------------------

    print("\n📊 Sample of Engineered Features:")
    print("=" * 70)

    # Show first 5 rows of key features
    sample_cols = [
        'close', 'rsi_14', 'macd', 'ema_30', 'bb_mid',
        'volume_ratio', 'returns', 'momentum_10'
    ]

    # Only show columns that exist
    sample_cols = [col for col in sample_cols if col in df_features.columns]

    print(df_features[sample_cols].head())

    print("\n" + "=" * 70)
    print("  Feature List (First 20):")
    print("=" * 70)

    for i, col in enumerate(df_features.columns[:20], 1):
        print(f"   {i:2d}. {col}")

    if len(df_features.columns) > 20:
        print(f"   ... and {len(df_features.columns) - 20} more features")

    # -------------------------------------------------------------------------
    # 5. STATISTICS
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("  FEATURE STATISTICS")
    print("=" * 70)

    # Check for any remaining NaN
    nan_check = df_features.isnull().sum().sum()
    print(f"\n✓ NaN values remaining: {nan_check}")

    # Check for infinite values
    inf_check = np.isinf(df_features.select_dtypes(include=[np.number])).sum().sum()
    print(f"✓ Infinite values: {inf_check}")

    # Feature value ranges (for first 10 features)
    print("\n📈 Feature Value Ranges (sample):")
    for col in df_features.columns[5:10]:  # Skip OHLCV, show engineered
        if df_features[col].dtype in [np.float64, np.int64]:
            print(f"   {col:30s}: [{df_features[col].min():.2f}, {df_features[col].max():.2f}]")

    # -------------------------------------------------------------------------
    # 6. RECOMMENDATIONS
    # -------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("  NEXT STEPS")
    print("=" * 70)

    print("\n💡 Your feature-engineered data is ready!")
    print("\n   1. Load the features:")
    print(f"      df = pd.read_csv('{output_file}', index_col='timestamp', parse_dates=True)")
    print("\n   2. Merge with sentiment data:")
    print("      fng = pd.read_csv('data/raw/fear_greed_index.csv', ...)")
    print("      df = df.join(fng, how='left')")
    print("\n   3. Normalize features:")
    print("      from sklearn.preprocessing import StandardScaler")
    print("      scaler = StandardScaler()")
    print("      X_scaled = scaler.fit_transform(df)")
    print("\n   4. Create sequences for LSTM:")
    print("      from utils.preprocessing import DataPreprocessor")
    print("      X, y = preprocessor.create_sequences(data, sequence_length=60)")
    print("\n   5. Build your model!")

    print("\n" + "=" * 70)
    print("  FEATURE ENGINEERING COMPLETE! ✅")
    print("=" * 70)
