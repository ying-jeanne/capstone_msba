"""
Quick Comparison: 3-Year vs 5-Year vs 11-Year Data
===================================================
Trains same model on different time periods and compares:
- Training performance
- Validation performance
- Test performance
- Overfitting metrics (train vs test gap)

Run this to determine optimal data period for Bitcoin prediction.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from pathlib import Path
import yfinance as yf
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.feature_engineering import engineer_technical_features, add_sentiment_features


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_acc = np.mean(direction_true == direction_pred) * 100

    return {'MAE': mae, 'MAPE': mape, 'R2': r2, 'Directional_Accuracy': directional_acc}


def train_and_evaluate(period_name, yahoo_period):
    """
    Train model on specific time period and return metrics

    Args:
        period_name: Display name (e.g., "3 years")
        yahoo_period: Yahoo Finance period string (e.g., "3y")

    Returns:
        dict: Metrics for this period
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING ON {period_name.upper()} DATA")
    print(f"{'='*70}")

    # Fetch data
    print(f"\nFetching {period_name} of Bitcoin data...")
    btc = yf.Ticker('BTC-USD')
    df = btc.history(period=yahoo_period)
    df.index = df.index.tz_localize(None)  # Remove timezone
    df = df.rename(columns=str.lower)
    df = df[['open', 'high', 'low', 'close', 'volume']]

    print(f"✓ Fetched {len(df)} days ({df.index[0].date()} to {df.index[-1].date()})")

    # Engineer features
    print("Engineering features...")
    df = engineer_technical_features(df)
    df = add_sentiment_features(df)
    print(f"✓ Created {len(df.columns)} features, {len(df)} samples after dropna")

    # Prepare data
    df['future_return_1d'] = df['close'].pct_change(1).shift(-1)
    df['current_price'] = df['close']
    df_clean = df.dropna(subset=['future_return_1d']).copy()

    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'current_price'] + \
                   [col for col in df_clean.columns if 'future_' in col]
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    X = df_clean[feature_cols].values
    y = df_clean['future_return_1d'].values
    current_prices = df_clean['current_price'].values

    # Scale
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Split 60/20/20
    split_1 = int(len(X_scaled) * 0.6)
    split_2 = int(len(X_scaled) * 0.8)

    X_train = X_scaled[:split_1]
    X_val = X_scaled[split_1:split_2]
    X_test = X_scaled[split_2:]

    y_train = y[:split_1]
    y_val = y[split_1:split_2]
    y_test = y[split_2:]

    prices_test = current_prices[split_2:]

    print(f"\nSplit: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # Train model
    print("Training XGBoost...")
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.3,
        'reg_alpha': 0.5,
        'reg_lambda': 2.0,
        'random_state': 42,
        'n_jobs': -1
    }

    # XGBoost 3.0+ uses early_stopping_rounds in params
    params['early_stopping_rounds'] = 20

    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Metrics
    train_metrics = calculate_metrics(y_train, y_pred_train)
    val_metrics = calculate_metrics(y_val, y_pred_val)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    # Price metrics
    pred_prices_test = prices_test * (1 + y_pred_test)
    actual_prices_test = prices_test * (1 + y_test)
    price_mape = mean_absolute_percentage_error(actual_prices_test, pred_prices_test) * 100

    # Overfitting
    train_val_gap = abs(train_metrics['MAPE'] - val_metrics['MAPE'])
    train_test_gap = abs(train_metrics['MAPE'] - test_metrics['MAPE'])

    # Print results
    print(f"\n✅ Results for {period_name}:")
    print(f"  Train MAPE: {train_metrics['MAPE']:.2f}%")
    print(f"  Val MAPE:   {val_metrics['MAPE']:.2f}%")
    print(f"  Test MAPE:  {test_metrics['MAPE']:.2f}%")
    print(f"  Test R²:    {test_metrics['R2']:.4f} {'✅' if test_metrics['R2'] > 0 else '❌'}")
    print(f"  Price MAPE: {price_mape:.2f}%")
    print(f"  Directional: {test_metrics['Directional_Accuracy']:.1f}%")
    print(f"  Overfitting: {train_test_gap:.2f}% {'✅ OK' if train_test_gap < 50 else '⚠️  HIGH'}")

    return {
        'period': period_name,
        'samples': len(df_clean),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'train_mape': train_metrics['MAPE'],
        'val_mape': val_metrics['MAPE'],
        'test_mape': test_metrics['MAPE'],
        'train_r2': train_metrics['R2'],
        'val_r2': val_metrics['R2'],
        'test_r2': test_metrics['R2'],
        'price_mape': price_mape,
        'directional_acc': test_metrics['Directional_Accuracy'],
        'train_test_gap': train_test_gap,
        'train_val_gap': train_val_gap
    }


def main():
    """Compare different time periods"""
    print("\n" + "="*70)
    print("  TIMEFRAME COMPARISON: 3Y vs 5Y vs 11Y")
    print("="*70)
    print("\nThis will train the same model on different data periods")
    print("to find the optimal timeframe for Bitcoin prediction.\n")

    results = []

    # Test each period
    periods = [
        ('3 years', '3y'),
        ('5 years', '5y'),
        ('11 years (max)', 'max')
    ]

    for period_name, yahoo_period in periods:
        try:
            metrics = train_and_evaluate(period_name, yahoo_period)
            results.append(metrics)
        except Exception as e:
            print(f"\n❌ Error with {period_name}: {e}")
            continue

    # Create comparison table
    if results:
        print("\n" + "="*70)
        print("  COMPARISON SUMMARY")
        print("="*70)

        df = pd.DataFrame(results)

        # Format for display
        print("\n📊 Performance Metrics:")
        print(df[['period', 'test_samples', 'test_mape', 'test_r2', 'price_mape', 'directional_acc']].to_string(index=False))

        print("\n🔍 Overfitting Analysis:")
        print(df[['period', 'train_mape', 'val_mape', 'test_mape', 'train_test_gap']].to_string(index=False))

        # Recommendations
        print("\n💡 Recommendations:")

        # Best R²
        best_r2 = df.loc[df['test_r2'].idxmax()]
        print(f"  • Best R² ({best_r2['test_r2']:.4f}): {best_r2['period']}")

        # Best MAPE
        best_mape = df.loc[df['test_mape'].idxmin()]
        print(f"  • Best MAPE ({best_mape['test_mape']:.2f}%): {best_mape['period']}")

        # Least overfitting
        best_overfit = df.loc[df['train_test_gap'].idxmin()]
        print(f"  • Least Overfitting ({best_overfit['train_test_gap']:.2f}%): {best_overfit['period']}")

        # Best directional
        best_dir = df.loc[df['directional_acc'].idxmax()]
        print(f"  • Best Directional ({best_dir['directional_acc']:.1f}%): {best_dir['period']}")

        # Overall recommendation
        print(f"\n🎯 Overall Recommendation:")
        # Count wins
        wins = {}
        for period in df['period']:
            wins[period] = 0
            if df[df['period'] == period]['test_r2'].values[0] == df['test_r2'].max():
                wins[period] += 1
            if df[df['period'] == period]['test_mape'].values[0] == df['test_mape'].min():
                wins[period] += 1
            if df[df['period'] == period]['train_test_gap'].values[0] == df['train_test_gap'].min():
                wins[period] += 1
            if df[df['period'] == period]['directional_acc'].values[0] == df['directional_acc'].max():
                wins[period] += 1

        best_overall = max(wins, key=wins.get)
        print(f"  → Use {best_overall} (wins {wins[best_overall]}/4 metrics)")

        # Save results
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        output_path = results_dir / 'timeframe_comparison.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved detailed results to: {output_path}")

    else:
        print("\n❌ No results to compare")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
