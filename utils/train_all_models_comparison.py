"""
Train Multiple Model Types for Comparison
==========================================
Trains XGBoost, Random Forest, and Gradient Boosting models for fair comparison
Tests on daily 1d, 3d, 7d horizons to see which algorithm performs best

Output:
- results/model_comparison.csv (comparison metrics)
- models/saved_models/comparison/ (all trained models)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from pathlib import Path
import joblib
import sys
import time
from catboost import CatBoostRegressor
import lightgbm
from lightgbm import LGBMRegressor

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.feature_engineering import engineer_technical_features, add_sentiment_features


def calculate_metrics(y_true, y_pred):
    """Calculate metrics for return predictions"""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    direction_true = np.sign(y_true)
    direction_pred = np.sign(y_pred)
    directional_acc = np.mean(direction_true == direction_pred) * 100
    
    return {
        'MAE': mae,
        'R2': r2,
        'Directional_Accuracy': directional_acc
    }


def prepare_data_for_horizon(df, horizon_days=1):
    """Prepare data for a specific prediction horizon"""
    # Create return target
    df[f'future_return_{horizon_days}d'] = df['close'].pct_change(horizon_days).shift(-horizon_days)
    df['current_price'] = df['close']
    
    # Drop rows with NaN targets
    df_clean = df.dropna(subset=[f'future_return_{horizon_days}d']).copy()
    
    # Select feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'current_price'] + \
                   [col for col in df_clean.columns if 'future_return' in col or 'future_price' in col]
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    X = df_clean[feature_cols].values
    y = df_clean[f'future_return_{horizon_days}d'].values
    current_prices = df_clean['current_price'].values
    
    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, feature_cols, scaler, current_prices


# def prepare_hourly_data(df, horizon_hours=1):
#     """Prepare hourly data for a specific prediction horizon."""
#     df[f'future_return_{horizon_hours}h'] = df['close'].pct_change(horizon_hours).shift(-horizon_hours)
#     df['current_price'] = df['close']

#     df_clean = df.dropna(subset=[f'future_return_{horizon_hours}h']).copy()

#     exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'current_price'] + \
#                    [col for col in df_clean.columns if 'future_return' in col or 'future_price' in col]
#     feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

#     X = df_clean[feature_cols].values
#     y = df_clean[f'future_return_{horizon_hours}h'].values
#     current_prices = df_clean['current_price'].values

#     scaler = RobustScaler()
#     X_scaled = scaler.fit_transform(X)

#     return X_scaled, y, feature_cols, scaler, current_prices


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model with regularization"""
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,    # L1 regularization
        reg_lambda=2.0,   # L2 regularization
        gamma=0.3,        # Min loss reduction
        random_state=42,
        tree_method='hist',
        early_stopping_rounds=20
    )
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - start_time
    
    return model, train_time


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return model, train_time


def train_gradient_boosting(X_train, y_train, X_val, y_val):
    """Train Gradient Boosting model"""
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        max_features='sqrt',
        random_state=42
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    return model, train_time


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model"""
    model = CatBoostRegressor(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=2.0,
        random_seed=42,
        verbose=False
    )
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=20,
        verbose=False
    )
    train_time = time.time() - start_time
    
    return model, train_time


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    model = LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1
    )
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lightgbm.early_stopping(20, verbose=False)]
    )
    train_time = time.time() - start_time
    
    return model, train_time


def evaluate_model(model, X_test, y_test, test_prices, model_name, horizon, timeframe):
    """Evaluate a trained model"""
    # Predict returns
    y_pred = model.predict(X_test)
    
    # Calculate return-based metrics
    return_metrics = calculate_metrics(y_test, y_pred)
    
    # Convert to prices for price-based metrics
    pred_prices = test_prices * (1 + y_pred)
    actual_prices = test_prices * (1 + y_test)
    
    price_mae = mean_absolute_error(actual_prices, pred_prices)
    price_mape = mean_absolute_percentage_error(actual_prices, pred_prices) * 100
    
    return {
        'Model': model_name,
        'Horizon': horizon,
        'Timeframe': timeframe,
        'Return_MAE': return_metrics['MAE'],
        'Return_R2': return_metrics['R2'],
        'Directional_Accuracy': return_metrics['Directional_Accuracy'],
        'Price_MAE': price_mae,
        'Price_MAPE': price_mape
    }


def main():
    """Train and compare all models"""
    print("=" * 80)
    print("TRAINING MODEL COMPARISON")
    print("=" * 80)
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    results_dir = base_dir / 'results'
    models_dir = base_dir / 'models' / 'saved_models' / 'comparison'
    
    results_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(data_dir / 'raw' / 'btc_yahoo_2y_daily.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"   Loaded {len(df)} rows")
    
    # Feature engineering
    print("\n2. Engineering features...")
    df = engineer_technical_features(df)
    
    # Set timestamp as index before adding sentiment
    df = df.set_index('timestamp')
    df = add_sentiment_features(df)
    df = df.reset_index()
    df = df.dropna()
    print(f"   After feature engineering: {len(df)} rows, {len(df.columns)} columns")
    
    # Train models for each horizon
    horizons = [1, 3, 7]
    all_results = []
    
    for horizon in horizons:
        print(f"\n{'=' * 80}")
        print(f"HORIZON: {horizon}-DAY PREDICTIONS")
        print(f"{'=' * 80}")
        
        # Prepare data
        print(f"\n3. Preparing data for {horizon}d horizon...")
        X, y, feature_cols, scaler, current_prices = prepare_data_for_horizon(df.copy(), horizon)
        
        # Train/val/test split (70/15/15)
        train_size = int(len(X) * 0.70)
        val_size = int(len(X) * 0.85)
        
        X_train, y_train, prices_train = X[:train_size], y[:train_size], current_prices[:train_size]
        X_val, y_val, prices_val = X[train_size:val_size], y[train_size:val_size], current_prices[train_size:val_size]
        X_test, y_test, prices_test = X[val_size:], y[val_size:], current_prices[val_size:]
        
        print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train XGBoost
        print(f"\n4a. Training XGBoost...")
        xgb_model, xgb_time = train_xgboost(X_train, y_train, X_val, y_val)
        xgb_results = evaluate_model(xgb_model, X_test, y_test, prices_test, 'XGBoost', f'{horizon}d', 'Daily')
        xgb_results['Train_Time'] = xgb_time
        all_results.append(xgb_results)
        print(f"   MAPE: {xgb_results['Price_MAPE']:.2f}%, Directional: {xgb_results['Directional_Accuracy']:.1f}%, Time: {xgb_time:.1f}s")
        
        # Train Random Forest
        print(f"\n4b. Training Random Forest...")
        rf_model, rf_time = train_random_forest(X_train, y_train, X_val, y_val)
        rf_results = evaluate_model(rf_model, X_test, y_test, prices_test, 'Random Forest', f'{horizon}d', 'Daily')
        rf_results['Train_Time'] = rf_time
        all_results.append(rf_results)
        print(f"   MAPE: {rf_results['Price_MAPE']:.2f}%, Directional: {rf_results['Directional_Accuracy']:.1f}%, Time: {rf_time:.1f}s")
        
        # Train Gradient Boosting
        print(f"\n4c. Training Gradient Boosting...")
        gb_model, gb_time = train_gradient_boosting(X_train, y_train, X_val, y_val)
        gb_results = evaluate_model(gb_model, X_test, y_test, prices_test, 'Gradient Boosting', f'{horizon}d', 'Daily')
        gb_results['Train_Time'] = gb_time
        all_results.append(gb_results)
        print(f"   MAPE: {gb_results['Price_MAPE']:.2f}%, Directional: {gb_results['Directional_Accuracy']:.1f}%, Time: {gb_time:.1f}s")
        
        # Train CatBoost
        print(f"\n4d. Training CatBoost...")
        cat_model, cat_time = train_catboost(X_train, y_train, X_val, y_val)
        cat_results = evaluate_model(cat_model, X_test, y_test, prices_test, 'CatBoost', f'{horizon}d', 'Daily')
        cat_results['Train_Time'] = cat_time
        all_results.append(cat_results)
        print(f"   MAPE: {cat_results['Price_MAPE']:.2f}%, Directional: {cat_results['Directional_Accuracy']:.1f}%, Time: {cat_time:.1f}s")
        
        # Train LightGBM
        print(f"\n4e. Training LightGBM...")
        lgbm_model, lgbm_time = train_lightgbm(X_train, y_train, X_val, y_val)
        lgbm_results = evaluate_model(lgbm_model, X_test, y_test, prices_test, 'LightGBM', f'{horizon}d', 'Daily')
        lgbm_results['Train_Time'] = lgbm_time
        all_results.append(lgbm_results)
        print(f"   MAPE: {lgbm_results['Price_MAPE']:.2f}%, Directional: {lgbm_results['Directional_Accuracy']:.1f}%, Time: {lgbm_time:.1f}s")
        
        # Save best model
        mapes = {
            'XGBoost': xgb_results['Price_MAPE'],
            'Random Forest': rf_results['Price_MAPE'],
            'Gradient Boosting': gb_results['Price_MAPE'],
            'CatBoost': cat_results['Price_MAPE'],
            'LightGBM': lgbm_results['Price_MAPE']
        }
        best_model_name = min(mapes, key=mapes.get)
        best_mape = mapes[best_model_name]
        
        print(f"\n5. {best_model_name} is best! Saving model...")
        if best_model_name == 'XGBoost':
            xgb_model.save_model(models_dir / f'xgboost_{horizon}d.json')
        elif best_model_name == 'Random Forest':
            joblib.dump(rf_model, models_dir / f'random_forest_{horizon}d.pkl')
        elif best_model_name == 'Gradient Boosting':
            joblib.dump(gb_model, models_dir / f'gradient_boosting_{horizon}d.pkl')
        elif best_model_name == 'CatBoost':
            cat_model.save_model(models_dir / f'catboost_{horizon}d.cbm')
        else:
            joblib.dump(lgbm_model, models_dir / f'lightgbm_{horizon}d.pkl')

    # Optional hourly benchmark using Random Forest
    print(f"\n{'=' * 80}")
    print("HOURLY RANDOM FOREST BENCHMARK")
    print(f"{'=' * 80}")

    hourly_path = data_dir / 'raw' / 'btc_cryptocompare_365d_1hour.csv'
    if not hourly_path.exists():
        hourly_path = data_dir / 'raw' / 'btc_binance_150d_1hour.csv'

    # if hourly_path.exists():
    #     print(f"\nLoading hourly dataset: {hourly_path.name}")
    #     hourly_df = pd.read_csv(hourly_path)

    #     if 'timestamp' in hourly_df.columns:
    #         hourly_df['timestamp'] = pd.to_datetime(hourly_df['timestamp'])
    #         hourly_df = hourly_df.sort_values('timestamp').reset_index(drop=True)
    #         hourly_df = hourly_df.set_index('timestamp')
    #     else:
    #         hourly_df.index = pd.to_datetime(hourly_df.index)

    #     hourly_df = engineer_technical_features(hourly_df)
    #     hourly_df = hourly_df.dropna()

    #     print(f"   Rows after feature engineering: {len(hourly_df)}")

    #     hourly_horizons = [1, 4, 12]
    #     for horizon in hourly_horizons:
    #         print(f"\nHourly horizon: {horizon}h (Random Forest)")
    #         X, y, feature_cols, scaler, current_prices = prepare_hourly_data(hourly_df.copy(), horizon)

    #         train_size = int(len(X) * 0.70)
    #         val_size = int(len(X) * 0.85)

    #         X_train, y_train = X[:train_size], y[:train_size]
    #         X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    #         X_test, y_test = X[val_size:], y[val_size:]

    #         prices_test = current_prices[val_size:]

    #         rf_model, rf_time = train_random_forest(X_train, y_train, X_val, y_val)
    #         rf_results = evaluate_model(rf_model, X_test, y_test, prices_test, 'Random Forest', f'{horizon}h', 'Hourly')
    #         rf_results['Train_Time'] = rf_time
    #         all_results.append(rf_results)

    #         print(f"   Random Forest → MAPE: {rf_results['Price_MAPE']:.2f}%, Directional: {rf_results['Directional_Accuracy']:.1f}%, Time: {rf_time:.1f}s")

    #         joblib.dump(rf_model, models_dir / f'random_forest_{horizon}h.pkl')
    # else:
    #     print(f"\n⚠️  Hourly benchmark skipped – data file not found: {hourly_path}")
    
    # Save comparison results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")
    
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(results_dir / 'model_comparison.csv', index=False)
    print(f"\nResults saved to: {results_dir / 'model_comparison.csv'}")
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}\n")
    print(results_df.to_string(index=False))
    
    # Determine overall winner
    print(f"\n{'=' * 80}")
    print("OVERALL WINNER")
    print(f"{'=' * 80}")
    
    avg_by_model = results_df.groupby('Model').agg({
        'Price_MAPE': 'mean',
        'Directional_Accuracy': 'mean',
        'Train_Time': 'mean'
    }).round(2)
    
    print(f"\nAverage Performance Across All Horizons:")
    print(avg_by_model.to_string())
    
    winner = avg_by_model['Price_MAPE'].idxmin()
    print(f"\n🏆 WINNER: {winner} (Lowest Average MAPE)")
    
    print(f"\n{'=' * 80}")
    print("TRAINING COMPLETE!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
