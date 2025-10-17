"""
Bitcoin Price Prediction - Flask Web Application
=================================================
Group Project Presentation Website

3-Page Structure:
1. Parameters & Methodology - Feature explanations and model approach
2. Test Results - Performance metrics and validation
3. Live Performance - Real-world results with smart contract integration

To run:
    python webapp/app.py
    Visit: http://localhost:5000
"""

from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

from utils.data_fetcher import get_latest_price
from utils.prediction_loader import PredictionLoader

# Try to import blockchain functions, but continue if they fail
try:
    from utils.blockchain_integration import (
        get_predictions_with_outcomes,
        get_performance_summary,
        CONTRACT_ADDRESS
    )
    BLOCKCHAIN_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Blockchain integration not available: {e}")
    BLOCKCHAIN_AVAILABLE = False
    CONTRACT_ADDRESS = "0x93b879c3b24AC7532Fd64C7F203D64Ab668bB42E"
    
    # Fallback functions
    def get_predictions_with_outcomes(count=30, use_demo=True, use_github=True):
        return []
    
    def get_performance_summary(use_demo=True, use_github=True):
        return {
            'total_predictions': 0,
            'avg_mape_1d': 0,
            'avg_mape_3d': 0,
            'avg_mape_7d': 0,
            'directional_accuracy_1d': 0,
            'directional_accuracy_3d': 0,
            'directional_accuracy_7d': 0
        }

app = Flask(__name__)

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / 'results'
USE_DEMO_DATA = True  # Set to False when using real blockchain data
USE_GITHUB = True  # Set to True to fetch from GitHub (for PythonAnywhere), False for local development

# Initialize prediction loader (fetches from GitHub with caching)
prediction_loader = PredictionLoader()


# ============================================================================
# Helper Functions
# ============================================================================

def get_feature_definitions():
    """Return all feature definitions organized by category."""
    return {
        "Technical Indicators": [
            {
                "name": "RSI (Relative Strength Index)",
                "features": ["rsi_10", "rsi_14", "rsi_30", "rsi_200"],
                "formula": "RSI = 100 - (100 / (1 + RS)), where RS = Avg Gain / Avg Loss",
                "description": "Measures overbought/oversold conditions (0-100). <30 = oversold (potential buy), >70 = overbought (potential sell)",
                "windows": "10, 14, 30, 200 periods"
            },
            {
                "name": "MACD (Moving Average Convergence Divergence)",
                "features": ["macd", "macd_signal", "macd_diff"],
                "formula": "MACD = EMA(12) - EMA(26), Signal = EMA(MACD, 9), Histogram = MACD - Signal",
                "description": "Shows relationship between two moving averages. MACD > Signal = bullish, crossovers indicate trend changes",
                "windows": "Fast: 12, Slow: 26, Signal: 9"
            },
            {
                "name": "EMA (Exponential Moving Average)",
                "features": ["ema_10", "ema_30", "ema_200"],
                "formula": "EMA_t = (Price_t × k) + (EMA_{t-1} × (1-k)), where k = 2/(N+1)",
                "description": "Smoothed price trends with more weight on recent prices. Price > EMA = uptrend",
                "windows": "10 (short), 30 (medium), 200 (long-term)"
            },
            {
                "name": "Bollinger Bands",
                "features": ["bb_high", "bb_low", "bb_mid", "bb_width"],
                "formula": "Upper = SMA + (2 × σ), Lower = SMA - (2 × σ), Width = (Upper - Lower) / Middle",
                "description": "Price envelope based on standard deviation. Price near upper band = overbought, near lower = oversold",
                "windows": "20-period SMA, 2 standard deviations"
            },
            {
                "name": "ATR (Average True Range)",
                "features": ["atr_14"],
                "formula": "ATR = EMA of TR, where TR = max(High-Low, |High-Close_prev|, |Low-Close_prev|)",
                "description": "Measures volatility. High ATR = high volatility, helps with position sizing",
                "windows": "14 periods"
            },
            {
                "name": "Stochastic Oscillator",
                "features": ["stoch_k", "stoch_d"],
                "formula": "%K = 100 × (Close - Low_n) / (High_n - Low_n), %D = SMA(%K, 3)",
                "description": "Compares closing price to price range. >80 = overbought, <20 = oversold",
                "windows": "14-period lookback, 3-period smoothing"
            }
        ],
        "Volume Indicators": [
            {
                "name": "Volume Moving Averages",
                "features": ["volume_ema_10", "volume_ema_30", "volume_ratio"],
                "formula": "Volume_EMA = EMA(volume, N), Ratio = Current_Volume / Volume_EMA_30",
                "description": "Volume confirms price movements. High volume ratio with price increase = strong trend",
                "windows": "10, 30 periods"
            }
        ],
        "Lag Features": [
            {
                "name": "Price Lags",
                "features": ["close_lag_1", "close_lag_2", "close_lag_3", "close_lag_5", "close_lag_7"],
                "formula": "close_lag_n = close[t-n]",
                "description": "Historical prices give model 'memory' of recent values. Captures patterns at different time steps",
                "windows": "1, 2, 3, 5, 7 periods back"
            },
            {
                "name": "Volume Lags",
                "features": ["volume_lag_1", "volume_lag_2", "volume_lag_3"],
                "formula": "volume_lag_n = volume[t-n]",
                "description": "Historical volume patterns help identify accumulation/distribution phases",
                "windows": "1, 2, 3 periods back"
            }
        ],
        "Rolling Statistics": [
            {
                "name": "Rolling Means (Moving Averages)",
                "features": ["close_rolling_mean_7", "close_rolling_mean_30", "close_rolling_mean_90"],
                "formula": "Rolling_Mean_n = mean(close[t-n:t])",
                "description": "Capture short/medium/long-term trends. Smooth out noise in price data",
                "windows": "7, 30, 90 periods"
            },
            {
                "name": "Rolling Standard Deviation (Volatility)",
                "features": ["close_rolling_std_7", "close_rolling_std_30"],
                "formula": "Rolling_Std_n = std(close[t-n:t])",
                "description": "Measures price volatility over time. High std = uncertain market conditions",
                "windows": "7, 30 periods"
            },
            {
                "name": "Rolling Min/Max (Support/Resistance)",
                "features": ["close_rolling_min_30", "close_rolling_max_30"],
                "formula": "Rolling_Min = min(close[t-30:t]), Rolling_Max = max(close[t-30:t])",
                "description": "Identify support and resistance levels. Price bounces at these levels",
                "windows": "30 periods"
            }
        ],
        "Returns & Momentum": [
            {
                "name": "Returns (Price Changes)",
                "features": ["returns", "log_returns", "returns_7d"],
                "formula": "returns = (close_t - close_{t-1}) / close_{t-1}, log_returns = ln(close_t / close_{t-1})",
                "description": "Normalize price changes as percentages. More meaningful than raw prices for ML",
                "windows": "1-period, 7-period"
            },
            {
                "name": "Momentum",
                "features": ["momentum_10", "momentum_30"],
                "formula": "momentum_n = close_t - close_{t-n}",
                "description": "Raw price change over N periods. Captures trend strength",
                "windows": "10, 30 periods"
            }
        ],
        "Time-based Features": [
            {
                "name": "Cyclical Time Encoding",
                "features": ["hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos"],
                "formula": "hour_sin = sin(2π × hour/24), hour_cos = cos(2π × hour/24)",
                "description": "Cyclical encoding ensures continuity (hour 23 is close to hour 0). Only for intraday data",
                "windows": "24 hours, 7 days"
            }
        ],
        "Interaction Features": [
            {
                "name": "Feature Interactions",
                "features": ["price_volume_interaction", "rsi_macd_interaction"],
                "formula": "price_volume = close × volume, rsi_macd = rsi_14 × macd_diff",
                "description": "Capture relationships between features that reveal hidden patterns",
                "windows": "N/A (computed from other features)"
            }
        ]
    }


def load_model_results():
    """Load REAL model results from your actual training."""
    results = []
    
    # Load Daily Models
    try:
        daily_metrics = RESULTS_DIR / 'daily_models_metrics.csv'
        if daily_metrics.exists():
            df = pd.read_csv(daily_metrics)
            for _, row in df.iterrows():
                results.append({
                    'Model': 'XGBoost',
                    'Timeframe': 'Daily',
                    'Horizon': row['horizon'],
                    'MAPE (%)': round(row['price_mape'], 2),
                    'MAE ($)': round(row['price_mae'], 2),
                    'R²': round(row['return_test_r2'], 4),
                    'Directional Accuracy (%)': round(row['directional_accuracy'], 1),
                    'Train Samples': int(row['train_samples']),
                    'Test Samples': int(row['test_samples'])
                })
    except Exception as e:
        print(f"Warning: Could not load daily metrics: {e}")
    
    # Load Hourly Models
    try:
        hourly_metrics = RESULTS_DIR / 'hourly_models_metrics.csv'
        if hourly_metrics.exists():
            df = pd.read_csv(hourly_metrics)
            for _, row in df.iterrows():
                results.append({
                    'Model': 'XGBoost',
                    'Timeframe': 'Hourly',
                    'Horizon': row['horizon'],
                    'MAPE (%)': round(row['price_mape'], 2),
                    'MAE ($)': round(row['price_mae'], 2),
                    'R²': round(row['return_test_r2'], 4),
                    'Directional Accuracy (%)': round(row['directional_accuracy'], 1),
                    'Train Samples': int(row['train_samples']),
                    'Test Samples': int(row['test_samples'])
                })
    except Exception as e:
        print(f"Warning: Could not load hourly metrics: {e}")
    
    if len(results) == 0:
        print("Warning: No metrics found, using placeholder data")
        return pd.DataFrame([{
            'Model': 'XGBoost', 
            'Timeframe': 'Daily',
            'Horizon': '1d', 
            'MAPE (%)': 1.53, 
            'MAE ($)': 1539, 
            'R²': -0.0000, 
            'Directional Accuracy (%)': 51.4,
            'Train Samples': 1138,
            'Test Samples': 245
        }])
    
    return pd.DataFrame(results)


def get_live_data():
    """Get current Bitcoin price."""
    try:
        result = get_latest_price()
        if result['status'] == 'success':
            return {
                'price': result['price'],
                'timestamp': result['timestamp'],
                'source': 'yahoo_finance'
            }
        else:
            print(f"⚠️  get_latest_price() returned error: {result.get('message')}")
    except Exception as e:
        print(f"⚠️  Exception in get_live_data(): {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Fallback: Try to get latest from CSV as backup
    try:
        csv_path = Path(__file__).parent.parent / 'data' / 'predictions' / 'daily_predictions.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                latest = df.iloc[-1]
                return {
                    'price': latest['current_price'],
                    'timestamp': latest['timestamp'],
                    'source': 'csv_backup'
                }
    except Exception as e:
        print(f"⚠️  CSV backup failed: {str(e)}")
    
    # Last resort mock data
    print("⚠️  WARNING: Using outdated mock price!")
    return {
        'price': 104886.59,  # Updated to recent price
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'mock'
    }


def get_local_predictions():
    """Get predictions from LOCAL prediction files (not GitHub)."""
    predictions = []
    predictions_dir = Path(__file__).parent.parent / 'data' / 'predictions'
    
    try:
        # Load Daily Predictions
        daily_file = predictions_dir / 'daily_predictions.csv'
        if daily_file.exists():
            daily_df = pd.read_csv(daily_file)
            for _, row in daily_df.iterrows():
                pred_price_1d = row['pred_1d_price']
                current_price = row['current_price']
                actual_price = row.get('actual_price_1d', float('nan'))

                if pd.isna(actual_price):
                    actual_price = pred_price_1d

                error = actual_price - pred_price_1d
                mape = (abs(error) / actual_price * 100) if actual_price else 0.0
                transaction_hash = row.get('transaction_hash', 'N/A')
                block_number = row.get('block_number', 0)

                try:
                    block_number = int(block_number)
                except (TypeError, ValueError):
                    block_number = 0

                predictions.append({
                    'date': row['timestamp'],
                    'current_price': round(current_price, 2),
                    'predicted_price_1d': round(pred_price_1d, 2),
                    'predicted_return_1d': round(row['pred_1d_return'] * 100, 2),
                    'predicted_price_3d': round(row['pred_3d_price'], 2),
                    'predicted_return_3d': round(row['pred_3d_return'] * 100, 2),
                    'predicted_price_7d': round(row['pred_7d_price'], 2),
                    'predicted_return_7d': round(row['pred_7d_return'] * 100, 2),
                    'generated_at': row.get('generated_at', 'N/A'),
                    'actual_price_1d': round(actual_price, 2) if actual_price else round(pred_price_1d, 2),
                    'error': round(error, 2) if actual_price else 0.0,
                    'mape': round(mape, 2) if actual_price else 0.0,
                    'transaction_hash': transaction_hash,
                    'block_number': block_number,
                    'timeframe': 'daily'
                })
        
        # Load Hourly Predictions
        hourly_file = predictions_dir / 'hourly_predictions.csv'
        if hourly_file.exists():
            hourly_df = pd.read_csv(hourly_file)
            latest_hourly = hourly_df.iloc[-1]
            hourly_pred_price = latest_hourly.get('pred_1h_price', 0)
            hourly_transaction = latest_hourly.get('transaction_hash', 'N/A')
            hourly_block = latest_hourly.get('block_number', 0)
            try:
                hourly_block = int(hourly_block)
            except (TypeError, ValueError):
                hourly_block = 0
            predictions.append({
                'date': latest_hourly['timestamp'],
                'current_price': round(latest_hourly['current_price'], 2),
                'predicted_price_1h': round(hourly_pred_price, 2),
                'predicted_return_1h': round(latest_hourly.get('pred_1h_return', 0) * 100, 2),
                'predicted_price_4h': round(latest_hourly.get('pred_4h_price', 0), 2),
                'predicted_return_4h': round(latest_hourly.get('pred_4h_return', 0) * 100, 2),
                'predicted_price_6h': round(latest_hourly.get('pred_6h_price', 0), 2),
                'predicted_return_6h': round(latest_hourly.get('pred_6h_return', 0) * 100, 2),
                'predicted_price_12h': round(latest_hourly.get('pred_12h_price', 0), 2),
                'predicted_return_12h': round(latest_hourly.get('pred_12h_return', 0) * 100, 2),
                'predicted_price_24h': round(latest_hourly.get('pred_24h_price', 0), 2),
                'predicted_return_24h': round(latest_hourly.get('pred_24h_return', 0) * 100, 2),
                'timeframe': 'hourly',
                'generated_at': latest_hourly.get('generated_at', 'N/A'),
                'actual_price_1d': round(hourly_pred_price, 2),
                'error': 0.0,
                'mape': 0.0,
                'transaction_hash': hourly_transaction,
                'block_number': hourly_block
            })
        
        return predictions
        
    except Exception as e:
        print(f"Error loading local predictions: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# Routes
# ============================================================================

@app.route('/')
def home():
    """Redirect to methodology page."""
    features = get_feature_definitions()
    return render_template('methodology.html', features=features)


@app.route('/methodology')
def methodology():
    """Page 1: Parameters & Methodology."""
    features = get_feature_definitions()
    return render_template('methodology.html', features=features)


@app.route('/results')
def results():
    """Page 2: Test Results - XGBoost Production + Model Comparison Study."""

    # Load daily models metrics (contains all 4 models: xgboost, random_forest, lightgbm, catboost)
    metrics_path = RESULTS_DIR / 'daily_models_metrics.csv'

    xgboost_results = []
    comparison_study = []
    model_averages = []
    xgb_summary = {}
    gradient_stats = {}
    random_forest_stats = {}
    total_models_tested = 0

    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            total_models_tested = df['model'].nunique()

            # Production Model: XGBoost only (for hero section and main display)
            xgb_df = df[df['model'] == 'xgboost'].copy()
            if not xgb_df.empty:
                order_map = {'1d': 0, '3d': 1, '7d': 2}
                xgb_df['sort_key'] = xgb_df['horizon'].map(order_map).fillna(99)
                xgb_df = xgb_df.sort_values('sort_key').drop(columns=['sort_key'])

                best_mape_row = xgb_df.loc[xgb_df['price_mape'].idxmin()]
                best_dir_row = xgb_df.loc[xgb_df['directional_accuracy'].idxmax()]

                horizons = [h.upper() for h in xgb_df['horizon'].tolist()]

                xgb_summary = {
                    'avg_mape': round(xgb_df['price_mape'].mean(), 2),
                    'avg_directional': round(xgb_df['directional_accuracy'].mean(), 1),
                    'best_mape': round(best_mape_row['price_mape'], 2),
                    'best_mape_horizon': best_mape_row['horizon'],
                    'best_directional': round(best_dir_row['directional_accuracy'], 1),
                    'best_directional_horizon': best_dir_row['horizon'],
                    'best_directional_edge': round(best_dir_row['directional_accuracy'] - 50, 1),
                    'horizons_label': ", ".join(horizons),
                    'horizon_count': len(horizons)
                }

            for _, row in xgb_df.iterrows():
                xgboost_results.append({
                    'horizon': row['horizon'],
                    'mape': round(row['price_mape'], 2),
                    'mae': round(row['price_mae'], 2),
                    'r2': round(row['return_test_r2'], 4),
                    'directional': round(row['directional_accuracy'], 1),
                    'train_samples': int(row['train_samples']),
                    'test_samples': int(row['test_samples'])
                })

            # Model Comparison Study: All 4 models (for validation section)
            for _, row in df.iterrows():
                comparison_study.append({
                    'model': row['model'],
                    'horizon': row['horizon'],
                    'mape': round(row['price_mape'], 2),
                    'mae': round(row['price_mae'], 2),
                    'r2': round(row['return_test_r2'], 4),
                    'directional': round(row['directional_accuracy'], 1)
                })

            # Calculate model averages
            for model_name in ['xgboost', 'random_forest', 'lightgbm', 'catboost']:
                model_df = df[df['model'] == model_name]
                if len(model_df) > 0:
                    avg_mape = model_df['price_mape'].mean()
                    avg_dir = model_df['directional_accuracy'].mean()
                    avg_r2 = model_df['return_test_r2'].mean()

                    # Determine rating
                    if avg_mape < 2.5 and avg_dir > 52:
                        rating = "⭐⭐⭐ Excellent"
                        verdict = "Production Ready"
                    elif avg_mape < 3.5 and avg_dir > 50:
                        rating = "✅✅ Good"
                        verdict = "Good Option"
                    elif avg_mape < 4.5:
                        rating = "✅ Acceptable"
                        verdict = "Use with Caution"
                    else:
                        rating = "❌ Poor"
                        verdict = "Not Recommended"

                    model_averages.append({
                        'model': model_name,
                        'avg_mape': round(avg_mape, 2),
                        'avg_directional': round(avg_dir, 1),
                        'avg_r2': round(avg_r2, 4),
                        'rating': rating,
                        'verdict': verdict
                    })

            # Sort model averages by MAPE
            model_averages = sorted(model_averages, key=lambda x: x['avg_mape'])

            # Gradient boosting summary (XGBoost, LightGBM, CatBoost)
            gradient_df = df[df['model'].isin(['xgboost', 'lightgbm', 'catboost'])]
            if not gradient_df.empty:
                gradient_means = gradient_df.groupby('model')['price_mape'].mean()
                gradient_stats = {
                    'avg_mape': round(gradient_means.mean(), 2),
                    'min_mape': round(gradient_means.min(), 2),
                    'max_mape': round(gradient_means.max(), 2)
                }

            # Random Forest diagnostics
            rf_df = df[df['model'] == 'random_forest']
            if not rf_df.empty:
                rf_avg_mape = rf_df['price_mape'].mean()
                rf_avg_dir = rf_df['directional_accuracy'].mean()
                rf_worst_r2 = rf_df['return_test_r2'].min()
                relative_worse = None
                if xgb_summary:
                    relative_worse = ((rf_avg_mape - xgb_summary['avg_mape']) / xgb_summary['avg_mape']) * 100

                random_forest_stats = {
                    'avg_mape': round(rf_avg_mape, 2),
                    'avg_directional': round(rf_avg_dir, 1),
                    'worst_r2': round(rf_worst_r2, 2),
                    'relative_mape_increase': round(relative_worse, 1) if relative_worse is not None else None
                }

        except Exception as e:
            print(f"Error loading model metrics: {e}")
            import traceback
            traceback.print_exc()

    return render_template('results.html',
                         xgboost_results=xgboost_results,
                         comparison_study=comparison_study,
                         model_averages=model_averages,
                         xgb_summary=xgb_summary,
                         gradient_stats=gradient_stats,
                         random_forest_stats=random_forest_stats,
                         total_models_tested=total_models_tested,
                         study_date='October 2025')


@app.route('/chart-test')
def chart_test():
    """Test page for Chart.js debugging."""
    return render_template('chart_test.html')


@app.route('/live')
def live():
    """Page 3: Live Performance & Blockchain Verification."""
    try:
        # Get current price
        current_data = get_live_data()

        # Get blockchain predictions with outcomes (only if available)
        blockchain_predictions = []
        summary = {
            'current_price': current_data['price'],
            'timestamp': current_data['timestamp'],
            'total_predictions': 0,
            'avg_mape_1d': 0,
            'directional_accuracy_1d': 0
        }
        
        if BLOCKCHAIN_AVAILABLE:
            blockchain_predictions = get_predictions_with_outcomes(count=30, use_demo=USE_DEMO_DATA, use_github=USE_GITHUB)
            summary = get_performance_summary(use_demo=USE_DEMO_DATA, use_github=USE_GITHUB)
            summary['current_price'] = current_data['price']
            summary['timestamp'] = current_data['timestamp']

        return render_template(
            'live.html',
            blockchain_predictions=blockchain_predictions,
            summary=summary,
            demo_mode=USE_DEMO_DATA,
            contract_address=CONTRACT_ADDRESS,
            blockchain_available=BLOCKCHAIN_AVAILABLE
        )
    except Exception as e:
        # Fallback to empty data on error
        print(f"Error in /live route: {e}")
        import traceback
        traceback.print_exc()

        return render_template(
            'live.html',
            blockchain_predictions=[],
            summary={
                'current_price': 0,
                'timestamp': '',
                'total_predictions': 0,
                'avg_mape_1d': 0,
                'directional_accuracy_1d': 0
            },
            demo_mode=USE_DEMO_DATA,
            contract_address=CONTRACT_ADDRESS,
            error=str(e)
        )


# ============================================================================
# API Endpoints
# ============================================================================

@app.route('/api/latest-price')
def api_latest_price():
    """Get current Bitcoin price."""
    data = get_live_data()
    return jsonify({'status': 'success', 'data': data})


@app.route('/api/model-results')
def api_model_results():
    """Get all model results."""
    results_df = load_model_results()
    return jsonify({
        'status': 'success',
        'results': results_df.to_dict('records')
    })


@app.route('/api/predictions')
def api_all_predictions():
    """Get all predictions (daily + hourly)."""
    predictions = get_local_predictions()
    return jsonify({
        'status': 'success',
        'predictions': predictions
    })


@app.route('/api/blockchain-predictions')
def api_blockchain_predictions():
    """Get blockchain predictions with actual outcomes."""
    try:
        if not BLOCKCHAIN_AVAILABLE:
            return jsonify({
                'status': 'error',
                'message': 'Blockchain integration not available. Please install web3 package.'
            }), 503
        
        predictions = get_predictions_with_outcomes(count=30, use_demo=USE_DEMO_DATA, use_github=USE_GITHUB)
        summary = get_performance_summary(use_demo=USE_DEMO_DATA, use_github=USE_GITHUB)
        
        # Clean NaN values - replace with None (null in JSON)
        def clean_nan(obj):
            """Recursively replace NaN with None for JSON serialization."""
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, float) and (obj != obj):  # NaN check
                return None
            else:
                return obj
        
        predictions = clean_nan(predictions)
        summary = clean_nan(summary)

        return jsonify({
            'status': 'success',
            'predictions': predictions,
            'summary': summary,
            'contract_address': CONTRACT_ADDRESS,
            'demo_mode': USE_DEMO_DATA,
            'explorer_url': f'https://moonbase.moonscan.io/address/{CONTRACT_ADDRESS}'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predictions/<timeframe>')
def api_predictions_by_timeframe(timeframe):
    """Get predictions for specific timeframe (daily, hourly)."""
    try:
        if timeframe not in ['daily', 'hourly']:
            return jsonify({'status': 'error', 'message': 'Invalid timeframe'}), 400
        
        df = prediction_loader.get_predictions(timeframe)
        
        if df is None or len(df) == 0:
            return jsonify({'status': 'error', 'message': 'No predictions available'}), 404
        
        # Convert DataFrame to list of dicts
        predictions = df.to_dict('records')
        
        return jsonify({
            'status': 'success',
            'timeframe': timeframe,
            'count': len(predictions),
            'predictions': predictions
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/feature-definitions')
def api_feature_definitions():
    """Get all feature definitions."""
    features = get_feature_definitions()
    return jsonify({
        'status': 'success',
        'features': features
    })


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("  BITCOIN PRICE PREDICTION - WEB APPLICATION")
    print("=" * 70)
    print("\nStarting Flask server...")
    print("Visit: http://localhost:5002")
    print("\nPages:")
    print("  1. Methodology:  http://localhost:5002/methodology")
    print("  2. Results:      http://localhost:5002/results")
    print("  3. Live:         http://localhost:5002/live")
    print("\nDebug:")
    print("  Chart Test:      http://localhost:5002/chart-test")
    print("\n" + "=" * 70 + "\n")

    # Use port 5002 to avoid conflict with AirPlay on macOS
    app.run(debug=True, host='0.0.0.0', port=5002)
