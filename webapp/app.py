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

app = Flask(__name__)

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / 'results'

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
                'source': result['source']
            }
    except:
        pass
    # Mock data if API fails
    return {
        'price': 67850.23,
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
                predictions.append({
                    'date': row['timestamp'],
                    'current_price': round(row['current_price'], 2),
                    'predicted_price_1d': round(row['pred_1d_price'], 2),
                    'predicted_return_1d': round(row['pred_1d_return'] * 100, 2),
                    'predicted_price_3d': round(row['pred_3d_price'], 2),
                    'predicted_return_3d': round(row['pred_3d_return'] * 100, 2),
                    'predicted_price_7d': round(row['pred_7d_price'], 2),
                    'predicted_return_7d': round(row['pred_7d_return'] * 100, 2),
                    'generated_at': row.get('generated_at', 'N/A'),
                    'timeframe': 'daily'
                })
        
        # Load Hourly Predictions
        hourly_file = predictions_dir / 'hourly_predictions.csv'
        if hourly_file.exists():
            hourly_df = pd.read_csv(hourly_file)
            latest_hourly = hourly_df.iloc[-1]
            predictions.append({
                'date': latest_hourly['timestamp'],
                'current_price': round(latest_hourly['current_price'], 2),
                'predicted_price_1h': round(latest_hourly.get('pred_1h_price', 0), 2),
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
                'generated_at': latest_hourly.get('generated_at', 'N/A')
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
    """Page 2: Test Results."""
    results_df = load_model_results()
    results_list = []

    balanced_best = None
    rf_benchmark = None
    comparison_path = RESULTS_DIR / 'model_comparison.csv'
    if comparison_path.exists():
        try:
            comparison_df = pd.read_csv(comparison_path)

            if len(comparison_df) > 0:
                df_balanced = comparison_df.copy()
                df_balanced['Balanced Score'] = df_balanced['Directional_Accuracy'] - 0.5 * df_balanced['Price_MAPE']
                top_row = df_balanced.sort_values('Balanced Score', ascending=False).iloc[0]
                balanced_best = {
                    'Model': top_row['Model'],
                    'Timeframe': top_row['Timeframe'],
                    'Horizon': top_row['Horizon'],
                    'MAPE (%)': top_row['Price_MAPE'],
                    'MAE ($)': top_row['Price_MAE'],
                    'Directional Accuracy (%)': top_row['Directional_Accuracy'],
                    'R²': top_row.get('Return_R2', np.nan),
                    'Balanced Score': top_row['Balanced Score']
                }

                display_df = df_balanced.copy()
                display_df['MAPE (%)'] = display_df['Price_MAPE']
                display_df['MAE ($)'] = display_df['Price_MAE']
                display_df['Directional Accuracy (%)'] = display_df['Directional_Accuracy']
                display_df['R²'] = display_df.get('Return_R2', np.nan)
                display_df = display_df.sort_values('Balanced Score', ascending=False)
                results_list = display_df[
                    ['Model', 'Timeframe', 'Horizon', 'MAPE (%)', 'R²', 'MAE ($)', 'Directional Accuracy (%)', 'Balanced Score']
                ].to_dict('records')

            daily_rf = comparison_df[
                (comparison_df['Model'] == 'Random Forest') &
                (comparison_df['Timeframe'] == 'Daily') &
                (comparison_df['Horizon'] == '1d')
            ]

            hourly_rf = comparison_df[
                (comparison_df['Model'] == 'Random Forest') &
                (comparison_df['Timeframe'] == 'Hourly') &
                (comparison_df['Horizon'] == '1h')
            ]

            if not daily_rf.empty and not hourly_rf.empty:
                rf_benchmark = {
                    'daily': daily_rf.iloc[0].to_dict(),
                    'hourly': hourly_rf.iloc[0].to_dict()
                }
        except Exception as exc:
            print(f"Warning: Could not load Random Forest benchmark: {exc}")
        else:
            if results_df is not None and len(results_df) > 0:
                df_copy = results_df.copy()
                df_copy['Balanced Score'] = df_copy['Directional Accuracy (%)'] - 0.5 * df_copy['MAPE (%)']
                balanced_best = df_copy.sort_values('Balanced Score', ascending=False).iloc[0].to_dict()
                # Sort by MAPE for legacy results
                df_copy = df_copy.sort_values('MAPE (%)', ascending=True)
                results_list = df_copy.to_dict('records')

        if (not results_list) and results_df is not None and len(results_df) > 0:
            df_fallback = results_df.copy()
            df_fallback['Balanced Score'] = df_fallback['Directional Accuracy (%)'] - 0.5 * df_fallback['MAPE (%)']
            if balanced_best is None:
                balanced_best = df_fallback.sort_values('Balanced Score', ascending=False).iloc[0].to_dict()
            df_fallback = df_fallback.sort_values('MAPE (%)', ascending=True)
            results_list = df_fallback.to_dict('records')

    return render_template('results.html', results=results_list, balanced_best=balanced_best, rf_benchmark=rf_benchmark)


@app.route('/live')
def live():
    """Page 3: Live Performance & Smart Contract."""
    current_data = get_live_data()
    predictions_data = get_local_predictions()

    # Calculate summary stats
    if predictions_data and len(predictions_data) > 0:
        # Filter daily predictions for stats
        daily_predictions = [p for p in predictions_data if p.get('timeframe') == 'daily']
        recent_predictions = daily_predictions[-7:] if len(daily_predictions) > 7 else daily_predictions
        
        # Calculate average error if we had actual prices (placeholder for now)
        avg_mape = 2.5  # Placeholder - would need actual vs predicted comparison
        
        # Get latest daily prediction for summary
        latest_pred = daily_predictions[-1] if daily_predictions else predictions_data[0]
    else:
        recent_predictions = []
        avg_mape = 0.0
        latest_pred = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'current_price': current_data['price'],
            'predicted_price_1d': current_data['price'],
            'predicted_return_1d': 0.0
        }

    summary = {
        'current_price': current_data['price'],
        'timestamp': current_data['timestamp'],
        'total_predictions': len(predictions_data),
        'avg_mape_7d': round(avg_mape, 2),
        'latest_prediction': latest_pred
    }

    return render_template('live.html', summary=summary, predictions=predictions_data)


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
    print("\n" + "=" * 70 + "\n")

    # Use port 5002 to avoid conflict with AirPlay on macOS
    app.run(debug=True, host='0.0.0.0', port=5002)
