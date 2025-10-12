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

app = Flask(__name__)

# Configuration
RESULTS_DIR = Path(__file__).parent.parent / 'results'


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
    """Load all model results from CSV files."""
    try:
        results_file = RESULTS_DIR / 'all_models_returns_combined.csv'
        if results_file.exists():
            return pd.read_csv(results_file)
        # Return mock data if file doesn't exist yet
        return create_mock_results()
    except Exception as e:
        print(f"Error loading results: {e}")
        return create_mock_results()


def create_mock_results():
    """Create mock results for demonstration."""
    return pd.DataFrame([
        # XGBoost
        {'Model': 'XGBoost', 'Horizon': '1d', 'MAPE (%)': 1.16, 'R²': 0.865, 'MAE ($)': 850, 'Directional Accuracy (%)': 68.5},
        {'Model': 'XGBoost', 'Horizon': '3d', 'MAPE (%)': 2.34, 'R²': 0.721, 'MAE ($)': 1650, 'Directional Accuracy (%)': 64.2},
        {'Model': 'XGBoost', 'Horizon': '7d', 'MAPE (%)': 3.89, 'R²': 0.598, 'MAE ($)': 2850, 'Directional Accuracy (%)': 61.8},
        # Random Forest
        {'Model': 'Random Forest', 'Horizon': '1d', 'MAPE (%)': 1.89, 'R²': 0.742, 'MAE ($)': 1320, 'Directional Accuracy (%)': 65.3},
        {'Model': 'Random Forest', 'Horizon': '3d', 'MAPE (%)': 3.12, 'R²': 0.658, 'MAE ($)': 2180, 'Directional Accuracy (%)': 62.1},
        {'Model': 'Random Forest', 'Horizon': '7d', 'MAPE (%)': 4.78, 'R²': 0.512, 'MAE ($)': 3420, 'Directional Accuracy (%)': 59.4},
        # Gradient Boosting
        {'Model': 'Gradient Boosting', 'Horizon': '1d', 'MAPE (%)': 3.14, 'R²': 0.621, 'MAE ($)': 2150, 'Directional Accuracy (%)': 62.7},
        {'Model': 'Gradient Boosting', 'Horizon': '3d', 'MAPE (%)': 4.89, 'R²': 0.523, 'MAE ($)': 3450, 'Directional Accuracy (%)': 59.8},
        {'Model': 'Gradient Boosting', 'Horizon': '7d', 'MAPE (%)': 6.45, 'R²': 0.412, 'MAE ($)': 4680, 'Directional Accuracy (%)': 57.2}
    ])


def get_live_data():
    """Get current Bitcoin price."""
    try:
        result = get_latest_price(source='yahoo')
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


def get_blockchain_predictions():
    """Get predictions stored on blockchain (mock for now)."""
    # This will be replaced with actual smart contract calls
    mock_predictions = []
    base_date = datetime.now() - timedelta(days=30)

    for i in range(30):
        date = base_date + timedelta(days=i)
        current_price = 65000 + np.random.randn() * 2000
        predicted_return = np.random.randn() * 0.02
        predicted_price = current_price * (1 + predicted_return)
        actual_price = predicted_price + np.random.randn() * 500

        mock_predictions.append({
            'date': date.strftime('%Y-%m-%d'),
            'current_price': round(current_price, 2),
            'predicted_price_1d': round(predicted_price, 2),
            'actual_price_1d': round(actual_price, 2),
            'error': round(abs(actual_price - predicted_price), 2),
            'mape': round(abs(actual_price - predicted_price) / actual_price * 100, 2),
            'transaction_hash': f'0x{"".join(np.random.choice(list("0123456789abcdef"), 64))}',
            'block_number': 12000000 + i * 100
        })

    return mock_predictions


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
    results_list = results_df.to_dict('records') if results_df is not None else []
    return render_template('results.html', results=results_list)


@app.route('/live')
def live():
    """Page 3: Live Performance & Smart Contract."""
    current_data = get_live_data()
    blockchain_data = get_blockchain_predictions()

    # Calculate summary stats
    recent_predictions = blockchain_data[-7:]  # Last 7 days
    avg_mape = np.mean([p['mape'] for p in recent_predictions])

    summary = {
        'current_price': current_data['price'],
        'timestamp': current_data['timestamp'],
        'total_predictions': len(blockchain_data),
        'avg_mape_7d': round(avg_mape, 2),
        'latest_prediction': blockchain_data[-1] if blockchain_data else None
    }

    return render_template('live.html', summary=summary, predictions=blockchain_data)


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


@app.route('/api/blockchain-predictions')
def api_blockchain_predictions():
    """Get predictions from blockchain."""
    predictions = get_blockchain_predictions()
    return jsonify({
        'status': 'success',
        'predictions': predictions
    })


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
