"""Configuration for Bitcoin Price Prediction System"""

# GitHub Configuration
GITHUB_REPO = "ying-jeanne/capstone_msba"
GITHUB_BRANCH = "main"
GITHUB_RAW_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"

PREDICTION_URLS = {
    'daily': f"{GITHUB_RAW_BASE_URL}/data/predictions/daily_predictions.csv",
    'hourly': f"{GITHUB_RAW_BASE_URL}/data/predictions/hourly_predictions.csv",
    '15min': f"{GITHUB_RAW_BASE_URL}/data/predictions/15min_predictions.csv",
}

# Cache duration in seconds
CACHE_DURATION = {
    'daily': 300,
    'hourly': 60,
    '15min': 30,
}

# Flask configuration
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# Display settings
HISTORY_DISPLAY_LIMIT = 30
WEBHOOK_SECRET = "your-secret-key-here"
