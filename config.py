"""
Configuration for Bitcoin Price Prediction System
==================================================
Set your configuration here before deployment
"""

# ============================================================================
# GitHub Configuration (for reading predictions from raw URLs)
# ============================================================================

# Your GitHub repository information
# Format: "username/repository" or "organization/repository"
# Example: "john-doe/bitcoin-prediction"
GITHUB_REPO = "ying-jeanne/capstone_msba"

# Branch where predictions are stored (usually "main" or "master")
GITHUB_BRANCH = "main"

# Base URL for raw GitHub files
GITHUB_RAW_BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}"

# Prediction file paths
PREDICTION_URLS = {
    'daily': f"{GITHUB_RAW_BASE_URL}/data/predictions/daily_predictions.csv",
    'hourly': f"{GITHUB_RAW_BASE_URL}/data/predictions/hourly_predictions.csv",
    '15min': f"{GITHUB_RAW_BASE_URL}/data/predictions/15min_predictions.csv",
}

# ============================================================================
# Cache Configuration
# ============================================================================

# How long to cache predictions before fetching from GitHub again (in seconds)
CACHE_DURATION = {
    'daily': 300,    # 5 minutes (daily updates once per day)
    'hourly': 60,    # 1 minute (hourly updates every hour)
    '15min': 30,     # 30 seconds (15-min updates every 15 min)
}

# ============================================================================
# Application Configuration
# ============================================================================

# Flask configuration
DEBUG = True
HOST = '0.0.0.0'
PORT = 5000

# ============================================================================
# Display Configuration
# ============================================================================

# Show how many recent predictions in history
HISTORY_DISPLAY_LIMIT = 30

# ============================================================================
# API Configuration (optional - for future features)
# ============================================================================

# If you want to add webhook authentication later
WEBHOOK_SECRET = "your-secret-key-here"  # Change this for production!
