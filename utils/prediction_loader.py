"""
Prediction Loader - Fetch predictions from GitHub with caching
===============================================================
Reads prediction CSVs from GitHub raw URLs
Implements smart caching to avoid excessive requests
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Try to import config
try:
    import config
except ImportError:
    # Fallback for when config isn't set up yet
    print("Warning: config.py not found or not configured")
    config = None


class PredictionLoader:
    """Load predictions from GitHub with smart caching"""
    
    def __init__(self, github_repo=None, github_branch='main', use_local_fallback=True):
        """
        Initialize prediction loader
        
        Args:
            github_repo: GitHub repo in format "username/repo"
            github_branch: Branch name (default: main)
            use_local_fallback: Fall back to local files if GitHub fails
        """
        self.cache = {}
        self.cache_time = {}
        self.use_local_fallback = use_local_fallback
        
        # Get config from config.py or use provided values
        if config and hasattr(config, 'GITHUB_REPO'):
            self.github_repo = config.GITHUB_REPO
            self.github_branch = config.GITHUB_BRANCH
            self.prediction_urls = config.PREDICTION_URLS
            self.cache_duration = config.CACHE_DURATION
        else:
            self.github_repo = github_repo
            self.github_branch = github_branch
            
            # Build URLs manually if config not available
            base_url = f"https://raw.githubusercontent.com/{github_repo}/{github_branch}"
            self.prediction_urls = {
                'daily': f"{base_url}/data/predictions/daily_predictions.csv",
                'hourly': f"{base_url}/data/predictions/hourly_predictions.csv",
                '15min': f"{base_url}/data/predictions/15min_predictions.csv",
            }
            self.cache_duration = {
                'daily': 300,
                'hourly': 60,
                '15min': 30,
            }
    
    def _is_cache_valid(self, timeframe):
        """Check if cached data is still valid"""
        if timeframe not in self.cache_time:
            return False
        
        age = datetime.now() - self.cache_time[timeframe]
        max_age = timedelta(seconds=self.cache_duration.get(timeframe, 60))
        
        return age < max_age
    
    def _fetch_from_github(self, timeframe):
        """Fetch prediction CSV from GitHub raw URL"""
        url = self.prediction_urls.get(timeframe)
        
        if not url or 'YOUR_USERNAME' in url:
            raise ValueError(
                f"GitHub URL not configured for {timeframe}. "
                "Please set GITHUB_REPO in config.py"
            )
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to fetch from GitHub: {str(e)}")
    
    def _fetch_from_local(self, timeframe):
        """Fallback: fetch from local file"""
        local_path = Path(f'data/predictions/{timeframe}_predictions.csv')
        
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")
        
        return pd.read_csv(local_path)
    
    def get_predictions(self, timeframe):
        """
        Get predictions for specified timeframe with caching
        
        Args:
            timeframe: 'daily', 'hourly', or '15min'
            
        Returns:
            pandas DataFrame with predictions
        """
        # Check cache first
        if self._is_cache_valid(timeframe):
            return self.cache[timeframe].copy()
        
        # Try GitHub first
        try:
            df = self._fetch_from_github(timeframe)
            
            # Update cache
            self.cache[timeframe] = df
            self.cache_time[timeframe] = datetime.now()
            
            return df.copy()
            
        except Exception as e:
            print(f"Warning: Failed to fetch {timeframe} from GitHub: {e}")
            
            # Try local fallback
            if self.use_local_fallback:
                try:
                    df = self._fetch_from_local(timeframe)
                    print(f"Using local fallback for {timeframe}")
                    
                    # Cache local data too
                    self.cache[timeframe] = df
                    self.cache_time[timeframe] = datetime.now()
                    
                    return df.copy()
                except Exception as e2:
                    print(f"Local fallback also failed: {e2}")
            
            # Return cached data if available (even if expired)
            if timeframe in self.cache:
                print(f"Using expired cache for {timeframe}")
                return self.cache[timeframe].copy()
            
            # No data available
            raise Exception(f"Unable to load {timeframe} predictions from any source")
    
    def get_all_predictions(self):
        """Get predictions for all timeframes"""
        predictions = {}
        
        for timeframe in ['daily', 'hourly', '15min']:
            try:
                predictions[timeframe] = self.get_predictions(timeframe)
            except Exception as e:
                print(f"Failed to load {timeframe} predictions: {e}")
                predictions[timeframe] = None
        
        return predictions
    
    def get_latest_prediction(self, timeframe):
        """Get just the latest prediction for a timeframe"""
        df = self.get_predictions(timeframe)
        if df is not None and len(df) > 0:
            return df.iloc[-1].to_dict()
        return None
    
    def clear_cache(self, timeframe=None):
        """Clear cache for specific timeframe or all"""
        if timeframe:
            self.cache.pop(timeframe, None)
            self.cache_time.pop(timeframe, None)
        else:
            self.cache.clear()
            self.cache_time.clear()


# Global instance for easy import
_prediction_loader = None

def get_prediction_loader():
    """Get or create global prediction loader instance"""
    global _prediction_loader
    if _prediction_loader is None:
        _prediction_loader = PredictionLoader()
    return _prediction_loader


def load_predictions(timeframe):
    """Convenience function to load predictions"""
    loader = get_prediction_loader()
    return loader.get_predictions(timeframe)


def load_latest_prediction(timeframe):
    """Convenience function to load latest prediction"""
    loader = get_prediction_loader()
    return loader.get_latest_prediction(timeframe)


def load_all_predictions():
    """Convenience function to load all predictions"""
    loader = get_prediction_loader()
    return loader.get_all_predictions()


# Example usage
if __name__ == '__main__':
    print("Testing Prediction Loader...")
    print("="*70)
    
    try:
        # Test loading daily predictions
        print("\nLoading daily predictions...")
        daily = load_predictions('daily')
        print(f"✓ Loaded daily predictions: {len(daily)} records")
        print(daily.head())
        
        # Test loading hourly predictions
        print("\nLoading hourly predictions...")
        hourly = load_predictions('hourly')
        print(f"✓ Loaded hourly predictions: {len(hourly)} records")
        
        # Test loading 15-min predictions
        print("\nLoading 15-min predictions...")
        min15 = load_predictions('15min')
        print(f"✓ Loaded 15-min predictions: {len(min15)} records")
        
        print("\n✓ All prediction loading tests passed!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to:")
        print("1. Set GITHUB_REPO in config.py")
        print("2. Run training and prediction scripts first")
        print("3. Commit and push predictions to GitHub")
