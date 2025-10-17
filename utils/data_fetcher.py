"""
Crypto Data Fetcher - Fetch Bitcoin price data from free APIs
Sources: Yahoo Finance (daily), Cryptocompare (15-min, hourly)

Note: Uses direct Cryptocompare REST API (no SDK) to avoid IP blocking in GitHub Actions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import time
import requests
from pathlib import Path


def get_bitcoin_data_incremental(source='cryptocompare_1h', days=365, cache_dir='data/raw', verbose=True):
    """
    Fetch Bitcoin data with incremental updates

    If cached file exists:
      - Load existing data
      - Fetch only new data since last timestamp
      - Append and deduplicate

    If cached file doesn't exist:
      - Fetch full history (days parameter)
      - Save to cache

    Args:
        source (str): 'yahoo', or 'cryptocompare_1h' (hourly)
        days (int): Days to fetch if no cache exists (default: 365)
        cache_dir (str): Directory for cached files (default: 'data/raw')
        verbose (bool): Print status messages (default: True)

    Returns:
        pd.DataFrame or None

    Example:
        # First call: fetches 365 days
        df = get_bitcoin_data_incremental('cryptocompare_1h', days=365)

        # Later calls: only fetches new data since last timestamp
        df = get_bitcoin_data_incremental('cryptocompare_1h', days=365)
    """
    import os

    # Determine cache filename
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    fetch_source = source
    base_period = None
    cutoff_days = None

    if source == 'yahoo':
        cache_file = cache_path / 'btc_yahoo_2y_daily.csv'  # Keep same name for compatibility
        fetch_source = 'yahoo'
        base_period = '2y'
        cutoff_days = 730
    elif source == 'yahoo_5y':
        cache_file = cache_path / 'btc_yahoo_5y_daily.csv'  # 5-year cache for daily training
        fetch_source = 'yahoo'
        base_period = '5y'
        cutoff_days = 1825
    elif source == 'cryptocompare_1h':
        cache_file = cache_path / f'btc_cryptocompare_{days}d_1hour.csv'
        fetch_source = 'cryptocompare_1h'
        cutoff_days = days
    else:
        raise ValueError(f"Invalid source: {source}")

    # Check if cache exists
    if cache_file.exists():
        if verbose:
            print(f"✓ Found cached data: {cache_file}")

        # Load existing data
        existing_df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        last_timestamp = existing_df.index[-1]

        if verbose:
            print(f"  Last cached timestamp: {last_timestamp}")
            print(f"  Fetching new data since {last_timestamp}...")

        # Calculate how many days to fetch (from last timestamp to now)
        days_to_fetch = (datetime.now() - last_timestamp).days + 1  # +1 for safety
        days_to_fetch = max(2, days_to_fetch)  # At least 2 days to ensure overlap

        if verbose:
            print(f"  Fetching last {days_to_fetch} days to update cache...")

        # Fetch new data (pass verbose to see fallback messages)
        if fetch_source == 'yahoo':
            period_arg = f"{max(days_to_fetch, 2)}d"
            result = get_bitcoin_data(
                source='yahoo',
                period=period_arg,
                interval='1d',
                return_dict=True,
                verbose=verbose
            )
        else:
            fetch_days = max(2, min(days_to_fetch, days)) if days else max(2, days_to_fetch)
            result = get_bitcoin_data(
                source=fetch_source,
                days=fetch_days,
                return_dict=True,
                verbose=verbose
            )

        if result['status'] != 'success' or result['data'] is None:
            if verbose:
                print(f"  ⚠️  Failed to fetch new data, using cached version")
            return existing_df

        new_df = result['data']

        # Merge: append new data and remove duplicates
        combined_df = pd.concat([existing_df, new_df])
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)

        # Trim to desired length (keep last N days)
        if cutoff_days:
            cutoff = datetime.now() - timedelta(days=cutoff_days)
            combined_df = combined_df[combined_df.index >= cutoff]

        # Save updated cache
        combined_df.to_csv(cache_file)

        new_rows = len(combined_df) - len(existing_df)
        if verbose:
            print(f"  ✅ Updated cache: added {new_rows} new rows")
            print(f"  Total rows: {len(combined_df)} (from {combined_df.index[0]} to {combined_df.index[-1]})")

        return combined_df

    else:
        # No cache, fetch full history
        if verbose:
            print(f"  No cache found, fetching full history...")

        # Determine fetch parameters for initial download
        if fetch_source == 'yahoo':
            period_arg = base_period or '2y'
            result = get_bitcoin_data(
                source='yahoo',
                period=period_arg,
                interval='1d',
                return_dict=True,
                verbose=verbose
            )
        else:
            result = get_bitcoin_data(
                source=fetch_source,
                days=days,
                return_dict=True,
                verbose=verbose
            )

        if result['status'] != 'success' or result['data'] is None:
            if verbose:
                print(f"  ❌ Failed to fetch data")
            return None

        df = result['data']

        # Save to cache
        df.to_csv(cache_file)

        if verbose:
            print(f"  ✅ Saved cache: {len(df)} rows to {cache_file}")

        return df


class CryptoDataFetcher:
    """Fetch Bitcoin data from free APIs (no authentication needed)"""

    def __init__(self, verbose=False):
        """
        Args:
            verbose (bool): Print status messages (default: False)
        """
        self.verbose = verbose

    def _print(self, *args, **kwargs):
        """Conditionally print if verbose=True"""
        if self.verbose:
            print(*args, **kwargs)

    def _fetch_cryptocompare(self, interval='15m', days=60):
        """
        Fetch Bitcoin data from CryptoCompare API with pagination
        Primary source for GitHub Actions (no geo-blocking)
        
        Args:
            interval (str): '15m' or '1h'
            days (int): Days of history
            
        Returns:
            pd.DataFrame or None
        """
        try:
            if interval == '15m':
                endpoint = 'histominute'
                aggregate = 1
                samples_per_day = 96
                self._print(f"Fetching {days}d of 15-min data via CryptoCompare...")
            elif interval == '1h':
                endpoint = 'histohour'
                aggregate = 1
                samples_per_day = 24
                self._print(f"Fetching {days}d of hourly data via CryptoCompare...")
            else:
                self._print(f"Invalid interval: {interval}")
                return None
            
            total_samples_needed = days * samples_per_day
            max_per_request = 2000
            
            # If we need more than 2000 samples, paginate
            if total_samples_needed > max_per_request:
                num_requests = (total_samples_needed + max_per_request - 1) // max_per_request
                self._print(f"Need {total_samples_needed} samples, fetching in {num_requests} batches...")
                
                all_candles = []
                to_timestamp = None  # Start from now, go backwards
                
                for i in range(num_requests):
                    remaining = total_samples_needed - len(all_candles)
                    limit = min(remaining, max_per_request)
                    
                    url = f'https://min-api.cryptocompare.com/data/v2/{endpoint}'
                    params = {
                        'fsym': 'BTC',
                        'tsym': 'USD',
                        'limit': limit,
                        'aggregate': aggregate
                    }
                    
                    if to_timestamp is not None:
                        params['toTs'] = to_timestamp
                    
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    if data['Response'] != 'Success':
                        self._print(f"API Error: {data.get('Message', 'Unknown error')}")
                        return None
                    
                    batch_candles = data['Data']['Data']
                    
                    if not batch_candles:
                        break
                    
                    all_candles.extend(batch_candles)
                    
                    # Set next batch to end where this one started (go backwards in time)
                    to_timestamp = batch_candles[0]['time'] - 1
                    
                    self._print(f"  Batch {i+1}/{num_requests}: {len(batch_candles)} samples, total: {len(all_candles)}")
                    
                    if len(all_candles) >= total_samples_needed:
                        break
                
                candles = all_candles
                
            else:
                # Single request is enough
                url = f'https://min-api.cryptocompare.com/data/v2/{endpoint}'
                params = {
                    'fsym': 'BTC',
                    'tsym': 'USD',
                    'limit': total_samples_needed,
                    'aggregate': aggregate
                }
                
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                
                if data['Response'] != 'Success':
                    self._print(f"API Error: {data.get('Message', 'Unknown error')}")
                    return None
                
                candles = data['Data']['Data']
            
            if not candles:
                self._print("No data returned")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volumefrom']]
            df.rename(columns={'volumefrom': 'volume'}, inplace=True)
            
            # Sort by timestamp (pagination fetches backwards)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            df = df[~df.index.duplicated(keep='first')]
            
            self._print(f"Success: {len(df)} samples, ${df['close'].iloc[-1]:,.2f}")
            return df
            
        except Exception as e:
            self._print(f"CryptoCompare error: {e}")
            return None

    def fetch_cryptocompare_15min(self, symbol='BTCUSDT', days=365):
        """
        Fetch 15-minute Bitcoin data from CryptoCompare

        Args:
            symbol (str): Trading pair (ignored, kept for compatibility)
            days (int): Days of history (default: 60)

        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        return self._fetch_cryptocompare(interval='15m', days=days)

    def fetch_cryptocompare_1hour(self, symbol='BTCUSDT', days=365):
        """
        Fetch 1-hour Bitcoin data from CryptoCompare

        Args:
            symbol (str): Trading pair (ignored, kept for compatibility)
            days (int): Days of history (default: 365)

        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        return self._fetch_cryptocompare(interval='1h', days=days)

    def fetch_yahoo_daily(self, ticker='BTC-USD', period='2y', interval='1d'):
        """
        Fetch daily Bitcoin data from Yahoo Finance

        Args:
            ticker (str): Ticker symbol (default: BTC-USD)
            period (str): Time period - '1mo', '1y', '2y', '5y', 'max'
            interval (str): Interval - '1d', '1h', '1wk', '1mo'

        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        try:
            self._print(f"Fetching {period} {interval} data from Yahoo...")

            btc = yf.Ticker(ticker)
            df = btc.history(period=period, interval=interval)

            if df is None or df.empty:
                self._print("No data returned from Yahoo")
                return None

            self._print(f"Received {len(df)} candles")

            # Clean up
            df = df.reset_index()
            df.columns = [col.lower() if col != 'Date' else 'timestamp' for col in df.columns]

            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.drop('date', axis=1)

            available_cols = [col for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            if col in df.columns]
            df = df[available_cols]
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            df.index = df.index.tz_localize(None)

            self._print(f"Success: {len(df)} samples, ${df['close'].iloc[-1]:.2f}")
            return df

        except Exception as e:
            self._print(f"Error: {e}")
            return None

    def save_to_csv(self, df, filename, data_dir='data/raw'):
        """
        Save DataFrame to CSV

        Args:
            df: DataFrame to save
            filename: Output filename
            data_dir: Directory (default: data/raw)

        Returns:
            bool: Success status
        """
        if df is None or df.empty:
            self._print("Cannot save: DataFrame is empty")
            return False

        try:
            import os
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath, index=True)
            file_size = os.path.getsize(filepath) / 1024
            self._print(f"Saved: {filepath} ({file_size:.2f} KB, {len(df)} rows)")
            return True

        except Exception as e:
            self._print(f"Save error: {e}")
            return False

    def fetch_all(self, days=365, include_yahoo=True, yahoo_period='2y'):
        """
        Fetch from all sources

        Returns:
            dict: {'cryptocompare_1h': df, 'yahoo': df}
        """
        results = {}
        results['cryptocompare_1h'] = self._fetch_cryptocompare(interval='1h', days=days)
        if include_yahoo:
            time.sleep(1)
            results['yahoo'] = self.fetch_yahoo_daily(period=yahoo_period)
        return results


# =============================================================================
# BACKEND-FRIENDLY FUNCTIONS (Silent by default)
# =============================================================================

def get_bitcoin_data(source='yahoo', days=365, period='2y', symbol='BTCUSDT',
                     ticker='BTC-USD', interval='1d', return_dict=False, verbose=False):
    """
    Fetch Bitcoin data (for backend/API use)

    Args:
        source: 'yahoo' or 'cryptocompare_1h'
        days: Days for Cryptocompare (default: 365)
        period: Period for Yahoo - '1mo', '1y', '2y', '5y'
        symbol: Cryptocompare trading pair (default: BTCUSDT)
        ticker: Yahoo ticker (default: BTC-USD)
        interval: Yahoo interval (default: 1d)
        return_dict: Return dict with metadata (default: False)
        verbose: Print status messages (default: False)

    Returns:
        DataFrame or dict with status/metadata

    Example:
        df = get_bitcoin_data('yahoo', period='1y')
        df = get_bitcoin_data('cryptocompare_1h', days=365)  # 1-hour candles
        result = get_bitcoin_data('yahoo', period='1y', return_dict=True)
    """
    fetcher = CryptoDataFetcher(verbose=verbose)
    df = None

    try:
        if source.lower() == 'yahoo':
            df = fetcher.fetch_yahoo_daily(ticker=ticker, period=period, interval=interval)
        elif source.lower() == 'cryptocompare_1h':
            df = fetcher.fetch_cryptocompare_1hour(symbol=symbol, days=days)
        else:
            raise ValueError(f"Invalid source '{source}'. Valid options: 'yahoo', 'cryptocompare_1h'")

        if df is None or df.empty:
            if return_dict:
                return {'data': None, 'source': source, 'samples': 0,
                       'date_range': None, 'status': 'error',
                       'message': f'No data from {source}'}
            return None

        if return_dict:
            # Get latest price from 'close' column (Yahoo/Cryptocompare)
            if 'close' in df.columns:
                latest_price = float(df['close'].iloc[-1])
            elif 'price' in df.columns:
                latest_price = float(df['price'].iloc[-1])
            else:
                latest_price = None

            return {
                'data': df,
                'source': source,
                'samples': len(df),
                'date_range': {'start': str(df.index[0]), 'end': str(df.index[-1])},
                'status': 'success',
                'columns': list(df.columns),
                'latest_price': latest_price
            }

        return df

    except Exception as e:
        if return_dict:
            return {'data': None, 'source': source, 'samples': 0,
                   'date_range': None, 'status': 'error', 'message': str(e)}
        raise


def get_latest_price(ticker='BTC-USD'):
    """
    Get current Bitcoin price from Yahoo Finance (fast and reliable)
    
    Returns:
        dict: {
            'price': float,
            'timestamp': str (ISO format),
            'status': 'success' or 'error',
            'message': str (if error)
        }
    
    Example:
        result = get_current_btc_price()
        if result['status'] == 'success':
            print(f"BTC: ${result['price']:,.2f}")
    """
    try:
        # Method 1: Try fast API (Ticker.info) - instant
        btc = yf.Ticker(ticker)
        price = btc.info.get('regularMarketPrice') or btc.info.get('currentPrice')
        
        if price:
            return {
                'price': float(price),
                'timestamp': datetime.now().isoformat(),
                'status': 'success'
            }
        
        # Method 2: Fallback to last candle (1 day history) - ~1 second
        df = btc.history(period='1d', interval='1m')
        if not df.empty:
            return {
                'price': float(df['Close'].iloc[-1]),
                'timestamp': df.index[-1].isoformat(),
                'status': 'success'
            }
        
        # Both methods failed
        return {
            'price': None,
            'timestamp': None,
            'status': 'error',
            'message': 'No price data available'
        }
        
    except Exception as e:
        return {
            'price': None,
            'timestamp': None,
            'status': 'error',
            'message': str(e)
        }

def save_bitcoin_data(source='yahoo', filename=None, data_dir='data/raw', **kwargs):
    """
    Fetch and save Bitcoin data to CSV

    Args:
        source: 'yahoo' or 'cryptocompare_1h'
        filename: Output filename (auto-generated if None)
        data_dir: Save directory (default: data/raw)
        **kwargs: Passed to get_bitcoin_data()

    Returns:
        dict: {status, filepath, filename, file_size_kb, samples}

    Example:
        result = save_bitcoin_data('yahoo', period='2y')
    """
    import os

    try:
        result = get_bitcoin_data(source=source, return_dict=True, **kwargs)

        if result['status'] == 'error' or result['data'] is None:
            return {'status': 'error',
                   'message': result.get('message', 'Failed to fetch'),
                   'filepath': None}

        df = result['data']

        # Auto-generate filename
        if filename is None:
            period = kwargs.get('period', '2y')
            days = kwargs.get('days', 60)
            if source == 'yahoo':
                filename = f'btc_yahoo_{period}_daily.csv'
            elif source == 'cryptocompare_1h':
                filename = f'btc_cryptocompare_{days}d_1hour.csv'

        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, filename)
        df.to_csv(filepath, index=True)

        return {
            'status': 'success',
            'filepath': filepath,
            'filename': filename,
            'file_size_kb': round(os.path.getsize(filepath) / 1024, 2),
            'samples': len(df),
            'source': source,
            'date_range': result['date_range']
        }

    except Exception as e:
        return {'status': 'error', 'message': str(e), 'filepath': None}


def get_fear_greed_index(limit=730, verbose=False):
    """
    Fetch Fear & Greed Index from alternative.me API
    
    Crypto Fear & Greed Index (0-100):
    - 0-24: Extreme Fear (buy signal)
    - 25-49: Fear
    - 50-74: Greed
    - 75-100: Extreme Greed (sell signal)
    
    Args:
        limit (int): Number of days to fetch (default: 730 = 2 years)
        verbose (bool): Print detailed output (default: False)
    
    Returns:
        pd.DataFrame with columns:
            - timestamp (index): datetime
            - fear_greed_value: 0-100 sentiment score
            - fear_greed_class: text classification
    """
    try:
        if verbose:
            print(f"\n📊 Fetching Fear & Greed Index ({limit} days)...")
        
        # API endpoint (supports up to 1000+ records)
        url = f"https://api.alternative.me/fng/?limit={min(limit, 1000)}"
        
        # Make request
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            if verbose:
                print(f"   ❌ HTTP {response.status_code}")
            return None
        
        data = response.json()
        
        if 'data' not in data or not data['data']:
            if verbose:
                print("   ⚠️  No data returned")
            return None
        
        # Convert to DataFrame
        records = []
        for item in data['data']:
            records.append({
                'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                'fear_greed_value': int(item['value']),
                'fear_greed_class': item['value_classification']
            })
        
        df = pd.DataFrame(records)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        if verbose:
            print(f"   ✓ Fetched {len(df)} days")
            print(f"   Range: {df.index[0].date()} to {df.index[-1].date()}")
            print(f"   Current: {df['fear_greed_value'].iloc[-1]} ({df['fear_greed_class'].iloc[-1]})")
        
        return df
    
    except Exception as e:
        if verbose:
            print(f"   ❌ Error: {e}")
        return None

def get_all_sources(days=365, yahoo_period='2y', save_to_disk=False):
    """
    Fetch from all sources (Daily + Hourly only)

    Args:
        days: Days for Cryptocompare hourly (default: 365)
        yahoo_period: Period for Yahoo daily (default: '2y')
        save_to_disk: Save CSV files (default: False)

    Returns:
        dict: {'yahoo': result, 'cryptocompare_1h': result}

    Example:
        all_data = get_all_sources(days=365, yahoo_period='2y')
        if all_data['yahoo']['status'] == 'success':
            df = all_data['yahoo']['data']
    """
    results = {}

    # Daily data: Yahoo Finance (2 years)
    results['yahoo'] = get_bitcoin_data('yahoo', period=yahoo_period, return_dict=True)
    time.sleep(1)
    
    # Hourly data: CryptoCompare (365 days)
    results['cryptocompare_1h'] = get_bitcoin_data('cryptocompare_1h', days=days, return_dict=True)

    if save_to_disk:
        for source, data in results.items():
            if data['status'] == 'success' and data['data'] is not None:
                save_result = save_bitcoin_data(source=source, days=days, period=yahoo_period)
                results[source]['saved'] = save_result['status'] == 'success'
                results[source]['filepath'] = save_result.get('filepath')

    return results