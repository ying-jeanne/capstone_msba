"""
Crypto Data Fetcher - Fetch Bitcoin price data from free APIs
Sources: Yahoo Finance (daily), Binance (15-min, hourly)

Note: Uses direct Binance REST API (no SDK) to avoid IP blocking in GitHub Actions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import time
import requests
from pathlib import Path


def get_bitcoin_data_incremental(source='binance_1h', days=365, cache_dir='data/raw', verbose=True):
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
        source (str): 'yahoo', 'binance' (15-min), or 'binance_1h' (hourly)
        days (int): Days to fetch if no cache exists (default: 365)
        cache_dir (str): Directory for cached files (default: 'data/raw')
        verbose (bool): Print status messages (default: True)

    Returns:
        pd.DataFrame or None

    Example:
        # First call: fetches 365 days
        df = get_bitcoin_data_incremental('binance_1h', days=365)

        # Later calls: only fetches new data since last timestamp
        df = get_bitcoin_data_incremental('binance_1h', days=365)
    """
    import os

    # Determine cache filename
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    if source == 'yahoo':
        cache_file = cache_path / 'btc_yahoo_2y_daily.csv'
    elif source == 'binance':
        cache_file = cache_path / 'btc_binance_60d_15min.csv'
    elif source == 'binance_1h':
        cache_file = cache_path / 'btc_binance_365d_1hour.csv'
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
        result = get_bitcoin_data(source=source, days=days_to_fetch, return_dict=True, verbose=verbose)

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
        if source == 'yahoo':
            # Keep 2 years for Yahoo
            cutoff = datetime.now() - timedelta(days=730)
        elif source == 'binance':
            # Keep 60 days for 15-min
            cutoff = datetime.now() - timedelta(days=60)
        elif source == 'binance_1h':
            # Keep 365 days for hourly
            cutoff = datetime.now() - timedelta(days=365)

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
            print(f"  No cache found, fetching full {days} days...")

        result = get_bitcoin_data(source=source, days=days, period='2y', return_dict=True, verbose=verbose)

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

    def _fetch_binance_direct(self, symbol='BTCUSDT', interval='1h', days=60):
        """
        Fetch Binance data via direct REST API (no SDK, no geo-restrictions)

        This is a fallback when the Binance SDK fails (e.g., GitHub Actions)
        Uses public Binance API: https://api.binance.com/api/v3/klines

        Supports pagination to fetch more than 1000 candles (Binance limit per request)

        Args:
            symbol (str): Trading pair (default: BTCUSDT)
            interval (str): '15m' or '1h' (default: 1h)
            days (int): Days of history (default: 60)

        Returns:
            pd.DataFrame or None
        """
        try:
            self._print(f"Fetching {days}d of {interval} data via Binance REST API...")

            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            start_ts = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)

            # Binance limit: 1000 candles per request, need pagination
            all_klines = []
            url = 'https://api.binance.com/api/v3/klines'
            current_start = start_ts

            while current_start < end_ts:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': current_start,
                    'endTime': end_ts,
                    'limit': 1000  # Binance max per request
                }

                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                klines = response.json()

                if not klines:
                    break

                all_klines.extend(klines)

                # Move to next batch (last candle's close time + 1ms)
                current_start = int(klines[-1][6]) + 1  # close_time is index 6

                # Small delay to respect rate limits
                if current_start < end_ts:
                    time.sleep(0.1)

            if not all_klines:
                self._print("No data returned from Binance API")
                return None

            self._print(f"Received {len(all_klines)} candles (paginated)")

            # Convert to DataFrame
            # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
            df = pd.DataFrame(all_klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Clean up
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            # Remove duplicates if any
            df = df[~df.index.duplicated(keep='first')]

            self._print(f"Success: {len(df)} samples, ${df['close'].iloc[-1]:.2f}")
            return df

        except Exception as e:
            self._print(f"Error with direct Binance API: {e}")
            return None

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

    def fetch_binance_15min(self, symbol='BTCUSDT', days=60):
        """
        Fetch 15-minute Bitcoin data from CryptoCompare

        Args:
            symbol (str): Trading pair (ignored, kept for compatibility)
            days (int): Days of history (default: 60)

        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        return self._fetch_cryptocompare(interval='15m', days=days)

    def fetch_binance_1hour(self, symbol='BTCUSDT', days=365):
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

    def fetch_all(self, days=60, include_yahoo=True, yahoo_period='2y'):
        """
        Fetch from all sources

        Returns:
            dict: {'binance': df, 'binance_1h': df, 'yahoo': df}
        """
        results = {}
        results['binance'] = self.fetch_binance_15min(days=days)
        time.sleep(1)
        results['binance_1h'] = self.fetch_binance_1hour(days=days)
        if include_yahoo:
            time.sleep(1)
            results['yahoo'] = self.fetch_yahoo_daily(period=yahoo_period)
        return results


# =============================================================================
# BACKEND-FRIENDLY FUNCTIONS (Silent by default)
# =============================================================================

def get_bitcoin_data(source='yahoo', days=60, period='2y', symbol='BTCUSDT',
                     ticker='BTC-USD', interval='1d', return_dict=False, verbose=False):
    """
    Fetch Bitcoin data (for backend/API use)

    Args:
        source: 'yahoo', 'binance', or 'binance_1h'
        days: Days for Binance (default: 60)
        period: Period for Yahoo - '1mo', '1y', '2y', '5y'
        symbol: Binance trading pair (default: BTCUSDT)
        ticker: Yahoo ticker (default: BTC-USD)
        interval: Yahoo interval (default: 1d)
        return_dict: Return dict with metadata (default: False)
        verbose: Print status messages (default: False)

    Returns:
        DataFrame or dict with status/metadata

    Example:
        df = get_bitcoin_data('yahoo', period='1y')
        df = get_bitcoin_data('binance_1h', days=60)  # 1-hour candles
        df = get_bitcoin_data('binance', days=60)     # 15-min candles
        result = get_bitcoin_data('yahoo', period='1y', return_dict=True)
    """
    fetcher = CryptoDataFetcher(verbose=verbose)
    df = None

    try:
        if source.lower() == 'yahoo':
            df = fetcher.fetch_yahoo_daily(ticker=ticker, period=period, interval=interval)
        elif source.lower() == 'binance':
            df = fetcher.fetch_binance_15min(symbol=symbol, days=days)
        elif source.lower() == 'binance_1h':
            df = fetcher.fetch_binance_1hour(symbol=symbol, days=days)
        else:
            raise ValueError(f"Invalid source '{source}'. Valid options: 'yahoo', 'binance', 'binance_1h'")

        if df is None or df.empty:
            if return_dict:
                return {'data': None, 'source': source, 'samples': 0,
                       'date_range': None, 'status': 'error',
                       'message': f'No data from {source}'}
            return None

        if return_dict:
            # Get latest price from 'close' column (Yahoo/Binance)
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


def get_latest_price(source='yahoo', ticker='BTC-USD'):
    """
    Get latest Bitcoin price

    Args:
        source: 'yahoo' or 'binance'
        ticker: Ticker symbol (default: BTC-USD)

    Returns:
        dict: {price, timestamp, source, currency, status}

    Example:
        price_info = get_latest_price('yahoo')
        print(f"BTC: ${price_info['price']:,.2f}")
    """
    try:
        result = get_bitcoin_data(source=source, ticker=ticker, days=7, return_dict=True)

        if result['status'] == 'error' or result['data'] is None:
            return {'price': None, 'timestamp': None, 'source': source,
                   'currency': 'USD', 'status': 'error',
                   'message': result.get('message', 'Failed to fetch')}

        df = result['data']
        # Get price from either 'close' or 'price' column
        if 'close' in df.columns:
            price = float(df['close'].iloc[-1])
        elif 'price' in df.columns:
            price = float(df['price'].iloc[-1])
        else:
            price = None

        return {
            'price': price,
            'timestamp': str(df.index[-1]),
            'source': source,
            'currency': 'USD',
            'status': 'success'
        }

    except Exception as e:
        return {'price': None, 'timestamp': None, 'source': source,
               'currency': 'USD', 'status': 'error', 'message': str(e)}


def save_bitcoin_data(source='yahoo', filename=None, data_dir='data/raw', **kwargs):
    """
    Fetch and save Bitcoin data to CSV

    Args:
        source: 'yahoo', 'binance', or 'binance_1h'
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
            elif source == 'binance':
                filename = f'btc_binance_{days}d_15min.csv'
            elif source == 'binance_1h':
                filename = f'btc_binance_{days}d_1hour.csv'

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


def get_all_sources(days=60, yahoo_period='2y', save_to_disk=False):
    """
    Fetch from all sources

    Args:
        days: Days for Binance
        yahoo_period: Period for Yahoo
        save_to_disk: Save CSV files (default: False)

    Returns:
        dict: {'yahoo': result, 'binance': result, 'binance_1h': result}

    Example:
        all_data = get_all_sources(days=30, yahoo_period='1y')
        if all_data['yahoo']['status'] == 'success':
            df = all_data['yahoo']['data']
    """
    results = {}

    results['yahoo'] = get_bitcoin_data('yahoo', period=yahoo_period, return_dict=True)
    time.sleep(1)
    results['binance'] = get_bitcoin_data('binance', days=days, return_dict=True)
    time.sleep(1)
    results['binance_1h'] = get_bitcoin_data('binance_1h', days=days, return_dict=True)

    if save_to_disk:
        for source, data in results.items():
            if data['status'] == 'success' and data['data'] is not None:
                save_result = save_bitcoin_data(source=source, days=days, period=yahoo_period)
                results[source]['saved'] = save_result['status'] == 'success'
                results[source]['filepath'] = save_result.get('filepath')

    return results