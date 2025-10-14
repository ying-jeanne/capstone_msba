"""
Crypto Data Fetcher - Fetch Bitcoin price data from free APIs
Sources: Yahoo Finance (daily), Binance (15-min, hourly)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
import yfinance as yf
import time
import requests


class CryptoDataFetcher:
    """Fetch Bitcoin data from free APIs (no keys needed)"""

    def __init__(self, verbose=False):
        """
        Args:
            verbose (bool): Print status messages (default: False)
        """
        self.verbose = verbose
        self._binance_client = None

    @property
    def binance_client(self):
        """Lazy initialization of Binance client (may fail in geo-restricted locations)"""
        if self._binance_client is None:
            try:
                self._binance_client = Client(api_key=None, api_secret=None)
            except Exception as e:
                self._print(f"Warning: Binance SDK initialization failed: {e}")
                self._print("Will use direct REST API as fallback")
                # Don't raise - we'll use direct API instead
                pass
        return self._binance_client

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

    def fetch_binance_15min(self, symbol='BTCUSDT', days=60):
        """
        Fetch 15-minute Bitcoin data from Binance

        Args:
            symbol (str): Trading pair (default: BTCUSDT)
            days (int): Days of history (default: 60)

        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        try:
            self._print(f"Fetching {days}d of 15-min data from Binance...")

            # Try SDK first
            if self.binance_client is not None:
                # Calculate time range
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days)
                start_ts = int(start_time.timestamp() * 1000)
                end_ts = int(end_time.timestamp() * 1000)

                # Fetch klines (no limit - get all available data in time range)
                # Binance free tier: 6000 weight/min, this call = 1 weight
                klines = self.binance_client.get_historical_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_15MINUTE,
                    start_str=start_ts,
                    end_str=end_ts
                )

                if not klines:
                    self._print("No data returned from Binance")
                    return None

                self._print(f"Received {len(klines)} candles")

                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
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

                self._print(f"Success: {len(df)} samples, ${df['close'].iloc[-1]:.2f}")
                return df
            else:
                # SDK failed, use direct API
                self._print("SDK unavailable, using direct REST API...")
                return self._fetch_binance_direct(symbol=symbol, interval='15m', days=days)

        except Exception as e:
            self._print(f"SDK Error: {e}, trying direct API...")
            return self._fetch_binance_direct(symbol=symbol, interval='15m', days=days)

    def fetch_binance_1hour(self, symbol='BTCUSDT', days=60):
        """
        Fetch 1-hour Bitcoin data from Binance

        Args:
            symbol (str): Trading pair (default: BTCUSDT)
            days (int): Days of history (default: 60)

        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        try:
            self._print(f"Fetching {days}d of 1-hour data from Binance...")

            # Try SDK first
            if self.binance_client is not None:
                # Calculate time range
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days)
                start_ts = int(start_time.timestamp() * 1000)
                end_ts = int(end_time.timestamp() * 1000)

                # Fetch klines (no limit - get all available data in time range)
                # Binance free tier: 6000 weight/min, this call = 1 weight
                klines = self.binance_client.get_historical_klines(
                    symbol=symbol,
                    interval=Client.KLINE_INTERVAL_1HOUR,
                    start_str=start_ts,
                    end_str=end_ts
                )

                if not klines:
                    self._print("No data returned from Binance")
                    return None

                self._print(f"Received {len(klines)} candles")

                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
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

                self._print(f"Success: {len(df)} samples, ${df['close'].iloc[-1]:.2f}")
                return df
            else:
                # SDK failed, use direct API
                self._print("SDK unavailable, using direct REST API...")
                return self._fetch_binance_direct(symbol=symbol, interval='1h', days=days)

        except Exception as e:
            self._print(f"SDK Error: {e}, trying direct API...")
            return self._fetch_binance_direct(symbol=symbol, interval='1h', days=days)

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
                     ticker='BTC-USD', interval='1d', return_dict=False):
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

    Returns:
        DataFrame or dict with status/metadata

    Example:
        df = get_bitcoin_data('yahoo', period='1y')
        df = get_bitcoin_data('binance_1h', days=60)  # 1-hour candles
        df = get_bitcoin_data('binance', days=60)     # 15-min candles
        result = get_bitcoin_data('yahoo', period='1y', return_dict=True)
    """
    fetcher = CryptoDataFetcher(verbose=False)
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