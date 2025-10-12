"""
Crypto Data Fetcher - Fetch Bitcoin price data from free APIs
Sources: Yahoo Finance (daily), Binance (15-min), CoinGecko (hourly)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from pycoingecko import CoinGeckoAPI
import yfinance as yf
import time


class CryptoDataFetcher:
    """Fetch Bitcoin data from free APIs (no keys needed)"""

    def __init__(self, verbose=False):
        """
        Args:
            verbose (bool): Print status messages (default: False)
        """
        self.verbose = verbose
        self.binance_client = Client(api_key=None, api_secret=None)
        self.coingecko_client = CoinGeckoAPI()

    def _print(self, *args, **kwargs):
        """Conditionally print if verbose=True"""
        if self.verbose:
            print(*args, **kwargs)

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

        except Exception as e:
            self._print(f"Error: {e}")
            return None

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

        except Exception as e:
            self._print(f"Error: {e}")
            return None

    def fetch_coingecko_hourly(self, coin_id='bitcoin', vs_currency='usd', days=60):
        """
        Fetch hourly Bitcoin data from CoinGecko

        Args:
            coin_id (str): Coin ID (default: bitcoin)
            vs_currency (str): Currency (default: usd)
            days (int): Days of history (1-90, default: 60)
                        Note: CoinGecko only accepts: 1, 7, 14, 30, 90, 180, 365
                        If invalid value provided, will use nearest valid value

        Returns:
            pd.DataFrame: OHLCV data with timestamp index
        """
        try:
            # CoinGecko OHLC API only accepts specific values
            valid_days = [1, 7, 14, 30, 90, 180, 365]

            # Find nearest valid value
            if days not in valid_days:
                # Round up to next valid value
                valid_days_sorted = sorted(valid_days)
                days_adjusted = next((d for d in valid_days_sorted if d >= days), valid_days_sorted[-1])
                if self.verbose:
                    self._print(f"Note: CoinGecko OHLC API doesn't support days={days}, using {days_adjusted} instead")
                days = days_adjusted

            self._print(f"Fetching {days}d hourly data from CoinGecko...")

            # Fetch OHLC data (provides open, high, low, close)
            ohlc_data = self.coingecko_client.get_coin_ohlc_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )

            if not ohlc_data:
                self._print("No OHLC data returned from CoinGecko")
                return None

            self._print(f"Received {len(ohlc_data)} OHLC candles")

            # Convert to DataFrame
            # CoinGecko OHLC format: [timestamp, open, high, low, close]
            df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Add volume column (CoinGecko OHLC doesn't include volume, so we'll use a placeholder)
            # We'll fetch volume data separately
            market_data = self.coingecko_client.get_coin_market_chart_by_id(
                id=coin_id,
                vs_currency=vs_currency,
                days=days
            )

            if 'total_volumes' in market_data and market_data['total_volumes']:
                vol_df = pd.DataFrame(market_data['total_volumes'], columns=['timestamp', 'volume'])
                vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')

                # Merge volume data with OHLC
                df = df.merge(vol_df, on='timestamp', how='left')
            else:
                # If no volume data, use zero
                df['volume'] = 0.0

            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)

            self._print(f"Success: {len(df)} samples, ${df['close'].iloc[-1]:.2f}")
            return df

        except Exception as e:
            self._print(f"Error: {e}")
            return None

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
            dict: {'binance': df, 'coingecko': df, 'yahoo': df}
        """
        results = {}
        results['binance'] = self.fetch_binance_15min(days=days)
        time.sleep(1)
        results['coingecko'] = self.fetch_coingecko_hourly(days=days)
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
        source: 'yahoo', 'binance', 'binance_1h', or 'coingecko'
        days: Days for Binance/CoinGecko (default: 60)
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
        elif source.lower() == 'coingecko':
            df = fetcher.fetch_coingecko_hourly(days=days)
        else:
            raise ValueError(f"Invalid source '{source}'")

        if df is None or df.empty:
            if return_dict:
                return {'data': None, 'source': source, 'samples': 0,
                       'date_range': None, 'status': 'error',
                       'message': f'No data from {source}'}
            return None

        if return_dict:
            # Get latest price from either 'close' (Yahoo/Binance) or 'price' (CoinGecko)
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
        source: 'yahoo', 'binance', or 'coingecko'
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
        source: 'yahoo', 'binance', or 'coingecko'
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
            elif source == 'coingecko':
                filename = f'btc_coingecko_{days}d_hourly.csv'

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
        days: Days for Binance/CoinGecko
        yahoo_period: Period for Yahoo
        save_to_disk: Save CSV files (default: False)

    Returns:
        dict: {'yahoo': result, 'binance': result, 'coingecko': result}

    Example:
        all_data = get_all_sources(days=30, yahoo_period='1y')
        if all_data['yahoo']['status'] == 'success':
            df = all_data['yahoo']['data']
    """
    results = {}

    results['yahoo'] = get_bitcoin_data('yahoo', period=yahoo_period, return_dict=True)
    time.sleep(1)
    results['coingecko'] = get_bitcoin_data('coingecko', days=days, return_dict=True)
    time.sleep(1)
    results['binance'] = get_bitcoin_data('binance', days=days, return_dict=True)

    if save_to_disk:
        for source, data in results.items():
            if data['status'] == 'success' and data['data'] is not None:
                save_result = save_bitcoin_data(source=source, days=days, period=yahoo_period)
                results[source]['saved'] = save_result['status'] == 'success'
                results[source]['filepath'] = save_result.get('filepath')

    return results