"""
Sentiment Data Fetcher Module
Collects FREE sentiment indicators for Bitcoin price prediction

Sentiment data can be powerful predictors because:
1. Market psychology drives price movements
2. Search interest often precedes price changes
3. Fear & Greed indicates market extremes (buy/sell signals)
4. Social sentiment captures retail investor behavior

This module provides sentiment features to complement technical indicators.
"""

import pandas as pd
import numpy as np
import requests
from pytrends.request import TrendReq
from datetime import datetime, timedelta
import time
import json


class SentimentFetcher:
    """
    Fetch Bitcoin sentiment data from FREE sources

    Data Sources:
    1. Fear & Greed Index - Market sentiment indicator (0-100)
    2. Google Trends - Search interest over time

    Both are proven to correlate with Bitcoin price movements and provide
    valuable features for prediction models.
    """

    def __init__(self):
        """
        Initialize sentiment fetcher
        """
        print("✓ SentimentFetcher initialized (using FREE APIs)")


    def fetch_fear_greed_index(self, limit=90):
        """
        Fetch Fear & Greed Index from alternative.me API

        The Crypto Fear & Greed Index is calculated from:
        - Volatility (25%): Current volatility vs average (high vol = fear)
        - Market Momentum/Volume (25%): Current volume vs average
        - Social Media (15%): Twitter hashtags, engagement, mentions
        - Surveys (15%): Weekly crypto polls
        - Dominance (10%): Bitcoin dominance increase = fear of alts
        - Trends (10%): Google search trends for Bitcoin-related terms

        Index Values:
        - 0-24: Extreme Fear (potential buying opportunity)
        - 25-49: Fear
        - 50-74: Greed
        - 75-100: Extreme Greed (potential sell signal)

        Why it matters:
        - Extreme fear often precedes price bottoms (buy signal)
        - Extreme greed often precedes price tops (sell signal)
        - Helps identify market sentiment shifts
        - Mean reversion indicator

        Args:
            limit (int): Number of days of historical data (default: 90)

        Returns:
            pd.DataFrame: Fear & Greed data with columns:
                - timestamp: datetime index
                - fear_greed_value: Index value (0-100)
                - fear_greed_class: Classification (Extreme Fear, Fear, Greed, etc.)

        Notes:
            - Free API with no authentication required
            - Updated daily at ~00:00 UTC
            - Rate limit: Reasonable (no official limit, but be respectful)
            - Historical data available back to 2018
        """
        try:
            print(f"\n📊 Fetching Fear & Greed Index...")
            print(f"   Source: alternative.me API")
            print(f"   Limit: {limit} days")

            # API endpoint
            url = f"https://api.alternative.me/fng/?limit={limit}"

            print("   Fetching data...")

            # Make request
            response = requests.get(url, timeout=10)

            # Check response status
            if response.status_code != 200:
                print(f"   ❌ Error: HTTP {response.status_code}")
                return None

            # Parse JSON
            data = response.json()

            # Check if data exists
            if 'data' not in data or not data['data']:
                print("   ⚠️  No data returned from API")
                return None

            print(f"   ✓ Received {len(data['data'])} data points")

            # Convert to DataFrame
            records = []
            for item in data['data']:
                records.append({
                    'timestamp': datetime.fromtimestamp(int(item['timestamp'])),
                    'fear_greed_value': int(item['value']),
                    'fear_greed_class': item['value_classification']
                })

            df = pd.DataFrame(records)

            # Set timestamp as index
            df.set_index('timestamp', inplace=True)

            # Sort by timestamp (ascending)
            df.sort_index(inplace=True)

            # Print summary
            print(f"\n   📈 Fear & Greed Index Summary:")
            print(f"      Samples: {len(df)}")
            print(f"      Date range: {df.index[0]} to {df.index[-1]}")
            print(f"      Value range: {df['fear_greed_value'].min()} - {df['fear_greed_value'].max()}")
            print(f"      Current value: {df['fear_greed_value'].iloc[-1]} ({df['fear_greed_class'].iloc[-1]})")
            print(f"      Average: {df['fear_greed_value'].mean():.1f}")

            # Distribution
            print(f"\n   Distribution:")
            class_counts = df['fear_greed_class'].value_counts()
            for cls, count in class_counts.items():
                print(f"      {cls}: {count} days ({count/len(df)*100:.1f}%)")

            return df

        except requests.exceptions.Timeout:
            print(f"\n   ❌ Error: Request timeout")
            print(f"      → Check your internet connection")
            return None
        except requests.exceptions.RequestException as e:
            print(f"\n   ❌ Error fetching Fear & Greed Index: {e}")
            print(f"      Error type: {type(e).__name__}")
            return None
        except Exception as e:
            print(f"\n   ❌ Unexpected error: {e}")
            print(f"      Error type: {type(e).__name__}")
            return None


    def fetch_google_trends(self, keyword='Bitcoin', timeframe='today 3-m', geo=''):
        """
        Fetch Google Trends data for Bitcoin search interest

        Google Trends indicates:
        - Public interest in Bitcoin
        - Retail investor attention
        - Search spikes often precede price movements
        - Leading indicator (search -> awareness -> buying)

        Why it matters for prediction:
        - Increased search interest often predicts price increases
        - Search volume correlates with trading volume
        - Retail FOMO indicator (extreme interest = potential top)
        - Early warning of trend changes

        Research findings:
        - Studies show Google Trends can predict Bitcoin returns
        - Search interest leads price by 1-3 days
        - Particularly useful for short-term predictions

        Args:
            keyword (str): Search term (default: 'Bitcoin')
            timeframe (str): Time period
                - 'now 1-H' (last hour)
                - 'now 4-H' (last 4 hours)
                - 'now 1-d' (last day)
                - 'now 7-d' (last 7 days)
                - 'today 1-m' (past month)
                - 'today 3-m' (past 3 months)
                - 'today 12-m' (past year)
            geo (str): Country code (empty = worldwide)

        Returns:
            pd.DataFrame: Google Trends data with columns:
                - timestamp: datetime index
                - search_interest: Search interest (0-100, normalized)
                - search_interest_ma7: 7-day moving average (smoothed)

        Notes:
            - Free API (Google Trends)
            - No API key required
            - Rate limit: ~4 requests per second, be respectful
            - Always add delays between requests (2-3 seconds)
            - Values are normalized to 100 (peak popularity)
        """
        try:
            print(f"\n📊 Fetching Google Trends data...")
            print(f"   Keyword: '{keyword}'")
            print(f"   Timeframe: {timeframe}")
            print(f"   Geography: {'Worldwide' if not geo else geo}")

            # Initialize pytrends
            # hl = host language, tz = timezone offset (UTC)
            print("   Initializing Google Trends client...")
            pytrends = TrendReq(hl='en-US', tz=0)

            # Build payload
            print("   Building request...")
            pytrends.build_payload(
                kw_list=[keyword],
                timeframe=timeframe,
                geo=geo
            )

            # Get interest over time
            print("   Fetching search interest data (this may take 5-10 seconds)...")

            # Add delay to be respectful to Google's servers
            time.sleep(2)

            df = pytrends.interest_over_time()

            # Check if data was returned
            if df is None or df.empty:
                print("   ⚠️  No data returned from Google Trends")
                print("      → This might be due to rate limiting")
                print("      → Try again in 60 seconds")
                return None

            # Remove 'isPartial' column if it exists
            if 'isPartial' in df.columns:
                df = df.drop('isPartial', axis=1)

            # Rename column
            df.columns = ['search_interest']

            # Add 7-day moving average for smoothing
            df['search_interest_ma7'] = df['search_interest'].rolling(
                window=7, min_periods=1
            ).mean()

            # Reset index to make timestamp a column
            df = df.reset_index()
            df.columns = ['timestamp', 'search_interest', 'search_interest_ma7']

            # Set timestamp as index
            df.set_index('timestamp', inplace=True)

            # Sort by timestamp
            df.sort_index(inplace=True)

            print(f"   ✓ Received {len(df)} data points")

            # Print summary
            print(f"\n   📈 Google Trends Summary:")
            print(f"      Samples: {len(df)}")
            print(f"      Date range: {df.index[0]} to {df.index[-1]}")
            print(f"      Interest range: {df['search_interest'].min()} - {df['search_interest'].max()}")
            print(f"      Current interest: {df['search_interest'].iloc[-1]}")
            print(f"      Average interest: {df['search_interest'].mean():.1f}")
            print(f"      Current MA7: {df['search_interest_ma7'].iloc[-1]:.1f}")

            # Trend analysis
            recent_avg = df['search_interest'].tail(7).mean()
            older_avg = df['search_interest'].head(7).mean()
            trend = "↑ Increasing" if recent_avg > older_avg else "↓ Decreasing"
            print(f"      Trend: {trend} ({recent_avg:.1f} vs {older_avg:.1f})")

            return df

        except Exception as e:
            print(f"\n   ❌ Error fetching Google Trends: {e}")
            print(f"      Error type: {type(e).__name__}")

            # Common error messages
            if "429" in str(e) or "rate" in str(e).lower():
                print("      → Rate limit exceeded")
                print("      → Wait 60 seconds and retry")
            elif "timeout" in str(e).lower():
                print("      → Request timeout")
                print("      → Check internet connection")

            return None


    def save_to_csv(self, df, filename, data_dir='data/raw'):
        """
        Save sentiment data to CSV

        Args:
            df (pd.DataFrame): Data to save
            filename (str): Output filename
            data_dir (str): Directory to save (default: 'data/raw')

        Returns:
            bool: Success status
        """
        if df is None or df.empty:
            print(f"\n   ⚠️  Cannot save: DataFrame is empty")
            return False

        try:
            import os

            # Create directory if needed
            os.makedirs(data_dir, exist_ok=True)

            # Full path
            filepath = os.path.join(data_dir, filename)

            # Save to CSV
            df.to_csv(filepath, index=True)

            # File size
            file_size = os.path.getsize(filepath) / 1024  # KB

            print(f"\n   ✓ Data saved successfully!")
            print(f"      Location: {filepath}")
            print(f"      Size: {file_size:.2f} KB")
            print(f"      Rows: {len(df)}")
            print(f"      Columns: {list(df.columns)}")

            return True

        except Exception as e:
            print(f"\n   ❌ Error saving data: {e}")
            return False


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================

if __name__ == "__main__":
    """
    Main execution: Fetch sentiment indicators for Bitcoin

    This script collects:
    1. Fear & Greed Index (90 days) - Market sentiment
    2. Google Trends (3 months) - Search interest

    Both are FREE and provide valuable sentiment features for prediction models.

    NO API KEYS REQUIRED!
    """

    print("=" * 70)
    print("  BITCOIN SENTIMENT DATA COLLECTION")
    print("=" * 70)
    print("\nCollecting sentiment indicators for Bitcoin price prediction...")
    print("These are FREE sources that capture market psychology.\n")

    # Initialize fetcher
    fetcher = SentimentFetcher()

    print("\n" + "=" * 70)
    print("  STEP 1: Fear & Greed Index")
    print("=" * 70)
    print("\nWhat is Fear & Greed Index?")
    print("  • Composite sentiment score (0-100)")
    print("  • Based on: volatility, momentum, volume, social media, surveys")
    print("  • Extreme Fear (0-24) often = buying opportunity")
    print("  • Extreme Greed (75-100) often = selling opportunity")
    print("  • Updated daily by alternative.me")

    # Fetch Fear & Greed Index
    fng_df = fetcher.fetch_fear_greed_index(limit=90)

    # Save if successful
    if fng_df is not None:
        fetcher.save_to_csv(fng_df, 'fear_greed_index.csv')
    else:
        print("\n   ⚠️  Fear & Greed Index fetch failed")

    # Delay before next API call (be respectful)
    print("\n   Waiting 3 seconds before next API call...")
    time.sleep(3)

    print("\n" + "=" * 70)
    print("  STEP 2: Google Trends (Bitcoin Search Interest)")
    print("=" * 70)
    print("\nWhy Google Trends matters?")
    print("  • Search interest often precedes price movements")
    print("  • Indicates retail investor attention and FOMO")
    print("  • Studies show it can predict short-term returns")
    print("  • Leading indicator (search → awareness → buying)")
    print("  • Normalized to 100 (peak popularity)")

    # Fetch Google Trends
    trends_df = fetcher.fetch_google_trends(
        keyword='Bitcoin',
        timeframe='today 3-m'  # Last 3 months
    )

    # Save if successful
    if trends_df is not None:
        fetcher.save_to_csv(trends_df, 'bitcoin_trends.csv')
    else:
        print("\n   ⚠️  Google Trends fetch failed")
        print("      This is often due to rate limiting")
        print("      Try running the script again in 60 seconds")

    # Final summary
    print("\n" + "=" * 70)
    print("  SUMMARY - SENTIMENT DATA")
    print("=" * 70)

    total_samples = 0

    if fng_df is not None:
        print(f"\n✓ Fear & Greed Index:")
        print(f"  • Samples: {len(fng_df):,}")
        print(f"  • Date range: {fng_df.index[0]} to {fng_df.index[-1]}")
        print(f"  • Current: {fng_df['fear_greed_value'].iloc[-1]} ({fng_df['fear_greed_class'].iloc[-1]})")
        print(f"  • File: data/raw/fear_greed_index.csv")
        total_samples += len(fng_df)
    else:
        print(f"\n✗ Fear & Greed Index: FAILED")

    if trends_df is not None:
        print(f"\n✓ Google Trends:")
        print(f"  • Samples: {len(trends_df):,}")
        print(f"  • Date range: {trends_df.index[0]} to {trends_df.index[-1]}")
        print(f"  • Current interest: {trends_df['search_interest'].iloc[-1]}")
        print(f"  • File: data/raw/bitcoin_trends.csv")
        total_samples += len(trends_df)
    else:
        print(f"\n✗ Google Trends: FAILED")

    print("\n" + "=" * 70)
    print(f"  SENTIMENT DATA COLLECTION COMPLETE!")
    if total_samples > 0:
        print(f"  Total sentiment samples: {total_samples:,}")
    print("=" * 70)

    # Usage recommendations
    print("\n💡 How to use sentiment data:")
    print("   1. Merge with price data by timestamp")
    print("   2. Use Fear & Greed as a feature in your models")
    print("   3. Google Trends search_interest_ma7 smooths noise")
    print("   4. Extreme Fear/Greed can be used for trading signals")
    print("   5. Combine with technical indicators for better predictions")

    print("\n📊 Expected Impact on Model:")
    print("   • Sentiment features typically improve accuracy by 2-5%")
    print("   • Most useful for short-term predictions (1-7 days)")
    print("   • Help capture retail investor psychology")
    print("   • Particularly valuable during extreme market conditions")

    print("\n⚠️  Important Notes:")
    print("   • Fear & Greed updated daily (new data at ~00:00 UTC)")
    print("   • Google Trends rate limits: Wait 2-3 sec between requests")
    print("   • Both datasets are FREE forever (no API keys)")
    print("   • Sentiment is contrarian: Extreme Fear = Buy, Extreme Greed = Sell")

    print("\n🔄 Rate Limiting Best Practices:")
    print("   • Fear & Greed: No strict limit, but use delays (1-2 sec)")
    print("   • Google Trends: Max ~4 req/sec, use 2-3 sec delays")
    print("   • If rate limited, wait 60 seconds and retry")
    print("   • This script already includes appropriate delays")

    print("\n" + "=" * 70)
