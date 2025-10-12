"""
Backtesting Engine for Bitcoin Trading Strategies

This module provides a realistic backtesting framework that accounts for:
1. Transaction costs (commission, slippage, spread)
2. Market impact (can't always execute at exact price)
3. Position sizing and capital management
4. Comprehensive performance metrics

**Why Transaction Costs Matter Critically:**

In paper trading, strategies often look profitable. In real trading, transaction
costs can turn profitable strategies into losers.

Example:
- Strategy predicts +2% move, trades 100 times
- Paper profit: 100 × 2% = 200% gain
- With 0.6% costs per round-trip: 100 × (2% - 1.2%) = 80% gain
- With 1.0% costs per round-trip: 100 × (2% - 2.0%) = 0% gain (breakeven!)

**For Bitcoin specifically:**
- Higher volatility = higher slippage
- 24/7 market = spread varies widely
- Exchange fees: 0.1-0.5% per trade
- Market impact: larger orders = more slippage

**Key Insight:** A strategy must generate returns SIGNIFICANTLY above transaction
costs to be viable. Our conservative estimate of 0.6% per trade means you need
to predict moves >1.2% (round-trip) to profit.

Author: Bitcoin Price Prediction System
Date: 2025-10-05
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

# Set random seed
np.random.seed(42)


def backtest_strategy(df, signals, initial_capital=10000, costs=None):
    """
    Backtest trading strategy with realistic transaction costs.

    This function simulates actual trading by:
    1. Starting with cash (no BTC)
    2. Executing BUY signals (convert cash → BTC)
    3. Executing SELL signals (convert BTC → cash)
    4. Applying realistic costs to each trade
    5. Tracking portfolio value over time

    **Transaction Cost Components:**

    1. **Commission (0.2% default):**
       - Exchange fees (Binance: 0.1%, Coinbase: 0.5%)
       - Network fees for withdrawals
       - Trading platform costs

    2. **Slippage (0.3% default):**
       - Market impact: your order moves the price
       - Example: Try to buy at $65,000, actually execute at $65,195
       - Larger in volatile markets
       - Larger for bigger orders

    3. **Spread (0.1% default):**
       - Bid-ask spread: difference between buy and sell price
       - Always buy at ask (higher), sell at bid (lower)
       - Wider during low liquidity or high volatility

    **Total Cost = 0.6% per trade (conservative estimate)**
    **Round-trip cost = 1.2% (buy + sell)**

    Parameters:
    -----------
    df : pd.DataFrame
        Price data with columns: [timestamp, open, high, low, close, volume]
        Must have datetime index or timestamp column
    signals : np.ndarray or list
        Trading signals ('BUY', 'SELL', 'HOLD') for each row in df
        Must be same length as df
    initial_capital : float, default=10000
        Starting cash in USD
        Typical: $10,000-$100,000 for retail traders
    costs : dict, optional
        Transaction cost structure. If None, uses conservative defaults:
        {
            'commission': 0.002,  # 0.2% exchange fee
            'slippage': 0.003,    # 0.3% market impact
            'spread': 0.001       # 0.1% bid-ask spread
        }

    Returns:
    --------
    results : dict
        Complete backtesting results including:
        - performance_metrics: dict of calculated metrics
        - trades: list of all executed trades
        - portfolio_values: time series of portfolio value
        - daily_returns: time series of daily returns
        - initial_capital: starting amount
        - final_capital: ending amount
        - total_return_pct: overall return percentage

    trades : list of dict
        Each trade contains:
        - timestamp: when trade occurred
        - action: 'BUY' or 'SELL'
        - price: market price at trade time
        - effective_price: actual execution price (with costs)
        - btc_amount: BTC quantity traded
        - cost_usd: transaction costs in USD
        - portfolio_value: total portfolio value after trade

    portfolio_values : pd.Series
        Portfolio value at each timestamp (cash + BTC value)

    Notes:
    ------
    - Assumes full capital deployment (all-in or all-out)
    - No partial positions or position sizing
    - Cannot short (only long positions)
    - Ignores margin/leverage
    - Assumes perfect execution (order always fills)
    """
    # Set default costs if not provided
    if costs is None:
        costs = {
            'commission': 0.002,  # 0.2%
            'slippage': 0.003,    # 0.3%
            'spread': 0.001       # 0.1%
        }

    # Validate inputs
    if len(df) != len(signals):
        raise ValueError(f"Length mismatch: df has {len(df)} rows, signals has {len(signals)}")

    # Calculate total cost per trade
    total_cost_pct = costs['commission'] + costs['slippage'] + costs['spread']

    print(f"\n{'='*70}")
    print(f"  BACKTESTING TRADING STRATEGY")
    print(f"{'='*70}")
    print(f"\n💰 Initial Capital: ${initial_capital:,.2f}")
    print(f"\n📊 Transaction Costs:")
    print(f"   Commission:  {costs['commission']*100:.2f}%")
    print(f"   Slippage:    {costs['slippage']*100:.2f}%")
    print(f"   Spread:      {costs['spread']*100:.2f}%")
    print(f"   ─────────────────────")
    print(f"   Total/trade: {total_cost_pct*100:.2f}%")
    print(f"   Round-trip:  {total_cost_pct*2*100:.2f}%")

    # Initialize state
    cash = initial_capital
    position = 0.0  # BTC holdings
    trades = []
    portfolio_values = []
    timestamps = []

    # Get price data
    if 'timestamp' in df.columns:
        timestamps_col = df['timestamp']
    else:
        timestamps_col = df.index

    prices = df['close'].values

    # Simulate trading
    for i in range(len(df)):
        timestamp = timestamps_col.iloc[i] if hasattr(timestamps_col, 'iloc') else timestamps_col[i]
        current_price = prices[i]
        signal = signals[i]

        # Execute BUY signal
        if signal == 'BUY' and position == 0 and cash > 0:
            # Calculate effective buy price (includes all costs)
            # We pay MORE than market price when buying
            effective_price = current_price * (1 + total_cost_pct)

            # Calculate BTC amount we can buy
            btc_amount = cash / effective_price

            # Calculate transaction cost in USD
            cost_usd = cash - (btc_amount * current_price)

            # Update position
            position = btc_amount
            cash = 0.0

            # Record trade
            trades.append({
                'timestamp': timestamp,
                'action': 'BUY',
                'price': current_price,
                'effective_price': effective_price,
                'btc_amount': btc_amount,
                'cost_usd': cost_usd,
                'portfolio_value': position * current_price
            })

        # Execute SELL signal
        elif signal == 'SELL' and position > 0:
            # Calculate effective sell price (includes all costs)
            # We receive LESS than market price when selling
            effective_price = current_price * (1 - total_cost_pct)

            # Calculate cash we receive
            cash_received = position * effective_price

            # Calculate transaction cost in USD
            cost_usd = (position * current_price) - cash_received

            # Update position
            cash = cash_received
            position = 0.0

            # Record trade
            trades.append({
                'timestamp': timestamp,
                'action': 'SELL',
                'price': current_price,
                'effective_price': effective_price,
                'btc_amount': position,  # Amount we sold
                'cost_usd': cost_usd,
                'portfolio_value': cash
            })

        # Calculate current portfolio value
        # Portfolio = cash + (BTC × current_price)
        portfolio_value = cash + (position * current_price)
        portfolio_values.append(portfolio_value)

    # Convert to pandas Series for easier manipulation
    portfolio_values = pd.Series(portfolio_values, index=df.index)

    # Calculate final capital
    final_capital = portfolio_values.iloc[-1]
    total_return_pct = (final_capital - initial_capital) / initial_capital * 100

    print(f"\n📊 Backtest Summary:")
    first_ts = timestamps_col[0] if isinstance(timestamps_col, pd.DatetimeIndex) else timestamps_col.iloc[0]
    last_ts = timestamps_col[-1] if isinstance(timestamps_col, pd.DatetimeIndex) else timestamps_col.iloc[-1]
    print(f"   Period: {first_ts} to {last_ts}")
    print(f"   Total trades: {len(trades)}")
    print(f"   Buy trades: {sum(1 for t in trades if t['action'] == 'BUY')}")
    print(f"   Sell trades: {sum(1 for t in trades if t['action'] == 'SELL')}")
    print(f"\n💰 Final Results:")
    print(f"   Initial Capital: ${initial_capital:,.2f}")
    print(f"   Final Capital:   ${final_capital:,.2f}")
    print(f"   Total Return:    {total_return_pct:+.2f}%")

    # Package results
    results = {
        'trades': trades,
        'portfolio_values': portfolio_values,
        'initial_capital': initial_capital,
        'final_capital': final_capital,
        'total_return_pct': total_return_pct,
        'costs': costs,
        'total_cost_pct': total_cost_pct
    }

    return results


def calculate_metrics(portfolio_values, trades, initial_capital):
    """
    Calculate comprehensive performance metrics.

    **Metrics Explained:**

    1. **Total Return:** Simple percentage gain/loss
       - Formula: (Final - Initial) / Initial × 100
       - Example: $10,000 → $12,000 = +20%

    2. **Win Rate:** Percentage of profitable trades
       - Formula: Winning Trades / Total Trades × 100
       - Good: >50%, Excellent: >60%
       - NOTE: Win rate alone is misleading! Better to have 40% win rate
         with winners 2x larger than losers

    3. **Max Drawdown:** Largest peak-to-trough decline
       - Formula: (Trough - Peak) / Peak × 100
       - Example: Peak $15,000 → Trough $12,000 = -20% DD
       - Critical for risk assessment
       - Most traders can't psychologically handle >30% DD

    4. **Sharpe Ratio:** Risk-adjusted return
       - Formula: (Mean Return - Risk-Free Rate) / Std Dev × √252
       - Annualized for daily data (252 trading days)
       - >1.0 = Good, >2.0 = Excellent, >3.0 = Exceptional
       - Measures return per unit of risk

    5. **Sortino Ratio:** Like Sharpe but only penalizes downside
       - Formula: Mean Return / Downside Std Dev × √252
       - Better than Sharpe because upside volatility is good
       - Generally higher than Sharpe ratio

    6. **Profit Factor:** Gross profit / Gross loss
       - >1.0 = Profitable overall
       - 1.5 = Good, 2.0 = Excellent
       - Shows how much you make per dollar lost

    7. **Calmar Ratio:** Return / Max Drawdown
       - Measures return per unit of worst-case loss
       - >1.0 = Good, >3.0 = Excellent

    Parameters:
    -----------
    portfolio_values : pd.Series
        Time series of portfolio values
    trades : list
        List of executed trades
    initial_capital : float
        Starting capital

    Returns:
    --------
    metrics : dict
        Dictionary of all calculated metrics
    """
    print(f"\n{'='*70}")
    print(f"  CALCULATING PERFORMANCE METRICS")
    print(f"{'='*70}")

    metrics = {}

    # 1. Total Return
    final_value = portfolio_values.iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital * 100
    metrics['total_return_pct'] = total_return

    # 2. Win Rate
    if len(trades) >= 2:
        # Pair buy and sell trades
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        profitable_trades = 0
        total_round_trips = min(len(buy_trades), len(sell_trades))

        for i in range(total_round_trips):
            buy_price = buy_trades[i]['effective_price']
            sell_price = sell_trades[i]['effective_price']

            if sell_price > buy_price:
                profitable_trades += 1

        win_rate = (profitable_trades / total_round_trips * 100) if total_round_trips > 0 else 0
        metrics['win_rate_pct'] = win_rate
        metrics['total_round_trips'] = total_round_trips
        metrics['winning_trades'] = profitable_trades
        metrics['losing_trades'] = total_round_trips - profitable_trades
    else:
        metrics['win_rate_pct'] = 0
        metrics['total_round_trips'] = 0
        metrics['winning_trades'] = 0
        metrics['losing_trades'] = 0

    # 3. Maximum Drawdown
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    metrics['max_drawdown_pct'] = max_drawdown

    # Find when max drawdown occurred
    max_dd_idx = drawdown.idxmin()
    metrics['max_dd_date'] = max_dd_idx

    # 4. Sharpe Ratio (annualized)
    daily_returns = portfolio_values.pct_change().dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        # Assume 0% risk-free rate for simplicity
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        metrics['sharpe_ratio'] = sharpe_ratio
    else:
        metrics['sharpe_ratio'] = 0

    # 5. Sortino Ratio (annualized)
    negative_returns = daily_returns[daily_returns < 0]

    if len(negative_returns) > 1 and negative_returns.std() > 0:
        sortino_ratio = (daily_returns.mean() / negative_returns.std()) * np.sqrt(252)
        metrics['sortino_ratio'] = sortino_ratio
    else:
        metrics['sortino_ratio'] = 0

    # 6. Profit Factor
    if len(trades) >= 2:
        buy_trades = [t for t in trades if t['action'] == 'BUY']
        sell_trades = [t for t in trades if t['action'] == 'SELL']

        gross_profit = 0
        gross_loss = 0

        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_price = buy_trades[i]['effective_price']
            sell_price = sell_trades[i]['effective_price']
            btc_amount = buy_trades[i]['btc_amount']

            pnl = (sell_price - buy_price) * btc_amount

            if pnl > 0:
                gross_profit += pnl
            else:
                gross_loss += abs(pnl)

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        metrics['profit_factor'] = profit_factor
        metrics['gross_profit'] = gross_profit
        metrics['gross_loss'] = gross_loss
    else:
        metrics['profit_factor'] = 0
        metrics['gross_profit'] = 0
        metrics['gross_loss'] = 0

    # 7. Calmar Ratio (Return / Max Drawdown)
    if max_drawdown < 0:
        calmar_ratio = abs(total_return / max_drawdown)
        metrics['calmar_ratio'] = calmar_ratio
    else:
        metrics['calmar_ratio'] = 0

    # Additional metrics
    metrics['total_trades'] = len(trades)
    metrics['total_cost_usd'] = sum(t['cost_usd'] for t in trades)

    # Print metrics
    print(f"\n📊 Performance Metrics:")
    print(f"\n   Returns:")
    print(f"      Total Return:     {metrics['total_return_pct']:+8.2f}%")
    print(f"\n   Risk:")
    print(f"      Max Drawdown:     {metrics['max_drawdown_pct']:8.2f}%")
    print(f"      Max DD Date:      {metrics['max_dd_date']}")
    print(f"\n   Risk-Adjusted:")
    print(f"      Sharpe Ratio:     {metrics['sharpe_ratio']:8.2f}")
    print(f"      Sortino Ratio:    {metrics['sortino_ratio']:8.2f}")
    print(f"      Calmar Ratio:     {metrics['calmar_ratio']:8.2f}")
    print(f"\n   Trading:")
    print(f"      Win Rate:         {metrics['win_rate_pct']:8.2f}%")
    print(f"      Profit Factor:    {metrics['profit_factor']:8.2f}")
    print(f"      Total Trades:     {metrics['total_trades']:8d}")
    print(f"      Round Trips:      {metrics['total_round_trips']:8d}")
    print(f"\n   Costs:")
    print(f"      Total Costs:      ${metrics['total_cost_usd']:,.2f}")

    return metrics


def compare_to_baseline(strategy_results, df, initial_capital):
    """
    Compare strategy to buy-and-hold baseline.

    **Buy & Hold Strategy:**
    - Simplest possible strategy: buy at start, sell at end
    - No trading decisions required
    - Only pays transaction costs twice (buy + sell)
    - Often hard to beat after costs!

    **Why Compare to Buy & Hold:**
    1. Sanity check: If you can't beat buy & hold, why trade?
    2. Risk assessment: Trading strategy might have higher returns but much higher drawdown
    3. Opportunity cost: Time and effort spent trading vs. passive investing
    4. Transaction costs: Active trading pays costs many times

    **Typical Results:**
    - In bull markets: Buy & hold often wins
    - In ranging markets: Active trading can outperform
    - In bear markets: Active trading can reduce losses (if it sells)

    Parameters:
    -----------
    strategy_results : dict
        Results from backtest_strategy()
    df : pd.DataFrame
        Price data
    initial_capital : float
        Starting capital

    Returns:
    --------
    comparison : dict
        Comparison metrics
    """
    print(f"\n{'='*70}")
    print(f"  COMPARISON TO BUY & HOLD BASELINE")
    print(f"{'='*70}")

    # Calculate buy & hold performance
    first_price = df['close'].iloc[0]
    last_price = df['close'].iloc[-1]

    # Apply transaction costs (buy at start, sell at end)
    total_cost_pct = strategy_results['total_cost_pct']
    buy_price = first_price * (1 + total_cost_pct)
    sell_price = last_price * (1 - total_cost_pct)

    # Calculate BTC amount bought
    btc_amount = initial_capital / buy_price

    # Calculate final value
    final_value = btc_amount * sell_price

    # Calculate return
    bh_return = (final_value - initial_capital) / initial_capital * 100

    # Calculate buy & hold drawdown
    prices = df['close'].values
    bh_portfolio = (initial_capital / buy_price) * prices
    bh_running_max = pd.Series(bh_portfolio).expanding().max()
    bh_drawdown = (pd.Series(bh_portfolio) - bh_running_max) / bh_running_max
    bh_max_drawdown = bh_drawdown.min() * 100

    # Compare
    strategy_return = strategy_results['total_return_pct']
    strategy_max_dd = 0  # Will calculate if metrics available

    print(f"\n📊 Buy & Hold Baseline:")
    print(f"   Buy Price:        ${buy_price:,.2f} (with costs)")
    print(f"   Sell Price:       ${sell_price:,.2f} (with costs)")
    print(f"   BTC Amount:       {btc_amount:.6f}")
    print(f"   Final Value:      ${final_value:,.2f}")
    print(f"   Return:           {bh_return:+.2f}%")
    print(f"   Max Drawdown:     {bh_max_drawdown:.2f}%")

    print(f"\n📊 Strategy vs Buy & Hold:")
    print(f"   Strategy Return:  {strategy_return:+.2f}%")
    print(f"   B&H Return:       {bh_return:+.2f}%")
    print(f"   ───────────────────────────────")
    print(f"   Difference:       {strategy_return - bh_return:+.2f}%")

    if strategy_return > bh_return:
        print(f"\n   ✅ Strategy OUTPERFORMED buy & hold by {strategy_return - bh_return:.2f}%")
    else:
        print(f"\n   ❌ Strategy UNDERPERFORMED buy & hold by {abs(strategy_return - bh_return):.2f}%")

    comparison = {
        'bh_return_pct': bh_return,
        'bh_max_drawdown_pct': bh_max_drawdown,
        'bh_final_value': final_value,
        'strategy_return_pct': strategy_return,
        'outperformance_pct': strategy_return - bh_return
    }

    return comparison


def plot_results(portfolio_values, df, trades, save_path='results/backtest_equity_curve.png'):
    """
    Create comprehensive backtest visualization.

    Plots:
    1. Portfolio value over time (equity curve)
    2. Bitcoin price overlay
    3. Buy/sell markers
    4. Drawdown chart

    Parameters:
    -----------
    portfolio_values : pd.Series
        Portfolio value time series
    df : pd.DataFrame
        Price data
    trades : list
        Trade history
    save_path : str
        Path to save plot
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Equity Curve
    ax1.plot(portfolio_values.index, portfolio_values.values, linewidth=2, label='Portfolio Value')
    ax1.axhline(y=portfolio_values.iloc[0], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')

    # Mark trades
    buy_trades = [t for t in trades if t['action'] == 'BUY']
    sell_trades = [t for t in trades if t['action'] == 'SELL']

    if buy_trades:
        buy_times = [t['timestamp'] for t in buy_trades]
        buy_values = [portfolio_values.loc[t['timestamp']] if t['timestamp'] in portfolio_values.index else None for t in buy_trades]
        buy_values = [v for v in buy_values if v is not None]
        ax1.scatter(buy_times[:len(buy_values)], buy_values, color='green', marker='^', s=100, label='Buy', zorder=5)

    if sell_trades:
        sell_times = [t['timestamp'] for t in sell_trades]
        sell_values = [portfolio_values.loc[t['timestamp']] if t['timestamp'] in portfolio_values.index else None for t in sell_trades]
        sell_values = [v for v in sell_values if v is not None]
        ax1.scatter(sell_times[:len(sell_values)], sell_values, color='red', marker='v', s=100, label='Sell', zorder=5)

    ax1.set_xlabel('Date', fontsize=11)
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.set_title('Backtest Results - Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bitcoin Price
    ax2.plot(df.index, df['close'], linewidth=2, color='orange', label='Bitcoin Price')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Bitcoin Price ($)', fontsize=11)
    ax2.set_title('Bitcoin Price Movement', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Drawdown
    running_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - running_max) / running_max * 100

    ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
    ax3.plot(drawdown.index, drawdown.values, linewidth=2, color='red')
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_ylabel('Drawdown (%)', fontsize=11)
    ax3.set_title('Portfolio Drawdown', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n📊 Equity curve plot saved: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  BACKTESTING ENGINE - BITCOIN TRADING STRATEGY")
    print("="*70)

    # Load data
    print("\n📂 Loading data...")
    data_dir = Path('data/processed')
    results_dir = Path('results')

    # Load price data (y_test has the target prices, we'll use as "actual" prices)
    y_test = pd.read_csv(data_dir / 'y_test.csv', index_col='timestamp')
    y_test.index = pd.to_datetime(y_test.index)

    # Create price DataFrame
    # Use target_1d as the "close" price for simplicity
    df = pd.DataFrame({
        'close': y_test['target_1d'].values
    }, index=y_test.index)

    # Add other OHLCV columns (simplified - just use close for all)
    df['open'] = df['close']
    df['high'] = df['close'] * 1.01  # Simulate 1% high
    df['low'] = df['close'] * 0.99   # Simulate 1% low
    df['volume'] = 1000000  # Dummy volume

    # Load trading signals
    signals_df = pd.read_csv(results_dir / 'trading_signals.csv')
    signals = signals_df['signal_final'].values

    print(f"   ✓ Loaded {len(df)} price points")
    print(f"   ✓ Loaded {len(signals)} signals")
    print(f"   ✓ Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")

    # Test multiple cost scenarios
    cost_scenarios = {
        'Optimistic (0.3%)': {
            'commission': 0.001,
            'slippage': 0.001,
            'spread': 0.001
        },
        'Realistic (0.6%)': {
            'commission': 0.002,
            'slippage': 0.003,
            'spread': 0.001
        },
        'Pessimistic (1.0%)': {
            'commission': 0.003,
            'slippage': 0.005,
            'spread': 0.002
        }
    }

    initial_capital = 10000
    all_results = {}

    for scenario_name, costs in cost_scenarios.items():
        print(f"\n{'─'*70}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"{'─'*70}")

        # Run backtest
        results = backtest_strategy(df, signals, initial_capital, costs)

        # Calculate metrics
        metrics = calculate_metrics(
            results['portfolio_values'],
            results['trades'],
            initial_capital
        )

        # Compare to baseline
        comparison = compare_to_baseline(results, df, initial_capital)

        # Store results
        all_results[scenario_name] = {
            'backtest': results,
            'metrics': metrics,
            'comparison': comparison
        }

    # Save results
    print(f"\n{'='*70}")
    print(f"  SAVING RESULTS")
    print(f"{'='*70}")

    # Create summary DataFrame
    summary_data = []
    for scenario_name, results in all_results.items():
        metrics = results['metrics']
        comparison = results['comparison']

        summary_data.append({
            'Scenario': scenario_name,
            'Total Return (%)': metrics['total_return_pct'],
            'B&H Return (%)': comparison['bh_return_pct'],
            'Outperformance (%)': comparison['outperformance_pct'],
            'Max Drawdown (%)': metrics['max_drawdown_pct'],
            'Sharpe Ratio': metrics['sharpe_ratio'],
            'Sortino Ratio': metrics['sortino_ratio'],
            'Win Rate (%)': metrics['win_rate_pct'],
            'Profit Factor': metrics['profit_factor'],
            'Total Trades': metrics['total_trades'],
            'Total Costs ($)': metrics['total_cost_usd']
        })

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_path = results_dir / 'backtest_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\n💾 Summary saved: {summary_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  BACKTEST SUMMARY - ALL SCENARIOS")
    print(f"{'='*70}")
    print(summary_df.to_string(index=False))

    # Save detailed results to JSON
    # Convert non-serializable objects for JSON
    json_results = {}
    for scenario_name in all_results:
        json_results[scenario_name] = {
            'metrics': all_results[scenario_name]['metrics'],
            'comparison': all_results[scenario_name]['comparison'],
            'backtest': {
                'initial_capital': all_results[scenario_name]['backtest']['initial_capital'],
                'final_capital': all_results[scenario_name]['backtest']['final_capital'],
                'total_return_pct': all_results[scenario_name]['backtest']['total_return_pct'],
                'total_trades': len(all_results[scenario_name]['backtest']['trades'])
            }
        }

    json_path = results_dir / 'backtest_detailed.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"💾 Detailed results saved: {json_path}")

    # Plot realistic scenario (only if there are trades)
    print(f"\n📊 Creating visualizations...")
    realistic_results = all_results['Realistic (0.6%)']['backtest']
    if len(realistic_results['trades']) > 0:
        plot_results(
            realistic_results['portfolio_values'],
            df,
            realistic_results['trades']
        )
    else:
        print(f"   ⚠️  No trades executed - skipping equity curve plot")
        print(f"   Reason: All signals were SELL, but no initial BTC position")

    print(f"\n{'='*70}")
    print(f"  ✅ BACKTESTING COMPLETE")
    print(f"{'='*70}")
    print(f"\n💡 Key Insights:")
    print(f"   1. Transaction costs significantly impact returns")
    print(f"   2. Even 0.3% difference in costs changes profitability")
    print(f"   3. Buy & hold is hard to beat with high transaction costs")
    print(f"   4. Win rate alone doesn't determine profitability")
    print(f"\n📋 Files Generated:")
    print(f"   • results/backtest_summary.csv")
    print(f"   • results/backtest_detailed.json")
    print(f"   • results/backtest_equity_curve.png")
    print(f"\n📊 Next Steps:")
    print(f"   1. Analyze why strategy underperformed (if applicable)")
    print(f"   2. Optimize signal thresholds to reduce trades")
    print(f"   3. Consider position sizing strategies")
    print(f"   4. Test on different market periods")
    print("="*70 + "\n")
