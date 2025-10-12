"""
Trading Signal Generation Module

This module converts Bitcoin price predictions into actionable trading signals (BUY/SELL/HOLD).

Key Concepts:
-------------
1. Threshold-based signals: Trade only when predicted return exceeds minimum threshold
2. Multi-horizon voting: Combine predictions from multiple timeframes for robust signals
3. Confidence scoring: Higher when multiple horizons agree on direction
4. Risk filtering: Avoid trades during high volatility or adverse conditions

Author: Bitcoin Price Prediction System
Date: 2025-10-05
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

# Set random seed
np.random.seed(42)


def generate_signals(predictions, current_prices, threshold=0.02, confidence_threshold=0.6):
    """
    Convert price predictions to Buy/Sell/Hold trading signals.

    This is the core signal generation function using a simple threshold-based approach.
    It calculates expected returns and generates signals when the predicted price movement
    exceeds the minimum threshold.

    **Why Threshold-Based Signals Work:**
    - Filters out noise: Small predicted movements may be within model error
    - Transaction costs: Need sufficient movement to overcome fees (typically 0.1-0.5%)
    - Risk management: Only trade when potential reward justifies risk
    - False positive reduction: Higher threshold = fewer but higher quality signals

    **Example:**
    If Bitcoin is $65,000 and model predicts $66,500:
    - Predicted return = (66,500 - 65,000) / 65,000 = 2.3%
    - If threshold = 2%, this generates BUY signal
    - If threshold = 5%, this is ignored (HOLD)

    Parameters:
    -----------
    predictions : np.ndarray
        Array of shape (n_samples, n_horizons) with predicted prices
        Example: [[66500, 67000, 68000], ...]  for [1d, 3d, 7d] horizons
    current_prices : np.ndarray
        Array of shape (n_samples,) with current Bitcoin prices
        Example: [65000, 65500, 64800, ...]
    threshold : float, default=0.02
        Minimum predicted return (as fraction) to generate BUY/SELL signal
        0.02 = 2% minimum price movement required
        Recommended: 0.01-0.05 (1-5%) depending on risk tolerance
    confidence_threshold : float, default=0.6
        Minimum confidence score (0-1) to execute trade
        0.6 = require 60% confidence to trade
        Lower = more trades but lower quality
        Higher = fewer trades but higher quality

    Returns:
    --------
    signals : np.ndarray
        Array of trading signals: 'BUY', 'SELL', or 'HOLD'
    signal_strength : np.ndarray
        Confidence score for each signal (0-1)
        Higher = more confidence in the prediction
    expected_returns : np.ndarray
        Predicted returns for primary horizon (useful for position sizing)

    Notes:
    ------
    - Uses only the first horizon (shortest timeframe) for signal generation
    - This is because short-term predictions are typically more accurate
    - Multi-horizon voting (see below) provides additional confirmation
    """
    n_samples = len(current_prices)
    signals = np.array(['HOLD'] * n_samples)
    signal_strength = np.zeros(n_samples)
    expected_returns = np.zeros(n_samples)

    # Calculate predicted returns for primary (first) horizon
    # This is the shortest timeframe (1-day in our case)
    pred_return = (predictions[:, 0] - current_prices) / current_prices

    print(f"\n{'='*70}")
    print(f"  SIGNAL GENERATION - THRESHOLD METHOD")
    print(f"{'='*70}")
    print(f"\n📊 Parameters:")
    print(f"   Return threshold: {threshold*100:.1f}%")
    print(f"   Confidence threshold: {confidence_threshold*100:.0f}%")
    print(f"   Samples: {n_samples}")

    for i in range(n_samples):
        expected_returns[i] = pred_return[i]

        # Calculate confidence based on magnitude of predicted return
        # Logic: Larger predicted movements = higher confidence (if model is certain)
        # We use absolute return scaled to 0-1 range
        # Cap at 1.0 for extreme predictions
        raw_confidence = min(abs(pred_return[i]) / (threshold * 3), 1.0)

        # Generate signal based on threshold
        if pred_return[i] > threshold:
            # Predicted price increase > threshold
            signals[i] = 'BUY'
            signal_strength[i] = raw_confidence

        elif pred_return[i] < -threshold:
            # Predicted price decrease > threshold
            signals[i] = 'SELL'
            signal_strength[i] = raw_confidence

        else:
            # Predicted movement too small to trade
            signals[i] = 'HOLD'
            signal_strength[i] = 0.0

        # Apply confidence filter
        # If confidence too low, override to HOLD (risk management)
        if signal_strength[i] < confidence_threshold and signals[i] != 'HOLD':
            signals[i] = 'HOLD'
            signal_strength[i] = 0.0

    # Calculate signal distribution
    buy_pct = np.sum(signals == 'BUY') / n_samples * 100
    sell_pct = np.sum(signals == 'SELL') / n_samples * 100
    hold_pct = np.sum(signals == 'HOLD') / n_samples * 100

    print(f"\n📊 Signal Distribution:")
    print(f"   BUY:  {np.sum(signals == 'BUY'):3d} ({buy_pct:5.1f}%)")
    print(f"   SELL: {np.sum(signals == 'SELL'):3d} ({sell_pct:5.1f}%)")
    print(f"   HOLD: {np.sum(signals == 'HOLD'):3d} ({hold_pct:5.1f}%)")

    print(f"\n📊 Expected Returns (for active signals):")
    active_returns = expected_returns[(signals == 'BUY') | (signals == 'SELL')]
    if len(active_returns) > 0:
        print(f"   Mean: {np.mean(active_returns)*100:+.2f}%")
        print(f"   Std:  {np.std(active_returns)*100:.2f}%")
        print(f"   Min:  {np.min(active_returns)*100:+.2f}%")
        print(f"   Max:  {np.max(active_returns)*100:+.2f}%")

    return signals, signal_strength, expected_returns


def multi_horizon_voting(predictions, current_prices, threshold=0.02):
    """
    Generate robust signals using multi-horizon voting.

    **Why Multi-Horizon Voting Reduces False Signals:**
    1. Diversification: Different timeframes capture different patterns
    2. Confirmation: When multiple horizons agree, signal is more reliable
    3. Noise reduction: Random fluctuations unlikely to affect all horizons
    4. Trend identification: Consistent direction across horizons = strong trend

    **Example Scenario:**
    Current price: $65,000
    Predictions: [1d: $66,300, 3d: $66,800, 7d: $67,500]

    1d: +2.0% → BUY
    3d: +2.8% → BUY
    7d: +3.8% → BUY

    Agreement: 3/3 = 100% → Strong BUY signal

    **Counter-Example (Conflicting Signals):**
    Current price: $65,000
    Predictions: [1d: $66,300, 3d: $64,500, 7d: $66,000]

    1d: +2.0% → BUY
    3d: -0.8% → HOLD (below threshold)
    7d: +1.5% → HOLD (below threshold)

    Agreement: 1/3 = 33% → HOLD (not enough consensus)

    Parameters:
    -----------
    predictions : np.ndarray
        Array of shape (n_samples, n_horizons) with predicted prices
    current_prices : np.ndarray
        Array of current prices
    threshold : float, default=0.02
        Minimum return threshold for each horizon

    Returns:
    --------
    aggregated_signals : np.ndarray
        Final signals after voting: 'BUY', 'SELL', or 'HOLD'
    agreement_scores : np.ndarray
        Agreement score 0-1 (proportion of horizons that agree)
    horizon_votes : np.ndarray
        Individual votes from each horizon (for transparency)

    Notes:
    ------
    Voting Rules:
    - Each horizon votes: +1 (BUY), -1 (SELL), or 0 (HOLD)
    - Sum votes: positive = BUY, negative = SELL, zero = HOLD
    - Agreement = |votes for winner| / total horizons
    - Require majority (>50%) agreement to trade
    """
    n_samples = len(current_prices)
    n_horizons = predictions.shape[1]

    aggregated_signals = np.array(['HOLD'] * n_samples)
    agreement_scores = np.zeros(n_samples)
    horizon_votes = np.zeros((n_samples, n_horizons))

    print(f"\n{'='*70}")
    print(f"  SIGNAL GENERATION - MULTI-HORIZON VOTING")
    print(f"{'='*70}")
    print(f"\n📊 Voting with {n_horizons} horizons")

    for i in range(n_samples):
        votes = []

        # Collect votes from each horizon
        for h in range(n_horizons):
            pred_return = (predictions[i, h] - current_prices[i]) / current_prices[i]

            if pred_return > threshold:
                vote = +1  # BUY
            elif pred_return < -threshold:
                vote = -1  # SELL
            else:
                vote = 0   # HOLD

            votes.append(vote)
            horizon_votes[i, h] = vote

        # Count votes
        buy_votes = sum(v == 1 for v in votes)
        sell_votes = sum(v == -1 for v in votes)
        hold_votes = sum(v == 0 for v in votes)

        # Determine winner by majority
        if buy_votes > sell_votes and buy_votes > hold_votes:
            aggregated_signals[i] = 'BUY'
            agreement_scores[i] = buy_votes / n_horizons

        elif sell_votes > buy_votes and sell_votes > hold_votes:
            aggregated_signals[i] = 'SELL'
            agreement_scores[i] = sell_votes / n_horizons

        else:
            # No clear majority or tied
            aggregated_signals[i] = 'HOLD'
            agreement_scores[i] = max(buy_votes, sell_votes, hold_votes) / n_horizons

        # Additional filter: require >50% agreement to trade
        # This prevents trading on weak signals
        if agreement_scores[i] <= 0.5 and aggregated_signals[i] != 'HOLD':
            aggregated_signals[i] = 'HOLD'

    # Calculate signal distribution
    buy_pct = np.sum(aggregated_signals == 'BUY') / n_samples * 100
    sell_pct = np.sum(aggregated_signals == 'SELL') / n_samples * 100
    hold_pct = np.sum(aggregated_signals == 'HOLD') / n_samples * 100

    print(f"\n📊 Signal Distribution (After Voting):")
    print(f"   BUY:  {np.sum(aggregated_signals == 'BUY'):3d} ({buy_pct:5.1f}%)")
    print(f"   SELL: {np.sum(aggregated_signals == 'SELL'):3d} ({sell_pct:5.1f}%)")
    print(f"   HOLD: {np.sum(aggregated_signals == 'HOLD'):3d} ({hold_pct:5.1f}%)")

    print(f"\n📊 Agreement Scores:")
    active_agreements = agreement_scores[aggregated_signals != 'HOLD']
    if len(active_agreements) > 0:
        print(f"   Mean agreement: {np.mean(active_agreements)*100:.1f}%")
        print(f"   High confidence (>80%): {np.sum(active_agreements > 0.8)}")
        print(f"   Medium confidence (60-80%): {np.sum((active_agreements > 0.6) & (active_agreements <= 0.8))}")
        print(f"   Low confidence (50-60%): {np.sum((active_agreements > 0.5) & (active_agreements <= 0.6))}")

    return aggregated_signals, agreement_scores, horizon_votes


def apply_risk_filters(signals, signal_strength, prices, volatility=None,
                       max_volatility_multiplier=2.0, max_consecutive_losses=3):
    """
    Apply risk management filters to trading signals.

    **Why Risk Filtering is Critical:**
    1. High Volatility Protection: Models trained on normal conditions may fail in extreme volatility
    2. Drawdown Management: Stop trading after consecutive losses (model may be in bad regime)
    3. Position Management: Avoid overtrading or conflicting positions
    4. Capital Preservation: Better to miss opportunities than lose capital

    **Common Risk Scenarios:**

    Scenario 1: Flash Crash
    - Bitcoin drops 20% in 1 hour due to exchange issue
    - Volatility spikes to 10x normal
    - Risk filter: Block all trades until volatility normalizes
    - Reason: Model predictions unreliable in extreme conditions

    Scenario 2: Losing Streak
    - 3 consecutive losing trades
    - Could indicate market regime change or model degradation
    - Risk filter: Stop trading temporarily
    - Reason: Preserve capital, re-evaluate model

    Scenario 3: Over-leveraged
    - Already have open position
    - New signal conflicts with current position
    - Risk filter: Don't add to position (or close first)
    - Reason: Risk management, position sizing

    Parameters:
    -----------
    signals : np.ndarray
        Original trading signals
    signal_strength : np.ndarray
        Confidence scores
    prices : np.ndarray
        Price series for volatility calculation
    volatility : np.ndarray, optional
        Pre-calculated volatility (e.g., ATR)
        If None, will calculate from price returns
    max_volatility_multiplier : float, default=2.0
        Block trades when volatility > multiplier * average
        Example: If avg volatility = 3%, block when > 6%
    max_consecutive_losses : int, default=3
        Number of consecutive losses before stopping trading

    Returns:
    --------
    filtered_signals : np.ndarray
        Signals after risk filtering
    filter_reasons : list
        Reasons why signals were filtered (for analysis)

    Notes:
    ------
    Filters are applied in order:
    1. Volatility filter (market condition)
    2. Consecutive loss filter (performance tracking)
    3. Position management (not implemented - would need portfolio state)
    """
    filtered_signals = signals.copy()
    filter_reasons = []

    print(f"\n{'='*70}")
    print(f"  RISK FILTERING")
    print(f"{'='*70}")

    n_samples = len(signals)
    n_filtered = 0

    # Calculate volatility if not provided
    if volatility is None:
        # Use rolling standard deviation of returns as volatility proxy
        returns = np.diff(prices) / prices[:-1]
        # Pad first value
        returns = np.concatenate([[0], returns])

        # Calculate rolling volatility (20-period window)
        window = min(20, len(returns) // 2)
        volatility = pd.Series(returns).rolling(window=window, min_periods=1).std().values

    # Calculate average volatility
    avg_volatility = np.mean(volatility)
    volatility_threshold = avg_volatility * max_volatility_multiplier

    print(f"\n📊 Volatility Analysis:")
    print(f"   Average volatility: {avg_volatility*100:.2f}%")
    print(f"   Volatility threshold: {volatility_threshold*100:.2f}%")
    print(f"   Max observed: {np.max(volatility)*100:.2f}%")

    # Filter 1: High Volatility
    high_vol_mask = volatility > volatility_threshold
    high_vol_count = 0

    for i in range(n_samples):
        if high_vol_mask[i] and signals[i] != 'HOLD':
            filtered_signals[i] = 'HOLD'
            filter_reasons.append(f"Sample {i}: High volatility ({volatility[i]*100:.2f}%)")
            n_filtered += 1
            high_vol_count += 1

    print(f"\n🛡️  Filter 1: High Volatility")
    print(f"   Signals blocked: {high_vol_count}")

    # Filter 2: Simulate consecutive losses
    # Note: In real implementation, this would track actual trade outcomes
    # For demonstration, we'll use signal strength as a proxy
    # (Low strength = higher chance of loss)

    consecutive_losses = 0
    loss_filter_count = 0
    min_strength_for_win = 0.7  # Signals below this are considered "losses"

    for i in range(n_samples):
        # Simulate outcome based on signal strength
        # In real trading, you'd track actual P&L
        if signal_strength[i] < min_strength_for_win and signals[i] != 'HOLD':
            consecutive_losses += 1
        else:
            consecutive_losses = 0  # Reset on strong signal

        # Block trading after max consecutive losses
        if consecutive_losses >= max_consecutive_losses and signals[i] != 'HOLD':
            if filtered_signals[i] != 'HOLD':  # Not already filtered
                filtered_signals[i] = 'HOLD'
                filter_reasons.append(f"Sample {i}: Max consecutive losses ({consecutive_losses})")
                n_filtered += 1
                loss_filter_count += 1

    print(f"\n🛡️  Filter 2: Consecutive Losses")
    print(f"   Signals blocked: {loss_filter_count}")

    # Summary
    original_active = np.sum(signals != 'HOLD')
    filtered_active = np.sum(filtered_signals != 'HOLD')

    print(f"\n📊 Risk Filtering Summary:")
    print(f"   Original active signals: {original_active}")
    print(f"   Filtered active signals: {filtered_active}")
    print(f"   Signals blocked: {n_filtered} ({n_filtered/original_active*100:.1f}%)")

    return filtered_signals, filter_reasons


def compare_signal_methods(predictions, current_prices, threshold=0.02):
    """
    Compare threshold-based vs multi-horizon voting methods.

    This function helps evaluate which signal generation method is more appropriate
    for your trading strategy.

    Parameters:
    -----------
    predictions : np.ndarray
        Predicted prices for all horizons
    current_prices : np.ndarray
        Current Bitcoin prices
    threshold : float
        Return threshold for signals

    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison of both methods
    """
    print(f"\n{'='*70}")
    print(f"  COMPARING SIGNAL GENERATION METHODS")
    print(f"{'='*70}")

    # Method 1: Simple threshold
    signals_simple, strength_simple, returns_simple = generate_signals(
        predictions, current_prices, threshold=threshold
    )

    # Method 2: Multi-horizon voting
    signals_voting, agreement_voting, votes = multi_horizon_voting(
        predictions, current_prices, threshold=threshold
    )

    # Calculate agreement between methods
    agreement = np.sum(signals_simple == signals_voting) / len(signals_simple)

    print(f"\n📊 Method Comparison:")
    print(f"   Agreement between methods: {agreement*100:.1f}%")

    # Count disagreements
    disagreements = signals_simple != signals_voting
    print(f"   Disagreements: {np.sum(disagreements)}")

    if np.sum(disagreements) > 0:
        print(f"\n   Disagreement breakdown:")
        for i in np.where(disagreements)[0][:5]:  # Show first 5
            print(f"      Sample {i}: Simple={signals_simple[i]}, Voting={signals_voting[i]}")

    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'current_price': current_prices,
        'pred_1d': predictions[:, 0] if predictions.shape[1] > 0 else np.nan,
        'signal_simple': signals_simple,
        'strength_simple': strength_simple,
        'signal_voting': signals_voting,
        'agreement_voting': agreement_voting,
        'methods_agree': signals_simple == signals_voting
    })

    return comparison_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TRADING SIGNAL GENERATION")
    print("="*70)

    # Load test predictions and actual prices
    print("\n📂 Loading test data and predictions...")

    data_dir = Path('data/processed')
    results_dir = Path('results')

    # Load actual test prices (from y_test)
    y_test = pd.read_csv(data_dir / 'y_test.csv', index_col='timestamp')
    current_prices = y_test['target_1d'].values  # Use 1d target as "current" price

    # Load model predictions
    # We'll use the ensemble model predictions as example
    ensemble_results = joblib.load(results_dir / 'ensemble_results.pkl')

    # Get predictions for all horizons
    predictions = np.column_stack([
        ensemble_results['1d']['predictions'],
        ensemble_results['3d']['predictions'],
        ensemble_results['7d']['predictions']
    ])

    print(f"   ✓ Loaded {len(current_prices)} test samples")
    print(f"   ✓ Predictions shape: {predictions.shape}")
    print(f"   ✓ Price range: ${current_prices.min():.2f} - ${current_prices.max():.2f}")

    # Generate signals using both methods
    print(f"\n{'─'*70}")
    print("  METHOD 1: THRESHOLD-BASED SIGNALS")
    print(f"{'─'*70}")

    signals_simple, strength_simple, returns_simple = generate_signals(
        predictions=predictions,
        current_prices=current_prices,
        threshold=0.02,  # 2% minimum return
        confidence_threshold=0.6  # 60% minimum confidence
    )

    print(f"\n{'─'*70}")
    print("  METHOD 2: MULTI-HORIZON VOTING")
    print(f"{'─'*70}")

    signals_voting, agreement_voting, horizon_votes = multi_horizon_voting(
        predictions=predictions,
        current_prices=current_prices,
        threshold=0.02
    )

    # Apply risk filters to voting signals (more robust method)
    print(f"\n{'─'*70}")
    print("  APPLYING RISK FILTERS")
    print(f"{'─'*70}")

    signals_filtered, filter_reasons = apply_risk_filters(
        signals=signals_voting,
        signal_strength=agreement_voting,
        prices=current_prices,
        max_volatility_multiplier=2.0,
        max_consecutive_losses=3
    )

    # Compare methods
    comparison_df = compare_signal_methods(predictions, current_prices, threshold=0.02)

    # Save signals to CSV
    signals_df = pd.DataFrame({
        'timestamp': y_test.index,
        'current_price': current_prices,
        'pred_1d': predictions[:, 0],
        'pred_3d': predictions[:, 1],
        'pred_7d': predictions[:, 2],
        'signal_threshold': signals_simple,
        'signal_strength': strength_simple,
        'expected_return': returns_simple,
        'signal_voting': signals_voting,
        'agreement_score': agreement_voting,
        'signal_final': signals_filtered
    })

    output_path = results_dir / 'trading_signals.csv'
    signals_df.to_csv(output_path, index=False)
    print(f"\n💾 Trading signals saved to: {output_path}")

    # Save detailed comparison
    comparison_path = results_dir / 'signal_method_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"💾 Method comparison saved to: {comparison_path}")

    # Print final summary
    print(f"\n{'='*70}")
    print(f"  FINAL TRADING SIGNALS SUMMARY")
    print(f"{'='*70}")

    print(f"\n📊 Signal Distribution (Final Filtered):")
    buy_count = np.sum(signals_filtered == 'BUY')
    sell_count = np.sum(signals_filtered == 'SELL')
    hold_count = np.sum(signals_filtered == 'HOLD')

    print(f"   BUY:  {buy_count:3d} ({buy_count/len(signals_filtered)*100:5.1f}%)")
    print(f"   SELL: {sell_count:3d} ({sell_count/len(signals_filtered)*100:5.1f}%)")
    print(f"   HOLD: {hold_count:3d} ({hold_count/len(signals_filtered)*100:5.1f}%)")

    print(f"\n📊 Signal Quality Metrics:")
    active_mask = signals_filtered != 'HOLD'
    if np.sum(active_mask) > 0:
        active_agreement = agreement_voting[active_mask]
        print(f"   Active signals: {np.sum(active_mask)}")
        print(f"   Avg agreement: {np.mean(active_agreement)*100:.1f}%")
        print(f"   High confidence signals (>80%): {np.sum(active_agreement > 0.8)}")

    print(f"\n💡 Recommendations:")
    print(f"   1. Use multi-horizon voting for more robust signals")
    print(f"   2. Always apply risk filters before trading")
    print(f"   3. Monitor agreement scores - higher is better")
    print(f"   4. Consider position sizing based on signal strength")
    print(f"   5. Backtest signals before live trading")

    print(f"\n📋 Next Steps:")
    print(f"   1. Backtest signals on historical data")
    print(f"   2. Calculate Sharpe ratio and max drawdown")
    print(f"   3. Optimize threshold and confidence parameters")
    print(f"   4. Implement in trading dashboard")

    print("="*70 + "\n")
