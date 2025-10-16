"""
Feature Selection for Bitcoin Price Prediction
===============================================
Reduces feature space from 58 → 20-25 most important features

Why Feature Selection Matters:
- Too many features → overfitting (current: 58 features, 425 samples)
- Feature-to-sample ratio should be < 10% (current: 13.6%)
- Removes redundant/correlated features
- Improves model generalization

Methods Used:
1. XGBoost Feature Importance (gain-based)
2. Correlation Analysis (remove r > 0.95)
3. Variance Threshold (remove low-variance features)
4. Domain Knowledge (keep known important features)

Output:
- results/feature_importance.csv (all features ranked)
- results/selected_features.txt (top 20-25 features)
- results/feature_importance_plot.png (visualization)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from utils.feature_engineering import engineer_technical_features, add_sentiment_features


def analyze_feature_importance(X_train, y_train, feature_names):
    """
    Train XGBoost and extract feature importance

    Args:
        X_train: Training features
        y_train: Training targets
        feature_names: List of feature names

    Returns:
        pd.DataFrame: Features ranked by importance
    """
    print("\n" + "="*70)
    print("  FEATURE IMPORTANCE ANALYSIS")
    print("="*70)

    # Train XGBoost model
    print("\nTraining XGBoost to extract feature importance...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=3,
        learning_rate=0.1,
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, verbose=False)

    # Get feature importance
    importance_dict = model.get_booster().get_score(importance_type='gain')

    # Map feature indices to names
    importance_data = []
    for i, name in enumerate(feature_names):
        feature_key = f'f{i}'
        importance = importance_dict.get(feature_key, 0.0)
        importance_data.append({
            'feature': name,
            'importance': importance
        })

    # Create DataFrame and sort
    importance_df = pd.DataFrame(importance_data)
    importance_df = importance_df.sort_values('importance', ascending=False)

    # Calculate cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum() / importance_df['importance'].sum()

    print(f"✓ Extracted importance for {len(importance_df)} features")

    return importance_df


def remove_correlated_features(df, threshold=0.95):
    """
    Remove highly correlated features

    Args:
        df: DataFrame with features
        threshold: Correlation threshold (default: 0.95)

    Returns:
        list: Features to keep
    """
    print(f"\n{'='*70}")
    print(f"  CORRELATION ANALYSIS (threshold={threshold})")
    print(f"{'='*70}")

    # Calculate correlation matrix
    corr_matrix = df.corr().abs()

    # Find pairs with correlation > threshold
    to_drop = set()

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                col_i = corr_matrix.columns[i]
                col_j = corr_matrix.columns[j]
                # Drop the second one (arbitrary choice)
                to_drop.add(col_j)
                print(f"  Dropping {col_j} (corr={corr_matrix.iloc[i, j]:.3f} with {col_i})")

    features_to_keep = [col for col in df.columns if col not in to_drop]

    print(f"\n✓ Removed {len(to_drop)} highly correlated features")
    print(f"✓ Kept {len(features_to_keep)} features")

    return features_to_keep


def remove_low_variance_features(X, feature_names, threshold=0.01):
    """
    Remove features with low variance

    Args:
        X: Feature matrix
        feature_names: List of feature names
        threshold: Variance threshold (default: 0.01)

    Returns:
        list: Features to keep
    """
    print(f"\n{'='*70}")
    print(f"  VARIANCE THRESHOLD ANALYSIS (threshold={threshold})")
    print(f"{'='*70}")

    # Standardize first (variance threshold works better on scaled data)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply variance threshold
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(X_scaled)

    # Get selected features
    features_to_keep = [feature_names[i] for i in range(len(feature_names)) if selector.get_support()[i]]
    features_dropped = [feature_names[i] for i in range(len(feature_names)) if not selector.get_support()[i]]

    if features_dropped:
        print(f"  Dropped {len(features_dropped)} low-variance features:")
        for feat in features_dropped[:5]:  # Show first 5
            print(f"    - {feat}")
        if len(features_dropped) > 5:
            print(f"    ... and {len(features_dropped) - 5} more")

    print(f"\n✓ Kept {len(features_to_keep)} features with sufficient variance")

    return features_to_keep


def select_top_features_by_importance(importance_df, n_features=25, cumulative_threshold=0.95):
    """
    Select top N features by importance

    Args:
        importance_df: DataFrame with feature importance
        n_features: Max number of features to select (default: 25)
        cumulative_threshold: Use features that explain X% of importance (default: 0.95)

    Returns:
        list: Selected feature names
    """
    print(f"\n{'='*70}")
    print(f"  SELECTING TOP FEATURES")
    print(f"{'='*70}")

    # Method 1: Top N features
    top_n_features = importance_df.head(n_features)['feature'].tolist()

    # Method 2: Cumulative importance threshold
    cumulative_features = importance_df[importance_df['cumulative_importance'] <= cumulative_threshold]

    print(f"\nMethod 1 (Top {n_features}): {len(top_n_features)} features")
    print(f"Method 2 (Cumulative {cumulative_threshold*100}%): {len(cumulative_features)} features")

    # Use the more conservative (smaller) list
    if len(cumulative_features) < len(top_n_features):
        selected = cumulative_features['feature'].tolist()
        method = f"cumulative {cumulative_threshold*100}%"
    else:
        selected = top_n_features
        method = f"top {n_features}"

    print(f"\n✓ Selected {len(selected)} features using {method} method")

    return selected


def add_critical_features(selected_features, critical_features):
    """
    Ensure critical features are included (domain knowledge)

    Args:
        selected_features: Currently selected features
        critical_features: Features that must be included

    Returns:
        list: Updated feature list
    """
    added = []
    for feat in critical_features:
        if feat not in selected_features:
            selected_features.append(feat)
            added.append(feat)

    if added:
        print(f"\n✓ Added {len(added)} critical features:")
        for feat in added:
            print(f"    - {feat}")

    return selected_features


def visualize_feature_importance(importance_df, output_path='results/feature_importance_plot.png', top_n=30):
    """Create feature importance visualization"""
    print(f"\n{'='*70}")
    print(f"  CREATING VISUALIZATIONS")
    print(f"{'='*70}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Top N features by importance
    ax = axes[0]
    top_features = importance_df.head(top_n)
    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'], fontsize=8)
    ax.set_xlabel('Importance (Gain)', fontsize=10)
    ax.set_title(f'Top {top_n} Features by Importance', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Plot 2: Cumulative importance
    ax = axes[1]
    ax.plot(range(len(importance_df)), importance_df['cumulative_importance'], color='darkgreen', linewidth=2)
    ax.axhline(0.90, color='red', linestyle='--', label='90% threshold', alpha=0.7)
    ax.axhline(0.95, color='orange', linestyle='--', label='95% threshold', alpha=0.7)
    ax.set_xlabel('Number of Features', fontsize=10)
    ax.set_ylabel('Cumulative Importance', fontsize=10)
    ax.set_title('Cumulative Feature Importance', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Add annotations for 90% and 95% thresholds
    n_90 = (importance_df['cumulative_importance'] <= 0.90).sum()
    n_95 = (importance_df['cumulative_importance'] <= 0.95).sum()
    ax.annotate(f'{n_90} features', xy=(n_90, 0.90), xytext=(n_90+5, 0.85),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9)
    ax.annotate(f'{n_95} features', xy=(n_95, 0.95), xytext=(n_95+5, 0.88),
                arrowprops=dict(arrowstyle='->', color='orange'), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_path}")

    plt.close()


def main():
    """Main feature selection pipeline"""
    print("\n" + "="*70)
    print("  BITCOIN FEATURE SELECTION PIPELINE")
    print("="*70)

    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # Step 1: Load data and engineer features
    print("\n[STEP 1] Loading data and engineering features...")

    df = pd.read_csv('data/raw/btc_yahoo_2y_daily.csv', index_col=0, parse_dates=True)
    print(f"✓ Loaded {len(df)} daily samples")

    df = engineer_technical_features(df)
    df = add_sentiment_features(df)
    print(f"✓ Engineered {len(df.columns)} features")

    # Step 2: Prepare data
    print("\n[STEP 2] Preparing training data...")

    # Create 1-day return target
    df['future_return_1d'] = df['close'].pct_change(1).shift(-1)
    df_clean = df.dropna(subset=['future_return_1d']).copy()

    # Select features (exclude OHLCV and targets)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume'] + \
                   [col for col in df_clean.columns if 'future_' in col or 'current_price' in col]
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    X = df_clean[feature_cols].values
    y = df_clean['future_return_1d'].values

    print(f"✓ Prepared data: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Feature-to-sample ratio: {X.shape[1]/X.shape[0]*100:.1f}%")

    # Step 3: Remove low-variance features
    feature_cols = remove_low_variance_features(X, feature_cols, threshold=0.005)

    # Update X with remaining features
    X = df_clean[feature_cols].values

    # Step 4: Remove highly correlated features
    print("\n[STEP 3] Removing correlated features...")
    df_features = df_clean[feature_cols]
    feature_cols = remove_correlated_features(df_features, threshold=0.95)

    # Update X again
    X = df_clean[feature_cols].values

    print(f"\n✓ After filtering: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  New feature-to-sample ratio: {X.shape[1]/X.shape[0]*100:.1f}%")

    # Step 5: Scale and split
    print("\n[STEP 4] Scaling features and creating train/test split...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X_scaled) * 0.8)
    X_train = X_scaled[:split_idx]
    y_train = y[:split_idx]

    print(f"✓ Train: {len(X_train)} samples")

    # Step 6: Analyze feature importance
    print("\n[STEP 5] Analyzing feature importance...")
    importance_df = analyze_feature_importance(X_train, y_train, feature_cols)

    # Save full importance report
    importance_path = results_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Saved importance to: {importance_path}")

    # Step 7: Select top features
    print("\n[STEP 6] Selecting top features...")
    selected_features = select_top_features_by_importance(
        importance_df,
        n_features=25,
        cumulative_threshold=0.95
    )

    # Step 8: Add critical features (domain knowledge)
    print("\n[STEP 7] Ensuring critical features are included...")
    critical_features = [
        'rsi_14',           # Standard RSI
        'macd_diff',        # MACD histogram
        'returns',          # Simple returns
        'fear_greed_value', # Sentiment
        'close_lag_1',      # Yesterday's close
        'bb_width',         # Volatility
        'atr_14',           # True range
        'volume_ratio'      # Volume signal
    ]

    # Only add critical features that exist in our feature set
    critical_features = [f for f in critical_features if f in feature_cols]
    selected_features = add_critical_features(selected_features, critical_features)

    # Step 9: Save selected features
    print("\n[STEP 8] Saving selected features...")
    selected_path = results_dir / 'selected_features.txt'
    with open(selected_path, 'w') as f:
        f.write("# Selected Features for Bitcoin Price Prediction\n")
        f.write(f"# Generated: {pd.Timestamp.now()}\n")
        f.write(f"# Total: {len(selected_features)} features\n\n")
        for i, feat in enumerate(selected_features, 1):
            importance = importance_df[importance_df['feature'] == feat]['importance'].values
            if len(importance) > 0:
                f.write(f"{i:2d}. {feat:30s} (importance: {importance[0]:.2f})\n")
            else:
                f.write(f"{i:2d}. {feat:30s} (importance: N/A - critical feature)\n")

    print(f"✓ Saved {len(selected_features)} selected features to: {selected_path}")

    # Step 10: Create visualizations
    print("\n[STEP 9] Creating visualizations...")
    visualize_feature_importance(importance_df, output_path=results_dir / 'feature_importance_plot.png')

    # Step 11: Summary
    print("\n" + "="*70)
    print("  FEATURE SELECTION SUMMARY")
    print("="*70)

    print(f"\nOriginal features: {len(df.columns)} (after engineering)")
    print(f"After variance filter: {len(feature_cols)}")
    print(f"After correlation filter: {len(feature_cols)}")
    print(f"Final selected features: {len(selected_features)}")

    print(f"\nReduction: {len(df.columns)} → {len(selected_features)} ({len(selected_features)/len(df.columns)*100:.1f}%)")
    print(f"Feature-to-sample ratio: {len(selected_features)/X.shape[0]*100:.1f}% (target: <10%)")

    print(f"\n📊 Top 10 Features by Importance:")
    for i, row in importance_df.head(10).iterrows():
        selected_marker = "✓" if row['feature'] in selected_features else " "
        print(f"  {selected_marker} {i+1:2d}. {row['feature']:30s} ({row['importance']:8.2f})")

    print(f"\n✅ Feature selection complete!")
    print(f"   Use these features in your training pipeline:")
    print(f"   selected_features = {selected_path}")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
