"""
Comprehensive Model Comparison - Return-Based Predictions
=========================================================

Compares all return-based models:
- XGBoost (best performance expected)
- Random Forest
- Gradient Boosting

Author: Bitcoin Price Prediction System
Date: 2025-10-05
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


REQUIRED_COLUMNS = [
    'Model',
    'Horizon',
    'MAPE (%)',
    'MAE ($)',
    'Directional Acc (%)',
    'R²'
]


def load_all_results():
    """Load results from all return-based models."""
    results_dir = Path('results')
    comparison_path = results_dir / 'model_comparison.csv'

    if comparison_path.exists():
        df = pd.read_csv(comparison_path)

        # Focus on daily horizons (1d, 3d, 7d)
        df = df[df['Timeframe'].isin(['Daily'])]
        df = df[df['Horizon'].isin(['1d', '3d', '7d'])]

        # Rename columns to legacy format expected downstream
        df = df.rename(columns={
            'Price_MAPE': 'MAPE (%)',
            'Price_MAE': 'MAE ($)',
            'Directional_Accuracy': 'Directional Acc (%)',
            'Return_R2': 'R²'
        })

        # Ensure expected columns exist
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        df['Balanced Score'] = df['Directional Acc (%)'] - 0.5 * df['MAPE (%)']

        # Reorder
        ordered_cols = REQUIRED_COLUMNS + ['Balanced Score'] + [c for c in df.columns if c not in REQUIRED_COLUMNS + ['Balanced Score']]
        df = df[ordered_cols]
        return df.reset_index(drop=True)

    # Fallback to legacy CSVs if available
    legacy_files = {
        'XGBoost': results_dir / 'xgboost_returns_summary.csv',
        'Random Forest': results_dir / 'rf_returns_summary.csv',
        'Gradient Boosting': results_dir / 'gb_returns_summary.csv'
    }

    frames = []
    for model_name, path in legacy_files.items():
        if path.exists():
            model_df = pd.read_csv(path)
            model_df['Model'] = model_name
            frames.append(model_df)

    if not frames:
        raise FileNotFoundError(
            "No comparison files found. Expected 'results/model_comparison.csv' or legacy summary CSVs."
        )

    all_results = pd.concat(frames, ignore_index=True)
    all_results['Balanced Score'] = all_results['Directional Acc (%)'] - 0.5 * all_results['MAPE (%)']
    return all_results.reset_index(drop=True)


def create_comparison_visualizations(df):
    """Create comprehensive comparison visualizations."""

    # Prepare data
    models = df['Model'].unique()
    horizons = ['1d', '3d', '7d']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Define colors for each model
    palette = plt.get_cmap('tab10')
    default_colors = {
        'XGBoost': '#1f77b4',
        'Random Forest': '#ff7f0e',
        'Gradient Boosting': '#2ca02c',
        'CatBoost': '#8b5cf6',
        'LightGBM': '#10b981'
    }

    colors = {}
    for idx, model in enumerate(models):
        colors[model] = default_colors.get(model, palette(idx % 10))

    x = np.arange(len(horizons))
    width = max(0.15, 0.65 / max(len(models), 1))

    # Helper to plot grouped bars
    def plot_metric(ax, metric_key, title, ylabel, highlight_line=None):
        for i, model in enumerate(models):
            model_data = df[df['Model'] == model]
            values = []
            for h in horizons:
                rows = model_data[model_data['Horizon'] == h]
                if metric_key in rows.columns and not rows.empty:
                    values.append(rows.iloc[0][metric_key])
                else:
                    values.append(np.nan)
            ax.bar(x + i * width, values, width, label=model, color=colors[model], alpha=0.85)

        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        center_offset = ((len(models) - 1) / 2) * width
        ax.set_xticks(x + center_offset)
        ax.set_xticklabels(horizons)
        ax.grid(True, alpha=0.3, axis='y')
        if highlight_line is not None:
            ax.axhline(y=highlight_line[0], color=highlight_line[1], linestyle='--', alpha=0.5, linewidth=1)

    plot_metric(axes[0, 0], 'MAPE (%)', 'MAPE Comparison (Lower is Better)', 'MAPE (%)', highlight_line=(5, 'green'))
    plot_metric(axes[0, 1], 'R²', 'R² Score Comparison', 'R² Score', highlight_line=(0, 'black'))
    plot_metric(axes[1, 0], 'MAE ($)', 'Mean Absolute Error Comparison', 'MAE ($)')
    plot_metric(axes[1, 1], 'Balanced Score', 'Balanced Score Comparison (Higher is Better)', 'Balanced Score', highlight_line=(0, 'black'))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(len(models), 5), frameon=False, fontsize=11)

    plt.suptitle('Return-Based Models - Comprehensive Performance Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = Path('results/all_models_returns_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Visualization saved: {save_path}")


def create_ranking_table(df):
    """Create a ranking table for each metric."""

    print(f"\n{'='*80}")
    print(f"  MODEL RANKINGS BY METRIC")
    print(f"{'='*80}")

    horizons = ['1d', '3d', '7d']

    for horizon in horizons:
        print(f"\n{horizon.upper()} AHEAD PREDICTIONS:")
        print(f"{'-'*80}")

        horizon_data = df[df['Horizon'] == horizon].copy()

        # Rank by MAPE (lower is better)
        horizon_data_sorted = horizon_data.sort_values('MAPE (%)')
        print(f"\n  📊 By MAPE (Lower is Better):")
        for i, (_, row) in enumerate(horizon_data_sorted.iterrows(), 1):
            print(f"     {i}. {row['Model']}: {row['MAPE (%)']:.2f}%")

        # Rank by R² (higher is better)
        horizon_data_sorted = horizon_data.sort_values('R²', ascending=False)
        print(f"\n  📊 By R² (Higher is Better):")
        for i, (_, row) in enumerate(horizon_data_sorted.iterrows(), 1):
            print(f"     {i}. {row['Model']}: {row['R²']:.4f}")

        # Rank by Directional Accuracy (higher is better)
        horizon_data_sorted = horizon_data.sort_values('Directional Acc (%)', ascending=False)
        print(f"\n  📊 By Directional Accuracy (Higher is Better):")
        for i, (_, row) in enumerate(horizon_data_sorted.iterrows(), 1):
            print(f"     {i}. {row['Model']}: {row['Directional Acc (%)']:.1f}%")

        if 'Balanced Score' in horizon_data.columns:
            horizon_data_sorted = horizon_data.sort_values('Balanced Score', ascending=False)
            print(f"\n  📊 By Balanced Score (Higher is Better):")
            for i, (_, row) in enumerate(horizon_data_sorted.iterrows(), 1):
                print(f"     {i}. {row['Model']}: {row['Balanced Score']:.2f}")


def print_best_model_summary(df):
    """Identify and print the best model for each horizon."""

    print(f"\n{'='*80}")
    print(f"  BEST MODEL RECOMMENDATIONS")
    print(f"{'='*80}")

    horizons = ['1d', '3d', '7d']

    for horizon in horizons:
        horizon_data = df[df['Horizon'] == horizon].copy()

        # Best by MAPE (primary metric)
        best_mape = horizon_data.loc[horizon_data['MAPE (%)'].idxmin()]

        # Best by R²
        best_r2 = horizon_data.loc[horizon_data['R²'].idxmax()]

        print(f"\n{horizon.upper()} AHEAD:")
        print(f"  🏆 Best Overall (by MAPE): {best_mape['Model']}")
        print(f"     MAPE: {best_mape['MAPE (%)']:.2f}%, R²: {best_mape['R²']:.4f}, Dir Acc: {best_mape['Directional Acc (%)']:.1f}%, Balanced Score: {best_mape['Balanced Score']:.2f}")

        if best_r2['Model'] != best_mape['Model']:
            print(f"  📊 Best R²: {best_r2['Model']}")
            print(f"     MAPE: {best_r2['MAPE (%)']:.2f}%, R²: {best_r2['R²']:.4f}, Dir Acc: {best_r2['Directional Acc (%)']:.1f}%, Balanced Score: {best_r2['Balanced Score']:.2f}")


def analyze_improvements(df):
    """Analyze improvements from old price-based to new return-based predictions."""

    print(f"\n{'='*80}")
    print(f"  IMPROVEMENT ANALYSIS (OLD PRICE-BASED → NEW RETURN-BASED)")
    print(f"{'='*80}")

    # Old results (estimated from previous runs)
    old_results = {
        '1d': {'R²': -14.22, 'MAPE': 16.2, 'MAE': 17838},
        '3d': {'R²': -30.46, 'MAPE': 23.2, 'MAE': 25488},
        '7d': {'R²': -22.46, 'MAPE': 19.3, 'MAE': 21250}
    }

    # Get best new results for each horizon
    horizons = ['1d', '3d', '7d']

    for horizon in horizons:
        horizon_data = df[df['Horizon'] == horizon]
        best_new = horizon_data.loc[horizon_data['MAPE (%)'].idxmin()]

        old = old_results[horizon]

        r2_improvement = best_new['R²'] - old['R²']
        mape_improvement = old['MAPE'] - best_new['MAPE (%)']
        mae_improvement = old['MAE'] - best_new['MAE ($)']

        print(f"\n{horizon.upper()} AHEAD (Best: {best_new['Model']}):")
        print(f"  R²:   {old['R²']:+7.2f} → {best_new['R²']:+7.4f}  (improvement: {r2_improvement:+.2f})")
        print(f"  MAPE: {old['MAPE']:7.2f}% → {best_new['MAPE (%)']:7.2f}%  (improvement: {mape_improvement:+.2f}%)")
        print(f"  MAE:  ${old['MAE']:7,.0f} → ${best_new['MAE ($)']:7,.0f}  (improvement: ${mae_improvement:+,.0f})")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("  COMPREHENSIVE MODEL COMPARISON - RETURN-BASED PREDICTIONS")
    print("="*80)

    # Load all results
    print("\n📂 Loading results from all models...")
    all_results = load_all_results()

    # Save combined results
    combined_path = Path('results/all_models_returns_combined.csv')
    all_results.to_csv(combined_path, index=False)
    print(f"   ✓ Combined results saved: {combined_path}")

    # Display full comparison table
    print(f"\n📊 Complete Results Table:")
    summary_cols = ['Model', 'Horizon', 'MAPE (%)', 'MAE ($)', 'Directional Acc (%)', 'R²', 'Balanced Score']
    summary_cols = [col for col in summary_cols if col in all_results.columns]
    display_df = all_results.copy()
    for col in ['MAPE (%)', 'MAE ($)', 'Directional Acc (%)', 'Balanced Score']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    if 'R²' in display_df.columns:
        display_df['R²'] = display_df['R²'].round(4)
    print(display_df[summary_cols].to_string(index=False))

    # Create visualizations
    print(f"\n📊 Creating comparison visualizations...")
    create_comparison_visualizations(all_results)

    # Create rankings
    create_ranking_table(all_results)

    # Best model recommendations
    print_best_model_summary(all_results)

    # Improvement analysis
    analyze_improvements(all_results)

    # Final summary
    print(f"\n{'='*80}")
    print(f"  KEY FINDINGS")
    print(f"{'='*80}")

    # Find overall best model
    best_1d = all_results[all_results['Horizon'] == '1d'].loc[all_results[all_results['Horizon'] == '1d']['MAPE (%)'].idxmin()]
    best_3d = all_results[all_results['Horizon'] == '3d'].loc[all_results[all_results['Horizon'] == '3d']['MAPE (%)'].idxmin()]
    best_7d = all_results[all_results['Horizon'] == '7d'].loc[all_results[all_results['Horizon'] == '7d']['MAPE (%)'].idxmin()]

    print(f"\n✅ Best Models by Horizon:")
    print(f"   1d: {best_1d['Model']} (MAPE: {best_1d['MAPE (%)']:.2f}%, R²: {best_1d['R²']:.4f}, Balanced Score: {best_1d['Balanced Score']:.2f})")
    print(f"   3d: {best_3d['Model']} (MAPE: {best_3d['MAPE (%)']:.2f}%, R²: {best_3d['R²']:.4f}, Balanced Score: {best_3d['Balanced Score']:.2f})")
    print(f"   7d: {best_7d['Model']} (MAPE: {best_7d['MAPE (%)']:.2f}%, R²: {best_7d['R²']:.4f}, Balanced Score: {best_7d['Balanced Score']:.2f})")

    print(f"\n✅ Success Criteria Check:")
    models_with_positive_r2_1d = all_results[(all_results['Horizon'] == '1d') & (all_results['R²'] > 0)]['Model'].tolist()
    models_with_mape_under_5_1d = all_results[(all_results['Horizon'] == '1d') & (all_results['MAPE (%)'] < 5)]['Model'].tolist()

    print(f"   Positive R² for 1d: {', '.join(models_with_positive_r2_1d)}")
    print(f"   MAPE < 5% for 1d: {', '.join(models_with_mape_under_5_1d)}")

    print(f"\n💡 Recommendations:")
    print(f"   1. Use {best_1d['Model']} for 1-day predictions (MAPE: {best_1d['MAPE (%)']:.2f}%)")
    print(f"   2. Use {best_3d['Model']} for 3-day predictions (MAPE: {best_3d['MAPE (%)']:.2f}%)")
    print(f"   3. Use {best_7d['Model']} for 7-day predictions (MAPE: {best_7d['MAPE (%)']:.2f}%)")
    print(f"   4. Consider ensemble combining top performers")

    print(f"\n📊 Files Generated:")
    print(f"   - results/all_models_returns_combined.csv")
    print(f"   - results/all_models_returns_comparison.png")

    print("\n" + "="*80 + "\n")
