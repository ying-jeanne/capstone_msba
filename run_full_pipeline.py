#!/usr/bin/env python3
"""
What this script is doing:
1. Fetch latest data from APIs
2. Engineer features
3. Prepare return-based targets
4. Train all models (XGBoost, Random Forest, Gradient Boosting)
5. Generate comparison reports
6. Diagnose bias

This script automatically updates with the latest Bitcoin data and retrains models.

Usage:
    python run_full_pipeline.py
"""

import sys
import os
from pathlib import Path
import subprocess
from datetime import datetime

# Import data fetcher functions
from utils.data_fetcher import get_all_sources

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(message):
    """Print colored header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f"  {message}")
    print(f"{'='*70}{Colors.ENDC}")


def print_step(step_num, message):
    """Print step number and message"""
    print(f"\n{Colors.OKBLUE}{Colors.BOLD}[STEP {step_num}] {message}{Colors.ENDC}")


def print_success(message):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_warning(message):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def print_error(message):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def run_script(script_path, description):
    """
    Run a Python script and handle errors

    Args:
        script_path (str): Path to the Python script
        description (str): Description of what the script does

    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n   Running: {description}")
    print(f"   Script: {script_path}")

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        # Check for errors
        if result.returncode != 0:
            print_error(f"Script failed with return code {result.returncode}")
            if result.stderr:
                print(f"   Error: {result.stderr}")
            return False

        print_success(f"Completed: {description}")
        return True

    except subprocess.TimeoutExpired:
        print_error(f"Script timed out after 10 minutes")
        return False
    except Exception as e:
        print_error(f"Error running script: {str(e)}")
        return False


def main():
    """Run the complete Bitcoin prediction pipeline"""
    # Track which steps completed successfully
    completed_steps = []

    # ========================================================================
    # STEP 1: Fetch Latest Data
    # ========================================================================
    print_step(1, "FETCH LATEST DATA")
    print("   This will download the latest Bitcoin price data from: Yahoo Finance, CoinGecko, and CryptoCompare.")
    try:
        results = get_all_sources(days=60, yahoo_period='2y', save_to_disk=True)

        # Check results from each source
        success_count = 0
        for source, result in results.items():
            if result['status'] == 'success':
                samples = result['samples']
                date_range = result['date_range']
                filepath = result.get('filepath', 'N/A')
                print_success(f"{source.capitalize()}: {samples} samples ({date_range['start']} to {date_range['end']})")
                if filepath != 'N/A':
                    print(f"   Saved to: {filepath}")
                success_count += 1
            else:
                print_warning(f"{source.capitalize()}: Failed - {result.get('message', 'Unknown error')}")

        if success_count > 0:
            print_success(f"Data fetching completed: {success_count}/3 sources successful")
            completed_steps.append("Data Fetching")
        else:
            print_warning("All data sources failed, but continuing with existing data...")

    except Exception as e:
        print_error(f"Data fetching error: {str(e)}")
        print_warning("Continuing with existing data...")

    # ========================================================================
    # STEP 2: Feature Engineering
    # ========================================================================
    print_step(2, "FEATURE ENGINEERING")
    print("   Creating 55 technical indicators from raw OHLCV data:")
    print("   - Trend indicators (SMA, EMA, MACD, ADX)")
    print("   - Momentum indicators (RSI, Stochastic, ROC)")
    print("   - Volatility indicators (Bollinger Bands, ATR)")
    print("   - Statistical features (lags, rolling stats)")

    if run_script("utils/feature_engineering.py", "Create 55 technical indicators"):
        completed_steps.append("Feature Engineering")
    else:
        print_error("Feature engineering failed. Cannot continue.")
        return

    # ========================================================================
    # STEP 3: Data Preparation (Return-Based Targets)
    # ========================================================================
    print_step(3, "DATA PREPARATION (RETURN-BASED)")
    print("   Preparing data with return-based targets:")
    print("   - Creating return targets (for training)")
    print("   - Creating price targets (for evaluation)")
    print("   - Chronological train/val/test split (70/15/15)")
    print("   - RobustScaler normalization")
    print("   - Multi-horizon targets (1, 3, 7 days)")

    if run_script("utils/data_preparation_returns.py", "Prepare return-based targets"):
        completed_steps.append("Data Preparation")
    else:
        print_error("Data preparation failed. Cannot continue.")
        return

    # ========================================================================
    # STEP 4: Train XGBoost (Best Model)
    # ========================================================================
    print_step(4, "TRAIN XGBOOST (BEST MODEL)")
    print("   Training XGBoost with return-based prediction:")
    print("   - Expected MAPE: ~1.16% (1-day)")
    print("   - Expected R²: ~0.865 (1-day)")
    print("   - Training 3 models: 1-day, 3-day, 7-day")

    if run_script("models/xgboost_returns.py", "Train XGBoost models"):
        completed_steps.append("XGBoost Training")
    else:
        print_warning("XGBoost training failed, but continuing...")

    # ========================================================================
    # STEP 5: Train Random Forest & Gradient Boosting
    # ========================================================================
    print_step(5, "TRAIN RANDOM FOREST & GRADIENT BOOSTING")
    print("   Training additional models for comparison:")
    print("   - Random Forest (expected MAPE: ~1.89%)")
    print("   - Gradient Boosting (expected MAPE: ~3.14%)")
    print("   - 3 models each: 1-day, 3-day, 7-day")

    if run_script("models/sklearn_models_returns.py", "Train RF & GB models"):
        completed_steps.append("RF & GB Training")
    else:
        print_warning("RF & GB training failed, but continuing...")

    # ========================================================================
    # STEP 6: Compare All Models
    # ========================================================================
    print_step(6, "COMPARE ALL MODELS")
    print("   Generating comprehensive comparison reports:")
    print("   - Performance metrics (MAPE, R², MAE, Directional Accuracy)")
    print("   - Rankings by metric")
    print("   - Visualizations")

    if run_script("utils/compare_all_models.py", "Compare all models"):
        completed_steps.append("Model Comparison")
    else:
        print_warning("Model comparison failed, but continuing...")

    # ========================================================================
    # STEP 7: Diagnose Bias
    # ========================================================================
    print_step(7, "DIAGNOSE BIAS")
    print("   Verifying systematic bias has been eliminated:")
    print("   - Before/after comparison")
    print("   - Success criteria verification")
    print("   - Diagnostic visualizations")

    if run_script("utils/diagnose_bias.py", "Verify bias elimination"):
        completed_steps.append("Bias Diagnosis")
    else:
        print_warning("Bias diagnosis failed, but continuing...")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("PIPELINE EXECUTION SUMMARY")

    print(f"\n{Colors.BOLD}Completed Steps:{Colors.ENDC}")
    for i, step in enumerate(completed_steps, 1):
        print(f"   {Colors.OKGREEN}✓{Colors.ENDC} {step}")

    total_steps = 7
    success_rate = (len(completed_steps) / total_steps) * 100

    print(f"\n{Colors.BOLD}Success Rate: {success_rate:.0f}% ({len(completed_steps)}/{total_steps} steps){Colors.ENDC}")

    if success_rate == 100:
        print(f"\n{Colors.OKGREEN}{Colors.BOLD}🎉 ALL STEPS COMPLETED SUCCESSFULLY!{Colors.ENDC}")
    elif success_rate >= 70:
        print(f"\n{Colors.WARNING}{Colors.BOLD}⚠️  MOSTLY SUCCESSFUL - Some steps failed but core pipeline completed{Colors.ENDC}")
    else:
        print(f"\n{Colors.FAIL}{Colors.BOLD}❌ PIPELINE FAILED - Multiple critical steps failed{Colors.ENDC}")

    # Show where results are
    print(f"\n{Colors.BOLD}Results Location:{Colors.ENDC}")
    print(f"   📊 Performance Metrics: {Colors.OKCYAN}results/all_models_returns_combined.csv{Colors.ENDC}")
    print(f"   📈 Visualizations: {Colors.OKCYAN}results/all_models_returns_comparison.png{Colors.ENDC}")
    print(f"   🔍 Bias Analysis: {Colors.OKCYAN}results/bias_fix_diagnostic.png{Colors.ENDC}")
    print(f"   🤖 Trained Models: {Colors.OKCYAN}models/saved_models/*_returns_*.json/.pkl{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Pipeline End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"\n{'='*70}\n")

    # Show next steps
    print(f"{Colors.BOLD}NEXT STEPS:{Colors.ENDC}")
    print(f"   1. View results: {Colors.OKCYAN}open results/{Colors.ENDC}")
    print(f"   2. Check models: {Colors.OKCYAN}ls -lh models/saved_models/{Colors.ENDC}")
    print(f"   3. Read documentation: {Colors.OKCYAN}cat README.md{Colors.ENDC}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Pipeline interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.FAIL}Pipeline failed with error: {str(e)}{Colors.ENDC}")
        sys.exit(1)
