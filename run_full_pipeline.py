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
from utils.data_fetcher import get_bitcoin_data_incremental

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
    # STEP 1: Fetch Latest Data (INCREMENTAL)
    # ========================================================================
    print_step(1, "FETCH LATEST DATA (Incremental)")
    print("   This will update cached Bitcoin data with only new bars since last fetch.")
    print("   - Yahoo Finance: 5 years daily (incremental)")
    try:
        # Fetch Yahoo 5-year data incrementally
        print("\n📊 Fetching Yahoo Finance daily data (5 years)...")
        yahoo_df = get_bitcoin_data_incremental(source='yahoo_5y', days=1825, verbose=True)

        if yahoo_df is not None:
            print_success(f"Yahoo: {len(yahoo_df)} samples ({yahoo_df.index[0].date()} to {yahoo_df.index[-1].date()})")
            completed_steps.append("Yahoo Data Fetch")
        else:
            print_warning("Yahoo data fetch failed, using cached data if available")

    except Exception as e:
        print_error(f"Data fetching error: {str(e)}")
        print_warning("Continuing with existing cached data...")

    # ========================================================================
    # STEP 2: Train Daily Models (1d, 3d, 7d)
    # ========================================================================
    print_step(2, "TRAIN DAILY MODELS (1d, 3d, 7d)")
    print("   Training daily models with multiple algorithms:")
    print("   - Uses Yahoo Finance 5y daily data (see 10Y_VS_5Y_COMPARISON.md)")
    print("   - Includes sentiment features (Fear & Greed)")
    print("   - Models: XGBoost, Random Forest, LightGBM, CatBoost")
    print("   - Expected MAPE: ~1.53% (1-day)")
    print("   - Training 3 horizons: 1-day, 3-day, 7-day")

    if run_script("utils/train_daily_models.py", "Train daily XGBoost models"):
        completed_steps.append("Daily Models Training")
    else:
        print_warning("Daily models training failed, but continuing...")

    # # ========================================================================
    # # STEP 3: Train Hourly Models (1h, 4h, 6h, 12h, 24h)
    # # ========================================================================
    # print_step(3, "TRAIN HOURLY MODELS (1h, 4h, 6h, 12h, 24h)")
    # print("   Training hourly XGBoost models:")
    # print("   - Uses Cryptocompare 365d hourly data")
    # print("   - NO sentiment (Fear & Greed is daily only)")
    # print("   - Training 5 models: 1h, 4h, 6h, 12h, 24h")

    # if run_script("utils/train_hourly_models.py", "Train hourly XGBoost models"):
    #     completed_steps.append("Hourly Models Training")
    # else:
    #     print_warning("Hourly models training failed, but continuing...")

    # ========================================================================
    # STEP 3: Generate Daily Predictions
    # ========================================================================
    print_step(3, "GENERATE DAILY PREDICTIONS")
    print("   Generating predictions using trained daily models:")
    print("   - 1-day, 3-day, 7-day predictions")
    print("   - Saves to data/predictions/daily_predictions.csv")

    if run_script("utils/predict_daily.py", "Generate daily predictions"):
        completed_steps.append("Daily Predictions")
    else:
        print_warning("Daily predictions failed, but continuing...")

    # ========================================================================
    # STEP 4: Generate Hourly Predictions
    # ========================================================================
    # print_step(4, "GENERATE HOURLY PREDICTIONS")
    # print("   Generating predictions using trained hourly models:")
    # print("   - 1h, 4h, 6h, 12h, 24h predictions")
    # print("   - Saves to data/predictions/hourly_predictions.csv")

    # if run_script("utils/predict_hourly.py", "Generate hourly predictions"):
    #     completed_steps.append("Hourly Predictions")
    # else:
    #     print_warning("Hourly predictions failed, but continuing...")



    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_header("PIPELINE EXECUTION SUMMARY")

    print(f"\n{Colors.BOLD}Completed Steps:{Colors.ENDC}")
    for i, step in enumerate(completed_steps, 1):
        print(f"   {Colors.OKGREEN}✓{Colors.ENDC} {step}")

    total_steps = 3  # Updated: Data, Features, DataPrep, Daily, Hourly
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
    print(f"   📊 Daily Models: {Colors.OKCYAN}models/saved_models/daily/{Colors.ENDC}")
    print(f"   📊 Hourly Models: {Colors.OKCYAN}models/saved_models/hourly/{Colors.ENDC}")
    print(f"   📈 Daily Predictions: {Colors.OKCYAN}data/predictions/daily_predictions.csv{Colors.ENDC}")
    # print(f"   📈 Hourly Predictions: {Colors.OKCYAN}data/predictions/hourly_predictions.csv{Colors.ENDC}")
    print(f"   � Daily Metrics: {Colors.OKCYAN}results/daily_models_metrics.csv{Colors.ENDC}")
    # print(f"   📋 Hourly Metrics: {Colors.OKCYAN}results/hourly_models_metrics.csv{Colors.ENDC}")

    print(f"\n{Colors.BOLD}Pipeline End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.ENDC}")
    print(f"\n{'='*70}\n")

    # Show next steps
    print(f"{Colors.BOLD}NEXT STEPS:{Colors.ENDC}")
    print(f"   1. View daily predictions: {Colors.OKCYAN}cat data/predictions/daily_predictions.csv{Colors.ENDC}")
    # print(f"   2. View hourly predictions: {Colors.OKCYAN}cat data/predictions/hourly_predictions.csv{Colors.ENDC}")
    print(f"   3. Check model performance: {Colors.OKCYAN}cat results/*_metrics.csv{Colors.ENDC}")
    print(f"   4. Read documentation: {Colors.OKCYAN}cat DATASOURCE.md{Colors.ENDC}")
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
