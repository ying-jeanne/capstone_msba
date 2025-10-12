#!/usr/bin/env python3
"""
System Health Check
===================
Verifies that all components of the prediction system are working

Run this before deploying to check everything is set up correctly
"""

import sys
from pathlib import Path
from datetime import datetime

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}{Colors.END}\n")

def check_pass(text):
    print(f"{Colors.GREEN}✓{Colors.END} {text}")

def check_fail(text):
    print(f"{Colors.RED}✗{Colors.END} {text}")

def check_warn(text):
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def check_config():
    """Check configuration file"""
    print_header("1. Checking Configuration")
    
    try:
        import config
        
        # Check if GitHub repo is configured
        if hasattr(config, 'GITHUB_REPO'):
            if 'YOUR_USERNAME' in config.GITHUB_REPO:
                check_fail(f"GitHub repo not configured: {config.GITHUB_REPO}")
                print(f"   → Please update GITHUB_REPO in config.py")
                return False
            else:
                check_pass(f"GitHub repo configured: {config.GITHUB_REPO}")
                return True
        else:
            check_fail("GITHUB_REPO not found in config.py")
            return False
            
    except ImportError:
        check_fail("config.py not found")
        print("   → Create config.py from the template")
        return False


def check_models():
    """Check if trained models exist"""
    print_header("2. Checking Trained Models")
    
    models_dir = Path('models/saved_models')
    
    if not models_dir.exists():
        check_fail("models/saved_models/ directory not found")
        print("   → Run training scripts first:")
        print("   → python utils/train_daily_models.py")
        print("   → python utils/train_hourly_models.py")
        print("   → python utils/train_15min_models.py")
        return False
    
    # Check daily models
    daily_models = list((models_dir / 'daily').glob('*.json')) if (models_dir / 'daily').exists() else []
    if len(daily_models) >= 3:
        check_pass(f"Daily models found: {len(daily_models)} files")
    else:
        check_fail(f"Daily models missing or incomplete ({len(daily_models)}/3)")
        print("   → Run: python utils/train_daily_models.py")
        return False
    
    # Check hourly models
    hourly_models = list((models_dir / 'hourly').glob('*.json')) if (models_dir / 'hourly').exists() else []
    if len(hourly_models) >= 3:
        check_pass(f"Hourly models found: {len(hourly_models)} files")
    else:
        check_warn(f"Hourly models missing or incomplete ({len(hourly_models)}/3)")
        print("   → Run: python utils/train_hourly_models.py")
    
    # Check 15-min models
    min15_models = list((models_dir / '15min').glob('*.json')) if (models_dir / '15min').exists() else []
    if len(min15_models) >= 3:
        check_pass(f"15-min models found: {len(min15_models)} files")
    else:
        check_warn(f"15-min models missing or incomplete ({len(min15_models)}/3)")
        print("   → Run: python utils/train_15min_models.py")
    
    return len(daily_models) >= 3


def check_predictions():
    """Check if prediction files exist"""
    print_header("3. Checking Predictions")
    
    pred_dir = Path('data/predictions')
    
    if not pred_dir.exists():
        check_warn("data/predictions/ directory not found")
        print("   → Predictions will be created when scripts run")
        return True
    
    # Check for prediction files
    daily_pred = pred_dir / 'daily_predictions.csv'
    hourly_pred = pred_dir / 'hourly_predictions.csv'
    min15_pred = pred_dir / '15min_predictions.csv'
    
    if daily_pred.exists():
        check_pass("Daily predictions found")
    else:
        check_warn("Daily predictions not found")
        print("   → Run: python utils/predict_daily.py")
    
    if hourly_pred.exists():
        check_pass("Hourly predictions found")
    else:
        check_warn("Hourly predictions not found")
    
    if min15_pred.exists():
        check_pass("15-min predictions found")
    else:
        check_warn("15-min predictions not found")
    
    if hourly_pred.exists() or min15_pred.exists():
        print("   → Run: python utils/predict_hourly_and_15min.py")
    
    return True


def check_github_actions():
    """Check if GitHub Actions workflows exist"""
    print_header("4. Checking GitHub Actions Workflows")
    
    workflows_dir = Path('.github/workflows')
    
    if not workflows_dir.exists():
        check_fail(".github/workflows/ directory not found")
        return False
    
    workflows = {
        'train_models_weekly.yml': 'Weekly training',
        'predict_daily.yml': 'Daily predictions',
        'predict_intraday.yml': 'Intraday predictions'
    }
    
    all_exist = True
    for filename, description in workflows.items():
        if (workflows_dir / filename).exists():
            check_pass(f"{description}: {filename}")
        else:
            check_fail(f"{description} workflow not found: {filename}")
            all_exist = False
    
    return all_exist


def check_dependencies():
    """Check if required Python packages are installed"""
    print_header("5. Checking Python Dependencies")
    
    required = [
        'pandas',
        'numpy',
        'xgboost',
        'sklearn',
        'requests',
        'flask',
        'yfinance',
        'pycoingecko',
        'python-binance'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.replace('-', '_'))
            check_pass(f"{package}")
        except ImportError:
            check_fail(f"{package} not installed")
            missing.append(package)
    
    if missing:
        print(f"\n   → Install missing packages:")
        print(f"   → pip install {' '.join(missing)}")
        return False
    
    return True


def check_github_url():
    """Check if GitHub raw URLs are accessible"""
    print_header("6. Checking GitHub URL Access")
    
    try:
        import config
        import requests
        
        if 'YOUR_USERNAME' in config.GITHUB_REPO:
            check_warn("GitHub repo not configured yet")
            print("   → URLs will work after you push to GitHub")
            return True
        
        # Try to fetch daily predictions
        url = config.PREDICTION_URLS.get('daily')
        
        check_warn("Testing GitHub raw URL access...")
        print(f"   URL: {url}")
        
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                check_pass("GitHub raw URLs are accessible!")
            elif response.status_code == 404:
                check_warn("Predictions not on GitHub yet")
                print("   → Push your code and run prediction scripts")
                print("   → Then commit and push the predictions")
            else:
                check_warn(f"Unexpected status code: {response.status_code}")
        except requests.exceptions.Timeout:
            check_warn("GitHub request timed out (might be network issue)")
        except requests.exceptions.RequestException as e:
            check_warn(f"Could not reach GitHub: {e}")
            
        return True
        
    except ImportError:
        check_warn("Cannot test URLs (config.py not found)")
        return True
    except Exception as e:
        check_warn(f"Error testing URLs: {e}")
        return True


def main():
    """Run all health checks"""
    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"  BITCOIN PREDICTION SYSTEM - HEALTH CHECK")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}{Colors.END}")
    
    checks = [
        ("Configuration", check_config),
        ("Trained Models", check_models),
        ("Predictions", check_predictions),
        ("GitHub Actions", check_github_actions),
        ("Dependencies", check_dependencies),
        ("GitHub URLs", check_github_url),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n{Colors.RED}Error in {name}: {e}{Colors.END}")
            results.append((name, False))
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        if result:
            print(f"{Colors.GREEN}✓{Colors.END} {name}")
        else:
            print(f"{Colors.RED}✗{Colors.END} {name}")
    
    print(f"\n{Colors.BOLD}Score: {passed}/{total} checks passed{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 All checks passed! System is ready!{Colors.END}\n")
        print("Next steps:")
        print("  1. Commit and push to GitHub")
        print("  2. Enable GitHub Actions")
        print("  3. Deploy to PythonAnywhere")
        return 0
    elif passed >= total * 0.7:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}⚠️  System mostly ready, but some issues found{Colors.END}\n")
        print("Review the warnings above and fix critical issues")
        return 1
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}❌ System not ready - multiple issues found{Colors.END}\n")
        print("Fix the errors above before deploying")
        return 2


if __name__ == '__main__':
    sys.exit(main())
