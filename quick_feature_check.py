"""
Quick Feature Importance Check
Run this after training to see which features matter
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import xgboost as xgb
import joblib
import numpy as np

# Daily models
print("\n=== DAILY MODELS ===")
models_dir = Path('models/saved_models/daily')
features = joblib.load(models_dir / 'feature_cols_daily.pkl')

for horizon in [1, 3, 7]:
    model = xgb.XGBRegressor()
    model.load_model(str(models_dir / f'xgboost_{horizon}d.json'))
    
    importance = model.feature_importances_
    top_idx = np.argsort(importance)[::-1][:10]
    
    print(f"\n{horizon}d - Top 10 features:")
    for i, idx in enumerate(top_idx, 1):
        print(f"  {i}. {features[idx]}: {importance[idx]:.4f}")

print(f"\nTotal features: {len(features)}")
print(f"Features with importance > 0.01: {np.sum(importance > 0.01)}")
print(f"Features with importance > 0.001: {np.sum(importance > 0.001)}")
