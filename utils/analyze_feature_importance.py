"""
Analyze Feature Importance from Trained Models
===============================================
Identifies which features are most predictive
Helps remove noisy/useless features

Run after training models to see feature importance
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
import joblib


def analyze_daily_models():
    """Analyze feature importance for daily models"""
    print("\n" + "="*70)
    print("  DAILY MODELS - FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    models_dir = Path('models/saved_models/daily')
    
    # Load feature names
    feature_cols = joblib.load(models_dir / 'feature_cols_daily.pkl')
    print(f"\n📊 Total features: {len(feature_cols)}")
    
    # Aggregate importance across all horizons
    all_importances = []
    
    for horizon in [1, 3, 7]:
        model_path = models_dir / f'xgboost_{horizon}d.json'
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        
        importance = model.feature_importances_
        all_importances.append(importance)
        
        print(f"\n{horizon}-day model:")
        print(f"  Non-zero features: {np.sum(importance > 0)}/{len(feature_cols)}")
        print(f"  Mean importance: {importance.mean():.6f}")
    
    # Average importance across all models
    avg_importance = np.mean(all_importances, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': avg_importance,
        '1d': all_importances[0],
        '3d': all_importances[1],
        '7d': all_importances[2]
    }).sort_values('importance', ascending=False)
    
    print(f"\n{'='*70}")
    print("  TOP 20 MOST IMPORTANT FEATURES")
    print(f"{'='*70}")
    print(importance_df.head(20).to_string(index=False))
    
    print(f"\n{'='*70}")
    print("  BOTTOM 20 LEAST IMPORTANT FEATURES (Candidates for Removal)")
    print(f"{'='*70}")
    print(importance_df.tail(20).to_string(index=False))
    
    # Analyze by importance threshold
    print(f"\n{'='*70}")
    print("  FEATURE SELECTION RECOMMENDATIONS")
    print(f"{'='*70}")
    
    thresholds = [0.001, 0.005, 0.01, 0.02]
    for threshold in thresholds:
        n_features = np.sum(avg_importance > threshold)
        pct = (n_features / len(feature_cols)) * 100
        print(f"  Threshold {threshold:.3f}: Keep {n_features}/{len(feature_cols)} features ({pct:.1f}%)")
    
    # Recommended threshold (keep 70-80% of features)
    recommended_threshold = np.percentile(avg_importance[avg_importance > 0], 20)
    n_recommended = np.sum(avg_importance > recommended_threshold)
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"  Use threshold: {recommended_threshold:.6f}")
    print(f"  Keep top {n_recommended} features ({n_recommended/len(feature_cols)*100:.1f}%)")
    print(f"  Remove {len(feature_cols) - n_recommended} low-importance features")
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    importance_df.to_csv(results_dir / 'feature_importance_daily.csv', index=False)
    print(f"\n✅ Saved to: results/feature_importance_daily.csv")
    
    return importance_df, recommended_threshold


def analyze_hourly_models():
    """Analyze feature importance for hourly models"""
    print("\n" + "="*70)
    print("  HOURLY MODELS - FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    models_dir = Path('models/saved_models/hourly')
    
    # Load feature names
    feature_cols = joblib.load(models_dir / 'feature_cols_hourly.pkl')
    print(f"\n📊 Total features: {len(feature_cols)}")
    
    # Aggregate importance across all horizons
    all_importances = []
    
    for horizon in [1, 4, 6, 12, 24]:
        model_path = models_dir / f'xgboost_{horizon}h.json'
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))
        
        importance = model.feature_importances_
        all_importances.append(importance)
        
        print(f"\n{horizon}-hour model:")
        print(f"  Non-zero features: {np.sum(importance > 0)}/{len(feature_cols)}")
        print(f"  Mean importance: {importance.mean():.6f}")
    
    # Average importance across all models
    avg_importance = np.mean(all_importances, axis=0)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': avg_importance,
        '1h': all_importances[0],
        '4h': all_importances[1],
        '6h': all_importances[2],
        '12h': all_importances[3],
        '24h': all_importances[4]
    }).sort_values('importance', ascending=False)
    
    print(f"\n{'='*70}")
    print("  TOP 20 MOST IMPORTANT FEATURES")
    print(f"{'='*70}")
    print(importance_df.head(20).to_string(index=False))
    
    print(f"\n{'='*70}")
    print("  BOTTOM 20 LEAST IMPORTANT FEATURES (Candidates for Removal)")
    print(f"{'='*70}")
    print(importance_df.tail(20).to_string(index=False))
    
    # Analyze by importance threshold
    print(f"\n{'='*70}")
    print("  FEATURE SELECTION RECOMMENDATIONS")
    print(f"{'='*70}")
    
    thresholds = [0.001, 0.005, 0.01, 0.02]
    for threshold in thresholds:
        n_features = np.sum(avg_importance > threshold)
        pct = (n_features / len(feature_cols)) * 100
        print(f"  Threshold {threshold:.3f}: Keep {n_features}/{len(feature_cols)} features ({pct:.1f}%)")
    
    # Recommended threshold (keep 70-80% of features)
    recommended_threshold = np.percentile(avg_importance[avg_importance > 0], 20)
    n_recommended = np.sum(avg_importance > recommended_threshold)
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"  Use threshold: {recommended_threshold:.6f}")
    print(f"  Keep top {n_recommended} features ({n_recommended/len(feature_cols)*100:.1f}%)")
    print(f"  Remove {len(feature_cols) - n_recommended} low-importance features")
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    importance_df.to_csv(results_dir / 'feature_importance_hourly.csv', index=False)
    print(f"\n✅ Saved to: results/feature_importance_hourly.csv")
    
    return importance_df, recommended_threshold


def main():
    """Run feature importance analysis"""
    print("\n" + "="*70)
    print("  🔍 FEATURE IMPORTANCE ANALYZER")
    print("="*70)
    print("\n📋 This will analyze which features are most predictive")
    print("   and recommend which features to remove")
    
    try:
        # Analyze both daily and hourly models
        daily_df, daily_threshold = analyze_daily_models()
        hourly_df, hourly_threshold = analyze_hourly_models()
        
        print("\n" + "="*70)
        print("  ✅ ANALYSIS COMPLETE")
        print("="*70)
        print("\n📊 Next Steps:")
        print("  1. Review: results/feature_importance_daily.csv")
        print("  2. Review: results/feature_importance_hourly.csv")
        print("  3. Consider removing low-importance features")
        print("  4. Re-train models with selected features")
        print("  5. Compare performance (price MAPE and directional accuracy)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you've trained models first:")
        print("   python run_full_pipeline.py")
        return
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
