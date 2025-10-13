# validate.py
# Python validation script for Amazon ML Challenge 2025
# Works with XGBoost/LightGBM .pkl files instead of PyTorch .pt files

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def find_model_files(checkpoint_dir: Path) -> Dict[str, List[Path]]:
    """Find all trained model files"""
    models = {
        'xgboost': [],
        'lightgbm': [],
        'catboost': []
    }
    
    if not checkpoint_dir.exists():
        return models
    
    # Look for XGBoost models
    for file in checkpoint_dir.glob("xgboost_fold*.pkl"):
        models['xgboost'].append(file)
    
    # Look for LightGBM models
    for file in checkpoint_dir.glob("lightgbm_fold*.pkl"):
        models['lightgbm'].append(file)
    
    # Look for CatBoost models
    for file in checkpoint_dir.glob("catboost_fold*.pkl"):
        models['catboost'].append(file)
    
    return models

def validate_model_file(model_path: Path, model_type: str) -> Dict:
    """Validate a single model file"""
    validation_result = {
        'file': model_path.name,
        'exists': model_path.exists(),
        'size_mb': 0,
        'loadable': False,
        'error': None
    }
    
    if not model_path.exists():
        validation_result['error'] = "File does not exist"
        return validation_result
    
    # Check file size
    validation_result['size_mb'] = model_path.stat().st_size / (1024 * 1024)
    
    # Try to load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        validation_result['loadable'] = True
        
        # Check model type specific attributes
        if model_type == 'xgboost':
            if hasattr(model, 'predict'):
                validation_result['valid_model'] = True
            else:
                validation_result['error'] = "Not a valid XGBoost model"
        
        elif model_type == 'lightgbm':
            if hasattr(model, 'predict'):
                validation_result['valid_model'] = True
            else:
                validation_result['error'] = "Not a valid LightGBM model"
                
    except Exception as e:
        validation_result['error'] = str(e)
    
    return validation_result

def check_predictions(checkpoint_dir: Path, features_dir: Path) -> Dict:
    """Check if prediction files exist or can be generated"""
    prediction_status = {
        'xgboost_predictions': False,
        'lightgbm_predictions': False,
        'test_features_available': False,
        'can_generate_predictions': False
    }
    
    # Check for existing predictions
    xgb_pred_path = checkpoint_dir / "xgboost_predictions.npy"
    lgb_pred_path = checkpoint_dir / "lightgbm_predictions.npy"
    
    prediction_status['xgboost_predictions'] = xgb_pred_path.exists()
    prediction_status['lightgbm_predictions'] = lgb_pred_path.exists()
    
    # Check for test features
    test_features_path = features_dir / "test_features.parquet"
    prediction_status['test_features_available'] = test_features_path.exists()
    
    # Can generate predictions if we have models and test features
    models = find_model_files(checkpoint_dir)
    has_models = len(models['xgboost']) > 0 or len(models['lightgbm']) > 0
    
    prediction_status['can_generate_predictions'] = (
        has_models and prediction_status['test_features_available']
    )
    
    return prediction_status

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate SMAPE (Symmetric Mean Absolute Percentage Error)"""
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def validate_training_results(output_dir: Path) -> Dict:
    """Validate training results and cross-validation scores"""
    results = {
        'cv_scores_available': False,
        'cv_scores': {},
        'training_logs': False
    }
    
    # Look for CV results
    cv_results_path = output_dir / "cv_results.json"
    if cv_results_path.exists():
        try:
            import json
            with open(cv_results_path, 'r') as f:
                cv_data = json.load(f)
            results['cv_scores_available'] = True
            results['cv_scores'] = cv_data
        except Exception as e:
            results['cv_error'] = str(e)
    
    # Look for training logs
    log_files = list(output_dir.glob("*.log")) + list(output_dir.glob("training_*.txt"))
    results['training_logs'] = len(log_files) > 0
    results['log_files'] = [f.name for f in log_files]
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Validate Amazon ML Challenge 2025 models')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--features', default='dataset/features', help='Features directory')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔍 Amazon ML Challenge 2025 - Model Validation")
    print("=" * 60)
    
    # Setup paths
    output_dir = Path(args.output)
    checkpoint_dir = output_dir / "checkpoints"
    features_dir = Path(args.features)
    
    print(f"📁 Output directory: {output_dir}")
    print(f"📁 Checkpoint directory: {checkpoint_dir}")
    print(f"📁 Features directory: {features_dir}")
    print()
    
    # 1. Check directory structure
    print("📋 Directory Structure:")
    print(f"  Output dir exists: {output_dir.exists()}")
    print(f"  Checkpoint dir exists: {checkpoint_dir.exists()}")
    print(f"  Features dir exists: {features_dir.exists()}")
    print()
    
    # 2. Find and validate model files
    print("🤖 Model Files:")
    models = find_model_files(checkpoint_dir)
    
    total_models = 0
    valid_models = 0
    
    for model_type, model_files in models.items():
        if model_files:
            print(f"\n  {model_type.upper()} Models:")
            for model_file in sorted(model_files):
                result = validate_model_file(model_file, model_type)
                total_models += 1
                
                status = "✅" if result.get('loadable', False) else "❌"
                size = f"{result['size_mb']:.1f}MB" if result['size_mb'] > 0 else "0MB"
                
                print(f"    {status} {result['file']} ({size})")
                
                if result.get('error'):
                    print(f"       Error: {result['error']}")
                else:
                    valid_models += 1
        else:
            print(f"  {model_type.upper()}: No models found")
    
    print(f"\n📊 Model Summary: {valid_models}/{total_models} models valid")
    
    # 3. Check predictions
    print("\n🎯 Predictions:")
    pred_status = check_predictions(checkpoint_dir, features_dir)
    
    for key, status in pred_status.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {key.replace('_', ' ').title()}")
    
    # 4. Check training results
    print("\n📈 Training Results:")
    training_results = validate_training_results(output_dir)
    
    if training_results['cv_scores_available']:
        print("  ✅ Cross-validation scores available:")
        for model, scores in training_results['cv_scores'].items():
            if isinstance(scores, dict) and 'mean' in scores:
                mean_score = scores['mean']
                print(f"    📊 {model}: {mean_score:.2f}% SMAPE")
            elif isinstance(scores, (int, float)):
                print(f"    📊 {model}: {scores:.2f}% SMAPE")
    else:
        print("  ❌ No cross-validation scores found")
    
    if training_results['training_logs']:
        print(f"  ✅ Training logs available: {training_results['log_files']}")
    else:
        print("  ❌ No training logs found")
    
    # 5. Submission readiness
    print("\n🚀 Submission Readiness:")
    
    can_submit = False
    if pred_status['xgboost_predictions'] or pred_status['lightgbm_predictions']:
        print("  ✅ Predictions available - can generate submission")
        can_submit = True
    elif pred_status['can_generate_predictions']:
        print("  ⚠️ Can generate predictions from models")
        can_submit = True
    else:
        print("  ❌ Cannot generate submission - need models or predictions")
    
    # Check submission file
    submission_dir = Path("submissions")
    if submission_dir.exists():
        submissions = list(submission_dir.glob("*.csv"))
        if submissions:
            print(f"  ✅ Found {len(submissions)} existing submission files")
            for sub in submissions:
                try:
                    df = pd.read_csv(sub)
                    print(f"    📄 {sub.name}: {len(df)} rows")
                except:
                    print(f"    ❌ {sub.name}: Invalid format")
        else:
            print("  ❌ No submission files found")
    
    # 6. Performance targets
    print(f"\n🎯 Performance Targets:")
    print(f"  🥇 Top 10: ~30% SMAPE")
    print(f"  🥈 Top 50: ~35% SMAPE") 
    print(f"  🥉 Competitive: ~45% SMAPE")
    print(f"  📊 Baseline: ~80% SMAPE")
    
    if training_results['cv_scores']:
        best_score = min(
            score.get('mean', 100) if isinstance(score, dict) else score 
            for score in training_results['cv_scores'].values()
        )
        
        if best_score < 35:
            print(f"  🎉 Your best score ({best_score:.1f}%) is competitive!")
        elif best_score < 50:
            print(f"  👍 Your best score ({best_score:.1f}%) is decent, room for improvement")
        else:
            print(f"  📈 Your best score ({best_score:.1f}%) needs improvement for top ranking")
    
    # 7. Next steps
    print(f"\n💡 Recommendations:")
    
    if not checkpoint_dir.exists() or total_models == 0:
        print("  1. Run training: python -m src --model enhanced --folds 5")
    elif not pred_status['xgboost_predictions'] and not pred_status['lightgbm_predictions']:
        print("  1. Generate predictions: python create_submission.py")
    elif not can_submit:
        print("  1. Create submission: python create_submission.py")
    else:
        print("  1. ✅ Ready to submit!")
    
    if valid_models > 0 and training_results['cv_scores']:
        best_score = min(
            score.get('mean', 100) if isinstance(score, dict) else score 
            for score in training_results['cv_scores'].values()
        )
        if best_score > 30:
            print("  2. For top 50: Try enhanced/ultra models or feature engineering")
    
    print(f"\n{'='*60}")
    
    if can_submit:
        print("🎉 VALIDATION PASSED - Ready for submission!")
    else:
        print("⚠️ VALIDATION INCOMPLETE - See recommendations above")
    
    return can_submit

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)