# create_submission.py
# Amazon ML Challenge 2025 - Final Submission Generation

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Optional
import warnings
import os
warnings.filterwarnings('ignore')

def load_predictions(checkpoint_dir: Path) -> Dict:
    """Load saved predictions from checkpoint directory"""
    predictions = {}
    
    # Look for saved prediction arrays
    xgb_pred_path = checkpoint_dir / "xgboost_predictions.npy"
    lgb_pred_path = checkpoint_dir / "lightgbm_predictions.npy"
    
    if xgb_pred_path.exists():
        predictions['xgboost'] = np.load(xgb_pred_path)
        print(f"✅ Loaded XGBoost predictions: {predictions['xgboost'].shape}")
    else:
        print(f"⚠️ XGBoost predictions not found at {xgb_pred_path}")
    
    if lgb_pred_path.exists():
        predictions['lightgbm'] = np.load(lgb_pred_path)
        print(f"✅ Loaded LightGBM predictions: {predictions['lightgbm'].shape}")
    else:
        print(f"⚠️ LightGBM predictions not found at {lgb_pred_path}")
    
    return predictions

def load_models_and_predict(checkpoint_dir: Path, features_dir: Path) -> Dict:
    """Load saved models and generate predictions"""
    print("\n🔄 Loading models and generating predictions...")
    
    # Load test features
    test_features_path = features_dir / "test_features.parquet"
    if not test_features_path.exists():
        print(f"❌ Test features not found at {test_features_path}")
        return {}
    
    test_features = pd.read_parquet(test_features_path)
    print(f"✅ Loaded test features: {test_features.shape}")
    
    # Prepare features (exclude sample_id)
    feature_cols = [c for c in test_features.columns if c not in ['sample_id']]
    X_test = test_features[feature_cols].fillna(0)
    
    predictions = {}
    
    # Load and predict with XGBoost models
    print("\n📊 Loading XGBoost models...")
    xgb_preds = []
    for fold in range(1, 6):
        model_path = checkpoint_dir / f"xgboost_fold{fold}.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                pred = model.predict(X_test)
                xgb_preds.append(pred)
                print(f"  ✅ Fold {fold}: {pred.shape}")
            except Exception as e:
                print(f"  ❌ Error loading fold {fold}: {e}")
    
    if xgb_preds:
        # Average across folds and convert from log scale
        xgb_final = np.mean(xgb_preds, axis=0)
        xgb_final = np.expm1(xgb_final)  # Convert from log(price+1) to price
        xgb_final = np.maximum(xgb_final, 0.01)  # Ensure positive
        predictions['xgboost'] = xgb_final
        
        # Save for future use
        np.save(checkpoint_dir / "xgboost_predictions.npy", xgb_final)
        print(f"💾 Saved XGBoost predictions: mean=${xgb_final.mean():.2f}")
    
    # Load and predict with LightGBM models
    print("\n📊 Loading LightGBM models...")
    lgb_preds = []
    for fold in range(1, 6):
        model_path = checkpoint_dir / f"lightgbm_fold{fold}.pkl"
        if model_path.exists():
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                pred = model.predict(X_test, num_iteration=model.best_iteration)
                lgb_preds.append(pred)
                print(f"  ✅ Fold {fold}: {pred.shape}")
            except Exception as e:
                print(f"  ❌ Error loading fold {fold}: {e}")
    
    if lgb_preds:
        # Average across folds and convert from log scale
        lgb_final = np.mean(lgb_preds, axis=0)
        lgb_final = np.expm1(lgb_final)  # Convert from log(price+1) to price
        lgb_final = np.maximum(lgb_final, 0.01)  # Ensure positive
        predictions['lightgbm'] = lgb_final
        
        # Save for future use
        np.save(checkpoint_dir / "lightgbm_predictions.npy", lgb_final)
        print(f"💾 Saved LightGBM predictions: mean=${lgb_final.mean():.2f}")
    
    return predictions

def ensemble_predictions(predictions_dict: Dict, weights: Dict = None) -> np.ndarray:
    """Create weighted ensemble of predictions"""
    
    if weights is None:
        # Use inverse SMAPE as weights (better model gets higher weight)
        weights = {
            'xgboost': 1.0 / 57.53,   # Your XGBoost SMAPE
            'lightgbm': 1.0 / 58.27   # Your LightGBM SMAPE
        }
    
    # Normalize weights for available models
    available_models = [k for k in weights.keys() if k in predictions_dict]
    if not available_models:
        print("❌ No available models for ensemble!")
        return None
    
    # Recalculate weights for available models
    total_weight = sum(weights[k] for k in available_models)
    normalized_weights = {k: weights[k]/total_weight for k in available_models}
    
    print(f"📊 Ensemble weights: {normalized_weights}")
    
    # Create ensemble
    ensemble = np.zeros_like(list(predictions_dict.values())[0])
    for model_name in available_models:
        preds = predictions_dict[model_name]
        weight = normalized_weights[model_name]
        ensemble += preds * weight
        print(f"  Added {model_name} with weight {weight:.3f}")
    
    return ensemble

def post_process_predictions(predictions: np.ndarray, 
                           train_prices: np.ndarray = None) -> np.ndarray:
    """Apply post-processing to improve predictions"""
    
    print("\n🔧 Applying post-processing...")
    
    # 1. Ensure positive values
    predictions = np.maximum(predictions, 0.01)
    print(f"  Ensured positive values (min: ${predictions.min():.2f})")
    
    # 2. Cap extreme values
    upper_limit = 50000  # Reasonable max price for Amazon products
    lower_limit = 0.01   # Minimum price
    
    too_high = (predictions > upper_limit).sum()
    too_low = (predictions < lower_limit).sum()
    
    predictions = np.clip(predictions, lower_limit, upper_limit)
    
    if too_high > 0:
        print(f"  Capped {too_high} high values to ${upper_limit}")
    if too_low > 0:
        print(f"  Capped {too_low} low values to ${lower_limit}")
    
    # 3. Round to 2 decimal places (standard for prices)
    predictions = np.round(predictions, 2)
    
    return predictions

def create_submission():
    """Create submission file from predictions"""
    
    print("=" * 60)
    print("🚀 Amazon ML Challenge 2025 - Creating Final Submission")
    print("=" * 60)
    
    # Setup paths
    root = Path.cwd()
    checkpoint_dir = root / "output" / "checkpoints"
    features_dir = root / "dataset" / "features"
    submission_dir = root / "submissions"
    
    # Create submission directory
    submission_dir.mkdir(exist_ok=True)
    
    print("🔍 Available checkpoint files:")
    for f in checkpoint_dir.glob("*"):
        print(f"  ✅ {f.name}")
    
    # Load test data for sample IDs
    test_path = root / "dataset" / "test.csv"
    if not test_path.exists():
        print(f"❌ Test data not found at {test_path}")
        return None
    
    test_df = pd.read_csv(test_path)
    print(f"✅ Loaded test data: {len(test_df)} samples")
    
    # Load training data for price stats
    train_path = root / "dataset" / "train.csv"
    train_prices = None
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        train_prices = train_df['price'].values
        print(f"✅ Training price stats: mean=${train_prices.mean():.2f}")
    
    # Try to load predictions first
    predictions_dict = load_predictions(checkpoint_dir)
    
    # If no predictions found, try loading models and predicting
    if not predictions_dict:
        print("\n⚠️ No saved predictions found. Loading models...")
        predictions_dict = load_models_and_predict(checkpoint_dir, features_dir)
    
    if not predictions_dict:
        print("❌ No predictions or models found!")
        return None
    
    print(f"\n📊 Available predictions: {list(predictions_dict.keys())}")
    
    # Create ensemble if multiple models
    if len(predictions_dict) > 1:
        print("\n🎯 Creating weighted ensemble...")
        final_predictions = ensemble_predictions(predictions_dict)
    else:
        print("\n📊 Using single model predictions")
        final_predictions = list(predictions_dict.values())[0]
    
    if final_predictions is None:
        print("❌ Failed to create ensemble!")
        return None
    
    # Post-process predictions
    final_predictions = post_process_predictions(final_predictions, train_prices)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': final_predictions
    })
    
    # Save submission with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_path = submission_dir / f"submission_{timestamp}.csv"
    latest_path = submission_dir / "final_submission.csv"
    
    submission.to_csv(submission_path, index=False)
    submission.to_csv(latest_path, index=False)  # Also save as latest
    
    # Print comprehensive statistics
    print("\n" + "=" * 60)
    print("📊 SUBMISSION STATISTICS")
    print("=" * 60)
    print(f"Total samples: {len(submission):,}")
    print(f"Mean price: ${submission['price'].mean():.2f}")
    print(f"Median price: ${submission['price'].median():.2f}")
    print(f"Min price: ${submission['price'].min():.2f}")
    print(f"Max price: ${submission['price'].max():.2f}")
    print(f"Std deviation: ${submission['price'].std():.2f}")
    
    # Validation checks
    print(f"\n✅ VALIDATION CHECKS:")
    print(f"  No NaN values: {submission['price'].isna().sum() == 0}")
    print(f"  All positive: {(submission['price'] > 0).all()}")
    print(f"  Correct sample count: {len(submission) == len(test_df)}")
    print(f"  Valid price range: {submission['price'].min() > 0 and submission['price'].max() < 100000}")
    print(f"  Sample IDs match: {submission['sample_id'].equals(test_df['sample_id'])}")
    
    print("\n" + "=" * 60)
    print(f"✅ Submission saved to:")
    print(f"   📁 {submission_path}")
    print(f"   📁 {latest_path}")
    print("\n🎉 READY TO SUBMIT TO LEADERBOARD!")
    print("=" * 60)
    
    # Show sample predictions
    print(f"\n📋 Sample predictions:")
    sample_display = submission.head(10).copy()
    sample_display['price'] = sample_display['price'].apply(lambda x: f"${x:.2f}")
    print(sample_display.to_string(index=False))
    
    # Expected performance
    print(f"\n🎯 EXPECTED PERFORMANCE:")
    print(f"  Local CV SMAPE: ~57-58%")
    print(f"  Expected Leaderboard: ~55-60%")
    print(f"  Models used: {', '.join(predictions_dict.keys())}")
    
    return str(submission_path)

if __name__ == "__main__":
    submission_path = create_submission()
    if submission_path:
        print("\n🚀 Submission generation completed successfully!")
    else:
        print("\n❌ Submission generation failed!")