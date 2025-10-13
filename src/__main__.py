#!/usr/bin/env python3
"""
Amazon ML Challenge 2025 - Training Module Entry Point
This file enables running: python -m src
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

# Suppress warnings before any imports
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def check_dependencies():
    """Check if required dependencies are installed"""
    missing = []
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'torch': 'torch',
        'transformers': 'transformers',
        'tqdm': 'tqdm'
    }
    
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies detected!")
        print(f"Please install: {', '.join(missing)}")
        print(f"\nRun: pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    """Main entry point for training"""
    print("="*60)
    print("🚀 Amazon ML Challenge 2025 - Training Pipeline")
    print("="*60)
    
    # Parse arguments FIRST (before heavy imports)
    parser = argparse.ArgumentParser(description='Train models for Amazon ML Challenge')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--features', type=str, default='features',
                       help='Path to features directory')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory')
    parser.add_argument('--folds', type=int, default=5,
                       help='Number of CV folds')
    parser.add_argument('--model', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'both'],
                       help='Model type to train')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (quick training)')
    
    args = parser.parse_args()
    
    print(f"📋 Config: {args.config}")
    print(f"📁 Features: {args.features}")
    print(f"📁 Output: {args.output}")
    print(f"🔢 Folds: {args.folds}")
    print(f"🤖 Model: {args.model}")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies available")
    
    # Now import heavy modules (AFTER argument parsing)
    print("\n📚 Loading modules...")
    try:
        import pandas as pd
        import numpy as np
        from pathlib import Path
        
        # Import our modules
        from src.train_simple import train_xgboost_cv, train_lightgbm_cv
        from src.utils import set_seed, calculate_smape
        
        print("✅ Modules loaded successfully")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTrying fallback import...")
        
        # Fallback to simpler training if complex modules fail
        try:
            from .train_simple import train_xgboost_cv, train_lightgbm_cv
            from .utils import set_seed, calculate_smape
        except:
            print("❌ Cannot import training modules")
            print("Please ensure all files are in src/ directory")
            sys.exit(1)
    
    # Setup paths
    features_dir = Path(args.features)
    output_dir = Path(args.output)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Checkpoints will be saved to: {checkpoint_dir}")
    
    # Load features
    print("\n📊 Loading features...")
    train_features_path = features_dir / "train_features.parquet"
    test_features_path = features_dir / "test_features.parquet"
    
    if not train_features_path.exists():
        print(f"❌ Features not found at {train_features_path}")
        print("Please run: python pipeline.py features")
        sys.exit(1)
    
    train_features = pd.read_parquet(train_features_path)
    test_features = pd.read_parquet(test_features_path)
    
    print(f"✅ Loaded train features: {train_features.shape}")
    print(f"✅ Loaded test features: {test_features.shape}")
    
    # Set seed
    set_seed(42)
    
    # Train models
    print("\n🎯 Starting training...")
    print("This will take some time...")
    
    results = {}
    
    if args.model in ['xgboost', 'both']:
        print("\n📈 Training XGBoost...")
        xgb_scores, xgb_predictions = train_xgboost_cv(
            train_features, 
            test_features,
            n_folds=args.folds,
            output_dir=checkpoint_dir,
            debug=args.debug
        )
        results['xgboost'] = {
            'scores': xgb_scores,
            'predictions': xgb_predictions,
            'mean_cv': np.mean(xgb_scores)
        }
        print(f"✅ XGBoost CV SMAPE: {np.mean(xgb_scores):.2f}%")
    
    if args.model in ['lightgbm', 'both']:
        print("\n📈 Training LightGBM...")
        lgb_scores, lgb_predictions = train_lightgbm_cv(
            train_features,
            test_features,
            n_folds=args.folds,
            output_dir=checkpoint_dir,
            debug=args.debug
        )
        results['lightgbm'] = {
            'scores': lgb_scores,
            'predictions': lgb_predictions,
            'mean_cv': np.mean(lgb_scores)
        }
        print(f"✅ LightGBM CV SMAPE: {np.mean(lgb_scores):.2f}%")
    
    # Save predictions
    print("\n💾 Saving predictions...")
    for model_name, result in results.items():
        pred_path = checkpoint_dir / f"{model_name}_predictions.npy"
        np.save(pred_path, result['predictions'])
        print(f"  Saved {model_name} predictions to {pred_path}")
    
    # Verify checkpoints were created
    checkpoint_files = list(checkpoint_dir.glob("*.pkl")) + list(checkpoint_dir.glob("*.pt"))
    if checkpoint_files:
        print(f"\n✅ Created {len(checkpoint_files)} checkpoint files:")
        for f in checkpoint_files[:5]:  # Show first 5
            print(f"  - {f.name}")
    else:
        print("\n⚠ Warning: No checkpoint files created!")
    
    print("\n" + "="*60)
    print("🎉 Training completed successfully!")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())