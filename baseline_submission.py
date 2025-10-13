# baseline_submission.py
# Create baseline submission without training models

import numpy as np
import pandas as pd
from pathlib import Path

def create_baseline_submission():
    """Create a baseline submission using simple heuristics"""
    
    print("🚀 Creating Baseline Submission")
    print("=" * 40)
    
    # Load test data
    test_df = pd.read_csv("dataset/test.csv")
    print(f"✅ Loaded test data: {len(test_df)} samples")
    
    # Load training data for price distribution
    train_df = pd.read_csv("dataset/train.csv")
    train_prices = train_df['price'].values
    
    print(f"📊 Training price stats:")
    print(f"  Mean: ${train_prices.mean():.2f}")
    print(f"  Median: ${np.median(train_prices):.2f}")
    print(f"  Range: ${train_prices.min():.2f} - ${train_prices.max():.2f}")
    
    # Create baseline predictions
    n_test = len(test_df)
    
    # Strategy: Use training mean with some noise
    baseline_price = train_prices.mean()
    
    # Add some variation based on test data if available
    try:
        # Try to load features to make slightly better predictions
        test_features = pd.read_parquet("dataset/features/test_features.parquet")
        
        if 'ipq' in test_features.columns:
            # Use IPQ (items per query) as a simple predictor
            ipq = test_features['ipq'].fillna(1.0)
            
            # Simple heuristic: higher IPQ might mean lower individual price
            price_predictions = baseline_price * (1.0 / (1.0 + 0.1 * np.log1p(ipq)))
            
            print("✅ Using IPQ-based predictions")
        else:
            # Fallback: use training mean with random variation
            np.random.seed(42)  # For reproducibility
            price_predictions = np.random.normal(
                baseline_price, 
                train_prices.std() * 0.3, 
                n_test
            )
            print("✅ Using random variation around mean")
            
    except Exception as e:
        print(f"⚠️ Could not load features ({e}), using simple baseline")
        
        # Simple baseline: training mean with small random variation
        np.random.seed(42)
        price_predictions = np.random.normal(
            baseline_price, 
            train_prices.std() * 0.2, 
            n_test
        )
    
    # Ensure reasonable price range
    price_predictions = np.clip(price_predictions, 0.01, 1000.0)
    price_predictions = np.round(price_predictions, 2)
    
    print(f"🎯 Prediction stats:")
    print(f"  Mean: ${price_predictions.mean():.2f}")
    print(f"  Range: ${price_predictions.min():.2f} - ${price_predictions.max():.2f}")
    
    # Create submission
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': price_predictions
    })
    
    # Save submission
    Path("submissions").mkdir(exist_ok=True)
    submission.to_csv("submissions/baseline_submission.csv", index=False)
    submission.to_csv("submissions/final_submission.csv", index=False)
    
    # Validation
    print(f"\n✅ Validation checks:")
    print(f"  Total samples: {len(submission)} (expected: 75000)")
    print(f"  No NaN values: {submission['price'].notna().all()}")
    print(f"  All positive: {(submission['price'] > 0).all()}")
    print(f"  Sample IDs match: {submission['sample_id'].equals(test_df['sample_id'])}")
    
    print(f"\n📁 Submission saved to:")
    print(f"  📄 submissions/baseline_submission.csv")
    print(f"  📄 submissions/final_submission.csv")
    
    print(f"\n📋 Sample predictions:")
    print(submission.head(10).to_string(index=False))
    
    print(f"\n🎯 Expected SMAPE: ~80-90% (baseline)")
    print("💡 To improve: Install xgboost/lightgbm and run proper training")
    
    return "submissions/baseline_submission.csv"

if __name__ == "__main__":
    create_baseline_submission()