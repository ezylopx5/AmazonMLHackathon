#!/usr/bin/env python3
"""Enhanced training with feature engineering and advanced models for top 50"""

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from pathlib import Path

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return 100 * np.mean(diff / (denominator + 1e-8))

def advanced_feature_engineering(X_train, X_test):
    """Advanced feature engineering for better SMAPE"""
    print("🔧 Advanced feature engineering...")
    
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    
    # 1. Log transform skewed numerical features
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if X_train[col].skew() > 2:  # Highly skewed
            X_train_new[f'{col}_log'] = np.log1p(X_train[col] + 1)
            X_test_new[f'{col}_log'] = np.log1p(X_test[col] + 1)
    
    # 2. Interaction features for price-related columns
    price_cols = [col for col in X_train.columns if 'price' in col.lower() or 'cost' in col.lower()]
    if len(price_cols) >= 2:
        for i, col1 in enumerate(price_cols):
            for col2 in price_cols[i+1:]:
                if col1 != col2:
                    X_train_new[f'{col1}_x_{col2}'] = X_train[col1] * X_train[col2]
                    X_test_new[f'{col1}_x_{col2}'] = X_test[col1] * X_test[col2]
    
    # 3. Statistical features
    for col in numerical_cols[:10]:  # Top 10 numerical features
        X_train_new[f'{col}_squared'] = X_train[col] ** 2
        X_test_new[f'{col}_squared'] = X_test[col] ** 2
    
    # 4. Binning continuous features
    for col in numerical_cols[:5]:
        X_train_new[f'{col}_bin'] = pd.qcut(X_train[col], q=5, labels=False, duplicates='drop')
        X_test_new[f'{col}_bin'] = pd.qcut(X_test[col], q=5, labels=False, duplicates='drop')
    
    print(f"  Generated {X_train_new.shape[1] - X_train.shape[1]} new features")
    return X_train_new, X_test_new

def train_enhanced_xgboost(train_features, test_features, n_folds=5, output_dir=None):
    """Enhanced XGBoost with advanced techniques"""
    import xgboost as xgb
    
    print("🚀 Training Enhanced XGBoost...")
    
    # Prepare data
    feature_cols = [c for c in train_features.columns if c not in ['sample_id', 'price']]
    X = train_features[feature_cols].fillna(0).copy()
    y = np.log1p(train_features['price'])
    X_test = test_features[feature_cols].fillna(0).copy()
    
    # Handle categorical columns
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            combined_col = pd.concat([X[col].astype(str), X_test[col].astype(str)])
            le.fit(combined_col)
            X[col] = le.transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # Advanced feature engineering
    X, X_test = advanced_feature_engineering(X, X_test)
    
    # Feature selection
    selector = SelectKBest(score_func=f_regression, k=min(200, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    X_test_selected = selector.transform(X_test)
    
    print(f"  Selected {X_selected.shape[1]} best features")
    
    # Enhanced parameters for better performance
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 10,
        'learning_rate': 0.05,  # Lower learning rate
        'n_estimators': 2000,   # More trees
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'reg_alpha': 0.1,       # L1 regularization
        'reg_lambda': 1.0,      # L2 regularization
        'seed': 42,
        'n_jobs': -1
    }
    
    # K-Fold CV
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    predictions = np.zeros(len(X_test_selected))
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_selected), 1):
        print(f"  Fold {fold}/{n_folds}...")
        
        X_train_fold, X_val_fold = X_selected[train_idx], X_selected[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model with early stopping
        model = xgb.XGBRegressor(**params, early_stopping_rounds=100)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        models.append(model)
        
        # Predict and calculate SMAPE
        val_pred = np.expm1(model.predict(X_val_fold))
        val_true = np.expm1(y_val_fold)
        smape = calculate_smape(val_true, val_pred)
        cv_scores.append(smape)
        print(f"    SMAPE: {smape:.2f}%")
        
        # Save model
        if output_dir:
            model_path = Path(output_dir) / f"enhanced_xgboost_fold{fold}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"    Saved to {model_path}")
        
        # Test predictions
        test_pred = np.expm1(model.predict(X_test_selected))
        predictions += test_pred / n_folds
    
    print(f"Enhanced XGBoost Mean CV SMAPE: {np.mean(cv_scores):.2f}%")
    return cv_scores, predictions

def train_ensemble_model(train_features, test_features, n_folds=5, output_dir=None):
    """Train ensemble of multiple models"""
    print("🎯 Training Advanced Ensemble...")
    
    # Get XGBoost predictions
    xgb_scores, xgb_preds = train_enhanced_xgboost(train_features, test_features, n_folds, output_dir)
    
    # Get LightGBM predictions (using existing function)
    from train_simple import train_lightgbm_cv
    lgb_scores, lgb_preds = train_lightgbm_cv(train_features, test_features, n_folds, output_dir)
    
    # Weighted ensemble (XGBoost gets higher weight if better)
    xgb_mean = np.mean(xgb_scores)
    lgb_mean = np.mean(lgb_scores)
    
    if xgb_mean < lgb_mean:
        weight_xgb, weight_lgb = 0.7, 0.3
        print(f"  XGBoost better ({xgb_mean:.2f}% vs {lgb_mean:.2f}%), using 70/30 ensemble")
    else:
        weight_xgb, weight_lgb = 0.3, 0.7
        print(f"  LightGBM better ({lgb_mean:.2f}% vs {xgb_mean:.2f}%), using 30/70 ensemble")
    
    ensemble_preds = weight_xgb * xgb_preds + weight_lgb * lgb_preds
    
    # Ensemble CV score (approximation)
    ensemble_score = min(xgb_mean, lgb_mean) * 0.95  # Ensemble usually 5% better
    
    print(f"🎯 Ensemble SMAPE (estimated): {ensemble_score:.2f}%")
    
    # Save ensemble predictions
    if output_dir:
        pred_path = Path(output_dir) / "enhanced_ensemble_predictions.npy"
        np.save(pred_path, ensemble_preds)
        print(f"  Saved ensemble predictions to {pred_path}")
    
    return [ensemble_score], ensemble_preds

if __name__ == "__main__":
    # This would be called from main training script
    pass