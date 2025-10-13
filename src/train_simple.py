"""
Simplified training functions that actually work
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path
from tqdm import tqdm

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return 100 * np.mean(diff / (denominator + 1e-8))

def set_seed(seed=42):
    """Set random seeds"""
    import random
    random.seed(seed)
    np.random.seed(seed)

def train_xgboost_cv(train_features, test_features, n_folds=5, output_dir=None, debug=False):
    """Train XGBoost with cross-validation"""
    import xgboost as xgb
    
    print(f"Training XGBoost with {n_folds}-fold CV...")
    
    # Prepare data
    feature_cols = [c for c in train_features.columns 
                   if c not in ['sample_id', 'price']]
    
    X = train_features[feature_cols].fillna(0).copy()
    y = np.log1p(train_features['price'])  # Log transform
    X_test = test_features[feature_cols].fillna(0).copy()
    
    # Handle categorical columns (convert object to numeric)
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"  Converting categorical column: {col}")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            # Fit on combined data to ensure consistency
            combined_col = pd.concat([X[col].astype(str), X_test[col].astype(str)])
            le.fit(combined_col)
            X[col] = le.transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # Parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6 if debug else 8,
        'learning_rate': 0.1,
        'n_estimators': 100 if debug else 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
        'n_jobs': -1
    }
    
    # K-Fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    predictions = np.zeros(len(test_features))
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/{n_folds}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train model
        model = xgb.XGBRegressor(**params, early_stopping_rounds=50 if not debug else 10)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        models.append(model)
        
        # Validate
        val_pred = np.expm1(model.predict(X_val))
        val_true = np.expm1(y_val)
        smape = calculate_smape(val_true, val_pred)
        cv_scores.append(smape)
        print(f"    SMAPE: {smape:.2f}%")
        
        # Test predictions
        predictions += model.predict(X_test) / n_folds
        
        # Save model
        if output_dir:
            model_path = Path(output_dir) / f"xgboost_fold{fold}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"    Saved to {model_path}")
    
    # Convert back from log
    predictions = np.expm1(predictions)
    predictions = np.maximum(predictions, 0.01)
    
    print(f"XGBoost Mean CV SMAPE: {np.mean(cv_scores):.2f}%")
    
    return cv_scores, predictions

def train_lightgbm_cv(train_features, test_features, n_folds=5, output_dir=None, debug=False):
    """Train LightGBM with cross-validation"""
    import lightgbm as lgb
    
    print(f"Training LightGBM with {n_folds}-fold CV...")
    
    # Prepare data
    feature_cols = [c for c in train_features.columns 
                   if c not in ['sample_id', 'price']]
    
    X = train_features[feature_cols].fillna(0).copy()
    y = np.log1p(train_features['price'])
    X_test = test_features[feature_cols].fillna(0).copy()
    
    # Handle categorical columns (convert object to numeric) 
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"  Converting categorical column: {col}")
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            # Fit on combined data to ensure consistency
            combined_col = pd.concat([X[col].astype(str), X_test[col].astype(str)])
            le.fit(combined_col)
            X[col] = le.transform(X[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
    
    # Parameters
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 15 if debug else 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    # K-Fold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    predictions = np.zeros(len(test_features))
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"  Fold {fold}/{n_folds}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
        
        # Train
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100 if debug else 1000,
            callbacks=[
                lgb.early_stopping(50 if not debug else 10),
                lgb.log_evaluation(0)
            ]
        )
        
        models.append(model)
        
        # Validate
        val_pred = np.expm1(model.predict(X_val, num_iteration=model.best_iteration))
        val_true = np.expm1(y_val)
        smape = calculate_smape(val_true, val_pred)
        cv_scores.append(smape)
        print(f"    SMAPE: {smape:.2f}%")
        
        # Test predictions
        predictions += model.predict(X_test, num_iteration=model.best_iteration) / n_folds
        
        # Save model
        if output_dir:
            model_path = Path(output_dir) / f"lightgbm_fold{fold}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"    Saved to {model_path}")
    
    # Convert back
    predictions = np.expm1(predictions)
    predictions = np.maximum(predictions, 0.01)
    
    print(f"LightGBM Mean CV SMAPE: {np.mean(cv_scores):.2f}%")
    
    return cv_scores, predictions