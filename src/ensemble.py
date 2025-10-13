"""
Advanced ensemble strategies with model stacking
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from scipy import stats

logger = logging.getLogger(__name__)

class AdvancedEnsemble:
    """Advanced ensemble with multiple strategies and stacking"""
    
    def __init__(self, config, methods: List[str] = None):
        self.config = config
        self.methods = methods or ['weighted_average', 'median', 'stacking']
        self.stacking_model = None
        self.fold_weights = None
        self.confidence_scores = None
        
    def fit_stacking_model(self, base_predictions: np.ndarray, targets: np.ndarray, 
                          features: pd.DataFrame = None) -> Dict:
        """Train stacking model using base predictions"""
        
        try:
            from sklearn.linear_model import Ridge
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import cross_val_score
            
            logger.info("Training stacking model...")
            
            # Prepare stacking features
            stacking_features = self._create_stacking_features(base_predictions, features)
            
            # Try different stacking models
            stacking_models = {
                'ridge': Ridge(alpha=1.0),
                'gbm': GradientBoostingRegressor(
                    n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
                )
            }
            
            best_model = None
            best_score = float('inf')
            results = {}
            
            for name, model in stacking_models.items():
                # Cross-validate
                cv_scores = cross_val_score(
                    model, stacking_features, targets, 
                    cv=3, scoring='neg_mean_squared_error'
                )
                mean_score = -cv_scores.mean()
                
                results[name] = {
                    'cv_score': mean_score,
                    'cv_std': cv_scores.std(),
                    'model': model
                }
                
                logger.info(f"Stacking {name}: {mean_score:.4f} ± {cv_scores.std():.4f}")
                
                if mean_score < best_score:
                    best_score = mean_score
                    best_model = model
            
            # Train best model on full data
            self.stacking_model = best_model
            self.stacking_model.fit(stacking_features, targets)
            
            logger.info(f"Best stacking model selected with score: {best_score:.4f}")
            
            return results
            
        except ImportError:
            logger.warning("Scikit-learn not available, using weighted average only")
            return {}
        except Exception as e:
            logger.error(f"Failed to train stacking model: {e}")
            return {}
    
    def _create_stacking_features(self, predictions: np.ndarray, features: pd.DataFrame = None) -> np.ndarray:
        """Create enhanced features for stacking model"""
        
        stacking_features = []
        
        # Base predictions (each fold)
        if predictions.ndim == 2:
            # Multiple fold predictions
            stacking_features.append(predictions)  # [n_samples, n_folds]
            
            # Statistical features from predictions
            stacking_features.append(np.mean(predictions, axis=1, keepdims=True))
            stacking_features.append(np.std(predictions, axis=1, keepdims=True))
            stacking_features.append(np.median(predictions, axis=1, keepdims=True))
            stacking_features.append(np.min(predictions, axis=1, keepdims=True))
            stacking_features.append(np.max(predictions, axis=1, keepdims=True))
            
            # Prediction confidence (inverse of std)
            pred_std = np.std(predictions, axis=1, keepdims=True)
            confidence = 1.0 / (pred_std + 1e-6)
            stacking_features.append(confidence)
            
        else:
            # Single prediction
            stacking_features.append(predictions.reshape(-1, 1))
        
        # Feature interactions if available
        if features is not None:
            key_features = ['ipq', 'has_premium_brand', 'word_count', 'spec_density']
            
            for feat_name in key_features:
                if feat_name in features.columns:
                    feat_values = features[feat_name].fillna(0).values.reshape(-1, 1)
                    stacking_features.append(feat_values)
                    
                    # Feature-prediction interactions
                    if predictions.ndim == 2:
                        mean_pred = np.mean(predictions, axis=1, keepdims=True)
                        interaction = feat_values * mean_pred
                        stacking_features.append(interaction)
            
            # Category features (top categories only)
            category_cols = [col for col in features.columns if col.startswith('cat_')][:5]
            for cat_col in category_cols:
                if cat_col in features.columns:
                    cat_values = features[cat_col].values.reshape(-1, 1)
                    stacking_features.append(cat_values)
        
        return np.hstack(stacking_features)
    
    def predict(self, base_predictions: np.ndarray, features: pd.DataFrame = None, 
                method: str = 'auto') -> Tuple[np.ndarray, Dict]:
        """Generate ensemble predictions using specified method"""
        
        if method == 'auto':
            method = 'stacking' if self.stacking_model is not None else 'weighted_average'
        
        results = {}
        
        if method == 'stacking' and self.stacking_model is not None:
            stacking_features = self._create_stacking_features(base_predictions, features)
            predictions = self.stacking_model.predict(stacking_features)
            results['method'] = 'stacking'
            
        elif method == 'weighted_average':
            predictions = self._weighted_average(base_predictions)
            results['method'] = 'weighted_average'
            results['weights'] = self.fold_weights
            
        elif method == 'median':
            predictions = self._median_ensemble(base_predictions)
            results['method'] = 'median'
            
        elif method == 'rank_average':
            predictions = self._rank_average(base_predictions)
            results['method'] = 'rank_average'
            
        else:
            # Fallback to simple average
            predictions = np.mean(base_predictions, axis=1)
            results['method'] = 'simple_average'
        
        # Calculate prediction confidence
        if base_predictions.ndim == 2:
            pred_std = np.std(base_predictions, axis=1)
            self.confidence_scores = 1.0 / (pred_std + 1e-6)
            self.confidence_scores = self.confidence_scores / np.max(self.confidence_scores)  # Normalize
            results['confidence_scores'] = self.confidence_scores
        
        return predictions, results
    
    def _weighted_average(self, predictions: np.ndarray) -> np.ndarray:
        """Weighted average based on fold performance"""
        
        if self.fold_weights is None:
            # Equal weights if no performance info
            return np.mean(predictions, axis=1)
        
        weights = np.array(self.fold_weights)
        weights = weights / weights.sum()  # Normalize
        
        return np.average(predictions, axis=1, weights=weights)
    
    def _median_ensemble(self, predictions: np.ndarray) -> np.ndarray:
        """Robust median ensemble"""
        return np.median(predictions, axis=1)
    
    def _rank_average(self, predictions: np.ndarray) -> np.ndarray:
        """Rank-based ensemble (good for different scales)"""
        
        # Convert predictions to ranks
        ranks = np.zeros_like(predictions)
        for i in range(predictions.shape[1]):
            ranks[:, i] = stats.rankdata(predictions[:, i])
        
        # Average ranks
        avg_ranks = np.mean(ranks, axis=1)
        
        # Convert back to prediction scale using overall distribution
        all_preds = predictions.flatten()
        sorted_preds = np.sort(all_preds)
        
        # Map ranks back to values
        rank_to_value = np.interp(
            avg_ranks, 
            np.arange(1, len(sorted_preds) + 1), 
            sorted_preds
        )
        
        return rank_to_value
    
    def set_fold_weights(self, fold_scores: List[float], method: str = 'inverse_error'):
        """Set weights for fold predictions based on validation scores"""
        
        scores = np.array(fold_scores)
        
        if method == 'inverse_error':
            # Weight inversely proportional to error (lower error = higher weight)
            weights = 1.0 / (scores + 1e-6)
        elif method == 'softmax':
            # Softmax of negative errors (lower error = higher weight)
            weights = np.exp(-scores / np.mean(scores))
        elif method == 'rank':
            # Rank-based weights
            ranks = stats.rankdata(-scores)  # Negative for descending order
            weights = ranks / ranks.sum()
        else:
            # Equal weights
            weights = np.ones(len(scores))
        
        # Normalize weights
        self.fold_weights = weights / weights.sum()
        
        logger.info(f"Fold weights ({method}): {self.fold_weights}")
    
    def create_diverse_ensemble(self, base_predictions: np.ndarray, 
                               features: pd.DataFrame = None) -> Dict[str, np.ndarray]:
        """Create multiple ensemble predictions for diversity"""
        
        ensemble_predictions = {}
        
        # Different ensemble methods
        methods = ['weighted_average', 'median', 'rank_average']
        if self.stacking_model is not None:
            methods.append('stacking')
        
        for method in methods:
            preds, _ = self.predict(base_predictions, features, method)
            ensemble_predictions[method] = preds
        
        # Meta-ensemble (ensemble of ensembles)
        if len(ensemble_predictions) > 1:
            meta_predictions = np.column_stack(list(ensemble_predictions.values()))
            meta_ensemble = np.mean(meta_predictions, axis=1)
            ensemble_predictions['meta_ensemble'] = meta_ensemble
        
        return ensemble_predictions
    
    def validate_ensemble(self, base_predictions: np.ndarray, targets: np.ndarray,
                         features: pd.DataFrame = None) -> Dict:
        """Validate different ensemble methods"""
        
        validation_results = {}
        
        methods = ['weighted_average', 'median', 'rank_average']
        if self.stacking_model is not None:
            methods.append('stacking')
        
        for method in methods:
            try:
                preds, _ = self.predict(base_predictions, features, method)
                
                # Calculate metrics
                mse = mean_squared_error(targets, preds)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(targets - preds))
                
                # SMAPE
                smape = np.mean(2 * np.abs(targets - preds) / (np.abs(targets) + np.abs(preds))) * 100
                
                validation_results[method] = {
                    'mse': mse,
                    'rmse': rmse, 
                    'mae': mae,
                    'smape': smape
                }
                
                logger.info(f"Ensemble {method}: SMAPE={smape:.2f}%, RMSE={rmse:.2f}")
                
            except Exception as e:
                logger.warning(f"Failed to validate {method}: {e}")
        
        return validation_results