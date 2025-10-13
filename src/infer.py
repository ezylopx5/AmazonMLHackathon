import os
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import logging
from typing import List, Dict, Optional, Tuple
from scipy import stats
import pickle

from .models import create_model
from .train import PriceDataset

logger = logging.getLogger(__name__)

class PostProcessor:
    """Advanced post-processing for predictions"""
    
    def __init__(self, train_prices: np.ndarray):
        self.train_prices = train_prices
        self.train_mean = np.mean(train_prices)
        self.train_std = np.std(train_prices)
        self.train_median = np.median(train_prices)
        
        # Calculate bounds
        q1, q3 = np.percentile(train_prices, [25, 75])
        iqr = q3 - q1
        self.lower_bound = max(0.01, q1 - 1.5 * iqr)
        self.upper_bound = q3 + 1.5 * iqr
        
        # Log statistics
        self.train_log_mean = np.mean(np.log1p(train_prices))
        self.train_log_std = np.std(np.log1p(train_prices))
        
        logger.info(f"Train price statistics:")
        logger.info(f"  Mean: ${self.train_mean:.2f}")
        logger.info(f"  Median: ${self.train_median:.2f}")
        logger.info(f"  Std: ${self.train_std:.2f}")
        logger.info(f"  Bounds: [${self.lower_bound:.2f}, ${self.upper_bound:.2f}]")
    
    def process(self, predictions: np.ndarray, 
                features_df: pd.DataFrame) -> np.ndarray:
        """Apply all post-processing steps"""
        
        processed = predictions.copy()
        
        # 1. Basic clipping
        processed = np.clip(processed, self.lower_bound * 0.5, self.upper_bound * 2)
        
        # 2. Distribution calibration
        processed = self.calibrate_distribution(processed)
        
        # 3. IPQ-based adjustments
        if 'ipq' in features_df.columns:
            processed = self.apply_ipq_rules(processed, features_df)
        
        # 4. Category-specific calibration
        processed = self.calibrate_by_category(processed, features_df)
        
        # 5. Outlier smoothing
        processed = self.smooth_outliers(processed)
        
        # 6. Final sanity checks
        processed = np.maximum(processed, 0.01)
        processed = np.minimum(processed, self.upper_bound * 3)
        
        return processed
    
    def calibrate_distribution(self, predictions: np.ndarray, 
                              strength: float = 0.3) -> np.ndarray:
        """Adjust prediction distribution to match training"""
        
        pred_mean = np.mean(predictions)
        pred_std = np.std(predictions)
        
        if pred_std > 0:
            # Z-score normalization
            z_scores = (predictions - pred_mean) / pred_std
            
            # Rescale to training distribution
            calibrated = z_scores * self.train_std + self.train_mean
            
            # Blend with original
            predictions = strength * calibrated + (1 - strength) * predictions
        
        return predictions
    
    def apply_ipq_rules(self, predictions: np.ndarray, 
                        features_df: pd.DataFrame) -> np.ndarray:
        """Apply IPQ-based logical rules"""
        
        ipq_values = features_df['ipq'].values
        
        # Get reference prices for single items
        single_mask = (ipq_values == 1)
        if single_mask.any():
            single_prices = predictions[single_mask]
            single_median = np.median(single_prices)
            single_mean = np.mean(single_prices)
            
            # Adjust multi-pack items
            for ipq in range(2, min(100, int(ipq_values.max()) + 1)):
                mask = (ipq_values == ipq)
                if not mask.any():
                    continue
                
                # Expected multiplier (sublinear scaling)
                multiplier = 1 + np.log1p(ipq) * 0.7
                
                # Expected price range
                expected_min = single_median * multiplier * 0.8
                expected_max = single_mean * multiplier * 1.5
                
                # Adjust predictions that are too low
                too_low = predictions[mask] < expected_min
                if too_low.any():
                    current = predictions[mask][too_low]
                    adjusted = 0.7 * expected_min + 0.3 * current
                    predictions[mask] = np.where(too_low, adjusted, predictions[mask])
        
        return predictions
    
    def calibrate_by_category(self, predictions: np.ndarray, 
                              features_df: pd.DataFrame) -> np.ndarray:
        """Apply category-specific calibrations"""
        
        # Define expected price ranges by category
        category_ranges = {
            'cat_electronics': (20, 5000),
            'cat_clothing': (5, 500),
            'cat_food': (1, 200),
            'cat_home': (10, 2000),
            'cat_sports': (10, 1000),
            'cat_beauty': (5, 300),
            'cat_toys': (5, 500),
            'cat_books': (5, 100)
        }
        
        for cat_col, (min_price, max_price) in category_ranges.items():
            if cat_col in features_df.columns:
                mask = features_df[cat_col] == 1
                if mask.any():
                    # Soft clipping for category
                    cat_preds = predictions[mask]
                    
                    # Adjust extreme values
                    too_low = cat_preds < min_price * 0.5
                    too_high = cat_preds > max_price * 2
                    
                    if too_low.any():
                        cat_preds[too_low] = min_price * 0.7
                    
                    if too_high.any():
                        cat_preds[too_high] = max_price * 1.5
                    
                    predictions[mask] = cat_preds
        
        return predictions
    
    def smooth_outliers(self, predictions: np.ndarray, 
                       threshold: float = 3.0) -> np.ndarray:
        """Smooth extreme outliers"""
        
        # Log transform for outlier detection
        log_preds = np.log1p(predictions)
        log_mean = np.mean(log_preds)
        log_std = np.std(log_preds)
        
        # Z-scores in log space
        z_scores = np.abs((log_preds - log_mean) / log_std)
        
        # Identify outliers
        outlier_mask = z_scores > threshold
        
        if outlier_mask.any():
            # Smooth outliers towards median
            median_price = np.median(predictions[~outlier_mask])
            predictions[outlier_mask] = 0.3 * median_price + 0.7 * predictions[outlier_mask]
        
        return predictions

class MLEnhancedPostProcessor(PostProcessor):
    """Machine Learning enhanced post-processing"""
    
    def __init__(self, train_prices: np.ndarray, train_features: pd.DataFrame = None):
        super().__init__(train_prices)
        self.train_features = train_features
        self.correction_model = None
        self.category_medians = {}
        self._initialize_corrections()
    
    def _initialize_corrections(self):
        """Initialize correction models and statistics"""
        if self.train_features is not None:
            # Calculate category medians for confidence-based smoothing
            category_cols = [col for col in self.train_features.columns if col.startswith('cat_')]
            for cat_col in category_cols:
                if cat_col in self.train_features.columns:
                    mask = self.train_features[cat_col] == 1
                    if mask.any() and 'price' in self.train_features.columns:
                        self.category_medians[cat_col] = np.median(self.train_features.loc[mask, 'price'])
    
    def train_correction_model(self, val_predictions: np.ndarray, val_targets: np.ndarray, 
                               val_features: pd.DataFrame):
        """Train a model to predict correction factors"""
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.model_selection import cross_val_score
            
            # Calculate residuals
            residuals = val_targets - val_predictions
            relative_errors = residuals / np.maximum(val_targets, 1e-6)
            
            # Prepare features for correction model
            correction_features = self._prepare_correction_features(val_predictions, val_features)
            
            # Train correction model
            self.correction_model = GradientBoostingRegressor(
                n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42
            )
            
            # Cross-validate
            scores = cross_val_score(self.correction_model, correction_features, 
                                   relative_errors, cv=3, scoring='neg_mean_squared_error')
            logger.info(f"Correction model CV scores: {-scores.mean():.4f} ± {scores.std():.4f}")
            
            # Train final model
            self.correction_model.fit(correction_features, relative_errors)
            
        except ImportError:
            logger.warning("Scikit-learn not available, using heuristic corrections only")
        except Exception as e:
            logger.warning(f"Failed to train correction model: {e}")
    
    def _prepare_correction_features(self, predictions: np.ndarray, features_df: pd.DataFrame):
        """Prepare features for correction model"""
        correction_features = []
        
        # Prediction-based features
        correction_features.extend([
            predictions,
            np.log1p(predictions),
            (predictions - self.train_mean) / self.train_std
        ])
        
        # Key feature interactions
        if 'ipq' in features_df.columns:
            ipq_values = features_df['ipq'].fillna(1).values
            correction_features.append(ipq_values)
            correction_features.append(predictions / np.maximum(ipq_values, 1))
        
        # Category features
        category_cols = [col for col in features_df.columns if col.startswith('cat_')]
        for cat_col in category_cols[:5]:  # Use top 5 categories
            if cat_col in features_df.columns:
                correction_features.append(features_df[cat_col].values)
        
        # Brand features
        if 'has_premium_brand' in features_df.columns:
            correction_features.append(features_df['has_premium_brand'].values)
        
        # Text features
        if 'word_count' in features_df.columns:
            correction_features.append(features_df['word_count'].fillna(0).values)
        
        return np.column_stack(correction_features)
    
    def process_enhanced(self, predictions: np.ndarray, features_df: pd.DataFrame, 
                        confidence_scores: Optional[np.ndarray] = None) -> np.ndarray:
        """Enhanced processing with ML corrections and confidence-based smoothing"""
        
        # Basic processing from parent
        processed = super().process(predictions, features_df)
        
        # ML-based correction if model is trained
        if self.correction_model is not None:
            try:
                # Prepare features for correction model
                correction_features = self._prepare_correction_features(processed, features_df)
                
                # Predict corrections
                corrections = self.correction_model.predict(correction_features)
                
                # Apply small corrections (limit impact to avoid overcorrection)
                corrections = np.clip(corrections, -0.2, 0.2)  # Limit to ±20%
                processed = processed * (1 + corrections)
                
            except Exception as e:
                logger.warning(f"ML correction failed: {e}")
        
        # Confidence-based smoothing
        if confidence_scores is not None:
            processed = self._smooth_low_confidence(processed, features_df, confidence_scores)
        
        # Category-aware price validation
        processed = self._category_aware_validation(processed, features_df)
        
        return processed
    
    def _smooth_low_confidence(self, predictions: np.ndarray, features_df: pd.DataFrame, 
                              confidence_scores: np.ndarray, threshold: float = 0.7) -> np.ndarray:
        """Smooth low-confidence predictions towards category medians"""
        
        low_confidence = confidence_scores < threshold
        if not low_confidence.any():
            return predictions
        
        smoothed = predictions.copy()
        
        # Get category information for low-confidence predictions
        category_cols = [col for col in features_df.columns if col.startswith('cat_')]
        
        for cat_col in category_cols:
            if cat_col in self.category_medians:
                cat_mask = (features_df[cat_col] == 1) & low_confidence
                if cat_mask.any():
                    # Blend with category median (stronger for lower confidence)
                    blend_strength = (threshold - confidence_scores[cat_mask]) / threshold
                    blend_strength = np.clip(blend_strength, 0, 0.5)  # Max 50% blend
                    
                    category_median = self.category_medians[cat_col]
                    smoothed[cat_mask] = (
                        (1 - blend_strength) * smoothed[cat_mask] + 
                        blend_strength * category_median
                    )
        
        return smoothed
    
    def _category_aware_validation(self, predictions: np.ndarray, 
                                  features_df: pd.DataFrame) -> np.ndarray:
        """Apply category-aware price validation with learned bounds"""
        
        # Enhanced category ranges based on analysis
        category_ranges = {
            'cat_electronics': (10, 3000),
            'cat_clothing': (3, 300),
            'cat_food': (0.5, 100),
            'cat_home': (5, 1500),
            'cat_sports': (8, 800),
            'cat_beauty': (3, 200),
            'cat_toys': (2, 150),
            'cat_books': (3, 80),
            'cat_automotive': (10, 2000),
            'cat_garden': (5, 500)
        }
        
        validated = predictions.copy()
        
        for cat_col, (min_price, max_price) in category_ranges.items():
            if cat_col in features_df.columns:
                mask = features_df[cat_col] == 1
                if mask.any():
                    cat_preds = validated[mask]
                    
                    # Soft clipping with exponential decay for extreme values
                    too_low = cat_preds < min_price
                    too_high = cat_preds > max_price
                    
                    if too_low.any():
                        # Exponential approach to minimum
                        ratio = cat_preds[too_low] / min_price
                        adjusted = min_price * (1 - np.exp(-ratio * 2))
                        validated[mask] = np.where(too_low, adjusted, validated[mask])
                    
                    if too_high.any():
                        # Logarithmic approach to maximum  
                        ratio = cat_preds[too_high] / max_price
                        adjusted = max_price * (1 + np.log1p(ratio - 1) * 0.5)
                        validated[mask] = np.where(too_high, adjusted, validated[mask])
        
        return validated

class EnsemblePredictor:
    """Ensemble predictions from multiple models"""
    
    def __init__(self, config, checkpoint_dir: Optional[str] = None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir or config.ckpt_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoints info
        self.checkpoints = self._find_checkpoints()
        logger.info(f"Found {len(self.checkpoints)} checkpoints for ensemble")
        
    def _find_checkpoints(self) -> List[str]:
        """Find all best checkpoints"""
        pattern = os.path.join(self.checkpoint_dir, "fold*_best.pt")
        checkpoints = glob.glob(pattern)
        checkpoints.sort()
        return checkpoints
    
    def predict(self, test_df: pd.DataFrame, 
                test_features: pd.DataFrame,
                image_dir: str) -> np.ndarray:
        """Generate ensemble predictions"""
        
        all_predictions = []
        fold_weights = []
        
        for checkpoint_path in self.checkpoints:
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            fold = checkpoint['fold']
            val_smape = checkpoint['val_smape']
            
            # Create model
            feature_cols = [col for col in test_features.columns 
                           if col not in ['sample_id', 'price']]
            num_features = len(feature_cols)
            
            model, text_encoder, image_encoder = create_model(self.config, num_features)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            
            # Create dataset and dataloader
            test_dataset = PriceDataset(
                test_df, test_features, image_dir,
                is_train=False, augment=False
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.config.batch_size * 2,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
            
            # Generate predictions
            predictions = self._predict_single_model(
                model, text_encoder, image_encoder, test_loader
            )
            
            all_predictions.append(predictions)
            
            # Weight based on validation performance
            if self.config.ensemble_weights == "inverse_smape":
                weight = 1.0 / (val_smape + 1e-6)
            elif self.config.ensemble_weights == "uniform":
                weight = 1.0
            else:
                weight = 1.0
            
            fold_weights.append(weight)
            
            logger.info(f"Fold {fold}: SMAPE={val_smape:.2f}%, Weight={weight:.4f}")
            
            # Clean up
            del model, text_encoder, image_encoder
            torch.cuda.empty_cache()
        
        # Ensemble predictions
        fold_weights = np.array(fold_weights)
        fold_weights = fold_weights / fold_weights.sum()
        
        ensemble_preds = np.zeros_like(all_predictions[0])
        for i, preds in enumerate(all_predictions):
            ensemble_preds += fold_weights[i] * preds
        
        logger.info(f"Ensemble weights: {fold_weights}")
        
        return ensemble_preds
    
    def _predict_single_model(self, model, text_encoder, image_encoder, dataloader):
        """Predict with a single model"""
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                # Encode
                if text_encoder:
                    text_embeddings = text_encoder.encode(batch['text'])
                else:
                    text_embeddings = torch.zeros(len(batch['text']), 4096).to(self.device)
                
                if image_encoder:
                    image_embeddings = image_encoder.encode(batch['image'])
                else:
                    image_embeddings = torch.zeros(len(batch['image']), 1024).to(self.device)
                
                features = batch['features'].to(self.device)
                
                # Predict
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        if text_encoder and image_encoder:
                            batch_preds = model(text_embeddings, image_embeddings, features)
                        else:
                            batch_preds = model(features)
                else:
                    if text_encoder and image_encoder:
                        batch_preds = model(text_embeddings, image_embeddings, features)
                    else:
                        batch_preds = model(features)
                
                predictions.extend(batch_preds.cpu().numpy())
        
        return np.array(predictions)
    
    def load_fold_models(self, num_folds: int = 5):
        """Load all fold models from checkpoints."""
        logger.info("🔄 Loading fold models...")
        
        self.models = []
        self.fold_weights = []
        
        for fold in range(num_folds):
            # Find best checkpoint for this fold
            fold_pattern = os.path.join(self.checkpoint_dir, f"fold{fold}_best.pt")
            
            if not os.path.exists(fold_pattern):
                logger.warning(f"⚠ No checkpoint found for fold {fold}")
                continue
            
            # Load checkpoint
            checkpoint = torch.load(fold_pattern, map_location=self.device)
            val_smape = checkpoint.get('val_smape', 100.0)
            
            logger.info(f"✅ Fold {fold}: Loading checkpoint (SMAPE: {val_smape:.2f}%)")
            
            # Extract model info
            fold_info = {
                'fold': fold,
                'val_smape': val_smape,
                'checkpoint_path': fold_pattern
            }
            
            self.fold_weights.append(1.0 / max(val_smape, 1e-6))  # Weight inversely proportional to SMAPE
        
        # Normalize weights
        if self.fold_weights:
            self.fold_weights = np.array(self.fold_weights)
            self.fold_weights = self.fold_weights / self.fold_weights.sum()
            logger.info(f"📊 Loaded {len(self.fold_weights)} models with weights: {self.fold_weights}")
        else:
            raise ValueError("No models could be loaded!")
    
    def predict_with_ensemble(self, test_df: pd.DataFrame, test_features: pd.DataFrame,
                             image_dir: str, use_stacking: bool = True) -> Tuple[np.ndarray, Dict]:
        """Generate ensemble predictions with multiple strategies"""
        
        # Generate base predictions from all folds
        all_predictions = []
        
        for checkpoint_path in self.checkpoints:
            fold_predictions = self._predict_single_fold(
                checkpoint_path, test_df, test_features, image_dir
            )
            all_predictions.append(fold_predictions)
        
        all_predictions = np.column_stack(all_predictions)  # [n_samples, n_folds]
        
        # Apply advanced ensemble strategy
        from .ensemble import AdvancedEnsemble
        
        ensemble_strategy = AdvancedEnsemble(self.config)
        ensemble_strategy.set_fold_weights(
            [self._get_fold_score(cp) for cp in self.checkpoints], 
            method=self.config.ensemble_weights
        )
        
        # Generate ensemble predictions
        if use_stacking and hasattr(self.config, 'use_stacking') and self.config.use_stacking:
            predictions, ensemble_info = ensemble_strategy.predict(
                all_predictions, test_features, method='stacking'
            )
        else:
            predictions, ensemble_info = ensemble_strategy.predict(
                all_predictions, test_features, method='weighted_average'
            )
        
        return predictions, ensemble_info
    
    def _predict_single_fold(self, checkpoint_path: str, test_df: pd.DataFrame,
                            test_features: pd.DataFrame, image_dir: str) -> np.ndarray:
        """Generate predictions for a single fold"""
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        feature_cols = [col for col in test_features.columns 
                       if col not in ['sample_id', 'price']]
        num_features = len(feature_cols)
        
        model, text_encoder, image_encoder = create_model(self.config, num_features)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Create dataset and dataloader
        test_dataset = PriceDataset(
            test_df, test_features, image_dir,
            is_train=False, augment=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Generate predictions
        predictions = self._predict_single_model(
            model, text_encoder, image_encoder, test_loader
        )
        
        # Cleanup
        del model, text_encoder, image_encoder
        torch.cuda.empty_cache()
        
        return predictions
    
    def _get_fold_score(self, checkpoint_path: str) -> float:
        """Extract validation score from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint.get('val_smape', 100.0)
        except:
            return 100.0

def generate_predictions(config):
    """Generate final predictions for test set"""
    
    # Load data
    logger.info("Loading test data...")
    test_df = pd.read_csv(config.test_csv)
    from .features import load_features
    test_features = load_features(config.feats_dir, 'test')
    
    # Load training prices for post-processing
    train_df = pd.read_csv(config.train_csv)
    train_prices = train_df['price'].values
    
    # Create ensemble predictor
    ensemble = EnsemblePredictor(config)
    
    # Generate predictions
    logger.info("Generating ensemble predictions...")
    predictions = ensemble.predict(test_df, test_features, config.img_test)
    
    # Enhanced post-processing
    if config.post_process:
        logger.info("Applying enhanced post-processing...")
        
        # Use ML-enhanced post-processor if enabled
        if getattr(config, 'use_ml_postprocessing', False):
            postprocessor = MLEnhancedPostProcessor(train_prices, train_df)
            
            # Try to load validation data for training correction model
            try:
                val_results_path = os.path.join(config.output_dir, 'validation_predictions.pkl')
                if os.path.exists(val_results_path):
                    with open(val_results_path, 'rb') as f:
                        val_data = pickle.load(f)
                    postprocessor.train_correction_model(
                        val_data['predictions'], val_data['targets'], val_data['features']
                    )
            except Exception as e:
                logger.warning(f"Could not load validation data for correction model: {e}")
            
            # Get confidence scores if available
            confidence_scores = getattr(ensemble, 'confidence_scores', None)
            predictions = postprocessor.process_enhanced(predictions, test_features, confidence_scores)
        else:
            postprocessor = PostProcessor(train_prices)
            predictions = postprocessor.process(predictions, test_features)
    
    # Save predictions
    os.makedirs(config.submission_dir, exist_ok=True)
    
    submission_df = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    submission_path = os.path.join(config.submission_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    
    logger.info(f"Saved predictions to {submission_path}")
    logger.info(f"Prediction statistics:")
    logger.info(f"  Mean: ${np.mean(predictions):.2f}")
    logger.info(f"  Median: ${np.median(predictions):.2f}")
    logger.info(f"  Std: ${np.std(predictions):.2f}")
    logger.info(f"  Min: ${np.min(predictions):.2f}")
    logger.info(f"  Max: ${np.max(predictions):.2f}")
    
    return submission_df

def validate_submission(submission_path: str, expected_rows: int = None) -> bool:
    """Validate submission file before uploading"""
    
    logger.info("🔍 Validating submission...")
    
    if not os.path.exists(submission_path):
        logger.error("❌ Submission file not found.")
        return False
    
    try:
        df = pd.read_csv(submission_path)
    except Exception as e:
        logger.error(f"❌ Error reading CSV: {e}")
        return False
    
    checks = []
    
    # Check columns
    if set(df.columns) == {'sample_id', 'price'}:
        checks.append(("Has correct columns", True))
    else:
        checks.append(("Has correct columns", False))
        logger.error(f"   Expected: ['sample_id', 'price'], Got: {list(df.columns)}")
    
    # Check row count
    if expected_rows is None or len(df) == expected_rows:
        checks.append(("Has correct row count", True))
    else:
        checks.append(("Has correct row count", False))
        logger.error(f"   Expected: {expected_rows}, Got: {len(df)}")
    
    # Check for missing values
    if df.isna().sum().sum() == 0:
        checks.append(("No missing values", True))
    else:
        checks.append(("No missing values", False))
        logger.error(f"   Missing values found: {df.isna().sum().to_dict()}")
    
    # Check all prices are positive
    if 'price' in df.columns and (df['price'] > 0).all():
        checks.append(("All prices positive", True))
    else:
        checks.append(("All prices positive", False))
        if 'price' in df.columns:
            neg_count = (df['price'] <= 0).sum()
            logger.error(f"   Found {neg_count} non-positive prices")
    
    # Check for duplicate sample_ids
    if 'sample_id' in df.columns and df['sample_id'].nunique() == len(df):
        checks.append(("No duplicate sample_ids", True))
    else:
        checks.append(("No duplicate sample_ids", False))
        if 'sample_id' in df.columns:
            logger.error(f"   Unique IDs: {df['sample_id'].nunique()}, Total rows: {len(df)}")
    
    # Print results
    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        logger.info(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("🎉 All checks passed! Ready to submit.")
    else:
        logger.warning("⚠ Some checks failed. Please fix before submitting.")
    
    return all_passed

def compare_distributions(train_prices: np.ndarray, pred_prices: np.ndarray) -> None:
    """Compare training and prediction distributions"""
    
    logger.info("📊 Distribution Comparison:")
    
    # Basic statistics
    stats = {
        'Train': {
            'mean': np.mean(train_prices),
            'median': np.median(train_prices), 
            'std': np.std(train_prices),
            'min': np.min(train_prices),
            'max': np.max(train_prices),
            'q25': np.percentile(train_prices, 25),
            'q75': np.percentile(train_prices, 75)
        },
        'Predictions': {
            'mean': np.mean(pred_prices),
            'median': np.median(pred_prices),
            'std': np.std(pred_prices), 
            'min': np.min(pred_prices),
            'max': np.max(pred_prices),
            'q25': np.percentile(pred_prices, 25),
            'q75': np.percentile(pred_prices, 75)
        }
    }
    
    for metric in ['mean', 'median', 'std', 'min', 'max', 'q25', 'q75']:
        train_val = stats['Train'][metric]
        pred_val = stats['Predictions'][metric]
        ratio = pred_val / train_val if train_val > 0 else float('inf')
        
        logger.info(f"  {metric:>6}: Train ${train_val:>8.2f} | Pred ${pred_val:>8.2f} | Ratio {ratio:.3f}")
    
    # Distribution overlap
    try:
        from scipy.stats import ks_2samp
        ks_stat, p_value = ks_2samp(train_prices, pred_prices)
        logger.info(f"  KS Test: statistic={ks_stat:.4f}, p-value={p_value:.4f}")
        
        if p_value > 0.05:
            logger.info("  ✅ Distributions appear similar (p > 0.05)")
        else:
            logger.warning("  ⚠ Distributions may be different (p <= 0.05)")
    except ImportError:
        logger.info("  scipy not available for KS test")