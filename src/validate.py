import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

from .loss import smape_loss
from .train import PriceDataset
from .models import create_model

logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive model validation and analysis"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def validate_single_model(self, checkpoint_path: str, 
                              val_df: pd.DataFrame,
                              val_features: pd.DataFrame,
                              image_dir: str) -> Dict:
        """Validate a single model checkpoint"""
        
        logger.info(f"Validating model: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model
        feature_cols = [col for col in val_features.columns 
                       if col not in ['sample_id', 'price']]
        num_features = len(feature_cols)
        
        model, text_encoder, image_encoder = create_model(self.config, num_features)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Create dataset
        val_dataset = PriceDataset(
            val_df, val_features, image_dir,
            is_train=False, augment=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Generate predictions
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
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
                target = batch['price'].to(self.device)
                
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
                targets.extend(target.cpu().numpy())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate metrics
        metrics = self.calculate_metrics(predictions, targets)
        metrics['checkpoint'] = checkpoint_path
        metrics['fold'] = checkpoint.get('fold', -1)
        
        # Cleanup
        del model, text_encoder, image_encoder
        torch.cuda.empty_cache()
        
        return metrics, predictions, targets
    
    def calculate_metrics(self, predictions: np.ndarray, 
                          targets: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        metrics = {}
        
        # SMAPE (primary metric)
        metrics['smape'] = smape_loss(torch.tensor(predictions), torch.tensor(targets)).item()
        
        # Standard regression metrics
        metrics['mae'] = mean_absolute_error(targets, predictions)
        metrics['mse'] = mean_squared_error(targets, predictions)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['r2'] = r2_score(targets, predictions)
        
        # Percentage-based metrics
        mape = np.mean(np.abs((targets - predictions) / np.maximum(targets, 1e-6))) * 100
        metrics['mape'] = mape
        
        # Log-based metrics (good for prices)
        log_targets = np.log1p(targets)
        log_preds = np.log1p(predictions)
        metrics['log_mae'] = mean_absolute_error(log_targets, log_preds)
        metrics['log_mse'] = mean_squared_error(log_targets, log_preds)
        
        # Correlation
        metrics['pearson_r'] = stats.pearsonr(targets, predictions)[0]
        metrics['spearman_r'] = stats.spearmanr(targets, predictions)[0]
        
        # Price range analysis
        for price_range, (min_p, max_p) in {
            'low': (0, 50),
            'medium': (50, 200),
            'high': (200, float('inf'))
        }.items():
            mask = (targets >= min_p) & (targets < max_p)
            if mask.any():
                range_smape = smape_loss(
                    torch.tensor(predictions[mask]), 
                    torch.tensor(targets[mask])
                ).item()
                metrics[f'smape_{price_range}'] = range_smape
                metrics[f'count_{price_range}'] = mask.sum()
        
        return metrics
    
    def analyze_errors(self, predictions: np.ndarray, 
                       targets: np.ndarray,
                       features_df: pd.DataFrame) -> Dict:
        """Analyze prediction errors in detail"""
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        rel_errors = abs_errors / np.maximum(targets, 1e-6)
        
        analysis = {
            'error_stats': {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'mean_abs_error': np.mean(abs_errors),
                'median_abs_error': np.median(abs_errors),
                'q75_abs_error': np.percentile(abs_errors, 75),
                'q95_abs_error': np.percentile(abs_errors, 95)
            }
        }
        
        # Find worst predictions
        worst_indices = np.argsort(abs_errors)[-20:]
        analysis['worst_predictions'] = []
        
        for idx in worst_indices:
            if idx < len(features_df):
                analysis['worst_predictions'].append({
                    'index': int(idx),
                    'prediction': float(predictions[idx]),
                    'target': float(targets[idx]),
                    'error': float(errors[idx]),
                    'rel_error': float(rel_errors[idx])
                })
        
        # Error by categories if available
        analysis['error_by_category'] = {}
        
        category_cols = [col for col in features_df.columns if col.startswith('cat_')]
        for cat_col in category_cols:
            if cat_col in features_df.columns:
                mask = features_df[cat_col] == 1
                if mask.any():
                    cat_errors = abs_errors[mask]
                    analysis['error_by_category'][cat_col] = {
                        'mean_error': float(np.mean(cat_errors)),
                        'median_error': float(np.median(cat_errors)),
                        'count': int(mask.sum())
                    }
        
        # Error by IPQ if available
        if 'ipq' in features_df.columns:
            analysis['error_by_ipq'] = {}
            for ipq in sorted(features_df['ipq'].unique()):
                if pd.notna(ipq):
                    mask = features_df['ipq'] == ipq
                    if mask.any():
                        ipq_errors = abs_errors[mask]
                        analysis['error_by_ipq'][f'ipq_{int(ipq)}'] = {
                            'mean_error': float(np.mean(ipq_errors)),
                            'count': int(mask.sum())
                        }
        
        return analysis
    
    def create_visualizations(self, predictions: np.ndarray, 
                              targets: np.ndarray,
                              output_dir: str,
                              model_name: str = "model"):
        """Create validation visualizations"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Predictions vs Targets scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(targets, predictions, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'{model_name} - Predictions vs Actual')
        plt.grid(True, alpha=0.3)
        
        # Add correlation text
        r = stats.pearsonr(targets, predictions)[0]
        plt.text(0.05, 0.95, f'R = {r:.3f}', transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_predictions_vs_actual.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Error distribution
        errors = predictions - targets
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Prediction Error ($)')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        abs_errors = np.abs(errors)
        plt.hist(abs_errors, bins=50, alpha=0.7, edgecolor='black', color='orange')
        plt.xlabel('Absolute Error ($)')
        plt.ylabel('Frequency')
        plt.title('Absolute Error Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_error_distribution.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Residual plot
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, errors, alpha=0.5, s=20)
        plt.axhline(0, color='red', linestyle='--', alpha=0.8)
        plt.xlabel('Predicted Price ($)')
        plt.ylabel('Residual (Pred - Actual) ($)')
        plt.title(f'{model_name} - Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_residuals.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # 4. Log-scale comparison
        plt.figure(figsize=(10, 8))
        log_targets = np.log1p(targets)
        log_predictions = np.log1p(predictions)
        
        plt.scatter(log_targets, log_predictions, alpha=0.5, s=20)
        
        min_val = min(log_targets.min(), log_predictions.min())
        max_val = max(log_targets.max(), log_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        plt.xlabel('Log(Actual Price + 1)')
        plt.ylabel('Log(Predicted Price + 1)')
        plt.title(f'{model_name} - Log Scale Comparison')
        plt.grid(True, alpha=0.3)
        
        r_log = stats.pearsonr(log_targets, log_predictions)[0]
        plt.text(0.05, 0.95, f'Log R = {r_log:.3f}', transform=plt.gca().transAxes,
                fontsize=12, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_log_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def validate_cross_validation_results(self, cv_results_path: str) -> Dict:
        """Analyze cross-validation results"""
        
        if not os.path.exists(cv_results_path):
            logger.warning(f"CV results not found: {cv_results_path}")
            return {}
        
        # Load CV results
        with open(cv_results_path, 'r') as f:
            cv_results = json.load(f)
        
        fold_scores = [result['val_smape'] for result in cv_results['fold_results']]
        
        analysis = {
            'mean_cv_smape': np.mean(fold_scores),
            'std_cv_smape': np.std(fold_scores),
            'min_cv_smape': np.min(fold_scores),
            'max_cv_smape': np.max(fold_scores),
            'cv_stability': np.std(fold_scores) / np.mean(fold_scores),  # Coefficient of variation
            'fold_scores': fold_scores
        }
        
        logger.info(f"Cross-validation analysis:")
        logger.info(f"  Mean SMAPE: {analysis['mean_cv_smape']:.2f}% ± {analysis['std_cv_smape']:.2f}%")
        logger.info(f"  CV Stability: {analysis['cv_stability']:.3f}")
        
        return analysis

def validate_model_performance(config, output_dir: str = None):
    """Complete model validation pipeline"""
    
    if output_dir is None:
        output_dir = os.path.join(config.output_dir, 'validation')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, 'validation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    validator = ModelValidator(config)
    
    # Load validation data
    val_df = pd.read_csv(config.val_csv) if hasattr(config, 'val_csv') else None
    if val_df is not None:
        from .features import load_features
        try:
            val_features = load_features(config.feats_dir, 'val')
        except FileNotFoundError:
            val_features = None
    else:
        val_features = None
    
    all_results = []
    
    if val_df is not None and val_features is not None:
        # Find model checkpoints
        checkpoint_pattern = os.path.join(config.ckpt_dir, "fold*_best.pt")
        import glob
        checkpoints = glob.glob(checkpoint_pattern)
        
        logger.info(f"Found {len(checkpoints)} checkpoints for validation")
        
        for checkpoint_path in checkpoints:
            try:
                metrics, predictions, targets = validator.validate_single_model(
                    checkpoint_path, val_df, val_features, config.img_val
                )
                
                # Error analysis
                error_analysis = validator.analyze_errors(predictions, targets, val_features)
                metrics.update(error_analysis)
                
                # Create visualizations
                fold_name = os.path.basename(checkpoint_path).replace('.pt', '')
                validator.create_visualizations(
                    predictions, targets, output_dir, fold_name
                )
                
                all_results.append(metrics)
                
                logger.info(f"Fold {metrics['fold']} validation complete:")
                logger.info(f"  SMAPE: {metrics['smape']:.2f}%")
                logger.info(f"  MAE: ${metrics['mae']:.2f}")
                logger.info(f"  R²: {metrics['r2']:.3f}")
                
            except Exception as e:
                logger.error(f"Error validating {checkpoint_path}: {e}")
                continue
    
    # Cross-validation analysis
    cv_results_path = os.path.join(config.output_dir, 'cv_results.json')
    cv_analysis = validator.validate_cross_validation_results(cv_results_path)
    
    # Save validation report
    validation_report = {
        'config': {
            'model_name': config.model_name,
            'num_folds': config.num_folds,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate
        },
        'individual_fold_results': all_results,
        'cross_validation_analysis': cv_analysis,
        'summary': {
            'mean_smape': np.mean([r['smape'] for r in all_results]) if all_results else None,
            'std_smape': np.std([r['smape'] for r in all_results]) if all_results else None,
            'best_fold_smape': min([r['smape'] for r in all_results]) if all_results else None
        }
    }
    
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(validation_report, f, indent=2)
    
    logger.info(f"Validation report saved to {report_path}")
    
    return validation_report