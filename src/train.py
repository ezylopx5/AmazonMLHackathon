import os
import gc
import glob
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional

# Make wandb optional
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from .models import create_model
from .loss import SMAPELoss, CombinedLoss
from .memory_utils import ResourceMonitor, MemoryOptimizedEncoder, GradientAccumulator, memory_efficient_context
from .utils import AverageMeter, EarlyStopping

logger = logging.getLogger(__name__)

class PriceDataset(Dataset):
    """Multi-modal dataset for price prediction"""
    
    def __init__(self, 
                 df: pd.DataFrame,
                 features_df: pd.DataFrame,
                 image_dir: str,
                 is_train: bool = True,
                 augment: bool = False):
        
        self.df = df
        self.features_df = features_df
        self.image_dir = image_dir
        self.is_train = is_train
        self.augment = augment
        
        # Merge dataframes
        self.data = pd.merge(df, features_df, on='sample_id', how='left')
        
        # Get feature columns
        self.feature_cols = [col for col in features_df.columns 
                            if col not in ['sample_id', 'price']]
        
        # Normalize features
        self.feature_means = self.data[self.feature_cols].mean()
        self.feature_stds = self.data[self.feature_cols].std() + 1e-8
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Text
        text = row.get('catalog_content', '')
        if pd.isna(text):
            text = ""
        
        # Image
        image_path = os.path.join(self.image_dir, f"{row['sample_id']}.jpg")
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Simple augmentation if training
            if self.is_train and self.augment and np.random.random() > 0.5:
                # Random horizontal flip
                if np.random.random() > 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                
                # Random rotation
                if np.random.random() > 0.7:
                    angle = np.random.randint(-10, 10)
                    image = image.rotate(angle)
                    
        except:
            # Create dummy image if loading fails
            image = Image.new('RGB', (224, 224), color='white')
        
        # Features (normalized)
        features = row[self.feature_cols].values.astype(np.float32)
        features = (features - self.feature_means.values) / self.feature_stds.values
        features = torch.tensor(features, dtype=torch.float32)
        
        # Add noise to features if augmenting
        if self.is_train and self.augment and np.random.random() > 0.7:
            noise = torch.randn_like(features) * 0.01
            features = features + noise
        
        # Output
        output = {
            'sample_id': row['sample_id'],
            'text': text,
            'image': image,
            'features': features
        }
        
        if self.is_train and 'price' in row:
            output['price'] = torch.tensor(row['price'], dtype=torch.float32)
        
        return output

class Trainer:
    """Main training class with memory optimization"""
    
    def __init__(self, config, model, text_encoder, image_encoder, device='cuda'):
        self.config = config
        self.model = model.to(device)
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.device = device
        
        # Initialize resource monitoring
        self.resource_monitor = ResourceMonitor(safety_margin_gb=2.0)
        self.resource_monitor.log_memory_usage("trainer_initialization")
        
        # Wrap encoders for memory optimization
        if self.text_encoder:
            self.text_encoder = MemoryOptimizedEncoder(self.text_encoder, "text")
        if self.image_encoder:
            self.image_encoder = MemoryOptimizedEncoder(self.image_encoder, "image")
        
        # Optimizer
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        self.scheduler = None  # Set per fold
        
        # Loss
        self.criterion = CombinedLoss(smape_weight=0.7, mse_weight=0.2, mae_weight=0.1)
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Metrics
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        
        # Checkpointing
        self.best_smape = float('inf')
        self.checkpoint_dir = config.ckpt_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch with memory optimization"""
        self.model.train()
        self.train_loss.reset()
        
        # Initialize gradient accumulator
        accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        accumulator = GradientAccumulator(accumulation_steps)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        
        with accumulator:
            for batch_idx, batch in enumerate(progress_bar):
                # Encode text and images with memory efficiency
                with memory_efficient_context():
                    with torch.no_grad():
                        if self.text_encoder:
                            text_embeddings = self.text_encoder.encode(batch['text'])
                        else:
                            text_embeddings = torch.zeros(len(batch['text']), 4096).to(self.device)
                        
                        if self.image_encoder:
                            image_embeddings = self.image_encoder.encode(batch['image'])
                        else:
                            image_embeddings = torch.zeros(len(batch['image']), 1024).to(self.device)
                
                features = batch['features'].to(self.device)
                prices = batch['price'].to(self.device)
                
                # Forward pass with gradient accumulation
                if self.config.mixed_precision and self.scaler:
                    with torch.cuda.amp.autocast():
                        if self.text_encoder and self.image_encoder:
                            predictions = self.model(text_embeddings, image_embeddings, features)
                        else:
                            predictions = self.model(features)
                        loss = self.criterion(predictions, prices)
                else:
                    if self.text_encoder and self.image_encoder:
                        predictions = self.model(text_embeddings, image_embeddings, features)
                    else:
                        predictions = self.model(features)
                    loss = self.criterion(predictions, prices)
                
                # Gradient accumulation and optimization
                should_step = accumulator.accumulate(loss, self.optimizer, self.scaler if self.config.mixed_precision else None)
                
                if should_step:
                    accumulator.step(
                        self.optimizer, 
                        self.scaler if self.config.mixed_precision else None,
                        max_norm=getattr(self.config, 'max_grad_norm', 1.0)
                    )
                    
                    if self.scheduler:
                        self.scheduler.step()
                
                # Update metrics
                self.train_loss.update(accumulator.get_average_loss(), len(batch['price']))
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{self.train_loss.avg:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to wandb (if available)
            if WANDB_AVAILABLE and self.config.upload_to_hub and batch_idx % self.config.log_interval == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        return self.train_loss.avg
    
    def validate(self, dataloader):
        """Validate the model"""
        self.model.eval()
        self.val_loss.reset()
        
        predictions_list = []
        targets_list = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                # Encode
                if self.text_encoder:
                    text_embeddings = self.text_encoder.encode(batch['text'])
                else:
                    text_embeddings = torch.zeros(len(batch['text']), 4096).to(self.device)
                
                if self.image_encoder:
                    image_embeddings = self.image_encoder.encode(batch['image'])
                else:
                    image_embeddings = torch.zeros(len(batch['image']), 1024).to(self.device)
                
                features = batch['features'].to(self.device)
                prices = batch['price'].to(self.device)
                
                # Predict
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        if self.text_encoder and self.image_encoder:
                            predictions = self.model(text_embeddings, image_embeddings, features)
                        else:
                            predictions = self.model(features)
                        loss = self.criterion(predictions, prices)
                else:
                    if self.text_encoder and self.image_encoder:
                        predictions = self.model(text_embeddings, image_embeddings, features)
                    else:
                        predictions = self.model(features)
                    loss = self.criterion(predictions, prices)
                
                self.val_loss.update(loss.item(), len(batch['price']))
                
                predictions_list.extend(predictions.cpu().numpy())
                targets_list.extend(prices.cpu().numpy())
        
        # Calculate SMAPE
        predictions_np = np.array(predictions_list)
        targets_np = np.array(targets_list)
        smape = self.calculate_smape(predictions_np, targets_np)
        
        return self.val_loss.avg, smape, predictions_np, targets_np
    
    @staticmethod
    def calculate_smape(y_pred, y_true, epsilon=1e-8):
        """Calculate SMAPE metric"""
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
        smape = np.mean(numerator / (denominator + epsilon)) * 100
        return smape
    
    def save_checkpoint(self, epoch, fold, val_smape, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'fold': fold,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_smape': val_smape,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        path = os.path.join(self.checkpoint_dir, f"fold{fold}_epoch{epoch}.pt")
        torch.save(checkpoint, path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"fold{fold}_best.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"💾 Saved best checkpoint: {best_path} (SMAPE: {val_smape:.2f}%)")
        
        # Upload to Hub if configured
        if self.config.upload_to_hub and self.config.hub_repo:
            self.upload_to_hub(path, fold, epoch)
    
    def upload_to_hub(self, checkpoint_path, fold, epoch):
        """Upload checkpoint to Hugging Face Hub"""
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=f"checkpoints/fold{fold}_epoch{epoch}.pt",
                repo_id=self.config.hub_repo,
                repo_type="model"
            )
            logger.info(f"☁️ Uploaded checkpoint to Hub: {self.config.hub_repo}")
        except Exception as e:
            logger.warning(f"Failed to upload to Hub: {e}")

def train_cv(config):
    """Run cross-validation training"""
    
    # Set seed
    from .data import set_seed
    set_seed(config.seed)
    
    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(config.train_csv)
    from .features import load_features
    train_features = load_features(config.feats_dir, 'train')
    
    # Get feature dimension
    feature_cols = [col for col in train_features.columns 
                   if col not in ['sample_id', 'price']]
    num_features = len(feature_cols)
    logger.info(f"Number of features: {num_features}")
    
    # Create stratified folds
    train_df['price_bin'] = pd.qcut(train_df['price'], q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    
    # Training
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['price_bin']), 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Fold {fold}/{config.folds}")
        logger.info(f"{'='*50}")
        
        # Split data
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        logger.info(f"Train samples: {len(train_fold)}, Val samples: {len(val_fold)}")
        
        # Create model
        model, text_encoder, image_encoder = create_model(config, num_features)
        
        # Create datasets
        train_dataset = PriceDataset(
            train_fold, train_features, config.img_train,
            is_train=True, augment=True
        )
        
        val_dataset = PriceDataset(
            val_fold, train_features, config.img_train,
            is_train=True, augment=False
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size * 2,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
        
        # Create trainer
        trainer = Trainer(config, model, text_encoder, image_encoder)
        
        # Setup scheduler
        total_steps = len(train_loader) * config.epochs
        trainer.scheduler = OneCycleLR(
            trainer.optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=0.1
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=3, mode='min')
        
        # Training loop
        best_smape = float('inf')
        
        for epoch in range(1, config.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{config.epochs}")
            
            # Train
            train_loss = trainer.train_epoch(train_loader, epoch)
            
            # Step scheduler
            if isinstance(trainer.scheduler, CosineAnnealingLR):
                trainer.scheduler.step()
            
            # Validate
            val_loss, val_smape, val_preds, val_targets = trainer.validate(val_loader)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val SMAPE: {val_smape:.2f}%")
            
            # Save checkpoint
            is_best = val_smape < best_smape
            if is_best:
                best_smape = val_smape
            
            if not config.save_best_only or is_best:
                trainer.save_checkpoint(epoch, fold, val_smape, is_best)
            
            # Early stopping
            if early_stopping(val_smape):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Log to wandb (if available)
            if WANDB_AVAILABLE and config.upload_to_hub:
                wandb.log({
                    f'fold{fold}_train_loss': train_loss,
                    f'fold{fold}_val_loss': val_loss,
                    f'fold{fold}_val_smape': val_smape,
                    f'fold{fold}_best_smape': best_smape,
                    'epoch': epoch
                })
        
        fold_scores.append(best_smape)
        logger.info(f"\n✅ Fold {fold} completed. Best SMAPE: {best_smape:.2f}%")
        
        # Clean up
        del model, text_encoder, image_encoder, trainer
        torch.cuda.empty_cache()
        gc.collect()
    
    # Print final results
    logger.info(f"\n{'='*50}")
    logger.info(f"FINAL RESULTS")
    logger.info(f"{'='*50}")
    for i, score in enumerate(fold_scores, 1):
        logger.info(f"Fold {i}: {score:.2f}%")
    logger.info(f"Mean SMAPE: {np.mean(fold_scores):.2f}% ± {np.std(fold_scores):.2f}%")
    
    # Save results
    results = {
        'fold_scores': fold_scores,
        'mean_smape': float(np.mean(fold_scores)),
        'std_smape': float(np.std(fold_scores)),
        'config': config.__dict__
    }
    
    with open(os.path.join(config.ckpt_dir, 'cv_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return fold_scores