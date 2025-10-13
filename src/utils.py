import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        
        if mode == 'min':
            self.is_better = lambda score, best: score < best - min_delta
        else:
            self.is_better = lambda score, best: score > best + min_delta
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
            return False

def calculate_smape(y_pred, y_true, epsilon=1e-8):
    """Calculate SMAPE metric"""
    import numpy as np
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.mean(numerator / (denominator + epsilon)) * 100
    return smape

def format_time(seconds):
    """Format time in seconds to readable format"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds//60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def save_predictions(predictions, sample_ids, output_path):
    """Save predictions in submission format"""
    import pandas as pd
    import os
    
    submission_df = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    submission_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")
    
    return submission_df