import torch
import torch.nn as nn
import torch.nn.functional as F

class SMAPELoss(nn.Module):
    """Symmetric Mean Absolute Percentage Error Loss"""
    
    def __init__(self, epsilon: float = 1e-8, reduction: str = 'mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate SMAPE loss"""
        numerator = torch.abs(predictions - targets)
        denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0
        
        smape = numerator / (denominator + self.epsilon)
        
        if self.reduction == 'mean':
            return 100.0 * torch.mean(smape)
        elif self.reduction == 'sum':
            return 100.0 * torch.sum(smape)
        else:
            return 100.0 * smape

class CombinedLoss(nn.Module):
    """Combined loss with SMAPE and auxiliary losses"""
    
    def __init__(self, smape_weight: float = 0.7,
                 mse_weight: float = 0.2,
                 mae_weight: float = 0.1,
                 epsilon: float = 1e-8):
        
        super().__init__()
        self.smape_weight = smape_weight
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        
        self.smape = SMAPELoss(epsilon=epsilon)
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss"""
        
        # Log transform for MSE (helps with scale)
        log_pred = torch.log1p(predictions)
        log_target = torch.log1p(targets)
        
        loss = (self.smape_weight * self.smape(predictions, targets) +
                self.mse_weight * self.mse(log_pred, log_target) * 100 +
                self.mae_weight * self.mae(predictions, targets))
        
        return loss

class FocalSMAPELoss(nn.Module):
    """Focal variant of SMAPE loss for hard examples"""
    
    def __init__(self, gamma: float = 2.0, epsilon: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate focal SMAPE loss"""
        
        numerator = torch.abs(predictions - targets)
        denominator = (torch.abs(targets) + torch.abs(predictions)) / 2.0
        
        smape = numerator / (denominator + self.epsilon)
        
        # Focal weighting
        focal_weight = smape ** self.gamma
        
        return 100.0 * torch.mean(focal_weight * smape)