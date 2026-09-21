import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional
from torch import Tensor

class FocalBCELoss(nn.Module):
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: Optional[Tensor] = None):
        super(FocalBCELoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        
        if targets.dim() == 1 and inputs.dim() == 2 and inputs.size(1) == 1:
            targets = targets.unsqueeze(1)
        mask = torch.isfinite(targets)

        if not mask.any():
            return inputs.sum() * 0.0

        targets_clean = torch.where(mask, targets, torch.zeros_like(targets)).to(dtype=inputs.dtype)

        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets_clean, pos_weight=self.pos_weight, reduction='none'
        )
        
        probs = torch.sigmoid(inputs)

      
        pt = targets_clean * probs + (1.0 - targets_clean) * (1.0 - probs)
        focal_weight = (1.0 - pt).clamp(min=0.0) ** self.gamma

        if self.alpha is not None:
            alpha = float(self.alpha)
            alpha_t = targets_clean * alpha + (1.0 - targets_clean) * (1.0 - alpha)
            focal_weight = focal_weight * alpha_t
        
        focal_loss = focal_weight * bce_loss

        mask_f = mask.to(dtype=focal_loss.dtype)
        return (focal_loss * mask_f).sum() / mask_f.sum().clamp(min=1)

class MaskedBCEWithLogitsLoss(nn.Module):
    
    
    def __init__(self, pos_weight: Optional[Tensor] = None, reduction: str = 'mean'):
        super(MaskedBCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        
        if targets.dim() == 1 and inputs.dim() == 2 and inputs.size(1) == 1:
            targets = targets.unsqueeze(1)
        mask = torch.isfinite(targets)
        
        if not mask.any():
            return inputs.sum() * 0.0
        
        # Apply task weights before masking; boolean indexing flattens task axes.
        safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
        pos_weight = self.pos_weight
        if pos_weight is not None:
            pos_weight = pos_weight.to(device=inputs.device, dtype=inputs.dtype)
        loss = F.binary_cross_entropy_with_logits(
            inputs, safe_targets, pos_weight=pos_weight, reduction='none'
        )
        valid_loss = loss[mask]
        if self.reduction == 'none':
            return valid_loss
        if self.reduction == 'sum':
            return valid_loss.sum()
        if self.reduction == 'mean':
            return valid_loss.mean()
        raise ValueError(f"Unsupported reduction: {self.reduction}")


def compute_pos_weight(labels: Tensor, smooth: float = 1.0) -> Tensor:
  
    valid_mask = ~torch.isnan(labels)
    valid_labels = labels[valid_mask]
    
    if len(valid_labels) == 0:
        return torch.tensor(1.0)
    
    pos_count = (valid_labels == 1).sum().float() + smooth
    neg_count = (valid_labels == 0).sum().float() + smooth
    
    pos_weight = neg_count / pos_count
    
    return pos_weight
