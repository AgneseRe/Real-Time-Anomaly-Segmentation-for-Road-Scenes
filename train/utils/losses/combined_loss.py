import torch.nn as nn
import torch.nn.functional as F
import torch

class CombinedLoss(nn.Module):
    """
    Combine three loss functions into a single loss with weighted contributions.

    This class combines a cross-entropy loss, a focal loss, and an EIM loss.
    The combined loss is computed as a weighted sum of the individual losses:
        total_loss = alpha * ce_loss + beta * focal_loss + gamma * eim_loss
    
    Parameters:
        - ce_loss (nn.Module): Cross-entropy loss function.
        - focal_loss (nn.Module): Focal loss function.
        - eim_loss (nn.Module): EIM loss function.
        - alpha (float): Weight for cross-entropy. Default is 1.0.
        - beta (float): Weight for focal-loss. Default is 1.0.
        - gamma (float): Weight for EIM loss. Default is 1.0.
    """
    def __init__(self, ce_loss = None, focal_loss = None, eim_loss = None, alpha = 1.0, beta = 1.0, gamma = 1.0):
        super(CombinedLoss, self).__init__()
        self.ce_loss = ce_loss
        self.focal_loss = focal_loss
        self.eim_loss = eim_loss
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, output, target):
        """
        Computes the weighted sum of the three loss functions.

        Parameters:
            - output (Tensor): Model predictions (logits or probabilities).
            - target (Tensor): Ground truth labels.

        Returns:
            Tensor: The combined loss value.
        """
        total_loss = 0.0

        if self.ce_loss is not None:
            total_loss += self.alpha * self.ce_loss(output, target)
        
        if self.focal_loss is not None:
            total_loss += self.beta * self.focal_loss(output, target)
        
        if self.eim_loss is not None:
            total_loss += self.gamma * self.eim_loss(output, target) 

        return total_loss