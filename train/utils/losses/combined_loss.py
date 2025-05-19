import torch.nn as nn
import torch.nn.functional as F
import torch

class CombinedLoss(nn.Module):
    """
    Combine two loss functions into a single loss with weighted contributions.

    This class allows the combination of a primary loss function (e.g., CrossEntropyLoss or FocalLoss)
    along with an auxiliary loss function (e.g., IsoMaxPlusLossSecondPart) during training. The combined 
    loss is computed as a weighted sum of the two losses, where the weights can be adjusted.
    
    The final loss is computed as:
        total_loss = alpha * base_loss + beta * iso_loss
    
    Parameters:
        - base_loss (nn.Module): The primary loss function.
        - iso_loss (nn.Module): The auxiliary loss function.
        - alpha (float): Weight for the base_loss. Default is 1.0.
        - beta (float): Weight for the iso_loss. Default is 1.0.
    """
    def __init__(self, base_loss, iso_loss, alpha = 1.0, beta = 1.0):
        super(CombinedLoss, self).__init__()
        self.base_loss = base_loss
        self.iso_loss = iso_loss
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        """
        Computes the weighted sum of the base and auxiliary loss functions.

        Parameters:
            - output (Tensor): Model predictions (logits or probabilities).
            - target (Tensor): Ground truth labels.

        Returns:
            Tensor: The combined loss value.
        """
        return self.alpha * self.base_loss(output, target) + self.beta * self.iso_loss(output, target)