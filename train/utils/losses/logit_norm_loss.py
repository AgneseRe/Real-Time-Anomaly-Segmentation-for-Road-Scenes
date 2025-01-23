import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitNormLoss(nn.Module):
    # TODO: implement logit norm loss
    def __init__(self, reduction='mean'):
        super(LogitNormLoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        
        # Normalize the logits
        inputs = F.normalize(inputs, p=1, dim=1)
        
        # Compute the loss
        loss = F.binary_cross_entropy(inputs, targets, reduction=self.reduction)
        
        return loss
    
    