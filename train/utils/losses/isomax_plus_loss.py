# =============================================================================================================================================
# IsoMaxPlus Loss Integration. Entropic Out-of-Distribution Detection
# GitHub: https://github.com/dlmacedo/entropic-out-of-distribution-detection/blob/9ad451ca815160e5339dc21319cea2b859e3e101/losses/isomaxplus.py
#
# This code is a modification of the original IsoMaxPlus loss implementation, adapted to semantic segmentation tasks. Changes include flattening
# and broadcasting to support spatial feature maps with shape [B, C, H, W], Batch size B, number of classes C, height H, and width W.
# =============================================================================================================================================

import torch.nn as nn
import torch.nn.functional as F
import torch

class IsoMaxPlusLossFirstPart(nn.Module):
    """ This part replaces the model classifier output layer nn.Linear() """
    def __init__(self, num_features, num_classes, temperature=1.0):
        super(IsoMaxPlusLossFirstPart, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature        
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1)) 
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        B, _, H, W = features.size()
        features_flat = features.permute(0, 2, 3, 1).flatten(0, 2)  # [B*H*W, C]
        features_norm = F.normalize(features_flat)
        prototypes_norm = F.normalize(self.prototypes)
        distances = torch.abs(self.distance_scale) * torch.cdist(features_norm, prototypes_norm, p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances.view(B, H, W, self.num_classes).permute(0, 3, 1, 2)  # [B, num_classes, H, W]
        # The temperature may be calibrated after training to improve uncertainty estimation.
        return logits / self.temperature


class IsoMaxPlusLossSecondPart(nn.Module):
    """ This part replaces the nn.CrossEntropyLoss() """
    def __init__(self, entropic_scale=10.0):
        super(IsoMaxPlusLossSecondPart, self).__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets, debug=False):
        """Probabilities and logarithms are calculated separately and sequentially"""
        """Therefore, nn.CrossEntropyLoss() must not be used to calculate the loss"""
        B, _, H, W = logits.size()
        logits = logits.permute(0, 2, 3, 1).flatten(0, 2)  # [B*H*W, num_classes]
        targets = targets.view(-1)  # [B*H*W]
        distances = -logits
        probabilities_for_training = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probabilities_at_targets = probabilities_for_training[range(distances.size(0)), targets] + 1e-12  # Add small value to avoid log(0)
        loss = -torch.log(probabilities_at_targets).mean()
        if not debug:
            return loss
        else:
            targets_one_hot = torch.eye(distances.size(1))[targets].long().cuda()
            intra_inter_distances = torch.where(targets_one_hot != 0, distances, torch.Tensor([float('Inf')]).cuda())
            inter_intra_distances = torch.where(targets_one_hot != 0, torch.Tensor([float('Inf')]).cuda(), distances)
            intra_distances = intra_inter_distances[intra_inter_distances != float('Inf')]
            inter_distances = inter_intra_distances[inter_intra_distances != float('Inf')]
            return loss, 1.0, intra_distances, inter_distances
        
    def __str__(self):
        return f"IsoMaxPlusLossSecondPart(entropic_scale={self.entropic_scale})"