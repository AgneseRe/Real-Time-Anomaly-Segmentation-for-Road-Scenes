import torch
import numpy as np
from torch.utils.data import DataLoader

# ========== ERFNET WEIGHTS ==========
def calculate_erfnet_weights(enc: bool, num_classes: int) -> torch.Tensor:
    """
    Calculate class weights for ErfNet model.

    This function generates a tensor of predefined weights, depending on
    wheter the model is being used in encoder or decoder mode.
    
    Parameters:
        - enc (bool): Boolean value for indicating if the model is in
            encoder (True) or decoder (False) mode.
        - num_classes (int): Number of classes in the dataset (19 + 1).

    Returns:
        - weights (torch.Tensor): Tensor containing weights for each class.
    """
    weights = torch.ones(num_classes)

    if (enc):
        weights[0] = 2.3653597831726	
        weights[1] = 4.4237880706787	
        weights[2] = 2.9691488742828	
        weights[3] = 5.3442072868347	
        weights[4] = 5.2983593940735	
        weights[5] = 5.2275490760803	
        weights[6] = 5.4394111633301	
        weights[7] = 5.3659925460815	
        weights[8] = 3.4170460700989	
        weights[9] = 5.2414722442627	
        weights[10] = 4.7376127243042	
        weights[11] = 5.2286224365234	
        weights[12] = 5.455126285553	
        weights[13] = 4.3019247055054	
        weights[14] = 5.4264230728149	
        weights[15] = 5.4331531524658	
        weights[16] = 5.433765411377	
        weights[17] = 5.4631009101868	
        weights[18] = 5.3947434425354
    else:
        weights[0] = 2.8149201869965	
        weights[1] = 6.9850029945374	
        weights[2] = 3.7890393733978	
        weights[3] = 9.9428062438965	
        weights[4] = 9.7702074050903	
        weights[5] = 9.5110931396484	
        weights[6] = 10.311357498169	
        weights[7] = 10.026463508606	
        weights[8] = 4.6323022842407	
        weights[9] = 9.5608062744141	
        weights[10] = 7.8698215484619	
        weights[11] = 9.5168733596802	
        weights[12] = 10.373730659485	
        weights[13] = 6.6616044044495	
        weights[14] = 10.260489463806	
        weights[15] = 10.287888526917	
        weights[16] = 10.289801597595	
        weights[17] = 10.405355453491	
        weights[18] = 10.138095855713	

    weights[19] = 1  # for void classifier

    return weights


# ========== ENET WEIGHTS ==========
def calculate_enet_weights(loader: DataLoader, num_classes: int, c: float = 1.02) -> torch.Tensor:
    """
    Calculate class weights for ENet model, according to the formula:
        w_class = 1 / ln(c + p_class)

    This function generates a tensor of weights, calculated on the basis of
    a custom weighing scheme. It is reported in the official paper 'ENet: A 
    Deep Neural Network Architecture for Real-Time Semantic Segmentation',
    available at the following link: https://arxiv.org/abs/1606.02147.
    
    Parameters:
        - loader (DataLoader): A data loader to iterate over the dataset.  
        - num_classes (int): Number of classes in the dataset (19 + 1).
        - c (int): An additional hyper-parameter (default 1.02).

    Returns:
        - weights (torch.Tensor): Tensor containing weights for each class.
    """
    class_counts = torch.zeros(num_classes)

    # Compute number of occurrences for each class (19 + 1)
    for _, labels in loader:    # features, labels
        labels = labels.view(-1)
        for c in range(num_classes):
            class_counts[c] += (labels == c).sum().item()

    # Compute class probabilities
    total_pixels = class_counts.sum()
    class_probabilities = class_counts / total_pixels

    # ENet weighting formula
    weights = 1.0 / torch.log(c + class_probabilities)
    
    # For classes that do not appear, so class_probabilities is equal to 0
    weights[class_probabilities == 0] = 0

    return weights