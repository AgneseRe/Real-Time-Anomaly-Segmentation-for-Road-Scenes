import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ========== ERFNET WEIGHTS FOR CLASS BALANCING BY HISTOGRAM ==========
def calculate_erfnet_weights(loader: DataLoader, num_classes: int, enc: bool) -> torch.Tensor:
    """
    Calculate class weights for ERFNet model.

    This function generates a tensor of weights, by processing dataset histogram, 
    depending on wheter the model is being used in encoder or decoder mode.
    
    Parameters:
        - loader (DataLoader): A data loader to iterate over the dataset.
        - num_classes (int): Number of classes in the dataset (19 + 1).
        - enc (bool): Boolean value for indicating if the model is in
            encoder (True) or decoder (False) mode.

    Returns:
        - weights (torch.Tensor): Tensor containing weights for each class.
    """
    class_counts = torch.zeros(num_classes)

    for _, label in loader:
        hist = torch.bincount(label.view(-1), minlength=num_classes)
        class_counts += hist
        class_counts[class_counts == 0] = 1  # Avoid division by zero

    mode = "encoder" if enc else "decoder"
    np.save(f"./utils/class_distribution/erfnet_class_frequency_{mode}.npy", class_counts.numpy())
    plot_class_histogram(class_counts, save_path=f"../plots/class_distribution_{mode}.png")

    class_weights = 1.0 / class_counts  # Inverse frequency weighting
    class_weights = class_weights / sum(class_weights) * num_classes
    np.save(f"./utils/class_distribution/erfnet_class_weights_{mode}.npy", class_weights.numpy())

    return class_weights

# ========== ERFNET WEIGHTS FOR CLASS BALANCING HARD-CODED ==========
def calculate_erfnet_weights_hard(enc: bool, num_classes: int) -> torch.Tensor:
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
        weights[0] = 2.8149201869965    #road
        weights[1] = 6.9850029945374	#sidewalk
        weights[2] = 3.7890393733978	#building
        weights[3] = 9.9428062438965	#wall
        weights[4] = 9.7702074050903	#fence
        weights[5] = 9.5110931396484	#pole
        weights[6] = 10.311357498169	#traffic light
        weights[7] = 10.026463508606	#traffic sign
        weights[8] = 4.6323022842407	#vegetation
        weights[9] = 9.5608062744141	#terrain
        weights[10] = 7.8698215484619   #sky
        weights[11] = 9.5168733596802	#person
        weights[12] = 10.373730659485	#rider
        weights[13] = 6.6616044044495	#car
        weights[14] = 10.260489463806	#truck
        weights[15] = 10.287888526917	#bus
        weights[16] = 10.289801597595	#train
        weights[17] = 10.405355453491	#motorcycle
        weights[18] = 10.138095855713	#bicycle

    weights[19] = 1  # for void classifier

    return weights


# ========== ENET WEIGHTS FOR CLASS BALANCING ==========
def calculate_enet_weights(loader: DataLoader, num_classes: int, c: float = 1.02) -> torch.Tensor:
    """
    Calculate class weights for ENet model, according to the formula:
        w_class = 1 / ln(c + p_class)

    This function generates a tensor of weights, calculated on the basis of a custom weighing scheme. 
    It is reported in the official paper 'ENet: A Deep Neural Network Architecture for Real-Time Semantic 
    Segmentation', available at the following link: https://arxiv.org/abs/1606.02147.
    
    Parameters:
        - loader (DataLoader): A data loader to iterate over the dataset.  
        - num_classes (int): Number of classes in the dataset (19 + 1).
        - c (int): An additional hyper-parameter (default 1.02).

    Returns:
        - weights (torch.Tensor): Tensor containing weights for each class.
    """
    class_counts = torch.zeros(num_classes)

    # Compute number of occurrences for each class (19 + 1)
    for _, label in loader:
        label = label.view(-1)
        for c in range(num_classes):
            class_counts[c] += (label == c).sum().item()

    # Compute class probabilities
    total_pixels = class_counts.sum()
    class_probabilities = class_counts / total_pixels

    # ENet weighting formula
    class_weights = 1.0 / torch.log(c + class_probabilities)
    
    # For classes that do not appear, so class_probabilities is equal to 0
    class_weights[class_probabilities == 0] = 0

    np.save("./utils/class_distribution/enet_class_weights.npy", class_weights.numpy())

    return class_weights


def plot_class_histogram(class_counts, class_names=None, save_path=None) -> None:
    """
    Plot a histogram of class frequencies for the Cityscapes train dataset.
    This function generates a bar plot showing the number of pixels for each 
    class in the dataset. It can be used to visualize the class distribution.

    Parameters:
        - class_counts (torch.Tensor): A tensor containing the pixel count for each class.
        - class_names (list[str]): If provided, the list will be used to label the x-axis ticks. 
            Otherwise, numeric indices are used.
        - save_path (str): If provided, the plot will be saved to the specified path. Otherwise, 
            the plot is shown interactively.

    Returns:
        None
    """
    num_classes = len(class_counts)
    indices = list(range(num_classes))

    plt.figure()
    plt.bar(indices, class_counts, tick_label=class_names if class_names else indices)
    plt.xlabel("Class")
    plt.ylabel("Pixel count")
    plt.title("Class distribution - Cityscapes Train Dataset")
    plt.xticks(rotation=45)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Saved histogram to {save_path}")
    else:
        plt.show()