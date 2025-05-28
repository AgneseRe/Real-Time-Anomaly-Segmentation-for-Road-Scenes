# ========================================================================
# Original GitHub repository: OOD Detection Metrics by Taylor Denouden
# URL: https://github.com/tayden/ood-metrics/blob/main/ood_metrics/plots.py
#
# plot_pr, plot_roc, plot_barcode function modifications in order to support 
# saving generated plots in a specified directory with a precise filename. 
# The original version of the functions only show the plot without saving it.
#
# Modifications:
# - Added 'save_dir' parameter to specify the directory to save the plot in.
# - Added 'file_name' parameter to specify the filename of the plot.
# - Used plt.savefig() to save the plot as a 'png' image.
# ========================================================================

import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from ood_metrics import aupr, auroc, fpr_at_95_tpr


def plot_roc(preds, labels, title="Receiver operating characteristic", save_dir=None, file_name=None):
    """Plot an ROC curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    fpr, tpr, _ = roc_curve(labels, preds)

    # Compute FPR (95% TPR)
    tpr95 = fpr_at_95_tpr(preds, labels)

    # Compute AUROC
    roc_auc = auroc(preds, labels)

    # Draw the plot
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUROC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0.95, 0.95], color='black', lw=lw, linestyle=':', label='FPR (95%% TPR) = %0.2f' % tpr95)
    plt.plot([tpr95, tpr95], [0, 1], color='black', lw=lw, linestyle=':')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Random detector ROC')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/{file_name}")
    else:
        plt.show()


def plot_pr(preds, labels, title="Precision recall curve", save_dir=None, file_name=None):
    """Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.
    
    preds: array, shape = [n_samples]
           Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    title: string, optional (default="Receiver operating characteristic")
           The title for the chart
    """

    # Compute values for curve
    precision, recall, _ = precision_recall_curve(labels, preds)
    prc_auc = auc(recall, precision)

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange',
             lw=lw, label='PRC curve (area = %0.2f)' % prc_auc)
    #     plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    if save_dir is not None:
        plt.savefig(f"{save_dir}/{file_name}")
    else:
        plt.show()


def plot_barcode(preds, labels, save_dir=None, file_name=None):
    """Plot a visualization showing inliers and outliers sorted by their prediction of novelty."""
    # the bar
    x = sorted([a for a in zip(preds, labels)], key=lambda x: x[0])
    x = np.array([[49, 163, 84] if a[1] == 1 else [173, 221, 142] for a in x])
    # x = np.array([a[1] for a in x]) # for bw image

    axprops = dict(xticks=[], yticks=[])
    barprops = dict(aspect='auto', cmap=plt.cm.binary_r, interpolation='nearest')

    fig = plt.figure()

    # a horizontal barcode
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.1], **axprops)
    ax.imshow(x.reshape((1, -1, 3)), **barprops)

    if save_dir is not None:
        plt.savefig(f"{save_dir}/{file_name}")
    else:
        plt.show()

# ============ VISUALIZATION OF ANOMALY SCORE IMAGES ============
def generate_colormap():
    """ Generate a colormap that gradually goes from blue to white to red """

    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    # Gradually go from red (index=0) to white (index=128)
    for i in range(128):
        ratio = i / 127
        b = int(255 * ratio)
        g = int(255 * ratio)
        r = 255
        colormap[i, 0] = [b, g, r]
    # Then go from white (index=128) to blue (index=255)
    for i in range(128, 256):
        ratio = (i - 128) / 127
        b = 255
        g = int(255 * (1 - ratio))
        r = int(255 * (1 - ratio))
        colormap[i, 0] = [b, g, r]
    return colormap

def save_colored_score_image(image_path, anomaly_score, save_path, file_name):
    """
    Save the image with the anomaly score colored in a new image.
    
    image_path: path to the input image
    anomaly_score: anomaly score for each pixel
    save_path: path to save the colored image
    """
    
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (anomaly_score.shape[1], anomaly_score.shape[0]))
    
    # Normalize the anomaly score
    anomaly_score = (anomaly_score - np.min(anomaly_score)) / (np.max(anomaly_score) - np.min(anomaly_score))
    
    # Apply the colormap
    anomaly_score = cv2.applyColorMap((anomaly_score * 255).astype(np.uint8), generate_colormap())
    
    # Combine the original image and the colored anomaly score
    # combined = cv2.addWeighted(image, 0.5, anomaly_score, 0.5, 0)
    
    # Save the image
    cv2.imwrite(f"{save_path}/{file_name}.png", cv2.cvtColor(anomaly_score, cv2.COLOR_RGB2BGR))