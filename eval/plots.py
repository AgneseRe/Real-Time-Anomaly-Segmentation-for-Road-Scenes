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