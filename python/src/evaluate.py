"""
evaluate.py

Functions to evaluate the various models using standard metrics and plots to compare models. 
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)


def compute_metrics(y_test, y_pred_proba, threshold=0.5):
    """Compute evaluation metrics for a binary classifier's probability output.

    Inputs: 
    y_test : shape (n_samples,); True binary labels.
    y_pred_proba : shape (n_samples, 2); model predictions 
    threshold : float, default=0.5

    Output: 
    dict of metric name and value
    """
    y_scores = y_pred_proba[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, y_scores),
        "pr_auc": average_precision_score(y_test, y_scores),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "threshold": threshold,
    }


def plot_evaluation(y_test, y_pred_proba, threshold=0.5):
    """
    Display ROC curve, precision-recall curve, and confusion matrix.
    """
    y_scores = y_pred_proba[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_scores):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_test, y_scores)
    axes[1].plot(rec, prec, label=f"AP = {average_precision_score(y_test, y_scores):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Readmit", "Readmit"]).plot(
        ax=axes[2], colorbar=True, cmap=plt.cm.Blues
    )
    axes[2].set_title(f"Confusion Matrix (threshold={threshold})")

    plt.tight_layout()
    plt.show()


def evaluate_model(y_test, y_pred_proba, threshold=0.5, plot=True):
    """
    Compute metrics and optionally display evaluation plots. Combines metrics calculation 
    and plot step as one function. 
    """
    metrics = compute_metrics(y_test, y_pred_proba, threshold=threshold)

    if plot:
        plot_evaluation(y_test, y_pred_proba, threshold=threshold)

    return metrics