"""
Model evaluation utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


def evaluate_wafer_classification(model, X_test, y_test, threshold=0.5):
    """
    Evaluate wafer classification model

    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
        threshold: Classification threshold

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Make predictions
    y_prob = model.predict(X_test)
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = (y_pred == y_test).mean()

    true_positives = ((y_pred == 1) & (y_test == 1)).sum()
    false_positives = ((y_pred == 1) & (y_test == 0)).sum()
    true_negatives = ((y_pred == 0) & (y_test == 0)).sum()
    false_negatives = ((y_pred == 0) & (y_test == 1)).sum()

    metrics['precision'] = true_positives / (true_positives + false_positives) if (
                                                                                              true_positives + false_positives) > 0 else 0
    metrics['recall'] = true_positives / (true_positives + false_negatives) if (
                                                                                           true_positives + false_negatives) > 0 else 0
    metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (
                metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
    metrics['specificity'] = true_negatives / (true_negatives + false_positives) if (
                                                                                                true_negatives + false_positives) > 0 else 0

    # Print metrics
    print("Model Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

    return metrics, y_prob, y_pred


def plot_confusion_matrix(y_true, y_pred, class_names=['No Scratch', 'Scratch'], figsize=(8, 6)):
    """
    Plot confusion matrix

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_prob, figsize=(8, 6)):
    """
    Plot ROC curve

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        figsize: Figure size
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def plot_precision_recall_curve(y_true, y_prob, figsize=(8, 6)):
    """
    Plot precision-recall curve

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        figsize: Figure size
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.show()