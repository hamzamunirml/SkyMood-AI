"""
Model Evaluation Module
Computes classification metrics and confusion matrices.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a fitted model on the test set.

    Returns a dict with accuracy, weighted precision/recall/F1, and predictions.
    """
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, preds, average="weighted", zero_division=0
    )
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "predictions": preds,
    }


def evaluate_all(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate every model in `models` and return a comparison DataFrame
    sorted by F1-Score (descending). Predictions are dropped from the
    returned table (kept only internally for confusion matrices).
    """
    rows = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        rows.append(result)

    df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return df


def print_classification_report(y_test, preds, model_name: str):
    print(f"\n{'='*60}\nClassification Report — {model_name}\n{'='*60}")
    print(classification_report(y_test, preds, zero_division=0))


def plot_confusion_matrix(y_test, preds, model_name: str, labels=None, save_path=None):
    """Plot (and optionally save) a confusion matrix heatmap for one model."""
    cm = confusion_matrix(y_test, preds, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
