"""
utils/evaluate.py
=================
All evaluation metrics, plots, and report generation for OOCSS.

Produces:
  - Confusion matrix (saved as PNG)
  - ROC curves per class (saved as PNG)
  - Training history curves (saved as PNG)
  - Full classification report printed to console
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")    # non-interactive backend — works without a display
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc as sklearn_auc,
)
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


# ── Collect all predictions from a DataLoader ──────────────────────────────────

@torch.no_grad()
def get_predictions(
    model  : nn.Module,
    loader : DataLoader,
) -> tuple:
    """
    Runs the model on all batches in loader.
    Returns:
      all_labels  — true integer class labels, shape [N]
      all_preds   — predicted class indices,   shape [N]
      all_probs   — softmax probabilities,     shape [N, 4]
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(config.DEVICE)
        logits = model(images)

        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(preds)
        all_probs.append(probs)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.vstack(all_probs)   # shape [N, 4]

    return all_labels, all_preds, all_probs


# ── Confusion matrix ───────────────────────────────────────────────────────────

def plot_confusion_matrix(
    labels : np.ndarray,
    preds  : np.ndarray,
    class_names: List[str] = config.CLASS_NAMES,
    save_path: Path = config.OUTPUT_DIR / "confusion_matrix.png",
):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # row-normalised

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=cm,           # show raw counts inside cells
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)
    ax.set_title("OOCSS — Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[✓] Confusion matrix saved → {save_path}")


# ── ROC curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(
    labels    : np.ndarray,
    probs     : np.ndarray,
    class_names: List[str] = config.CLASS_NAMES,
    save_path  : Path = config.OUTPUT_DIR / "roc_curves.png",
):
    """Plots one ROC curve per class (one-vs-rest)."""
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(labels, classes=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    for i, (cls, col) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc     = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title("OOCSS — Per-Class ROC Curves (Test Set)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[✓] ROC curves saved → {save_path}")


# ── Training history curves ────────────────────────────────────────────────────

def plot_training_history(
    history  : dict,
    save_path: Path = config.OUTPUT_DIR / "training_history.png",
):
    """Plots loss and accuracy over epochs for train and validation."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", ms=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"],   "r-o", ms=4, label="Val Loss")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Loss Curves", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", ms=4, label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"],   "r-o", ms=4, label="Val Accuracy")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_title("Accuracy Curves", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("OOCSS ResNet-50 — Training History",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[✓] Training history plot saved → {save_path}")


# ── Full evaluation report ─────────────────────────────────────────────────────

def full_evaluation_report(
    model      : nn.Module,
    test_loader: DataLoader,
    class_names: List[str] = config.CLASS_NAMES,
):
    """
    Runs all evaluation steps and prints a comprehensive report.
    Saves confusion matrix, ROC curves, and returns metrics dict.
    """
    print("\n── Running full evaluation on test set ──────────────────")
    labels, preds, probs = get_predictions(model, test_loader)

    # ── Console report ─────────────────────────────────────────────────────
    acc = (labels == preds).mean() * 100
    print(f"\nOverall Accuracy: {acc:.2f}%\n")
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    # ── Plots ──────────────────────────────────────────────────────────────
    plot_confusion_matrix(labels, preds, class_names)
    plot_roc_curves(labels, probs, class_names)

    # ── Per-class metrics dict ─────────────────────────────────────────────
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, labels=list(range(len(class_names)))
    )
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    y_bin = label_binarize(labels, classes=list(range(len(class_names))))
    auc_scores = [
        sklearn_auc(*roc_curve(y_bin[:, i], probs[:, i])[:2])
        for i in range(len(class_names))
    ]

    metrics = {}
    for i, cls in enumerate(class_names):
        metrics[cls] = {
            "precision" : round(precision[i] * 100, 2),
            "recall"    : round(recall[i]    * 100, 2),
            "f1"        : round(f1[i],              4),
            "auc"       : round(auc_scores[i],      4),
        }

    print("\n── Per-Class Summary ─────────────────────────────────────")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("─" * 50)
    for cls, m in metrics.items():
        print(f"{cls:<12} {m['precision']:>9.2f}% {m['recall']:>7.2f}% "
              f"{m['f1']:>8.4f} {m['auc']:>8.4f}")

    return metrics
