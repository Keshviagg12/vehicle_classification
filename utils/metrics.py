"""
utils/metrics.py — Evaluation helpers: classification report, confusion
matrix, per-class F1 bars, and training-curve plots.

Author: Keshvi Agarwal
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import config


# ── Palette ───────────────────────────────────────────────────────────────────
_PALETTE = {
    "bg":      "#0f1117",
    "panel":   "#1a1d27",
    "accent1": "#6c63ff",
    "accent2": "#00d4aa",
    "accent3": "#ff6b6b",
    "text":    "#e8e8f0",
    "muted":   "#6c7a89",
}

plt.rcParams.update({
    "figure.facecolor":  _PALETTE["bg"],
    "axes.facecolor":    _PALETTE["panel"],
    "axes.edgecolor":    _PALETTE["muted"],
    "axes.labelcolor":   _PALETTE["text"],
    "xtick.color":       _PALETTE["text"],
    "ytick.color":       _PALETTE["text"],
    "text.color":        _PALETTE["text"],
    "grid.color":        "#2a2d3a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
})


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder,
    tag: str = "",
    save_dir: str = config.RESULTS_DIR,
) -> dict:
    """
    Compute and print all classification metrics, then save plots.

    Parameters
    ----------
    y_true, y_pred : integer-encoded ground truth and predictions
    label_encoder  : fitted sklearn LabelEncoder
    tag            : filename prefix for saved figures
    save_dir       : directory to write PNGs into

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
    """
    os.makedirs(save_dir, exist_ok=True)
    class_names = list(label_encoder.classes_)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true,   y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true,       y_pred, average="weighted", zero_division=0)

    print(f"\n{'─'*52}")
    print(f"  Evaluation results  [{tag}]")
    print(f"{'─'*52}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}  (weighted)")
    print(f"  Recall    : {rec:.4f}  (weighted)")
    print(f"  F1-Score  : {f1:.4f}  (weighted)")
    print(f"\n  Per-class report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"{'─'*52}\n")

    _plot_confusion_matrix(y_true, y_pred, class_names, tag, save_dir)
    _plot_per_class_f1(y_true, y_pred, class_names, tag, save_dir)

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)


def _plot_confusion_matrix(
    y_true, y_pred, class_names: list, tag: str, save_dir: str
) -> None:
    cm   = confusion_matrix(y_true, y_pred)
    norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        norm,
        annot=cm,            # show raw counts …
        fmt="d",
        cmap="viridis",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        linecolor=_PALETTE["bg"],
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=8)
    ax.set_ylabel("True label",      fontsize=11, labelpad=8)
    ax.set_title(f"Confusion Matrix — {tag}", fontsize=13, pad=12, color=_PALETTE["text"])
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def _plot_per_class_f1(
    y_true, y_pred, class_names: list, tag: str, save_dir: str
) -> None:
    f1s    = f1_score(y_true, y_pred, average=None, zero_division=0)
    colors = [_PALETTE["accent1"], _PALETTE["accent2"], _PALETTE["accent3"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.barh(class_names, f1s, color=colors[:len(class_names)], height=0.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-Score", fontsize=11)
    ax.set_title(f"Per-class F1-Score — {tag}", fontsize=13, pad=10)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="x")

    for bar, val in zip(bars, f1s):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=10, color=_PALETTE["text"]
        )

    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_per_class_f1.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_training_curves(history, tag: str, save_dir: str = config.RESULTS_DIR) -> None:
    """Plot and save accuracy + loss curves from a Keras History object."""
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    metrics_pairs = [
        ("accuracy", "val_accuracy", "Accuracy", _PALETTE["accent1"], _PALETTE["accent2"]),
        ("loss",     "val_loss",     "Loss",      _PALETTE["accent3"], "#ffaa00"),
    ]

    for ax, (tr_key, val_key, title, c1, c2) in zip(axes, metrics_pairs):
        epochs = range(1, len(history.history[tr_key]) + 1)
        ax.plot(epochs, history.history[tr_key],  color=c1, lw=2, label="Train")
        ax.plot(epochs, history.history[val_key], color=c2, lw=2, label="Val",
                linestyle="--")
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Epoch")
        ax.legend(framealpha=0.3)
        ax.grid(True)

    fig.suptitle(f"Training Curves — {tag}", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, f"{tag}_training_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")
