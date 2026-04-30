"""
train.py — Entry point to train and evaluate vehicle classification models.

Usage
─────
  python train.py --mode raw      # Train CNN-BiLSTM on raw 3-axis signals
  python train.py --mode fe       # Train MLP on feature-engineered data
  python train.py --mode both     # Train and compare both (default)

Author: Keshvi Agarwal
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# ── Add project root to path (works from any working directory) ───────────────
sys.path.insert(0, os.path.dirname(__file__))

import config
from utils.seed        import set_global_seed
from utils.data_loader import load_raw_dataset, load_fe_dataset
from utils.metrics     import evaluate, plot_training_curves
from models.cnn_lstm   import build_cnn_lstm, get_callbacks, compute_class_weights
from models.mlp_fe     import build_mlp_fe


def train_raw_model(save: bool = True) -> dict:
    """Train the CNN-BiLSTM model on the raw 3-axis magnetic time-series."""
    print("\n" + "═" * 58)
    print("  CNN-BiLSTM  ·  Raw 3-axis Magnetic Signal")
    print("═" * 58)

    data = load_raw_dataset()

    model = build_cnn_lstm()
    model.summary()

    cw = compute_class_weights(data["y_train"])
    print(f"\n  Class weights: {cw}\n")

    history = model.fit(
        data["X_train"], data["yohe_train"],
        validation_data=(data["X_val"], data["yohe_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=cw,
        callbacks=get_callbacks("val_accuracy"),
        verbose=1,
    )

    plot_training_curves(history, tag="CNN_BiLSTM")

    y_pred = np.argmax(model.predict(data["X_test"]), axis=1)
    metrics = evaluate(
        data["y_test"], y_pred,
        label_encoder=data["label_encoder"],
        tag="CNN_BiLSTM",
    )

    if save:
        path = os.path.join(config.MODELS_DIR, "cnn_bilstm_vehicle.keras")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        model.save(path)
        print(f"  [saved] model → {path}")

    return metrics


def train_fe_model(save: bool = True) -> dict:
    """Train the MLP model on the 44-feature engineered dataset."""
    print("\n" + "═" * 58)
    print("  MLP  ·  44 Feature-Engineered Inputs")
    print("═" * 58)

    data = load_fe_dataset()

    model = build_mlp_fe()
    model.summary()

    from models.cnn_lstm import get_callbacks, compute_class_weights
    cw = compute_class_weights(data["y_train"])
    print(f"\n  Class weights: {cw}\n")

    history = model.fit(
        data["X_train"], data["yohe_train"],
        validation_data=(data["X_val"], data["yohe_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        class_weight=cw,
        callbacks=get_callbacks("val_accuracy"),
        verbose=1,
    )

    plot_training_curves(history, tag="MLP_FE")

    y_pred = np.argmax(model.predict(data["X_test"]), axis=1)
    metrics = evaluate(
        data["y_test"], y_pred,
        label_encoder=data["label_encoder"],
        tag="MLP_FE",
    )

    if save:
        path = os.path.join(config.MODELS_DIR, "mlp_fe_vehicle.keras")
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        model.save(path)
        print(f"  [saved] model → {path}")

    return metrics


def compare_models(raw_metrics: dict, fe_metrics: dict) -> None:
    """Print a side-by-side comparison table."""
    header  = f"{'Metric':<14} {'CNN-BiLSTM (raw)':>18} {'MLP (FE)':>12}"
    divider = "─" * len(header)
    print(f"\n{divider}")
    print("  Model Comparison")
    print(divider)
    print(header)
    print(divider)
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"  {key.capitalize():<12} {raw_metrics[key]:>18.4f} {fe_metrics[key]:>12.4f}")
    print(divider + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vehicle Type Classifier — Training")
    parser.add_argument(
        "--mode", choices=["raw", "fe", "both"], default="both",
        help="Which model(s) to train (default: both)"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip saving trained models to disk"
    )
    args = parser.parse_args()

    set_global_seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    raw_m = fe_m = None

    if args.mode in ("raw", "both"):
        raw_m = train_raw_model(save=not args.no_save)

    if args.mode in ("fe", "both"):
        fe_m = train_fe_model(save=not args.no_save)

    if raw_m and fe_m:
        compare_models(raw_m, fe_m)


if __name__ == "__main__":
    main()
