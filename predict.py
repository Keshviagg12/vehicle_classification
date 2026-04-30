"""
predict.py — Load a saved model and run inference on new samples.

Usage
─────
  # Predict a single CSV row (621 semicolon-separated raw values)
  python predict.py --model models/cnn_bilstm_vehicle.keras \\
                    --input sample.csv --mode raw

  # Predict from a 44-column FE CSV
  python predict.py --model models/mlp_fe_vehicle.keras \\
                    --input fe_sample.csv --mode fe

Author: Keshvi Agarwal
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(__file__))
import config
from utils.seed import set_global_seed


CLASS_NAMES = ["motorcycle", "passenger_car", "truck"]


def predict_raw(model_path: str, csv_path: str) -> None:
    """Run inference on raw 3-axis magnetic sensor data."""
    model = tf.keras.models.load_model(model_path)

    df = pd.read_csv(csv_path, header=None)
    if ";" in str(df.iloc[0, 0]):
        df = df[0].str.split(";", expand=True).iloc[:, :config.N_RAW_FEATS].astype(float)

    X = df.values.astype(float)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(-1, config.N_TIMESTEPS, config.N_AXES)

    probs  = model.predict(X)
    labels = np.argmax(probs, axis=1)

    for i, (label, prob) in enumerate(zip(labels, probs)):
        print(f"  Sample {i+1:3d} → {CLASS_NAMES[label]:>15s}  "
              f"(confidence: {prob[label]*100:.1f}%)")


def predict_fe(model_path: str, csv_path: str) -> None:
    """Run inference on feature-engineered data."""
    model = tf.keras.models.load_model(model_path)

    df = pd.read_csv(csv_path, header=None)
    X  = df.values[:, :config.N_FE_FEATS].astype(float)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    probs  = model.predict(X)
    labels = np.argmax(probs, axis=1)

    for i, (label, prob) in enumerate(zip(labels, probs)):
        print(f"  Sample {i+1:3d} → {CLASS_NAMES[label]:>15s}  "
              f"(confidence: {prob[label]*100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vehicle Type Classifier — Inference")
    parser.add_argument("--model", required=True, help="Path to .keras saved model")
    parser.add_argument("--input", required=True, help="CSV file with input samples")
    parser.add_argument("--mode",  choices=["raw", "fe"], default="raw",
                        help="Input format: raw time-series or FE features")
    args = parser.parse_args()

    set_global_seed(config.SEED)

    print(f"\n  Model   : {args.model}")
    print(f"  Input   : {args.input}")
    print(f"  Mode    : {args.mode}")
    print()

    if args.mode == "raw":
        predict_raw(args.model, args.input)
    else:
        predict_fe(args.model, args.input)


if __name__ == "__main__":
    main()
