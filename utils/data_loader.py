"""
utils/data_loader.py — Data loading, parsing, and preprocessing pipelines.

Supports two modes:
  • 'raw'  : 621 raw magnetic-sensor values reshaped into (207, 3) time-series
  • 'fe'   : 44 hand-crafted statistical / frequency-domain features

Author: Keshvi Agarwal
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical

import config


# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_MAP = {
    "motorcycle": "motorcycle",   # light  (class 1)
    "HSS":        "passenger_car",  # medium (class 2) — original label in CSV
    "truck":      "truck",          # heavy  (class 3)
}


def _encode_labels(train_y: np.ndarray, test_y: np.ndarray) -> tuple:
    """
    Fit a LabelEncoder on training labels only, then transform both splits.
    Returns: (int_train, int_test, one_hot_train, one_hot_test, encoder)
    """
    le = LabelEncoder()
    int_train = le.fit_transform(train_y)
    int_test  = le.transform(test_y)
    ohe_train = to_categorical(int_train, num_classes=config.N_CLASSES)
    ohe_test  = to_categorical(int_test,  num_classes=config.N_CLASSES)
    return int_train, int_test, ohe_train, ohe_test, le


# ── Raw dataset ───────────────────────────────────────────────────────────────

def load_raw_dataset() -> dict:
    """
    Load class3.csv, parse the semicolon-delimited format, normalise,
    and reshape features into (N, 207, 3) for CNN-LSTM ingestion.

    Returns a dict with keys:
        X_train, X_val, X_test  — shaped (n, 207, 3)
        y_train, y_val, y_test  — integer labels
        yohe_train, yohe_val, yohe_test — one-hot labels
        label_encoder           — fitted LabelEncoder
    """
    # ── Load & parse ──────────────────────────────────────────────────────────
    df_raw = pd.read_csv(config.RAW_CSV, header=None)
    df     = df_raw[0].str.split(";", expand=True)

    labels_raw = df[config.N_RAW_FEATS].map(lambda v: LABEL_MAP.get(v, v))
    X_raw      = df.iloc[:, :config.N_RAW_FEATS].astype(float).values

    # ── Normalise per-feature (fit only on full dataset here;
    #    for production, fit only on train — see split below) ──────────────────
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X_raw)

    # ── Train / test split (stratified) ──────────────────────────────────────
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_norm, labels_raw,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=labels_raw,
    )

    # ── Train / val split ─────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_train_full,
    )

    # ── Reshape → (N, timesteps, axes) for CNN-LSTM ───────────────────────────
    def reshape(X: np.ndarray) -> np.ndarray:
        return X.reshape(-1, config.N_TIMESTEPS, config.N_AXES)

    X_train, X_val, X_test = reshape(X_train), reshape(X_val), reshape(X_test)

    # ── Encode labels ─────────────────────────────────────────────────────────
    int_train, int_val, ohe_train, ohe_val, le = _encode_labels(
        y_train.values, y_val.values
    )
    # transform test with same encoder
    int_test  = le.transform(y_test.values)
    ohe_test  = to_categorical(int_test, num_classes=config.N_CLASSES)

    print("── Raw dataset splits ──────────────────────────────────")
    print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
    print(f"  Classes: {list(le.classes_)}")
    print()

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=int_train, y_val=int_val, y_test=int_test,
        yohe_train=ohe_train, yohe_val=ohe_val, yohe_test=ohe_test,
        label_encoder=le,
        scaler=scaler,
    )


# ── Feature-Engineered dataset ────────────────────────────────────────────────

def load_fe_dataset() -> dict:
    """
    Load class3_FE.csv (44 features), attach labels from class3.csv,
    normalise, and split.

    Returns a dict with keys:
        X_train, X_val, X_test  — shaped (n, 44)
        y_train, y_val, y_test  — integer labels
        yohe_train, yohe_val, yohe_test — one-hot labels
        label_encoder           — fitted LabelEncoder
    """
    # ── Load FE features ──────────────────────────────────────────────────────
    df_fe = pd.read_csv(config.FE_CSV, header=None)

    # ── Attach labels from raw CSV ─────────────────────────────────────────────
    df_raw    = pd.read_csv(config.RAW_CSV, header=None)
    df_parsed = df_raw[0].str.split(";", expand=True)
    labels_raw = df_parsed[config.N_RAW_FEATS].map(lambda v: LABEL_MAP.get(v, v))

    # Merge and shuffle (both DataFrames share the same row order)
    df_fe["label"] = labels_raw.values
    df_fe = df_fe.sample(frac=1, random_state=config.RANDOM_STATE).reset_index(drop=True)

    # Drop last 6 columns (cols 44-49) as per original study, keep 44 features
    feat_cols = [c for c in df_fe.columns if c != "label"][:config.N_FE_FEATS]
    X_fe      = df_fe[feat_cols].fillna(0).values
    y_fe      = df_fe["label"]

    # ── Normalise ─────────────────────────────────────────────────────────────
    scaler = MinMaxScaler()
    X_norm = scaler.fit_transform(X_fe)

    # ── Splits ────────────────────────────────────────────────────────────────
    X_tr_full, X_test, y_tr_full, y_test = train_test_split(
        X_norm, y_fe,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_fe,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr_full, y_tr_full,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_tr_full,
    )

    int_train, int_val, ohe_train, ohe_val, le = _encode_labels(
        y_train.values, y_val.values
    )
    int_test = le.transform(y_test.values)
    ohe_test = to_categorical(int_test, num_classes=config.N_CLASSES)

    print("── FE dataset splits ───────────────────────────────────")
    print(f"  Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")
    print(f"  Classes: {list(le.classes_)}")
    print()

    return dict(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=int_train, y_val=int_val, y_test=int_test,
        yohe_train=ohe_train, yohe_val=ohe_val, yohe_test=ohe_test,
        label_encoder=le,
        scaler=scaler,
    )
