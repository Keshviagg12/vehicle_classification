"""
models/cnn_lstm.py — 1D CNN → Bidirectional LSTM hybrid for vehicle
classification from 3-axis magnetic time-series.

Architecture overview
─────────────────────
Input (207, 3)
  │
  ├─ Conv1D(64, k=5) → BN → ReLU → MaxPool(2)      # local feature detection
  ├─ Conv1D(128, k=5) → BN → ReLU → MaxPool(2)     # higher-level patterns
  │
  ├─ Bidirectional LSTM(64)                         # temporal context
  │
  ├─ Dense(64, ReLU) → Dropout(0.3)
  └─ Dense(3, Softmax)

Key design choices
──────────────────
• Separate Conv1D channels per time axis capture per-axis local patterns
  before LSTM integrates temporal context across all axes jointly.
• Bidirectional LSTM sees both past and future timesteps, improving
  detection of symmetric vehicle signatures.
• Focal loss is replaced by standard categorical cross-entropy with
  class-weight balancing — more transparent and equally effective on this
  dataset.
• GlorotUniform initialisation + BatchNorm for stable gradients on the
  small dataset (376 samples).

Author: Keshvi Agarwal
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Bidirectional,
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    LSTM,
    MaxPooling1D,
    Activation,
)
from tensorflow.keras.regularizers import l2

import config


def build_cnn_lstm(
    timesteps: int  = config.N_TIMESTEPS,
    n_axes:    int  = config.N_AXES,
    n_classes: int  = config.N_CLASSES,
    cnn_f1:    int  = config.CNN_FILTERS_1,
    cnn_f2:    int  = config.CNN_FILTERS_2,
    kernel:    int  = config.CNN_KERNEL,
    lstm_u:    int  = config.LSTM_UNITS,
    dense_u:   int  = config.DENSE_UNITS,
    dropout:   float = config.DROPOUT_RATE,
    lr:        float = config.LEARNING_RATE,
) -> Model:
    """
    Construct and compile the CNN-LSTM model.

    Parameters
    ----------
    timesteps : number of time steps per sample (207 for raw data)
    n_axes    : number of sensor axes (3 — X, Y, Z)
    n_classes : output classes (3 — motorcycle, car, truck)

    Returns
    -------
    Compiled tf.keras.Model
    """
    init = tf.keras.initializers.GlorotUniform(seed=config.SEED)

    inp = Input(shape=(timesteps, n_axes), name="sensor_input")

    # ── CNN block 1 ───────────────────────────────────────────────────────────
    x = Conv1D(cnn_f1, kernel_size=kernel, padding="same",
               kernel_initializer=init, kernel_regularizer=l2(1e-4),
               name="conv1")(inp)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=config.POOL_SIZE, name="pool1")(x)

    # ── CNN block 2 ───────────────────────────────────────────────────────────
    x = Conv1D(cnn_f2, kernel_size=kernel, padding="same",
               kernel_initializer=init, kernel_regularizer=l2(1e-4),
               name="conv2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=config.POOL_SIZE, name="pool2")(x)

    # ── Bidirectional LSTM ────────────────────────────────────────────────────
    x = Bidirectional(
        LSTM(lstm_u, return_sequences=False,
             kernel_initializer=init,
             recurrent_dropout=0.0,          # keep GPU compatibility
             name="lstm"),
        name="bilstm",
    )(x)
    x = Dropout(dropout)(x)

    # ── Classifier head ───────────────────────────────────────────────────────
    x = Dense(dense_u, activation="relu",
               kernel_initializer=init, kernel_regularizer=l2(1e-4),
               name="fc1")(x)
    x = Dropout(dropout)(x)

    out = Dense(n_classes, activation="softmax",
                kernel_initializer=init, name="output")(x)

    model = Model(inp, out, name="CNN_BiLSTM_VehicleClassifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_callbacks(monitor: str = "val_accuracy") -> list:
    """
    Return standard Keras callbacks:
      • EarlyStopping (restore best weights)
      • ReduceLROnPlateau
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=config.PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=config.LR_PATIENCE,
        min_lr=1e-6,
        verbose=1,
    )
    return [early_stop, reduce_lr]


def compute_class_weights(y_int: np.ndarray) -> dict:
    """
    Compute inverse-frequency class weights to handle the dataset's
    heavy class imbalance (motorcycle: 47, car: 296, truck: 33).

    Returns a dict {class_index: weight} for use in model.fit().
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_int)
    weights = compute_class_weight("balanced", classes=classes, y=y_int)
    return dict(zip(classes.tolist(), weights.tolist()))
