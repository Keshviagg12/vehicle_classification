"""
models/mlp_fe.py — Multi-Layer Perceptron for the feature-engineered (FE)
dataset.  Uses the same modular style as cnn_lstm.py.

Architecture
────────────
Input (44,)
  Dense(128) → BN → ReLU → Dropout(0.3)
  Dense(64)  → BN → ReLU → Dropout(0.3)
  Dense(32)  → BN → ReLU → Dropout(0.2)
  Dense(3)   → Softmax

Author: Keshvi Agarwal
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Activation
from tensorflow.keras.regularizers import l2

import config


def build_mlp_fe(
    n_features: int  = config.N_FE_FEATS,
    n_classes:  int  = config.N_CLASSES,
    lr:         float = config.LEARNING_RATE,
) -> Model:
    """
    Build and compile an MLP suited for the 44-feature FE dataset.

    Parameters
    ----------
    n_features : number of input features
    n_classes  : number of output classes
    lr         : Adam learning rate

    Returns
    -------
    Compiled tf.keras.Model
    """
    init = tf.keras.initializers.GlorotUniform(seed=config.SEED)
    reg  = l2(1e-4)

    inp = Input(shape=(n_features,), name="fe_input")

    x = Dense(128, kernel_initializer=init, kernel_regularizer=reg, name="fc1")(inp)
    x = BatchNormalization(name="bn1")(x)
    x = Activation("relu")(x)
    x = Dropout(0.30)(x)

    x = Dense(64, kernel_initializer=init, kernel_regularizer=reg, name="fc2")(x)
    x = BatchNormalization(name="bn2")(x)
    x = Activation("relu")(x)
    x = Dropout(0.30)(x)

    x = Dense(32, kernel_initializer=init, kernel_regularizer=reg, name="fc3")(x)
    x = BatchNormalization(name="bn3")(x)
    x = Activation("relu")(x)
    x = Dropout(0.20)(x)

    out = Dense(n_classes, activation="softmax",
                kernel_initializer=init, name="output")(x)

    model = Model(inp, out, name="MLP_FE_VehicleClassifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
