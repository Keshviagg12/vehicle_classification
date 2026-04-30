"""
config.py — Central configuration for all project settings.
Author: Keshvi Agarwal
"""

# ── Data ──────────────────────────────────────────────────────────────────────
RAW_CSV      = "data/class3.csv"
FE_CSV       = "data/class3_FE.csv"
RESULTS_DIR  = "results"
MODELS_DIR   = "models"

# Sensor / Signal structure
N_TIMESTEPS  = 207        # raw signal length per axis
N_AXES       = 3          # X, Y, Z magnetic axes
N_RAW_FEATS  = 621        # N_TIMESTEPS * N_AXES
N_FE_FEATS   = 44         # engineered features kept (6 dropped as in original)
N_CLASSES    = 3

CLASS_NAMES  = ["motorcycle", "passenger_car", "truck"]   # L, M, H

# ── Preprocessing ─────────────────────────────────────────────────────────────
TEST_SIZE    = 0.30
VAL_SIZE     = 0.15       # fraction of train set used for validation
RANDOM_STATE = 42

# ── CNN-LSTM Model ────────────────────────────────────────────────────────────
CNN_FILTERS_1   = 64
CNN_FILTERS_2   = 128
CNN_KERNEL      = 5
POOL_SIZE       = 2
LSTM_UNITS      = 64
DENSE_UNITS     = 64
DROPOUT_RATE    = 0.30

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS          = 80
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-3
PATIENCE        = 12      # early stopping
LR_PATIENCE     = 5       # reduce-LR-on-plateau patience

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
