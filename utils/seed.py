"""
utils/seed.py — Global seed setter for full reproducibility.
Author: Keshvi Agarwal
"""

import os
import random
import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
