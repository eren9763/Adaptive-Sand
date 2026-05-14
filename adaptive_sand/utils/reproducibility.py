"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42) -> None:
    """Set Python, NumPy and TensorFlow seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

