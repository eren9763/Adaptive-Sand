"""Dataset transformation helpers."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


def normalize_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Normalize each batch independently.

    For stricter experimental reproducibility, consider computing train-set
    statistics once and applying the same statistics to train/val/test.
    """

    def _map(x, y):
        mean = tf.reduce_mean(x, axis=0)
        std = tf.math.reduce_std(x, axis=0) + 1e-6
        return (x - mean) / std, y

    return ds.map(_map)


def dataset_to_numpy(ds: tf.data.Dataset):
    """Convert a batched tf.data.Dataset to NumPy arrays."""
    x_list, y_list = [], []
    for x_batch, y_batch in ds:
        x_list.append(x_batch.numpy())
        y_list.append(y_batch.numpy())
    return np.concatenate(x_list), np.concatenate(y_list)

