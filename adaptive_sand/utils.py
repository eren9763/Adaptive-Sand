"""
Utility functions for Adaptive-SAND experiments.

Functions
---------
set_seeds
    Global determinism setup (call once at programme start).
reset_seeds
    Re-apply seeds after tf.keras.backend.clear_session().
dataset_to_numpy
    Convert a tf.data.Dataset of (x, y) batches to NumPy arrays.
load_prepared_dataset
    Thin wrapper around the original SAND ``get_dataset`` helper.
build_classifier_mlp
    Baseline MLP without feature selection, used for comparison.
"""

import os
import random

import numpy as np
import tensorflow as tf


def set_seeds(seed: int = 42):
    """Globally fix all random seeds for reproducibility.

    Should be called **before** any dataset loading or model construction.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass


def reset_seeds(seed: int = 42):
    """Re-apply seeds after ``tf.keras.backend.clear_session()``."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:
        print(f"Determinism could not be re-applied: {exc}")


def dataset_to_numpy(ds: tf.data.Dataset):
    """Materialise a batched tf.data.Dataset into (X, y) NumPy arrays."""
    xs, ys = [], []
    for xb, yb in ds:
        xs.append(xb.numpy())
        ys.append(yb.numpy())
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def load_prepared_dataset(
    dataset_name: str,
    val_ratio: float = 0.125,
    batch_size: int = 32,
    seed: int = 42,
) -> dict:
    """Load a dataset via the original SAND ``get_dataset`` helper.

    Returns a unified dict with both tf.data.Dataset objects and raw
    NumPy arrays for convenience.

    Parameters
    ----------
    dataset_name : str
        One of the dataset keys recognised by ``sand.experiments.datasets.dataset``.
    val_ratio : float, optional
        Fraction of training data used for validation.
    batch_size : int, optional
        Mini-batch size.
    seed : int, optional
        Random seed passed to the loader.
    """
    from sand.experiments.datasets.dataset import get_dataset  # type: ignore

    data = get_dataset(
        dataname=dataset_name,
        valratio=val_ratio,
        batchsize=batch_size,
        seed=seed,
    )

    return {
        "ds_train": data["dstrain"],
        "ds_val": data["dsval"],
        "ds_test": data["dstest"],
        "x_train": data["xtrain"],
        "y_train": data["ytrain"],
        "x_val": data["xval"],
        "y_val": data["yval"],
        "x_test": data["xtest"],
        "y_test": data["ytest"],
        "num_features": data["numfeatures"],
        "num_classes": data["numclasses"],
        "is_classification": data["isclassification"],
    }


def build_classifier_mlp(
    input_dim: int,
    num_classes: int,
    hidden_units: int = 67,
    seed: int = 42,
) -> tf.keras.Model:
    """Build a simple baseline MLP (no feature selection).

    Used as a comparison point against A-SAND variants.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    num_classes : int
        Number of output classes.
    hidden_units : int, optional
        Width of the single hidden layer.
    seed : int, optional
        Initialisation seed.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(
                hidden_units,
                activation=tf.keras.layers.LeakyReLU(negative_slope=0.0),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
            ),
            tf.keras.layers.Dense(
                num_classes,
                activation="softmax",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
            ),
        ],
        name="baseline_mlp",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    return model
