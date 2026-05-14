"""Baseline model builders."""

from __future__ import annotations

import tensorflow as tf


def build_lasso_mlp(input_dim: int, num_classes: int, hidden_dim: int = 67):
    """Build the MLP trained after LASSO feature selection."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )


def build_original_sand_model(
    num_features: int,
    num_classes: int,
    num_inputs_to_select: int = 60,
    hidden_layers: list[int] | None = None,
    learning_rate: float = 1e-4,
    decay_steps: int = 250,
    decay_rate: float = 1.0,
    alpha: float = 0.0,
    batch_norm: bool = False,
):
    """Build Original SAND from the external `sand` package."""
    from sand.experiments.models.mlp_sand import SANDModel

    return SANDModel(
        num_inputs=num_features,
        num_inputs_to_select=num_inputs_to_select,
        sigma=1.5,
        layer_sequence=hidden_layers or [67],
        is_classification=True,
        num_classes=num_classes,
        learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        alpha=alpha,
        batch_norm=batch_norm,
    )


def build_sequential_attention_model(
    num_features: int,
    num_classes: int,
    num_inputs_to_select: int = 60,
    num_train_steps: int = 1000,
    hidden_layers: list[int] | None = None,
    learning_rate: float = 1e-4,
    decay_steps: int = 250,
    decay_rate: float = 1.0,
    alpha: float = 0.0,
    batch_norm: bool = False,
):
    """Build Sequential Attention from the external `sand` package."""
    from sand.experiments.models.mlp_sa import SequentialAttentionModel

    return SequentialAttentionModel(
        num_inputs=num_features,
        num_inputs_to_select=num_inputs_to_select,
        num_train_steps=num_train_steps,
        num_inputs_to_select_per_step=1,
        layer_sequence=hidden_layers or [67],
        is_classification=True,
        num_classes=num_classes,
        learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        alpha=alpha,
        batch_norm=batch_norm,
    )

