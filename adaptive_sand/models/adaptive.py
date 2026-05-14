"""Adaptive SAND model.

This module contains the first-order adaptive feature selection variant.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class SANDLayerUniversal(tf.keras.layers.Layer):
    """Feature weighting and pruning layer for Adaptive SAND."""

    def __init__(self, num_inputs: int, initial_k: int | None = None):
        super().__init__()
        self.num_features = num_inputs
        self.k = tf.Variable(
            initial_k if initial_k is not None else num_inputs,
            trainable=False,
            dtype=tf.int32,
        )
        self.w = self.add_weight(
            shape=(num_inputs,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
            name="feature_weights",
        )
        self.mask = tf.Variable(tf.ones(num_inputs), trainable=False)
        self.best_mask = tf.Variable(tf.ones(num_inputs), trainable=False)
        self.best_k = tf.Variable(self.k.read_value(), trainable=False)
        self.pruning_active = True
        self.grad_accumulator = tf.Variable(tf.zeros(num_inputs), trainable=False)
        self.grad_count = tf.Variable(0.0, trainable=False)

    def enforce_mask(self) -> None:
        self.w.assign(self.w * self.mask)

    def call(self, x, training: bool = False):
        self.enforce_mask()
        w_abs = tf.minimum(tf.abs(self.w), 1.0)
        w_active = w_abs * self.mask
        norm = tf.sqrt(tf.reduce_sum(tf.square(w_active)) + 1e-8)
        wn = (w_active / norm) * tf.sqrt(tf.cast(self.k, tf.float32))
        x_weighted = x * wn

        if training:
            noise_std = 0.5 * tf.abs(1 - wn)
            noise = tf.random.normal(tf.shape(x_weighted), stddev=noise_std)
            return x_weighted + noise

        return x_weighted

    def record_gradients(self, grad) -> None:
        if grad is not None and self.pruning_active:
            importance = tf.abs(grad * self.w)
            self.grad_accumulator.assign_add(importance)
            self.grad_count.assign_add(1.0)

    def save_current_as_best(self) -> None:
        self.best_mask.assign(self.mask)
        self.best_k.assign(self.k)

    def restore_best(self) -> None:
        self.mask.assign(self.best_mask)
        self.k.assign(self.best_k)
        self.enforce_mask()

    def reset_importance_buffers(self) -> None:
        self.grad_accumulator.assign(tf.zeros(self.num_features))
        self.grad_count.assign(0.0)

    def get_first_order_scores(self) -> np.ndarray:
        current_mask = self.mask.numpy()
        if self.grad_count.numpy() > 0:
            scores = (self.grad_accumulator / self.grad_count).numpy()
        else:
            scores = np.abs(self.w.numpy())
        scores[current_mask == 0] = -np.inf
        return scores

    def prune_by_gradient(self, force_target_k: int | None = None) -> bool:
        if not self.pruning_active:
            return False

        current_mask = self.mask.numpy()
        k_old = int(np.sum(current_mask))
        scores = self.get_first_order_scores()

        if force_target_k is not None:
            num_to_keep = max(1, min(int(force_target_k), k_old))
        else:
            pruning_rate = 0.1
            num_to_keep = max(1, k_old - int(pruning_rate * k_old))

        top_indices = np.argsort(scores)[-num_to_keep:]
        new_mask = np.zeros_like(current_mask)
        new_mask[top_indices] = 1.0

        self.mask.assign(new_mask.astype(np.float32))
        self.k.assign(int(np.sum(new_mask)))
        self.enforce_mask()
        self.reset_importance_buffers()

        print(f"\nPruning: {k_old} -> {self.k.numpy()}")
        return True

    def get_feature_importance(self) -> dict:
        return {
            "num_selected": int(np.sum(self.mask.numpy())),
            "mask": self.mask.numpy(),
        }


class SANDAdaptiveModel(tf.keras.Model):
    """Adaptive SAND model with first-order pruning signals."""

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        initial_k: int | None = None,
        layer_sequence: list[int] | None = None,
        is_classification: bool = True,
        learning_rate: float = 1e-4,
        decay_steps: int = 250,
        decay_rate: float = 1.0,
        alpha: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()
        layer_sequence = layer_sequence or [67]

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.is_classification = is_classification
        self.selector = SANDLayerUniversal(num_inputs, initial_k)
        self.batch_norm = batch_norm

        if batch_norm:
            self.bn = tf.keras.layers.BatchNormalization()

        self.hidden = [
            tf.keras.layers.Dense(dim, activation=tf.keras.layers.LeakyReLU(alpha=alpha))
            for dim in layer_sequence
        ]

        if is_classification:
            self.out = tf.keras.layers.Dense(num_outputs, activation="softmax")
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
            self.acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        else:
            self.out = tf.keras.layers.Dense(num_outputs)
            self.loss_fn = tf.keras.losses.MeanSquaredError()
            self.acc_metric = None

        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=False,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        metrics = [self.loss_tracker]
        if self.acc_metric is not None:
            metrics.append(self.acc_metric)
        return metrics

    def call(self, x, training: bool = False):
        x = self.selector(x, training=training)
        if self.batch_norm:
            x = self.bn(x, training=training)
        for layer in self.hidden:
            x = layer(x)
        return self.out(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_fn(y, y_pred)

        train_vars = self.trainable_variables
        grads = tape.gradient(loss, train_vars)

        for var, grad in zip(train_vars, grads):
            if "feature_weights" in var.name and grad is not None:
                self.selector.record_gradients(grad)

        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.selector.enforce_mask()

        self.loss_tracker.update_state(loss)
        if self.acc_metric is not None:
            self.acc_metric.update_state(y, y_pred)

        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.loss_fn(y, y_pred)

        self.loss_tracker.update_state(loss)
        if self.acc_metric is not None:
            self.acc_metric.update_state(y, y_pred)

        return {metric.name: metric.result() for metric in self.metrics}

    def get_feature_importance(self) -> dict:
        return self.selector.get_feature_importance()

