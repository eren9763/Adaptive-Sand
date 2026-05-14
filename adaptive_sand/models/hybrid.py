"""Hybrid Adaptive SAND model.

This variant combines first-order gradient importance with a diagonal
second-order approximation for pruning.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from adaptive_sand.models.adaptive import SANDAdaptiveModel, SANDLayerUniversal


class SANDLayerUniversalHybrid(SANDLayerUniversal):
    """Adaptive SAND selector with first-order and second-order buffers."""

    def __init__(self, num_inputs: int, initial_k: int | None = None):
        super().__init__(num_inputs=num_inputs, initial_k=initial_k)
        self.hdiag_accumulator = tf.Variable(tf.zeros(num_inputs), trainable=False)
        self.hdiag_count = tf.Variable(0.0, trainable=False)

    def record_hdiag(self, hdiag) -> None:
        if hdiag is not None and self.pruning_active:
            hdiag = tf.where(tf.math.is_finite(hdiag), hdiag, tf.zeros_like(hdiag))
            hdiag = tf.maximum(hdiag, 0.0)
            self.hdiag_accumulator.assign_add(hdiag)
            self.hdiag_count.assign_add(1.0)

    def reset_importance_buffers(self) -> None:
        super().reset_importance_buffers()
        self.hdiag_accumulator.assign(tf.zeros(self.num_features))
        self.hdiag_count.assign(0.0)

    def get_second_order_scores(self) -> np.ndarray:
        current_mask = self.mask.numpy()
        if self.hdiag_count.numpy() > 0:
            hdiag = (self.hdiag_accumulator / self.hdiag_count).numpy()
        else:
            hdiag = np.ones_like(self.w.numpy())

        w_np = self.w.numpy()
        scores = 0.5 * hdiag * (w_np**2)
        scores[current_mask == 0] = -np.inf
        return scores

    @staticmethod
    def _zscore_finite(scores: np.ndarray) -> np.ndarray:
        finite = scores[np.isfinite(scores)]
        if len(finite) > 0 and np.std(finite) > 1e-12:
            return (scores - np.nanmean(finite)) / (np.nanstd(finite) + 1e-8)
        return scores

    def prune_by_hybrid(
        self,
        alpha: float = 0.7,
        force_target_k: int | None = None,
    ) -> bool:
        if not self.pruning_active:
            return False

        current_mask = self.mask.numpy()
        k_old = int(np.sum(current_mask))

        g_scores = self._zscore_finite(self.get_first_order_scores())
        h_scores = self._zscore_finite(self.get_second_order_scores())
        scores = alpha * g_scores + (1.0 - alpha) * h_scores
        scores[current_mask == 0] = -np.inf

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

        print(f"\nHybrid pruning: {k_old} -> {self.k.numpy()}")
        return True


class SANDAdaptiveModelHybrid(SANDAdaptiveModel):
    """Adaptive SAND model with hybrid first-order/second-order pruning."""

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
        super().__init__(
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            initial_k=initial_k,
            layer_sequence=layer_sequence,
            is_classification=is_classification,
            learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            alpha=alpha,
            batch_norm=batch_norm,
        )
        self.selector = SANDLayerUniversalHybrid(num_inputs, initial_k)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape_vars:
            tape_vars.watch(self.selector.w)
            with tf.GradientTape() as tape_first:
                y_pred = self(x, training=True)
                loss = self.loss_fn(y, y_pred)
            grad_w = tape_first.gradient(loss, self.selector.w)

        train_vars = self.trainable_variables
        grads = tape_vars.gradient(loss, train_vars)
        hdiag = tape_vars.gradient(grad_w, self.selector.w) if grad_w is not None else None
        del tape_vars

        for var, grad in zip(train_vars, grads):
            if "feature_weights" in var.name and grad is not None:
                self.selector.record_gradients(grad)

        if hdiag is not None:
            self.selector.record_hdiag(hdiag)

        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.selector.enforce_mask()

        self.loss_tracker.update_state(loss)
        if self.acc_metric is not None:
            self.acc_metric.update_state(y, y_pred)

        return {metric.name: metric.result() for metric in self.metrics}
