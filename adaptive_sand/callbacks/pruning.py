"""Pruning callbacks for Adaptive SAND models."""

from __future__ import annotations

import numpy as np
import tensorflow as tf


class BaseSmartPruningCallback(tf.keras.callbacks.Callback):
    """Shared bookkeeping for adaptive pruning callbacks."""

    prune_method_name = "prune_by_gradient"

    def __init__(
        self,
        start_epoch: int = 5,
        interval: int = 4,
        tolerance: float = 0.005,
        lambda_k: float = 0.15,
        min_val_accuracy: float = 0.95,
        cooldown: int = 3,
        max_rejections: int = 3,
    ):
        super().__init__()
        self.start_epoch = start_epoch
        self.interval = interval
        self.tolerance = tolerance
        self.lambda_k = lambda_k
        self.min_val_accuracy = min_val_accuracy
        self.cooldown = cooldown
        self.max_rejections = max_rejections

        self.best_score = -np.inf
        self.best_val_acc = 0.0
        self.pending_prune = False
        self.backup_mask = None
        self.backup_k = None
        self.backup_weights = None
        self.last_prune_epoch = -999
        self.rejected_targets = {}
        self.input_dim = None
        self.k_history = []
        self.acc_history = []
        self.score_history = []
        self.prune_epochs = []

    def compute_score(self, val_acc: float, k: int) -> float:
        ratio = k / self.input_dim
        return val_acc - self.lambda_k * (ratio**1.5)

    def on_train_begin(self, logs=None):
        self.input_dim = self.model.selector.num_features
        self.k_history.append(self.input_dim)

    def _call_prune(self, selector):
        getattr(selector, self.prune_method_name)()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get("val_accuracy", 0.0)
        selector = self.model.selector
        current_k = int(tf.reduce_sum(selector.mask).numpy())
        score = self.compute_score(val_acc, current_k)

        if self.pending_prune:
            score_at_old_k = self.compute_score(val_acc, self.backup_k)

            if score_at_old_k >= self.best_score - self.tolerance:
                print("Pruning accepted")
                self.best_score = self.compute_score(val_acc, current_k)
                self.best_val_acc = val_acc
                selector.save_current_as_best()
                self.k_history.append(current_k)
                self.rejected_targets.pop(int(self.backup_k), None)
            else:
                print("Pruning rejected; rolling back")
                selector.mask.assign(self.backup_mask)
                selector.k.assign(self.backup_k)
                selector.enforce_mask()
                self.model.set_weights(self.backup_weights)

                k_old = int(self.backup_k)
                self.rejected_targets[k_old] = self.rejected_targets.get(k_old, 0) + 1
                self.best_score = self.compute_score(val_acc, k_old)
                self.best_val_acc = val_acc

            self.pending_prune = False
            self.acc_history.append(val_acc)
            self.score_history.append(score)
            return

        if score > self.best_score:
            self.best_score = score
            self.best_val_acc = val_acc
            selector.save_current_as_best()

        k_old = current_k
        rejection_count = self.rejected_targets.get(k_old, 0)

        should_prune = (
            epoch >= self.start_epoch
            and (epoch - self.start_epoch) % self.interval == 0
            and (epoch - self.last_prune_epoch) >= self.cooldown
            and val_acc > self.min_val_accuracy
            and rejection_count < self.max_rejections
        )

        if should_prune:
            print("\nTrying pruning...")
            self.last_prune_epoch = epoch
            self.backup_mask = selector.mask.numpy().copy()
            self.backup_k = selector.k.numpy()
            self.backup_weights = self.model.get_weights()
            self._call_prune(selector)
            self.pending_prune = True
            self.prune_epochs.append(epoch)

        self.acc_history.append(val_acc)
        self.score_history.append(score)


class SANDSmartPruningCallback(BaseSmartPruningCallback):
    """First-order adaptive pruning callback."""

    prune_method_name = "prune_by_gradient"


class SANDSmartHybridPruningCallback(BaseSmartPruningCallback):
    """Hybrid first-order/second-order adaptive pruning callback."""

    prune_method_name = "prune_by_hybrid"

    def __init__(self, hybrid_alpha: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_alpha = hybrid_alpha

    def _call_prune(self, selector):
        selector.prune_by_hybrid(alpha=self.hybrid_alpha)

