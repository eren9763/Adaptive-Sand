"""
Adaptive SAND Feature Selection Layers

Classes
-------
SAND_Layer_Universal
    First-order (gradient-based) adaptive feature selection layer (A-SAND).

SANDLayerUniversalHybrid
    First + second-order (Hutchinson diagonal Hessian) adaptive feature
    selection layer (A-SAND-Hybrid).
"""

import numpy as np
import tensorflow as tf


# ============================================================
# A-SAND: First-order layer
# ============================================================

class SAND_Layer_Universal(tf.keras.layers.Layer):
    """Adaptive feature selection layer using first-order gradient importance.

    Parameters
    ----------
    num_inputs : int
        Total number of input features.
    initial_k : int, optional
        Initial number of features to keep. Defaults to num_inputs.
    seed : int, optional
        Random seed for weight initialisation and noise RNG. Default 42.
    """

    def __init__(self, num_inputs: int, initial_k: int = None, seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.num_features = int(num_inputs)

        self.k = tf.Variable(
            initial_k if initial_k is not None else self.num_features,
            trainable=False,
            dtype=tf.int32,
            name="selected_feature_count",
        )

        self.w = self.add_weight(
            shape=(self.num_features,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=seed),
            trainable=True,
            name="feature_weights",
        )

        self.noise_rng = tf.random.Generator.from_seed(seed)

        self.mask = tf.Variable(
            tf.ones(self.num_features, dtype=tf.float32),
            trainable=False,
            name="feature_mask",
        )
        self.best_mask = tf.Variable(
            tf.ones(self.num_features, dtype=tf.float32),
            trainable=False,
            name="best_feature_mask",
        )
        self.best_k = tf.Variable(
            initial_k if initial_k is not None else self.num_features,
            trainable=False,
            dtype=tf.int32,
            name="best_selected_feature_count",
        )

        self.pruning_active = True
        self.grad_accumulator = tf.Variable(
            tf.zeros(self.num_features), trainable=False, name="gradient_accumulator"
        )
        self.grad_count = tf.Variable(0.0, trainable=False, name="gradient_count")
        self.prune_rate = 0.1

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def enforce_mask(self):
        """Zero-out pruned weights in-place."""
        self.w.assign(self.w * self.mask)

    def call(self, x, training: bool = False):
        self.enforce_mask()

        w_abs = tf.minimum(tf.abs(self.w), 1.0)
        w_active = w_abs * self.mask
        norm = tf.sqrt(tf.reduce_sum(tf.square(w_active)) + 1e-8)
        wn = (w_active / norm) * tf.sqrt(tf.cast(self.k, tf.float32))
        x_weighted = x * wn

        if training:
            noise_std = 0.5 * tf.abs(1.0 - wn)
            noise = self.noise_rng.normal(tf.shape(x_weighted), stddev=noise_std)
            return x_weighted + noise

        return x_weighted

    # ------------------------------------------------------------------
    # Importance accumulation
    # ------------------------------------------------------------------

    def record_gradients(self, grad):
        """Accumulate absolute gradient * weight importance."""
        if grad is not None and self.pruning_active:
            importance = tf.abs(grad * self.w)
            self.grad_accumulator.assign_add(importance)
            self.grad_count.assign_add(1.0)

    # ------------------------------------------------------------------
    # State checkpointing
    # ------------------------------------------------------------------

    def save_current_as_best(self):
        self.best_mask.assign(self.mask)
        self.best_k.assign(self.k)

    def restore_best(self):
        self.mask.assign(self.best_mask)
        self.k.assign(self.best_k)
        self.enforce_mask()

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_by_gradient(self, force_target_k: int = None, prune_rate: float = 0.10) -> bool:
        """Prune features with lowest gradient-based importance scores."""
        if not self.pruning_active:
            return False

        current_mask = self.mask.numpy()
        k_old = int(np.sum(current_mask))

        scores = (
            (self.grad_accumulator / self.grad_count).numpy()
            if self.grad_count > 0
            else np.abs(self.w.numpy())
        )
        scores[current_mask == 0] = -np.inf

        if force_target_k is not None:
            num_to_keep = int(force_target_k)
        else:
            num_to_keep = max(1, k_old - int(prune_rate * k_old))

        top_idx = np.argsort(scores)[-num_to_keep:]
        new_mask = np.zeros_like(current_mask, dtype=np.float32)
        new_mask[top_idx] = 1.0

        self.mask.assign(new_mask)
        self.k.assign(int(np.sum(new_mask)))
        self.enforce_mask()

        self.grad_accumulator.assign(tf.zeros(self.num_features))
        self.grad_count.assign(0.0)

        print(f"\n✂️  Pruning: {k_old} → {self.k.numpy()}")
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict:
        return {
            "num_selected": int(np.sum(self.mask.numpy())),
            "mask": self.mask.numpy().copy(),
        }


# ============================================================
# A-SAND-Hybrid: First + second-order layer
# ============================================================

class SANDLayerUniversalHybrid(tf.keras.layers.Layer):
    """Adaptive feature selection layer combining first-order and
    Hutchinson-based diagonal Hessian (second-order) importance scores.

    Parameters
    ----------
    num_inputs : int
        Total number of input features.
    initial_k : int, optional
        Initial number of features to keep. Defaults to num_inputs.
    """

    def __init__(self, num_inputs: int, initial_k: int = None, **kwargs):
        super().__init__(**kwargs)

        self.num_features = int(num_inputs)

        self.k = tf.Variable(
            initial_k if initial_k is not None else self.num_features,
            trainable=False,
            dtype=tf.int32,
            name="selected_feature_count",
        )

        self.hessian_rng = tf.random.Generator.from_seed(2026)
        self.noise_rng = tf.random.Generator.from_seed(42)

        self.w = self.add_weight(
            shape=(self.num_features,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=42),
            trainable=True,
            name="feature_weights",
        )

        self.mask = tf.Variable(
            tf.ones(self.num_features, dtype=tf.float32),
            trainable=False,
            name="feature_mask",
        )
        self.best_mask = tf.Variable(
            tf.ones(self.num_features, dtype=tf.float32),
            trainable=False,
            name="best_feature_mask",
        )
        self.best_k = tf.Variable(
            initial_k if initial_k is not None else self.num_features,
            trainable=False,
            dtype=tf.int32,
            name="best_selected_feature_count",
        )

        self.pruning_active = True

        self.grad_accumulator = tf.Variable(
            tf.zeros(self.num_features, dtype=tf.float32),
            trainable=False,
            name="gradient_accumulator",
        )
        self.grad_count = tf.Variable(
            0.0, trainable=False, dtype=tf.float32, name="gradient_count"
        )
        self.hdiag_accumulator = tf.Variable(
            tf.zeros(self.num_features, dtype=tf.float32),
            trainable=False,
            name="hessian_diagonal_accumulator",
        )
        self.hdiag_count = tf.Variable(
            0.0, trainable=False, dtype=tf.float32, name="hessian_diagonal_count"
        )

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def enforce_mask(self):
        """Zero-out pruned weights in-place."""
        self.w.assign(self.w * self.mask)

    def call(self, x, training: bool = False):
        self.enforce_mask()

        w_abs = tf.minimum(tf.abs(self.w), 1.0)
        w_active = w_abs * self.mask
        norm = tf.sqrt(tf.reduce_sum(tf.square(w_active)) + 1e-8)
        normalized_weights = (w_active / norm) * tf.sqrt(tf.cast(self.k, tf.float32))
        x_weighted = x * normalized_weights

        if training:
            noise_std = 0.5 * tf.abs(1.0 - normalized_weights)
            noise = self.noise_rng.normal(
                shape=tf.shape(x_weighted),
                mean=0.0,
                stddev=noise_std,
                dtype=x_weighted.dtype,
            )
            return x_weighted + noise

        return x_weighted

    # ------------------------------------------------------------------
    # Importance accumulation
    # ------------------------------------------------------------------

    def record_gradients(self, gradients):
        """Accumulate absolute first-order importance for active features."""
        if gradients is None or not self.pruning_active:
            return

        gradients = tf.where(
            tf.math.is_finite(gradients), gradients, tf.zeros_like(gradients)
        )
        self.grad_accumulator.assign_add(tf.abs(gradients) * self.mask)
        self.grad_count.assign_add(1.0)

    def record_hdiag(self, hdiag):
        """Accumulate Hutchinson diagonal Hessian estimate."""
        if hdiag is None or not self.pruning_active:
            return

        hdiag = tf.where(tf.math.is_finite(hdiag), hdiag, tf.zeros_like(hdiag))
        hdiag = tf.maximum(hdiag, 0.0) * self.mask
        self.hdiag_accumulator.assign_add(hdiag)
        self.hdiag_count.assign_add(1.0)

    # ------------------------------------------------------------------
    # State checkpointing
    # ------------------------------------------------------------------

    def save_current_as_best(self):
        self.best_mask.assign(self.mask)
        self.best_k.assign(self.k)

    def restore_best(self):
        self.mask.assign(self.best_mask)
        self.k.assign(self.best_k)
        self.enforce_mask()

    def reset_importance_buffers(self):
        """Reset all accumulated importance statistics."""
        self.grad_accumulator.assign(tf.zeros(self.num_features, dtype=tf.float32))
        self.grad_count.assign(0.0)
        self.hdiag_accumulator.assign(tf.zeros(self.num_features, dtype=tf.float32))
        self.hdiag_count.assign(0.0)

    # ------------------------------------------------------------------
    # Score computation
    # ------------------------------------------------------------------

    def _first_order_scores(self) -> np.ndarray:
        mask = self.mask.numpy()
        if self.grad_count.numpy() > 0:
            scores = self.grad_accumulator.numpy() / self.grad_count.numpy()
        else:
            scores = np.abs(self.w.numpy())
        scores = scores.copy()
        scores[mask == 0] = -np.inf
        return scores

    def _second_order_scores(self) -> np.ndarray:
        if self.hdiag_count.numpy() > 0:
            hdiag = self.hdiag_accumulator.numpy() / self.hdiag_count.numpy()
        else:
            hdiag = np.zeros_like(self.w.numpy())
        scores = 0.5 * hdiag * np.square(self.w.numpy())
        scores = scores.copy()
        scores[self.mask.numpy() == 0] = -np.inf
        return scores

    def second_order_scores(self) -> np.ndarray:
        """Public alias for backward compatibility."""
        return self._second_order_scores()

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def _apply_prune(self, scores: np.ndarray, force_target_k: int = None, prune_rate: float = 0.10):
        current_mask = self.mask.numpy()
        k_old = int(np.sum(current_mask))

        if k_old <= 1:
            return k_old, k_old

        if force_target_k is not None:
            num_to_keep = int(force_target_k)
        else:
            num_to_prune = max(1, int(prune_rate * k_old))
            num_to_keep = k_old - num_to_prune

        num_to_keep = max(1, min(num_to_keep, k_old))

        # Deterministic tie-breaking
        tie_breaker = 1e-12 * np.arange(len(scores))
        top_idx = np.argsort(scores + tie_breaker)[-num_to_keep:]

        new_mask = np.zeros_like(current_mask, dtype=np.float32)
        new_mask[top_idx] = 1.0

        self.mask.assign(new_mask)
        self.k.assign(int(np.sum(new_mask)))
        self.enforce_mask()
        self.reset_importance_buffers()

        return k_old, int(self.k.numpy())

    def prune_by_gradient(self, force_target_k: int = None, prune_rate: float = 0.10) -> bool:
        """Prune using only first-order gradient importance."""
        if not self.pruning_active:
            return False
        k_old, k_new = self._apply_prune(
            scores=self._first_order_scores(),
            force_target_k=force_target_k,
            prune_rate=prune_rate,
        )
        print(f"\n✂️  Pruning: {k_old} → {k_new}")
        return True

    def prune_by_hybrid(self, alpha: float = 0.7, force_target_k: int = None, prune_rate: float = 0.10) -> bool:
        """Prune using a normalised mix of first- and second-order scores.

        Parameters
        ----------
        alpha : float
            Weight given to first-order score (0 = pure Hessian, 1 = pure gradient).
        """
        if not self.pruning_active:
            return False

        alpha = float(np.clip(alpha, 0.0, 1.0))

        g_scores = self._first_order_scores()
        h_scores = self._second_order_scores()

        fg = g_scores[np.isfinite(g_scores)]
        fh = h_scores[np.isfinite(h_scores)]

        g_norm = (g_scores - fg.mean()) / (fg.std() + 1e-8) if fg.size > 0 and fg.std() > 1e-12 else g_scores.copy()
        h_norm = (h_scores - fh.mean()) / (fh.std() + 1e-8) if fh.size > 0 and fh.std() > 1e-12 else h_scores.copy()

        hybrid = alpha * g_norm + (1.0 - alpha) * h_norm
        hybrid = hybrid.copy()
        hybrid[self.mask.numpy() == 0] = -np.inf

        k_old, k_new = self._apply_prune(
            scores=hybrid,
            force_target_k=force_target_k,
            prune_rate=prune_rate,
        )
        print(f"\n✂️  Hybrid Pruning: {k_old} → {k_new}")
        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict:
        return {
            "num_selected": int(np.sum(self.mask.numpy())),
            "mask": self.mask.numpy().copy(),
            "first_order_scores": self._first_order_scores(),
            "second_order_scores": self._second_order_scores(),
        }
