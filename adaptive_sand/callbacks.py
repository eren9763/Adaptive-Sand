"""
Adaptive SAND Pruning Callbacks

Classes
-------
_BasePlateauPruningCallback
    Shared plateau-detection and accept / reject logic.

SANDSmartPruningCallback
    First-order (gradient) plateau pruning for A-SAND.

SANDSmartHybridPruningCallback
    First + second-order plateau pruning for A-SAND-Hybrid.
"""

import numpy as np
import tensorflow as tf


class _BasePlateauPruningCallback(tf.keras.callbacks.Callback):
    """Plateau-aware pruning callback with accept/reject safety.

    A pruning step is triggered only when:
    - Training has passed ``start_epoch``.
    - The configured ``interval`` of epochs has elapsed since the last trigger.
    - A ``cooldown`` gap since the last accepted prune has been observed.
    - Validation accuracy has plateaued (max change ≤ ``plateau_tol`` over
      the last ``plateau_window`` epochs).
    - The current target-k has not been rejected more than ``max_rejections``
      times.

    After pruning, the next epoch's score is compared to the pre-prune score.
    If it drops by more than ``tolerance``, the prune is rolled back.

    Parameters
    ----------
    start_epoch : int
        Epoch from which pruning may begin.
    interval : int
        Minimum number of epochs between pruning checks.
    tolerance : float
        Maximum score drop tolerated before a prune is rejected.
    lambda_k : float
        Sparsity regularisation coefficient in the composite score.
    plateau_window : int
        Number of consecutive epochs used to detect a plateau.
    plateau_tol : float
        Maximum per-epoch accuracy change considered a plateau.
    cooldown : int
        Minimum epochs to wait after a prune before the next check.
    max_rejections : int
        How many times a specific target-k can be rejected before being abandoned.
    prune_rate : float
        Fraction of active features to prune per step.
    """

    def __init__(
        self,
        start_epoch: int = 5,
        interval: int = 4,
        tolerance: float = 0.005,
        lambda_k: float = 0.15,
        plateau_window: int = 3,
        plateau_tol: float = 0.003,
        cooldown: int = 3,
        max_rejections: int = 3,
        prune_rate: float = 0.10,
    ):
        super().__init__()
        self.start_epoch = start_epoch
        self.interval = interval
        self.tolerance = tolerance
        self.lambda_k = lambda_k
        self.plateau_window = plateau_window
        self.plateau_tol = plateau_tol
        self.cooldown = cooldown
        self.max_rejections = max_rejections
        self.prune_rate = prune_rate

        # State
        self.best_score: float = -np.inf
        self.pending_prune: bool = False
        self.backup_mask = self.backup_k = self.backup_weights = None
        self.backup_score: float = -np.inf
        self.last_prune_epoch: int = -999
        self.rejected_targets: dict = {}
        self.input_dim: int = None
        self.full_acc_history: list = []
        self.prune_epochs: list = []
        self.best_weights = None
        self.best_mask = None
        self.best_k = None

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def compute_score(self, val_acc: float, k: int) -> float:
        return val_acc - self.lambda_k * (k / self.input_dim) ** 1.5

    # ------------------------------------------------------------------
    # Plateau detection
    # ------------------------------------------------------------------

    def is_plateaued(self) -> bool:
        if len(self.full_acc_history) <= self.plateau_window + 1:
            return False
        recent = self.full_acc_history[-(self.plateau_window + 1):]
        diffs = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        return max(diffs) <= self.plateau_tol

    # ------------------------------------------------------------------
    # Keras hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, logs=None):
        self.input_dim = self.model.selector.num_features

    def _do_prune(self, selector, epoch: int):
        raise NotImplementedError

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        val_acc = float(logs.get("val_accuracy", 0.0))
        self.full_acc_history.append(val_acc)

        selector = self.model.selector
        current_k = int(selector.k.numpy())
        current_score = self.compute_score(val_acc, current_k)

        # ---- Accept / reject last prune ----
        if self.pending_prune:
            if current_score >= self.backup_score - self.tolerance:
                # Accept
                self.pending_prune = False
                self.rejected_targets.pop(int(self.backup_k), None)
                if self.best_weights is None or current_score > self.best_score:
                    self.save_best_state(selector, current_score)
            else:
                # Reject: full rollback
                self.model.set_weights(self.backup_weights)
                selector.mask.assign(self.backup_mask.astype(np.float32))
                selector.k.assign(self.backup_k)
                selector.enforce_mask()

                old_k = int(self.backup_k)
                self.rejected_targets[old_k] = self.rejected_targets.get(old_k, 0) + 1
                self.pending_prune = False
            return

        # ---- Track best ----
        if self.best_weights is None or current_score > self.best_score:
            self.save_best_state(selector, current_score)

        # ---- Check whether to prune ----
        rejection_count = self.rejected_targets.get(current_k, 0)
        can_prune = (
            epoch >= self.start_epoch
            and (epoch - self.start_epoch) % self.interval == 0
            and (epoch - self.last_prune_epoch) >= self.cooldown
            and self.is_plateaued()
            and rejection_count < self.max_rejections
            and current_k > 1
        )

        if not can_prune:
            return

        # ---- Backup and prune ----
        self.backup_weights = [w.copy() for w in self.model.get_weights()]
        self.backup_mask = selector.mask.numpy().copy()
        self.backup_k = int(selector.k.numpy())
        self.backup_score = self.compute_score(val_acc, self.backup_k)

        self._do_prune(selector, epoch)

        self.last_prune_epoch = epoch
        self.prune_epochs.append(epoch)
        self.pending_prune = True

    def on_train_end(self, logs=None):
        self.restore_best_state()

    # ------------------------------------------------------------------
    # Best-state helpers
    # ------------------------------------------------------------------

    def save_best_state(self, selector, score: float):
        self.best_score = float(score)
        self.best_weights = [w.copy() for w in self.model.get_weights()]
        self.best_mask = selector.mask.numpy().copy()
        self.best_k = int(selector.k.numpy())

    def restore_best_state(self):
        if self.best_weights is None:
            raise RuntimeError("Best model state was never initialised.")
        self.model.set_weights(self.best_weights)
        selector = self.model.selector
        selector.mask.assign(self.best_mask.astype(np.float32))
        selector.k.assign(self.best_k)
        selector.enforce_mask()


class SANDSmartPruningCallback(_BasePlateauPruningCallback):
    """First-order (gradient-based) plateau pruning callback for A-SAND."""

    def _do_prune(self, selector, epoch: int):
        selector.prune_by_gradient(prune_rate=self.prune_rate)


class SANDSmartHybridPruningCallback(_BasePlateauPruningCallback):
    """First + second-order plateau pruning callback for A-SAND-Hybrid.

    Parameters
    ----------
    hybrid_alpha : float
        Mixing weight for first-order score (0 = pure Hessian, 1 = pure gradient).
    """

    def __init__(self, hybrid_alpha: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.hybrid_alpha = hybrid_alpha

    def _do_prune(self, selector, epoch: int):
        selector.prune_by_hybrid(alpha=self.hybrid_alpha, prune_rate=self.prune_rate)
