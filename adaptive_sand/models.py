"""
Adaptive SAND Models

Classes
-------
SANDAdaptiveModel
    MLP with SAND_Layer_Universal (A-SAND, first-order pruning).

SANDAdaptiveModelHybrid
    MLP with SANDLayerUniversalHybrid (A-SAND-Hybrid, first + second-order pruning).
"""

import numpy as np
import tensorflow as tf

from .layers import SAND_Layer_Universal, SANDLayerUniversalHybrid


# ============================================================
# A-SAND Model
# ============================================================

class SANDAdaptiveModel(tf.keras.Model):
    """MLP with adaptive first-order feature selection (A-SAND).

    Parameters
    ----------
    num_inputs : int
        Number of input features.
    num_outputs : int
        Number of output classes (classification) or targets (regression).
    initial_k : int, optional
        Initial number of features passed to the SAND layer.
    layer_sequence : tuple of int, optional
        Hidden layer widths. Default (67,).
    is_classification : bool, optional
        If True, uses softmax output and cross-entropy loss.
    alpha : float, optional
        LeakyReLU negative slope. Default 0 (= ReLU).
    batch_norm : bool, optional
        Whether to apply BatchNormalization after the SAND layer.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        initial_k: int = None,
        layer_sequence=(67,),
        is_classification: bool = True,
        alpha: float = 0.0,
        batch_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.selector = SAND_Layer_Universal(num_inputs, initial_k)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(name="batch_normalization")

        self.hidden = [
            tf.keras.layers.Dense(
                d,
                activation=tf.keras.layers.LeakyReLU(negative_slope=alpha),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42 + i),
                bias_initializer="zeros",
                name=f"hidden_{i}",
            )
            for i, d in enumerate(layer_sequence)
        ]

        if is_classification:
            self.out = tf.keras.layers.Dense(
                num_outputs,
                activation="softmax",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=43),
                bias_initializer="zeros",
                name="output",
            )
        else:
            self.out = tf.keras.layers.Dense(
                num_outputs,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=43),
                bias_initializer="zeros",
                name="output",
            )

        # Fixed LR schedule matching the original SAND paper
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001,
            decay_steps=250,
            decay_rate=1.0,
            staircase=False,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def call(self, x, training: bool = False):
        x = self.selector(x, training=training)

        if self.batch_norm:
            x = self.bn(x, training=training)

        for layer in self.hidden:
            x = layer(x)

        return self.out(x)

    # ------------------------------------------------------------------
    # Custom train step
    # ------------------------------------------------------------------

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred)

        grads = tape.gradient(loss, self.trainable_variables)

        for var, grad in zip(self.trainable_variables, grads):
            if "feature_weights" in var.name and grad is not None:
                self.selector.record_gradients(grad)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.selector.enforce_mask()
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics} | {"loss": loss}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict:
        return self.selector.get_feature_importance()


# ============================================================
# A-SAND-Hybrid Model
# ============================================================

class SANDAdaptiveModelHybrid(tf.keras.Model):
    """MLP with adaptive first + second-order feature selection (A-SAND-Hybrid).

    Parameters
    ----------
    num_inputs : int
        Number of input features.
    num_outputs : int
        Number of output classes (classification) or targets (regression).
    initial_k : int, optional
        Initial number of features passed to the SAND layer.
    layer_sequence : tuple of int, optional
        Hidden layer widths. Default (67,).
    is_classification : bool, optional
        If True, uses softmax output and cross-entropy loss.
    alpha : float, optional
        LeakyReLU negative slope. Default 0 (= ReLU).
    batch_norm : bool, optional
        Whether to apply BatchNormalization after the SAND layer.
    learning_rate : float, optional
        Adam learning rate. Default 1e-4.
    """

    def __init__(
        self,
        num_inputs: int,
        num_outputs: int,
        initial_k: int = None,
        layer_sequence=(67,),
        is_classification: bool = True,
        alpha: float = 0.0,
        batch_norm: bool = False,
        learning_rate: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.is_classification = is_classification
        self.batch_norm = batch_norm
        self.learning_rate = learning_rate

        self.selector = SANDLayerUniversalHybrid(
            num_inputs=num_inputs, initial_k=initial_k
        )

        if self.batch_norm:
            self.bn = tf.keras.layers.BatchNormalization(name="batch_normalization")

        self.hidden = [
            tf.keras.layers.Dense(
                units=units,
                activation=tf.keras.layers.LeakyReLU(negative_slope=alpha),
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42 + i),
                name=f"hidden_{i}",
            )
            for i, units in enumerate(layer_sequence)
        ]

        if self.is_classification:
            self.out = tf.keras.layers.Dense(
                num_outputs,
                activation="softmax",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=100),
                name="classification_output",
            )
            self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
            self.acc_metric = tf.keras.metrics.CategoricalAccuracy(name="accuracy")
        else:
            self.out = tf.keras.layers.Dense(
                num_outputs,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=100),
                name="regression_output",
            )
            self.loss_fn = tf.keras.losses.MeanSquaredError()
            self.acc_metric = None

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    # ------------------------------------------------------------------
    # Compile override
    # ------------------------------------------------------------------

    def compile(self, optimizer=None, **kwargs):
        """Accepts compile() without arguments; falls back to Adam(learning_rate)."""
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        super().compile(optimizer=optimizer, **kwargs)

    # ------------------------------------------------------------------
    # Metrics property
    # ------------------------------------------------------------------

    @property
    def metrics(self):
        metrics = [self.loss_tracker]
        if self.acc_metric is not None:
            metrics.append(self.acc_metric)
        return metrics

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def call(self, x, training: bool = False):
        x = self.selector(x, training=training)

        if self.batch_norm:
            x = self.bn(x, training=training)

        for layer in self.hidden:
            x = layer(x, training=training)

        return self.out(x, training=training)

    # ------------------------------------------------------------------
    # Custom train step (computes first + second-order importance)
    # ------------------------------------------------------------------

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as outer_tape:
            with tf.GradientTape() as inner_tape:
                y_pred = self(x, training=True)
                loss = self.loss_fn(y, y_pred)

            grad_w = inner_tape.gradient(loss, self.selector.w)

        if grad_w is None:
            raise RuntimeError("Gradient for selector weights could not be computed.")

        # Hutchinson diagonal Hessian estimate
        rademacher = tf.where(
            self.selector.hessian_rng.uniform(
                shape=tf.shape(self.selector.w), minval=0.0, maxval=1.0
            ) < 0.5,
            -tf.ones_like(self.selector.w),
            tf.ones_like(self.selector.w),
        )

        directional_grad = tf.reduce_sum(grad_w * tf.stop_gradient(rademacher))
        hessian_vector_product = outer_tape.gradient(directional_grad, self.selector.w)

        gradients = outer_tape.gradient(loss, self.trainable_variables)
        del outer_tape

        self.selector.record_gradients(grad_w)
        if hessian_vector_product is not None:
            self.selector.record_hdiag(rademacher * hessian_vector_product)

        self.optimizer.apply_gradients(
            (g, v) for g, v in zip(gradients, self.trainable_variables) if g is not None
        )
        self.selector.enforce_mask()

        self.loss_tracker.update_state(loss)
        if self.acc_metric is not None:
            self.acc_metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    # ------------------------------------------------------------------
    # Custom test step
    # ------------------------------------------------------------------

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        loss = self.loss_fn(y, y_pred)

        self.loss_tracker.update_state(loss)
        if self.acc_metric is not None:
            self.acc_metric.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> dict:
        return self.selector.get_feature_importance()
