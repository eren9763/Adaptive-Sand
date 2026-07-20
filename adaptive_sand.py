

import numpy as np
import pandas as pd

import tensorflow as tf

import os
import subprocess
import numpy as np
import pandas as pd
import tensorflow as tf

import os
import random
import numpy as np
import tensorflow as tf
import time
import copy
from sklearn.metrics import accuracy_score

# ==========================================
# 1. REPRODUCIBILITY
# ==========================================
def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.enable_op_determinism()

set_seeds(42)
# ==========================================
# ADAPTIVE SAND LAYER (FIXED)
# ==========================================
class SAND_Layer_Universal(tf.keras.layers.Layer):
    def __init__(self, num_inputs, initial_k=None, seed=42):
        super().__init__()
        self.num_features = num_inputs

        self.k = tf.Variable(
            initial_k if initial_k is not None else num_inputs,
            trainable=False, dtype=tf.int32
        )

        self.w = self.add_weight(
            shape=(num_inputs,),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05, seed=seed),  # DÜZELTME
            trainable=True,
            name="feature_weights"
        )

        self.noise_rng = tf.random.Generator.from_seed(seed)  # DÜZELTME: kendi izole RNG'si

        self.mask = tf.Variable(tf.ones(num_inputs), trainable=False)
        self.best_mask = tf.Variable(tf.ones(num_inputs), trainable=False)
        self.best_k = tf.Variable(self.k.read_value(), trainable=False)

        self.pruning_active = True
        self.grad_accumulator = tf.Variable(tf.zeros(num_inputs), trainable=False)
        self.grad_count = tf.Variable(0.0, trainable=False)
        self.prune_rate = 0.1

    def enforce_mask(self):
        self.w.assign(self.w * self.mask)

    def call(self, x, training=False):
        self.enforce_mask()
        w_abs = tf.abs(self.w)
        w_abs = tf.minimum(w_abs, 1.0)
        w_active = w_abs * self.mask
        norm = tf.sqrt(tf.reduce_sum(tf.square(w_active)) + 1e-8)
        wn = (w_active / norm) * tf.sqrt(tf.cast(self.k, tf.float32))
        x_weighted = x * wn

        if training:
            noise_std = 0.5 * tf.abs(1 - wn)
            noise = self.noise_rng.normal(tf.shape(x_weighted), stddev=noise_std)  # DÜZELTME
            return x_weighted + noise
        else:
            return x_weighted

    def record_gradients(self, grad):
        if grad is not None and self.pruning_active:
            importance = tf.abs(grad * self.w)
            self.grad_accumulator.assign_add(importance)
            self.grad_count.assign_add(1.0)

    def save_current_as_best(self):
        self.best_mask.assign(self.mask)
        self.best_k.assign(self.k)

    def restore_best(self):
        self.mask.assign(self.best_mask)
        self.k.assign(self.best_k)
        self.enforce_mask()

    def prune_by_gradient(self, force_target_k=None, prune_rate = 0.1):
        if not self.pruning_active:
            return False

        current_mask = self.mask.numpy()
        k_old = int(np.sum(current_mask))

        if self.grad_count > 0:
            scores = (self.grad_accumulator / self.grad_count).numpy()
        else:
            scores = np.abs(self.w.numpy())

        scores[current_mask == 0] = -np.inf

        if force_target_k is not None:
            num_to_keep = force_target_k
        else:
            p_rate = prune_rate
            num_to_keep = max(1, k_old - int(p_rate * k_old))

        top_indices = np.argsort(scores)[-num_to_keep:]
        new_mask = np.zeros_like(current_mask)
        new_mask[top_indices] = 1.0

        self.mask.assign(new_mask.astype(np.float32))
        self.k.assign(int(np.sum(new_mask)))

        self.enforce_mask()
        self.grad_accumulator.assign(tf.zeros(self.num_features))
        self.grad_count.assign(0.0)

        print(f"\n✂️ Pruning: {k_old} → {self.k.numpy()}")
        return True

    def get_feature_importance(self):
        return {
            "num_selected": int(np.sum(self.mask.numpy())),
            "mask": self.mask.numpy()
        }

# ==========================================
# ADAPTIVE MODEL (FIXED)
# ==========================================
class SANDAdaptiveModel(tf.keras.Model):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        initial_k=None,
        layer_sequence=[67],
        is_classification=True,
        alpha=0,
        batch_norm=False
    ):
        super().__init__()

        self.selector = SAND_Layer_Universal(num_inputs, initial_k)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = tf.keras.layers.BatchNormalization()

        # 🔥 FIX: SAME ACTIVATION AS PAPER
        self.hidden = [
    tf.keras.layers.Dense(
        d,
        activation=tf.keras.layers.LeakyReLU(negative_slope=alpha),
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=42 + i),
        bias_initializer="zeros"
    )
    for i, d in enumerate(layer_sequence)
]

        if is_classification:
            self.out = tf.keras.layers.Dense(
        num_outputs,
        activation="softmax",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=43),
        bias_initializer="zeros"
            )
        else:
            self.out = tf.keras.layers.Dense(
        num_outputs,
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=43),
        bias_initializer="zeros"
            )

        # 🔥 FIX: SAME LR SCHEDULE
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0001,
            decay_steps=250,
            decay_rate=1.0,
            staircase=False,
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def call(self, x, training=False):
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
            loss = self.compiled_loss(y, y_pred)

        grads = tape.gradient(loss, self.trainable_variables)

        for var, grad in zip(self.trainable_variables, grads):
            if 'feature_weights' in var.name and grad is not None:
                self.selector.record_gradients(grad)

        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.selector.enforce_mask()

        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics} | {"loss": loss}

    def get_feature_importance(self):
        return self.selector.get_feature_importance()