"""Selection with Additive Noise Distortion (FAIR VERSION)"""

from sand.experiments.models.mlp import MLPModel
import tensorflow as tf
import numpy as np


class SAND_Layer(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_inputs_to_select, sigma=1.5):
        super(SAND_Layer, self).__init__()
        self.num_inputs = num_inputs
        self.num_inputs_to_select = num_inputs_to_select
        self.sigma = sigma

        self.w = self.add_weight(
            shape=(num_inputs,),
            initializer=tf.keras.initializers.Constant(
                np.ones((num_inputs,)) * tf.sqrt(num_inputs_to_select / num_inputs)
            ),
            trainable=True,
        )

    def call(self, inputs, training=False):
        # 🔹 normalize weights
        wabs = tf.abs(self.w)
        wabs = wabs - tf.nn.relu(wabs - 1)

        wn = wabs / (tf.sqrt(tf.reduce_sum(wabs**2)) + 1e-8)
        wn = wn * tf.sqrt(float(self.num_inputs_to_select))

        self.w.assign(wn)

        # =========================================================
        # 🔥 HARD MASK (FAIRNESS FIX)
        # =========================================================
        if not training:
            _, top_indices = tf.math.top_k(tf.abs(wn), self.num_inputs_to_select)

            mask = tf.scatter_nd(
                tf.reshape(top_indices, (-1, 1)),
                tf.ones_like(top_indices, dtype=wn.dtype),
                (self.num_inputs,),
            )

            wn = wn * mask  # ONLY top-k survive

        inpwn = inputs * tf.expand_dims(wn, 0)

        # =========================================================
        # 🔹 noise (only training)
        # =========================================================
        if training:
            noise = tf.random.normal(
                shape=tf.shape(inpwn),
                stddev=self.sigma * tf.abs(1 - tf.expand_dims(wn, 0)),
            )
            return inpwn + noise
        else:
            return inpwn

    def selected_indices(self):
        return tf.argsort(tf.abs(self.w), direction="DESCENDING")[
            : self.num_inputs_to_select
        ]


class SANDModel(MLPModel):
    """MLP with SAND (FAIR)."""

    def __init__(self, num_inputs, num_inputs_to_select, sigma, **kwargs):
        super(SANDModel, self).__init__(**kwargs)

        self.select = SAND_Layer(
            num_inputs=num_inputs,
            num_inputs_to_select=num_inputs_to_select,
            sigma=sigma,
        )

    def call(self, inputs, training=False):
        x = self.select(inputs, training=training)
        x = self.mlp_model(x)
        return self.mlp_predictor(x)

    def selected_indices(self):
        return self.select.selected_indices()