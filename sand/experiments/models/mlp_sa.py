"""Feature selection with Sequential Attention (FAIR VERSION)."""

from sand.experiments.models.mlp import MLPModel
from sand.sequential_attention import SequentialAttention
import tensorflow as tf


class SequentialAttentionModel(MLPModel):
    """MLP with Sequential Attention (FAIR)."""

    def __init__(
        self,
        num_inputs,
        num_inputs_to_select,
        num_train_steps,
        num_inputs_to_select_per_step=1,
        **kwargs,
    ):
        super(SequentialAttentionModel, self).__init__(**kwargs)

        self.num_inputs = num_inputs
        self.k = num_inputs_to_select

        self.seqatt = SequentialAttention(
            num_candidates=num_inputs,
            num_candidates_to_select=num_inputs_to_select,
            num_candidates_to_select_per_step=num_inputs_to_select_per_step,
        )

        self.num_train_steps = num_train_steps

    def call(self, inputs, training=False):
        if self.batch_norm:
            inputs = self.batch_norm_layer(inputs, training=training)

        training_percentage = self.optimizer.iterations / self.num_train_steps
        feature_weights = self.seqatt(training_percentage)

        # =========================================================
        # 🔥 HARD MASK (FAIRNESS FIX)
        # =========================================================
        if not training:
            _, top_indices = tf.math.top_k(feature_weights, self.k)

            mask = tf.scatter_nd(
                tf.reshape(top_indices, (-1, 1)),
                tf.ones_like(top_indices, dtype=feature_weights.dtype),
                (self.num_inputs,),
            )

            feature_weights = feature_weights * mask

        inputs = inputs * feature_weights

        x = self.mlp_model(inputs)
        return self.mlp_predictor(x)