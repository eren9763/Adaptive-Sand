"""Selection with Additive Noise Distortion"""

from sand.experiments.models.mlp import MLPModel
import tensorflow as tf
import numpy as np

class SAND_Layer(tf.keras.layers.Layer):
    def __init__(self, num_inputs, num_inputs_to_select, sigma=1.5):
        super(SAND_Layer, self).__init__()
        self.num_inputs_to_select = num_inputs_to_select
        self.sigma = sigma
        self.w = self.add_weight(
            shape=(num_inputs,),
            initializer=tf.keras.initializers.Constant(np.ones((num_inputs,))*tf.sqrt(num_inputs_to_select/num_inputs)),
            trainable=True,
        )
        self.batch_norm = False

    def call(self, inputs, training=False):
        wabs = tf.abs(self.w)
        wabs = wabs - tf.nn.relu(wabs - 1)
        
        wn = wabs / tf.sqrt(tf.reduce_sum(wabs**2)) * tf.sqrt( float(self.num_inputs_to_select) )
        self.w.assign(wn)
        
        inpwn = inputs * tf.expand_dims(wn, 0)
        
        if training:
          noise = tf.random.normal(shape=tf.shape(inpwn), stddev=self.sigma * tf.abs(1-tf.expand_dims(wn, 0)))
          return inpwn + noise
        else:
          return inpwn
        
    def zero_out_minor_gains(self):
        # keep the top feat_num gains and set the rest to 0
        _, top_indices = tf.math.top_k(tf.abs(self.w), self.num_inputs_to_select)
        mask = tf.scatter_nd(tf.reshape(top_indices, (-1, 1)), tf.ones_like(top_indices, dtype=self.w.dtype), self.w.shape)
        self.w.assign(self.w * mask)

    def binarize_gains(self):
        # set the top feat_num gains to 1 and the rest to 0
        _, top_indices = tf.math.top_k(tf.abs(self.w), self.num_inputs_to_select)
        binarized_gains = tf.zeros(self.w.shape)
        updates = tf.ones_like(top_indices, dtype=binarized_gains.dtype)
        binarized_gains = tf.tensor_scatter_nd_update(binarized_gains, tf.reshape(top_indices, (-1, 1)), updates)
        self.w.assign(binarized_gains)


class SANDModel(MLPModel):
  """MLP with SAND."""

  def __init__(self, num_inputs, num_inputs_to_select, sigma, **kwargs):
    """Initialize the model."""

    super(SANDModel, self).__init__(**kwargs)

    self.select = SAND_Layer(num_inputs=num_inputs, num_inputs_to_select=num_inputs_to_select, sigma=sigma)

  def call(self, inputs):
    inputs = self.select(inputs)
    representation = self.mlp_model(inputs) 
    prediction = self.mlp_predictor(representation)
    return prediction
  
  def selected_indices(self):
      return tf.argsort(tf.abs(self.select.w), direction='DESCENDING')[:self.select.num_inputs_to_select]