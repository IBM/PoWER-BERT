from __future__ import print_function
import os
import sys
import keras
import keras.backend as K
import tensorflow as tf
import numpy as np

if 'TF_KERAS' in os.environ and os.environ['TF_KERAS'] != '0':
    from tensorflow.python import keras
    TF_KERAS = True
else:
    import keras
    TF_KERAS = False

class ScaledDotProductAttention(keras.layers.Layer):

	def __init__(self,
		 return_attention=False,
		 history_only=False,
		 **kwargs):
		"""Initialize the layer.

		:param return_attention: Whether to return attention weights.
		:param history_only: Whether to only use history data.
		:param kwargs: Arguments for parent class.
		"""
		super(ScaledDotProductAttention, self).__init__(**kwargs)
		self.supports_masking = True
		self.return_attention = True
		self.history_only = history_only
		self.name = kwargs['name']

	def get_config(self):
		config = {
		    'return_attention': self.return_attention,
		    'history_only': self.history_only,
		}
		base_config = super(ScaledDotProductAttention, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def compute_output_shape(self, input_shape):
		if isinstance(input_shape, list):
		    query_shape, key_shape, value_shape = input_shape
		else:
		    query_shape = key_shape = value_shape = input_shape
		output_shape = query_shape[:-1] + value_shape[-1:]
		if self.return_attention:
		    attention_shape = query_shape[:2] + (key_shape[1],)
		    return [output_shape, attention_shape]
		return output_shape

	def compute_mask(self, inputs, mask=None):
		if isinstance(mask, list):
		    mask = mask[0]
		if self.return_attention:
		    return [mask, None]
		return mask

	def call(self, inputs, mask=None, **kwargs):
                if isinstance(inputs, list):
                    query, key, value = inputs
                else:
                    query = key = value = inputs
                if isinstance(mask, list):
                    mask = mask[1]
                feature_dim = K.shape(query)[-1]
                e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
                e = K.exp(e - K.max(e, axis=-1, keepdims=True))
                if self.history_only:
                    query_len, key_len = K.shape(query)[1], K.shape(key)[1]
                    indices = K.expand_dims(K.arange(0, key_len), axis=0)
                    upper = K.expand_dims(K.arange(0, query_len), axis=-1)
                    e *= K.expand_dims(K.cast(indices <= upper, K.floatx()), axis=0)
                if mask is not None:
                    e *= K.cast(K.expand_dims(mask, axis=-2), K.floatx())
                a = e / (K.sum(e, axis=-1, keepdims=True) + K.epsilon())
                v = K.batch_dot(a, value)
                if self.return_attention:
                    return [v, a]
                return v
