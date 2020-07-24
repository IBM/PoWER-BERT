import os
import sys
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K
import keras.constraints as Constraint
import keras.initializers as Initializer
from keras_layer_normalization import LayerNormalization
from keras_position_wise_feed_forward import FeedForward
from model.multi_head_attention import MultiHeadAttention


__all__ = [
    'get_custom_objects', 'get_encoders',
    'attention_builder', 'feed_forward_builder', 'get_encoder_component',
]

def get_custom_objects():
    return {
        'LayerNormalization': LayerNormalization,
        'MultiHeadAttention': MultiHeadAttention,
        'FeedForward': FeedForward,
    }

class Constraint(object):

    def __call__(self, w):
        return w

    def get_config(self):
        return {}

class CustomConstraint(Constraint):

    def __init__(self, min, max):
        self.min=min
        self.max=max    

    def __call__(self, w):      
        new_w = tf.clip_by_value(w, clip_value_min=self.min, clip_value_max=self.max)
        return new_w

    def get_config(self):
        return {'C': 0.0}

class Soft_Extract(keras.layers.Layer):

        def __init__(self, 
                atten=None, 
                kernel_initializer='glorot_normal',
                kernel_Regularizer=None,
                kernel_constraint=None,
                LAMBDA=None,
                **kwargs):

                self.atten = atten
                self.LAMBDA = LAMBDA
                self.kernel_initializer = keras.initializers.Constant(value=1)
                self.kernel_regularizer = keras.regularizers.l1(self.LAMBDA)
                self.kernel_constraint = CustomConstraint(0.0, 1.0) 
                self.name = kwargs['name']
                self.W = None
                super(Soft_Extract, self).__init__(**kwargs)

        def compute_output_shape(self, input_shape):
                return input_shape

        def compute_mask(self, inputs, mask=None):
                return None

        def calc_avg_atten(self, x, head_num):
                input_shape = K.shape(x)
                batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
                x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
                x = K.permute_dimensions(x, [0, 2, 1, 3])
                avg_atten = tf.reduce_mean(x, axis=2)
                return avg_atten

        def atten_col(self, avg_atten):
                row_sum = tf.reduce_sum(avg_atten, axis=1)
                diag_softmax = tf.linalg.diag_part(avg_atten)
                attended_by = tf.math.subtract(row_sum, diag_softmax)
                return attended_by

        def build(self, input_shape):
        
                self.W = self.add_weight(
                    shape=(input_shape[1],),
                    initializer=self.kernel_initializer,
                    regularizer=self.kernel_regularizer,
                    constraint=self.kernel_constraint,
                    name='%s_W' % self.name,
                )
                super(Soft_Extract, self).build(input_shape)


        def call(self, x, mask=None):

                avg_atten = self.calc_avg_atten(self.atten, 12)
                attended_by = self.atten_col(avg_atten)
                indices = tf.math.top_k(attended_by, k=tf.shape(x,out_type=tf.dtypes.int32)[1], sorted=True).indices
                indices = tf.cast(tf.math.multiply(-1.0, tf.cast(indices, dtype=tf.float32)),dtype=tf.int32)
                indices_inverted = tf.expand_dims(tf.cast(tf.math.top_k(indices, k=tf.shape(x,out_type=tf.dtypes.int32)[1], sorted=True).indices,dtype=tf.int32), axis=-1)
                W_broadcast = tf.broadcast_to(self.W, shape=[tf.shape(x,out_type=tf.dtypes.int32)[0],tf.shape(x,out_type=tf.dtypes.int32)[1]])
                W_inverted = tf.expand_dims(tf.gather_nd(W_broadcast, indices_inverted, batch_dims=1), axis=-1)
                output = tf.math.multiply(x, W_inverted)
                return output


class Hard_Extract(keras.layers.Layer):
        def __init__(self, index=None, atten=None, attention_mask=None, **kwargs):
                super(Hard_Extract, self).__init__(**kwargs)
                self.index = index
                self.supports_masking = True
                self.atten = atten

        def compute_output_shape(self, input_shape):
                input_shape = (input_shape[0], self.index, input_shape[2])
                return input_shape

        def compute_mask(self, inputs, mask=None):
                return None

        def calc_avg_atten(self, x, head_num):
                input_shape = K.shape(x)
                batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
                x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
                x = K.permute_dimensions(x, [0, 2, 1, 3])
                avg_atten = tf.reduce_mean(x, axis=2)
                return avg_atten

        def atten_col(self, avg_atten):
                row_sum = tf.reduce_sum(avg_atten, axis=1)
                diag_softmax = tf.linalg.diag_part(avg_atten)
                attended_by = tf.math.subtract(row_sum, diag_softmax)
                return attended_by

        def call(self, x, mask=None):

                avg_atten = self.calc_avg_atten(self.atten, 12)
                attended_by = self.atten_col(avg_atten)
                attended_by = attended_by[:,1:]
                indices = tf.cast(tf.math.top_k(attended_by, k=self.index-1, sorted=True).indices,dtype=tf.int32)
                indices = tf.expand_dims(tf.add(indices, tf.constant([1], dtype=tf.int32)), axis=-1)
                CLS_SEP = tf.broadcast_to(tf.constant([0], dtype=tf.int32),shape=[tf.shape(x,out_type=tf.dtypes.int32)[0],1])
                indices_CLS_SEP = tf.concat([indices, tf.expand_dims(CLS_SEP, axis=-1)], axis=1)
                indices_CLS_SEP = tf.sort(indices_CLS_SEP, axis=1, direction='ASCENDING')
                extract_layer = tf.gather_nd(x, indices_CLS_SEP, batch_dims=1)
                return extract_layer


def _wrap_layer(name,
                input_layer,
                build_func,
                dropout_rate=0.0,
                trainable=True,
                use_adapter=False,
                adapter_units=None,
                adapter_activation='relu',
                attention_mask=None,
                SEQ_LEN=None,
                retention_configuration=None,
                LAMBDA=None,
                FLAG_EXTRACT_LAYER=None,
                layer_idx=None,
                word_vector_elimination=None):
        """Wrap layers with residual, normalization and dropout.

        :param name: Prefix of names for internal layers.
        :param input_layer: Input layer.
        :param build_func: A callable that takes the input tensor and uenerates the output tensor.
        :param dropout_rate: Dropout rate.
        :param trainable: Whether the layers are trainable.
        :param use_adapter: Whether to use feed-forward adapters before each residual connections.
        :param adapter_units: The dimension of the first transformation in feed-forward adapter.
        :param adapter_activation: The activation after the first transformation in feed-forward adapter.
        :return: Output layer.
        """
        if word_vector_elimination:
            [build_output, atten] = build_func(input_layer)
        else:
            build_output = build_func(input_layer)

        if dropout_rate > 0.0:
                dropout_layer = keras.layers.Dropout(
                        rate=dropout_rate,
                        name='%s-Dropout' % name,
                )(build_output)
        else:
                dropout_layer = build_output
        if isinstance(input_layer, list):
                input_layer = input_layer[0]
        if use_adapter:
                adapter = FeedForward(
                    units=adapter_units,
                    activation=adapter_activation,
                    kernel_initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.001),
                    name='%s-Adapter' % name,
                )(dropout_layer)
                dropout_layer = keras.layers.Add(name='%s-Adapter-Add' % name)([dropout_layer, adapter])
        add_layer = keras.layers.Add(name='%s-Add' % name)([input_layer, dropout_layer])
        normal_layer = LayerNormalization(
                trainable=trainable,
                name='%s-Norm' % name,
                )(add_layer)

        if word_vector_elimination:
        
                if FLAG_EXTRACT_LAYER == 1:
                
                        extract_layer = Soft_Extract(atten=atten, LAMBDA=LAMBDA*(layer_idx**1.0), name='%s-Soft-Extract' % name)(normal_layer)
                        return extract_layer, attention_mask

                elif FLAG_EXTRACT_LAYER == 2:
        
                        extract_layer = Hard_Extract(atten=atten, index=retention_configuration[layer_idx-1], name='%s-Extract' % name)(normal_layer)
                        attention_mask = attention_mask[:,:retention_configuration[layer_idx-1]]
                        return extract_layer, attention_mask
        
        return normal_layer, attention_mask


def attention_builder(name,
                      head_num,
                      activation,
                      history_only,
                      trainable=True,
                      attention_mask=None):
    """Get multi-head self-attention builder.

    :param name: Prefix of names for internal layers.
    :param head_num: Number of heads in multi-head self-attention.
    :param activation: Activation for multi-head self-attention.
    :param history_only: Only use history data.
    :param trainable: Whether the layer is trainable.
    :return:
    """
    def _attention_builder(x):
        return MultiHeadAttention(
            head_num=head_num,
            activation=activation,
            history_only=history_only,
            trainable=trainable,
            name=name,
        )(inputs=x, mask=attention_mask)
    return _attention_builder


def feed_forward_builder(name,
                         hidden_dim,
                         activation,
                         trainable=True):
    """Get position-wise feed-forward layer builder.

    :param name: Prefix of names for internal layers.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param activation: Activation for feed-forward layer.
    :param trainable: Whether the layer is trainable.
    :return:
    """
    def _feed_forward_builder(x):
        return FeedForward(
            units=hidden_dim,
            activation=activation,
            trainable=trainable,
            name=name,
        )(x)
    return _feed_forward_builder


def get_encoder_component(name,
                          input_layer,
                          head_num,
                          hidden_dim,
                          attention_activation=None,
                          feed_forward_activation='relu',
                          dropout_rate=0.0,
                          trainable=True,
                          use_adapter=False,
                          adapter_units=None,
                          adapter_activation='relu',
                          SEQ_LEN=None,
                          retention_configuration=None,
                          LAMBDA=None,
                          FLAG_EXTRACT_LAYER=None,
                          layer_idx=None,
                          attention_mask=None):
    """Multi-head self-attention and feed-forward layer.

    :param name: Prefix of names for internal layers.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    attention_name = '%s-MultiHeadSelfAttention' % name
    feed_forward_name = '%s-FeedForward' % name
    attention_layer, attention_mask = _wrap_layer(
        name=attention_name,
        input_layer=input_layer,
        build_func=attention_builder(
            name=attention_name,
            head_num=head_num,
            activation=attention_activation,
            history_only=False,
            trainable=trainable,
            attention_mask=attention_mask,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
        attention_mask=attention_mask,
        SEQ_LEN=SEQ_LEN,
        retention_configuration=retention_configuration,
        LAMBDA=LAMBDA,
        FLAG_EXTRACT_LAYER=FLAG_EXTRACT_LAYER,
        layer_idx=layer_idx,
        word_vector_elimination=True,
    )
    feed_forward_layer, attention_mask = _wrap_layer(
        name=feed_forward_name,
        input_layer=attention_layer,
        build_func=feed_forward_builder(
            name=feed_forward_name,
            hidden_dim=hidden_dim,
            activation=feed_forward_activation,
            trainable=trainable,
        ),
        dropout_rate=dropout_rate,
        trainable=trainable,
        use_adapter=use_adapter,
        adapter_units=adapter_units,
        adapter_activation=adapter_activation,
        attention_mask=attention_mask,
        SEQ_LEN=SEQ_LEN,
        retention_configuration=retention_configuration,
        LAMBDA=LAMBDA,
        FLAG_EXTRACT_LAYER=FLAG_EXTRACT_LAYER,
        layer_idx=layer_idx,
        word_vector_elimination=False,
    )
    return feed_forward_layer, attention_mask


def get_encoders(encoder_num,
                 input_layer,
                 head_num,
                 hidden_dim,
                 attention_activation=None,
                 feed_forward_activation='relu',
                 dropout_rate=0.0,
                 trainable=True,
                 use_adapter=False,
                 adapter_units=None,
                 adapter_activation='relu',
                 SEQ_LEN=None,
                 retention_configuration=None,
                 LAMBDA=None,
                 FLAG_EXTRACT_LAYER=None,
                 attention_mask=None):
    """Get encoders.

    :param encoder_num: Number of encoder components.
    :param input_layer: Input layer.
    :param head_num: Number of heads in multi-head self-attention.
    :param hidden_dim: Hidden dimension of feed forward layer.
    :param attention_activation: Activation for multi-head self-attention.
    :param feed_forward_activation: Activation for feed-forward layer.
    :param dropout_rate: Dropout rate.
    :param trainable: Whether the layers are trainable.
    :param use_adapter: Whether to use feed-forward adapters before each residual connections.
    :param adapter_units: The dimension of the first transformation in feed-forward adapter.
    :param adapter_activation: The activation after the first transformation in feed-forward adapter.
    :return: Output layer.
    """
    last_layer = input_layer
    for i in range(encoder_num):
        last_layer, attention_mask = get_encoder_component(
            name='Encoder-%d' % (i + 1),
            input_layer=last_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
            use_adapter=use_adapter,
            adapter_units=adapter_units,
            adapter_activation=adapter_activation,
            attention_mask=attention_mask,
            SEQ_LEN=SEQ_LEN,
            retention_configuration=retention_configuration,
            LAMBDA=LAMBDA,
            FLAG_EXTRACT_LAYER=FLAG_EXTRACT_LAYER,
            layer_idx=i+1,
        )
    return last_layer

