import json
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras_bert.layers import get_inputs, get_embedding, TokenEmbedding, EmbeddingSimilarity, Masked, Extract
from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization
from model import get_encoders
from model import get_custom_objects as get_encoder_custom_objects


def gelu_tensorflow(x):
    from tensorflow.python.ops.math_ops import erf, sqrt
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))


def gelu_fallback(x):
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * K.pow(x, 3))))


if K.backend() == 'tensorflow':
    gelu = gelu_tensorflow
else:
    gelu = gelu_fallback



def get_checkpoint_model(token_num,
              pos_num=512,
              seq_len=512,
              embed_dim=768,
              transformer_num=12,
              head_num=12,
              feed_forward_dim=3072,
              dropout_rate=0.1,
              attention_activation=None,
              feed_forward_activation='gelu',
              training=True,
              finetuned=False,
              output_dim=2,
              trainable=None,
              output_layer_num=1,
              retention_configuration=None,
              LAMBDA=None,
              FLAG_EXTRACT_LAYER=None,
              TASK=None,
              ):
        """Get BERT model.
        :param token_num: Number of tokens.
        :param pos_num: Maximum position.
        :param seq_len: Maximum length of the input sequence or None.
        :param embed_dim: Dimensions of embeddings.
        :param transformer_num: Number of transformers.
        :param head_num: Number of heads in multi-head attention in each transformer.
        :param feed_forward_dim: Dimension of the feed forward layer in each transformer.
        :param dropout_rate: Dropout rate.
        :param attention_activation: Activation for attention layers.
        :param feed_forward_activation: Activation for feed-forward layers.
        :param trainable: Whether the model is trainable.
        :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `training` is `False`.
        :return: The built model.
        """
        if attention_activation == 'gelu':
                attention_activation = gelu
        if feed_forward_activation == 'gelu':
                feed_forward_activation = gelu
        if trainable is None:
                trainable = training
        def _trainable(_layer):
                if isinstance(trainable, (list, tuple, set)):
                    for prefix in trainable:
                        if _layer.name.startswith(prefix):
                            return True
                    return False
                return trainable

        inputs = get_inputs(seq_len=seq_len)
        attention_mask = inputs[2]
        embed_layer, embed_weights = get_embedding(
                inputs,
                token_num=token_num,
                embed_dim=embed_dim,
                pos_num=pos_num,
                dropout_rate=dropout_rate,
        )

        if dropout_rate > 0.0:
                dropout_layer = keras.layers.Dropout(
                    rate=dropout_rate,
                    name='Embedding-Dropout',
                )(embed_layer)
        else:
                dropout_layer = embed_layer
        embed_layer = LayerNormalization(
                trainable=trainable,
                    name='Embedding-Norm',
                )(dropout_layer)

        transformed = get_encoders(
                encoder_num=transformer_num,
                input_layer=embed_layer,
                head_num=head_num,
                hidden_dim=feed_forward_dim,
                attention_activation=attention_activation,
                feed_forward_activation=feed_forward_activation,
                dropout_rate=dropout_rate,
                attention_mask=attention_mask,
                SEQ_LEN=seq_len,
                retention_configuration=retention_configuration,
                LAMBDA=LAMBDA,
                FLAG_EXTRACT_LAYER=FLAG_EXTRACT_LAYER,
        )
        extract_layer = Extract(index=0, name='Extract')(transformed)
        nsp_dense_layer = keras.layers.Dense(
                units=embed_dim,
                activation='tanh',
                name='NSP-Dense',
        )(extract_layer)
        if TASK == 'sts-b':
            nsp_pred_layer = keras.layers.Dense(
                 units=output_dim,
                 name='NSP',
            )(nsp_dense_layer)
        else:
            nsp_pred_layer = keras.layers.Dense(
                 units=output_dim,
                 activation='softmax',
                 name='NSP',
            )(nsp_dense_layer)
        model = keras.models.Model(inputs=inputs, outputs=nsp_pred_layer)
        for layer in model.layers:
            layer.trainable = _trainable(layer)
        return model


def checkpoint_loader(checkpoint_file):
    def _loader(name):
        return tf.train.load_variable(checkpoint_file, name)
    return _loader


def build_model_from_config(config_file,
                            output_dim=2,
                            trainable=None,
                            output_layer_num=1,
                            seq_len=int(1e9),
                            retention_configuration=None,
                            LAMBDA=None,
                            FLAG_EXTRACT_LAYER=None,
                            TASK=None,
                            **kwargs
                            ):
    """Build the model from config file.
    :param config_file: The path to the JSON configuration file.
    :param training: If training, the whole model will be returned.
                     Otherwise, the MLM and NSP parts will be ignored.
    :param trainable: Whether the model is trainable.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `training` is `False`.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :return: model and config
    """
    with open(config_file, 'r') as reader:
        config = json.loads(reader.read())
    if seq_len is not None:
        config['max_position_embeddings'] = seq_len = min(seq_len, config['max_position_embeddings'])
    if trainable is None:
        trainable = True
    model = get_checkpoint_model(
        token_num=config['vocab_size'],
        pos_num=config['max_position_embeddings'],
        seq_len=seq_len,
        embed_dim=config['hidden_size'],
        transformer_num=config['num_hidden_layers'],
        head_num=config['num_attention_heads'],
        feed_forward_dim=config['intermediate_size'],
        feed_forward_activation=config['hidden_act'],
        output_dim=output_dim,
        trainable=trainable,
        output_layer_num=output_layer_num,
        retention_configuration=retention_configuration,
        LAMBDA=LAMBDA,
        FLAG_EXTRACT_LAYER=FLAG_EXTRACT_LAYER,
        TASK=TASK,
        **kwargs
    )
    return model, config



def load_checkpoint(model,
                    config,
                    checkpoint_file,
                    FLAG_BERT_PRETRAINED=False):

        """Load trained official model from checkpoint.
        :param model: Built keras model.
        :param config: Loaded configuration file.
        :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
        """

        loader = checkpoint_loader(checkpoint_file)
        model.get_layer(name='Embedding-Token').set_weights([
                loader('bert/embeddings/word_embeddings'),
        ])
        model.get_layer(name='Embedding-Position').set_weights([
                loader('bert/embeddings/position_embeddings')[:config['max_position_embeddings'], :],
        ])
        model.get_layer(name='Embedding-Segment').set_weights([
                loader('bert/embeddings/token_type_embeddings'),
        ])
        model.get_layer(name='Embedding-Norm').set_weights([
                loader('bert/embeddings/LayerNorm/gamma'),
                loader('bert/embeddings/LayerNorm/beta'),
        ])
        for i in range(config['num_hidden_layers']):
                try:
                    model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1))
                except ValueError as e:
                    continue
                model.get_layer(name='Encoder-%d-MultiHeadSelfAttention' % (i + 1)).set_weights([
                    loader('bert/encoder/layer_%d/attention/self/query/kernel' % i),
                    loader('bert/encoder/layer_%d/attention/self/query/bias' % i),
                    loader('bert/encoder/layer_%d/attention/self/key/kernel' % i),
                    loader('bert/encoder/layer_%d/attention/self/key/bias' % i),
                    loader('bert/encoder/layer_%d/attention/self/value/kernel' % i),
                    loader('bert/encoder/layer_%d/attention/self/value/bias' % i),
                    loader('bert/encoder/layer_%d/attention/output/dense/kernel' % i),
                    loader('bert/encoder/layer_%d/attention/output/dense/bias' % i),
                ])
                model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
                    loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
                    loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
                ])
                model.get_layer(name='Encoder-%d-MultiHeadSelfAttention-Norm' % (i + 1)).set_weights([
                    loader('bert/encoder/layer_%d/attention/output/LayerNorm/gamma' % i),
                    loader('bert/encoder/layer_%d/attention/output/LayerNorm/beta' % i),
                ])
                model.get_layer(name='Encoder-%d-FeedForward' % (i + 1)).set_weights([
                    loader('bert/encoder/layer_%d/intermediate/dense/kernel' % i),
                    loader('bert/encoder/layer_%d/intermediate/dense/bias' % i),
                    loader('bert/encoder/layer_%d/output/dense/kernel' % i),
                    loader('bert/encoder/layer_%d/output/dense/bias' % i),
                ])
                model.get_layer(name='Encoder-%d-FeedForward-Norm' % (i + 1)).set_weights([
                    loader('bert/encoder/layer_%d/output/LayerNorm/gamma' % i),
                    loader('bert/encoder/layer_%d/output/LayerNorm/beta' % i),
                ])
        model.get_layer(name='NSP-Dense').set_weights([
                loader('bert/pooler/dense/kernel'),
                loader('bert/pooler/dense/bias'),
        ])
        if not FLAG_BERT_PRETRAINED:
                model.get_layer(name='NSP').set_weights([
                        np.transpose(loader('output_weights')),
                        loader('output_bias'),
                ])
                

def load_model(config_file,
               checkpoint_file,
               FLAG_BERT_PRETRAINED=False,
               output_dim=2,
               trainable=True,
               output_layer_num=1,
               seq_len=int(1e9),
               retention_configuration=None,
               LAMBDA=None,
               FLAG_EXTRACT_LAYER=None,
               TASK=None,
               **kwargs):

    """Load trained official model from checkpoint.
    :param config_file: The path to the JSON configuration file.
    :param checkpoint_file: The path to the checkpoint files, should end with '.ckpt'.
    :param trainable: Whether the model is trainable. The default value is the same with `training`.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `training` is `False`.
    :param seq_len: If it is not None and it is shorter than the value in the config file, the weights in
                    position embeddings will be sliced to fit the new length.
    :return: model
    """
    model, config = build_model_from_config(
        config_file,
        output_dim=output_dim,
        trainable=trainable,
        output_layer_num=output_layer_num,
        seq_len=seq_len,
        retention_configuration=retention_configuration,
        LAMBDA=LAMBDA,
        FLAG_EXTRACT_LAYER=FLAG_EXTRACT_LAYER,
        TASK=TASK,
        **kwargs
    )
    
    load_checkpoint(model, 
                    config, 
                    checkpoint_file, 
                    FLAG_BERT_PRETRAINED=FLAG_BERT_PRETRAINED)
    
    return model
