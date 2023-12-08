# # Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
#
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Layer
#
# from torch.utils
#
#
#
# __all__ = [
#     "GAUAlphaModel",
#     "GAUAlphaForMaskedLM",
#     "GAUAlphaPretrainedModel",
#     "GAUAlphaForSequenceClassification",
#     "GAUAlphaForTokenClassification",
#     "GAUAlphaForQuestionAnswering",
#     "GAUAlphaForMultipleChoice",
# ]
#
# INF = 1e4
#
#
# class Norm(Layer):
#     def __init__(self, epsilon=1e-12):
#         super().__init__()
#         self._epsilon = epsilon
#
#     def forward(self, x):
#         variance = paddle.mean(paddle.square(x), axis=-1, keepdim=True)
#         return x / paddle.sqrt(variance + self._epsilon)
#
#
# def attention_normalize(a, mask=None, axis=-1, method="softmax"):
#     if method == "softmax":
#         return F.softmax(a, axis=axis)
#     else:
#         if mask is not None:
#             l = mask.sum(-1, keepdim=True)
#         else:
#             l = paddle.ones_like(a) * paddle.shape(a)[-2]
#         if method == "squared_relu":
#             return F.relu(a) ** 2 / l
#         elif method == "softmax_plus":
#             scale = paddle.log(l) / np.log(512)
#             # mask: 1 for not padding, 0 for padding
#             # padding position's scale is 1
#             if mask is not None:
#                 scale = scale * mask + 1 - mask
#             return F.softmax(a * scale, axis=axis)
#     return a
#
#
# class ScaleOffset(Layer):
#     def __init__(
#         self,
#         hidden_size=768,
#         scale=True,
#         offset=True,
#     ):
#         super().__init__()
#         self.scale = scale
#         self.offset = offset
#
#         if self.scale:
#             self.weight = self.create_parameter((hidden_size,), default_initializer=nn.initializer.Constant(1.0))
#         if self.offset:
#             self.bias = self.create_parameter((hidden_size,), is_bias=True)
#
#     def forward(self, inputs):
#         if self.scale:
#             inputs = inputs * self.weight
#         if self.offset:
#             inputs = inputs + self.bias
#
#         return inputs
#
#
# class GatedAttentionUnit(Layer):
#     """
#     https://github.com/ZhuiyiTechnology/GAU-alpha/blob/ea15e08a85d35652775c360218090cbaed98da18/models.py#L6-L85
#     """
#
#     def __init__(self, config: GAUAlphaConfig):
#         super().__init__()
#         self.activation = ACT2FN[config.activation]
#         self.intermediate_size = config.intermediate_size
#         self.attention_key_size = config.attention_key_size
#         self.use_bias = config.use_bias
#         self.normalization = config.normalization
#         self.attention_scale = config.attention_scale
#         self.attention_dropout = config.attention_probs_dropout_prob
#
#         self.i_dense = nn.Linear(
#             config.hidden_size,
#             2 * config.intermediate_size + config.attention_key_size,
#             bias_attr=self.use_bias,
#         )
#         self.o_dense = nn.Linear(config.intermediate_size, config.hidden_size, bias_attr=self.use_bias)
#
#         self.q_scaleoffset = ScaleOffset(config.attention_key_size, offset=self.use_bias)
#         self.k_scaleoffset = ScaleOffset(config.attention_key_size, offset=self.use_bias)
#         self.rotary = RotaryPositionEmbedding(config)
#
#     def forward(self, hidden_states, attention_mask=None):
#         x = self.i_dense(hidden_states)
#         u, v, qk = paddle.split(
#             self.activation(x),
#             [self.intermediate_size, self.intermediate_size, self.attention_key_size],
#             axis=-1,
#         )
#         q, k = self.q_scaleoffset(qk), self.k_scaleoffset(qk)
#
#         # apply_rotary
#         q, k = self.rotary(q), self.rotary(k)
#
#         # Attention
#         a = paddle.matmul(q, k, transpose_y=True)
#
#         if self.attention_scale:
#             a = a / self.attention_key_size**0.5
#
#         if attention_mask is not None:
#             a = a * attention_mask + (attention_mask - 1) * INF
#
#         A = attention_normalize(a, attention_mask, axis=-1, method=self.normalization)
#
#         A = F.dropout(A, p=self.attention_dropout, training=self.training)
#
#         o = self.o_dense(u * paddle.matmul(A, v))
#
#         return o
#
#
# class GAULayer(Layer):
#     def __init__(self, config: GAUAlphaConfig):
#         super().__init__()
#         self.gau = GatedAttentionUnit(config)
#         self.norm = Norm(config.norm_eps)
#         self.hidden_dropout = config.hidden_dropout_prob
#
#     def forward(self, hidden_states, attention_mask=None):
#         gau_output = self.gau(hidden_states, attention_mask=attention_mask)
#
#         # dropout and residual
#         o = F.dropout(gau_output[0], p=self.hidden_dropout, training=self.training)
#         o = self.norm(hidden_states + o)
#
#         return o
#
#
# def initializer(tensor, num_hidden_layers=12, order=2, gain=1.0):
#     """
#     https://github.com/bojone/bert4keras/blob/5572ed481a14f5a62be7107e3846c88a5d6b617d/bert4keras/models.py#L1226-L1235
#     """
#     shape = paddle.shape(tensor)
#     if shape[0] > 10000 or shape[0] < 10:
#         hidden_size = shape[1]
#     else:
#         hidden_size = shape[0]
#     gain *= num_hidden_layers ** (-1.0 / order)
#     std = 1.13684723 / hidden_size**0.5 * gain
#
#     return nn.initializer.TruncatedNormal(std=std)
#
#
# class RotaryPositionEmbedding(Layer):
#     def __init__(self, config: GAUAlphaConfig):
#         super().__init__()
#         inv_freq = 1.0 / (
#             10000
#             ** (
#                 paddle.arange(0, config.attention_key_size, 2, dtype=paddle.get_default_dtype())
#                 / config.attention_key_size
#             )
#         )
#         t = paddle.arange(config.max_position_embeddings, dtype=paddle.get_default_dtype())
#         freqs = paddle.matmul(t.unsqueeze(1), inv_freq.unsqueeze(0))
#         self.register_buffer("sin", freqs.sin(), persistable=False)
#         self.register_buffer("cos", freqs.cos(), persistable=False)
#
#     def forward(self, x, offset=0):
#         # x shape [batch_size, seqlen, dim]
#         seqlen = paddle.shape(x)[-2]
#         sin, cos = (
#             self.sin[offset : offset + seqlen, :],
#             self.cos[offset : offset + seqlen, :],
#         )
#         x1, x2 = x[..., 0::2], x[..., 1::2]
#         # [cos_nθ, -sin_nθ] [x1]
#         # [sin_nθ,  cos_nθ] [x2]
#         # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
#         return paddle.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).flatten(-2, -1)
#
#
# class GAUAlphaPretrainedModel(PretrainedModel):
#     """
#     An abstract class for pretrained GAU-alpha models. It provides GAU-alpha related
#     `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
#     `pretrained_resource_files_map`, `base_model_prefix` for downloading and
#     loading pretrained models.
#     See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
#     """
#
#     model_config_file = CONFIG_NAME
#     config_class = GAUAlphaConfig
#     resource_files_names = {"model_state": "model_state.pdparams"}
#     base_model_prefix = "gau_alpha"
#
#     pretrained_init_configuration = GAUAlPHA_PRETRAINED_INIT_CONFIGURATION
#     pretrained_resource_files_map = GAUAlPHA_PRETRAINED_RESOURCE_FILES_MAP
#
#     def _init_weights(self, layer):
#         """Initialization hook"""
#         if isinstance(layer, (nn.Linear, nn.Embedding)):
#             # In the dygraph mode, use the `set_value` to reset the parameter directly,
#             # and reset the `state_dict` to update parameter in static mode.
#             if isinstance(layer.weight, paddle.Tensor):
#                 initializer(layer.weight, self.config.num_hidden_layers, order=2, gain=1.0)
#             if isinstance(layer, nn.Linear):
#                 use_bias = self.use_bias if hasattr(self, "use_bias") else self.gau_alpha.config["use_bias"]
#                 if layer.bias is not None and not use_bias:
#                     layer.bias = None
#
#
# @register_base_model
# class GAUAlphaModel(GAUAlphaPretrainedModel):
#     """
#     The bare GAUAlpha Model transformer outputting raw hidden-states.
#
#     This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
#     Refer to the superclass documentation for the generic methods.
#
#     This model is also a Paddle `paddle.Layer <https://www.paddlepaddle.org.cn/documentation
#     /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
#     and refer to the Paddle documentation for all matter related to general usage and behavior.
#
#     Args:
#         vocab_size (int):
#             Vocabulary size of `inputs_ids` in `GAUAlphaModel`. Also is the vocab size of token embedding matrix.
#             Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `GAUAlphaModel`.
#         hidden_size (int, optional):
#             Dimensionality of the, encoder layers and pooler layer. Defaults to `768`.
#         intermediate_size (int, optional):
#             Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
#             to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
#             and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
#             Defaults to `3072`.
#         num_hidden_layers (int, optional):
#             Number of hidden layers in the gau_alpha encoder. Defaults to `12`.
#         max_position_embeddings (int, optional):
#             The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
#             sequence. Defaults to `512`.
#         type_vocab_size (int, optional):
#             The vocabulary size of `token_type_ids`.
#             Defaults to `2`.
#         attention_key_size (int, optional):
#             The dimensionality of the key used in the gau layer. Defaults to `128`.
#         norm_eps (float, optional):
#             The epsilon value used in the normalization layer.
#             Defaults to `1e-12`.
#         pad_token_id (int, optional):
#             The index of padding token in the token vocabulary.
#             Defaults to `0`.
#         hidden_dropout_prob (float, optional):
#             The dropout probability for all fully connected layers in the embeddings and encoder.
#             Defaults to `0.1`.
#         attention_probs_dropout_prob (float, optional):
#             The dropout probability used in gau in all encoder layers to drop some attention target.
#             Defaults to `0.1`.
#         hidden_act (str, optional):
#             The activation function used in gau layer. Defaults to `swish`.
#         use_bias (bool, optional):
#             Whether or not use bias.
#             Defaults to `False`.
#         normalization (str, optional):
#             The normalization method used in gau layer.
#             Defaults to `softmax_plus`.
#         attention_scale (bool, optional):
#             Whether or not to scale the attention scores.
#             Defaults to `True`.
#     """
#
#     def __init__(self, config: GAUAlphaConfig):
#         super(GAUAlphaModel, self).__init__(config)
#         self.pad_token_id = config.pad_token_id
#         self.norm_eps = config.norm_eps
#         self.num_hidden_layers = config.num_hidden_layers
#         self.use_bias = config.use_bias
#         self.embeddings = GAUAlphaEmbeddings(config)
#
#         self.encoder = GAUAlphaEncoder(config)
#
#     def get_input_embeddings(self):
#         return self.embeddings.word_embeddings
#
#     def set_input_embeddings(self, value):
#         self.embeddings.word_embeddings = value
#
#     def forward(
#         self,
#         input_ids=None,
#         attention_mask=None,
#         token_type_ids=None,
#     ):
#         r"""
#         The GAUAlphaModel forward method, overrides the `__call__()` special method.
#
#         Args:
#             input_ids (Tensor):
#                 Indices of input sequence tokens in the vocabulary. They are
#                 numerical representations of tokens that build the input sequence.
#                 Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
#             attention_mask (Tensor, optional):
#                 Mask used in gau to avoid performing attention on to some unwanted positions,
#                 usually the paddings or the subsequent positions.
#                 Its data type can be int, float and bool.
#                 When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
#                 When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
#                 It is a tensor with shape broadcasted to `[batch_size, sequence_length, sequence_length]`.
#                 Defaults to `None`, which means nothing needed to be prevented attention to.
#             token_type_ids (Tensor, optional):
#                 Segment token indices to indicate different portions of the inputs.
#                 Selected in the range ``[0, type_vocab_size - 1]``.
#                 If `type_vocab_size` is 2, which means the inputs have two portions.
#                 Indices can either be 0 or 1:
#
#                 - 0 corresponds to a *sentence A* token,
#                 - 1 corresponds to a *sentence B* token.
#
#                 Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
#                 Defaults to `None`, which means we don't add segment embeddings.
#
#         Returns:
#             tuple: Returns `last_hidden_state` (Tensor)
#                 Sequence of hidden-states at the last layer of the model.
#                 It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
#
#         Example:
#             .. code-block::
#
#                 import paddle
#                 from paddlenlp.transformers import GAUAlphaModel, GAUAlphaTokenizer
#
#                 tokenizer = GAUAlphaTokenizer.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#                 model = GAUAlphaModel.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#
#                 inputs = tokenizer("欢迎使用百度飞桨!")
#                 inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
#                 last_hidden_state = model(**inputs)
#
#         """
#
#         if attention_mask is None:
#             attention_mask = input_ids != self.pad_token_id
#         if attention_mask.ndim == 2:
#             attention_mask = attention_mask.unsqueeze(1)  # bs, 1, seqlen
#         attention_mask = attention_mask.astype(paddle.get_default_dtype())
#         attention_mask.stop_gradient = True
#
#         embedding_output = self.embeddings(
#             input_ids=input_ids,
#             token_type_ids=token_type_ids,
#         )
#
#         last_hidden_state = self.encoder(embedding_output, attention_mask=attention_mask)
#
#         return last_hidden_state
#
#
# class GAUAlphaEmbeddings(Layer):
#     """
#     Include embeddings from word and token_type embeddings
#     """
#
#     def __init__(self, config: GAUAlphaConfig):
#         super(GAUAlphaEmbeddings, self).__init__()
#         self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
#         self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
#         self.norm = Norm(config.norm_eps)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#     def forward(self, input_ids, token_type_ids=None):
#         if token_type_ids is None:
#             token_type_ids = paddle.zeros_like(input_ids, dtype="int64")
#
#         input_embedings = self.word_embeddings(input_ids)
#         token_type_embeddings = self.token_type_embeddings(token_type_ids)
#
#         embeddings = input_embedings + token_type_embeddings
#         embeddings = self.norm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings
#
#
# class GAUAlphaEncoder(Layer):
#     def __init__(self, config: GAUAlphaConfig):
#         super().__init__()
#         self.layer = nn.LayerList([GAULayer(config) for _ in range(config.num_hidden_layers)])
#
#     def forward(self, hidden_states, attention_mask=None):
#         for layer_module in self.layer:
#             hidden_states = layer_module(
#                 hidden_states,
#                 attention_mask,
#             )
#         return hidden_states
#
#
# class GAUAlphaForQuestionAnswering(GAUAlphaPretrainedModel):
#     """
#     GAUAlpha with a linear layer on top of the hidden-states output to compute `span_start_logits`
#     and `span_end_logits`, designed for question-answering tasks like SQuAD.
#
#     Args:
#         gau_alpha (:class:`GAUAlphaModel`):
#             An instance of GAUAlphaModel.
#         dropout (float, optional):
#             The dropout probability for output of GAUAlpha.
#             If None, use the same value as `hidden_dropout_prob` of `GAUAlphaModel`
#             instance `gau_alpha`. Defaults to `None`.
#     """
#
#     def __init__(self, config: GAUAlphaConfig):
#         super(GAUAlphaForQuestionAnswering, self).__init__(config)
#         self.gau_alpha = GAUAlphaModel(config)
#         self.dropout = nn.Dropout(
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.classifier = nn.Linear(config.hidden_size, 2)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None):
#         r"""
#         The GAUAlphaForQuestionAnswering forward method, overrides the __call__() special method.
#
#         Args:
#             input_ids (Tensor):
#                 See :class:`GAUAlphaModel`.
#             token_type_ids (Tensor, optional):
#                 See :class:`GAUAlphaModel`.
#             attention_mask (Tensor, optional):
#                 See :class:`GAUAlphaModel`.
#
#         Returns:
#             tuple: Returns tuple (`start_logits`, `end_logits`).
#
#             With the fields:
#
#             - `start_logits` (Tensor):
#                 A tensor of the input token classification logits, indicates the start position of the labelled span.
#                 Its data type should be float32 and its shape is [batch_size, sequence_length].
#
#             - `end_logits` (Tensor):
#                 A tensor of the input token classification logits, indicates the end position of the labelled span.
#                 Its data type should be float32 and its shape is [batch_size, sequence_length].
#
#         Example:
#             .. code-block::
#
#                 import paddle
#                 from paddlenlp.transformers import GAUAlphaForQuestionAnswering, GAUAlphaTokenizer
#
#                 tokenizer = GAUAlphaTokenizer.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#                 model = GAUAlphaForQuestionAnswering.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#
#                 inputs = tokenizer("欢迎使用百度飞桨!")
#                 inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
#                 outputs = model(**inputs)
#
#                 start_logits = outputs[0]
#                 end_logits = outputs[1]
#         """
#         sequence_output = self.gau_alpha(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#
#         logits = self.classifier(sequence_output)
#         start_logits, end_logits = paddle.unstack(logits, axis=-1)
#
#         return start_logits, end_logits
#
#
# class GAUAlphaForSequenceClassification(GAUAlphaPretrainedModel):
#     """
#     GAUAlpha Model with a linear layer on top of the output layer,
#     designed for sequence classification/regression tasks like GLUE tasks.
#
#     Args:
#         gau_alpha (`GAUAlphaModel`):
#             An instance of `paddlenlp.transformers.GAUAlphaModel`.
#         num_labels (int, optional):
#             The number of classes. Default to `2`.
#         dropout (float, optional):
#             The dropout probability for output of GAUAlpha.
#             If None, use the same value as `hidden_dropout_prob`
#             of `paddlenlp.transformers.GAUAlphaModel` instance. Defaults to `None`.
#     """
#
#     def __init__(self, config: GAUAlphaConfig):
#         super(GAUAlphaForSequenceClassification, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.gau_alpha = GAUAlphaModel(config)
#         self.dropout = nn.Dropout(
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None):
#         r"""
#         Args:
#             input_ids (Tensor):
#                 See :class:`GAUAlphaModel`.
#             token_type_ids (Tensor, optional):
#                 See :class:`GAUAlphaModel`.
#             attention_mask (Tensor, optional):
#                 See :class:`GAUAlphaModel`.
#
#         Returns:
#             Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
#             Shape as `[batch_size, num_labels]` and dtype as float32.
#
#         Example:
#             .. code-block::
#
#                 import paddle
#                 from paddlenlp.transformers import GAUAlphaForSequenceClassification, GAUAlphaTokenizer
#
#                 tokenizer = GAUAlphaTokenizer.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#                 model = GAUAlphaForSequenceClassification.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#
#                 inputs = tokenizer("欢迎使用百度飞桨!")
#                 inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
#                 logits = model(**inputs)
#
#         """
#         sequence_output = self.gau_alpha(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#         pooled_output = sequence_output[:, 0]
#
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         return logits
#
#
# class GAUAlphaForTokenClassification(GAUAlphaPretrainedModel):
#     """
#     GAUAlpha Model with a linear layer on top of the hidden-states output layer,
#     designed for token classification tasks like NER tasks.
#
#     Args:
#         gau_alpha (`GAUAlphaModel`):
#             An instance of `paddlenlp.transformers.GAUAlphaModel`.
#         num_labels (int, optional):
#             The number of classes. Default to `2`.
#         dropout (float, optional):
#             The dropout probability for output of GAUAlpha.
#             If None, use the same value as `hidden_dropout_prob`
#             of `paddlenlp.transformers.GAUAlphaModel` instance. Defaults to `None`.
#     """
#
#     def __init__(self, config: GAUAlphaConfig):
#         super(GAUAlphaForTokenClassification, self).__init__(config)
#         self.num_labels = config.num_labels
#         self.gau_alpha = GAUAlphaModel(config)  # allow gau_alpha to be config
#         self.dropout = nn.Dropout(
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.classifier = nn.Linear(config.hidden_size, config.num_labels)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None):
#         r"""
#         Args:
#             input_ids (Tensor):
#                 See :class:`GAUAlphaModel`.
#             token_type_ids (Tensor, optional):
#                 See :class:`GAUAlphaModel`.
#             attention_mask (Tensor, optional):
#                 See :class:`GAUAlphaModel`.
#
#         Returns:
#             Tensor: Returns tensor `logits`, a tensor of the input token classification logits.
#             Shape as `[batch_size, sequence_length, num_labels]` and dtype as `float32`.
#
#         Example:
#             .. code-block::
#
#                 import paddle
#                 from paddlenlp.transformers import GAUAlphaForTokenClassification, GAUAlphaTokenizer
#
#                 tokenizer = GAUAlphaTokenizer.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#                 model = GAUAlphaForTokenClassification.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#
#                 inputs = tokenizer("欢迎使用百度飞桨!")
#                 inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
#                 logits = model(**inputs)
#
#         """
#         sequence_output = self.gau_alpha(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#
#         sequence_output = self.dropout(sequence_output)
#         logits = self.classifier(sequence_output)
#         return logits
#
#
# class GAUAlphaForMultipleChoice(GAUAlphaPretrainedModel):
#     """
#     GAUAlpha Model with a linear layer on top of the hidden-states output layer,
#     designed for multiple choice tasks like RocStories/SWAG tasks.
#
#     Args:
#         gau_alpha (:class:`GAUAlphaModel`):
#             An instance of GAUAlphaModel.
#         num_choices (int, optional):
#             The number of choices. Defaults to `2`.
#         dropout (float, optional):
#             The dropout probability for output of GAUAlpha.
#             If None, use the same value as `hidden_dropout_prob` of `GAUAlphaModel`
#             instance `gau_alpha`. Defaults to None.
#     """
#
#     def __init__(self, config: GAUAlphaConfig):
#         super(GAUAlphaForMultipleChoice, self).__init__(config)
#         self.num_choices = config.num_choices
#         self.gau_alpha = GAUAlphaModel(config)
#         self.dropout = nn.Dropout(
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.classifier = nn.Linear(config.hidden_size, 1)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None):
#         r"""
#         The GAUAlphaForMultipleChoice forward method, overrides the __call__() special method.
#
#         Args:
#             input_ids (Tensor):
#                 See :class:`GAUAlphaModel` and shape as [batch_size, num_choice, sequence_length].
#             token_type_ids(Tensor, optional):
#                 See :class:`GAUAlphaModel` and shape as [batch_size, num_choice, sequence_length].
#             attention_mask (list, optional):
#                 See :class:`GAUAlphaModel` and shape as [batch_size, num_choice, sequence_length].
#
#         Returns:
#             Tensor: Returns tensor `reshaped_logits`, a tensor of the multiple choice classification logits.
#             Shape as `[batch_size, num_choice]` and dtype as `float32`.
#
#         Example:
#             .. code-block::
#
#                 import paddle
#                 from paddlenlp.transformers import GAUAlphaForMultipleChoice, GAUAlphaTokenizer
#                 from paddlenlp.data import Pad
#
#                 tokenizer = GAUAlphaTokenizer.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#                 model = GAUAlphaForMultipleChoice.from_pretrained('chinese_GAU-alpha-char_L-24_H-768', num_choices=2)
#
#                 data = [
#                     {
#                         "question": "如何打开ipad屏幕？",
#                         "answer1": "按音量按钮。",
#                         "answer2": "按下锁定按钮。",
#                         "label": 1,
#                     },
#                     {
#                         "question": "如何缩进一些文本？",
#                         "answer1": "在开始写之前留一些空格。",
#                         "answer2": "按空格键。",
#                         "label": 0,
#                     },
#                 ]
#
#                 text = []
#                 text_pair = []
#                 for d in data:
#                     text.append(d["question"])
#                     text_pair.append(d["answer1"])
#                     text.append(d["question"])
#                     text_pair.append(d["answer2"])
#
#                 inputs = tokenizer(text, text_pair)
#                 input_ids = Pad(axis=0, pad_val=tokenizer.pad_token_id)(inputs["input_ids"])
#                 token_type_ids = Pad(axis=0, pad_val=tokenizer.pad_token_type_id)(inputs["token_type_ids"])
#
#                 reshaped_logits = model(
#                     input_ids=paddle.to_tensor(input_ids, dtype="int64"),
#                     token_type_ids=paddle.to_tensor(token_type_ids, dtype="int64"),
#                 )
#                 print(reshaped_logits.shape)
#                 # [2, 2]
#
#         """
#         # input_ids: [bs, num_choice, seq_l]
#         input_ids = input_ids.reshape(shape=(-1, paddle.shape(input_ids)[-1]))  # flat_input_ids: [bs*num_choice,seq_l]
#
#         if token_type_ids is not None:
#             token_type_ids = token_type_ids.reshape(shape=(-1, paddle.shape(token_type_ids)[-1]))
#
#         if attention_mask is not None:
#             attention_mask = attention_mask.reshape(shape=(-1, paddle.shape(attention_mask)[-1]))
#
#         sequence_output = self.gau_alpha(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#
#         pooled_output = sequence_output[:, 0]
#         pooled_output = self.dropout(pooled_output)
#
#         logits = self.classifier(pooled_output)  # logits: (bs*num_choice,1)
#         reshaped_logits = logits.reshape(shape=(-1, self.num_choices))  # logits: (bs, num_choice)
#
#         return reshaped_logits
#
#
# class GAUAlphaLMPredictionHead(Layer):
#     def __init__(
#         self,
#         config: GAUAlphaConfig,
#         embedding_weights=None,
#     ):
#         super(GAUAlphaLMPredictionHead, self).__init__()
#         self.use_bias = config.use_bias
#         self.decoder_weight = (
#             self.create_parameter(shape=[config.vocab_size, config.hidden_size], dtype=self.transform.weight.dtype)
#             if embedding_weights is None
#             else embedding_weights
#         )
#         if self.use_bias:
#             self.decoder_bias = self.create_parameter(
#                 shape=[config.vocab_size], dtype=self.decoder_weight.dtype, is_bias=True
#             )
#
#     def forward(self, hidden_states):
#         hidden_states = paddle.matmul(hidden_states, self.decoder_weight, transpose_y=True)
#         if self.use_bias:
#             hidden_states = hidden_states + self.decoder_bias
#
#         return hidden_states
#
#
# class GAUAlphaForMaskedLM(GAUAlphaPretrainedModel):
#     """
#     GAUAlpha Model with a `masked language modeling` head on top.
#
#     Args:
#         gau_alpha (:class:`GAUAlphaModel`):
#             An instance of :class:`GAUAlphaModel`.
#
#     """
#
#     def __init__(
#         self,
#         config: GAUAlphaConfig,
#     ):
#         super(GAUAlphaForMaskedLM, self).__init__(config)
#         self.gau_alpha = GAUAlphaModel(config)
#         self.cls = GAUAlphaLMPredictionHead(
#             config=config,
#             embedding_weights=self.gau_alpha.embeddings.word_embeddings.weight,
#         )
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None):
#         r"""
#
#         Args:
#             input_ids (Tensor):
#                 See :class:`GAUAlphaModel`.
#             token_type_ids (Tensor, optional):
#                 See :class:`GAUAlphaModel`.
#             attention_mask (Tensor, optional):
#                 See :class:`GAUAlphaModel`.
#
#         Returns:
#             Tensor: Returns tensor `prediction_scores`, The scores of masked token prediction.
#             Its data type should be float32 and shape is [batch_size, sequence_length, vocab_size].
#
#         Example:
#             .. code-block::
#
#                 import paddle
#                 from paddlenlp.transformers import GAUAlphaForMaskedLM, GAUAlphaTokenizer
#
#                 tokenizer = GAUAlphaTokenizer.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#                 model = GAUAlphaForMaskedLM.from_pretrained('chinese_GAU-alpha-char_L-24_H-768')
#
#                 inputs = tokenizer("欢迎使用百度飞桨!")
#                 inputs = {k:paddle.to_tensor([v], dtype="int64") for (k, v) in inputs.items()}
#
#                 logits = model(**inputs)
#                 print(logits.shape)
#                 # [1, 11, 12000]
#
#         """
#         sequence_output = self.gau_alpha(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#
#         prediction_scores = self.cls(sequence_output)
#         return prediction_scores