# coding=utf-8
# Copyright 2018 Google AI, Google Brain and the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ALBERT model. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import copy
import math
import logging
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from transformers.modeling_utils import PreTrainedModel, prune_layer
from transformers.modeling_bert import BertSelfAttention, load_tf_weights_in_bert, BertPooler, BertIntermediate, BertOutput, BertPredictionHeadTransform, BertSelfOutput, BertLMPredictionHead, BertOnlyMLMHead, BertOnlyMLMHead, BertEmbeddings, BertOnlyNSPHead
from transformers import AutoModel, BertTokenizer, AlbertConfig
from transformers.modeling_albert import AlbertMLMHead, AlbertPreTrainedModel, load_tf_weights_in_albert
from torch.nn import CrossEntropyLoss, MSELoss
from activations import ACT2FN
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.
    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.
    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_linear_layer(layer: torch.nn.Linear, index: torch.LongTensor, dim: int = 0) -> torch.nn.Linear:
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.
    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    This function chunks the :obj:`input_tensors` into smaller input tensor parts of size :obj:`chunk_size` over the
    dimension :obj:`chunk_dim`. It then applies a layer :obj:`forward_fn` to each chunk independently to save memory.
    If the :obj:`forward_fn` is independent across the :obj:`chunk_dim` this function will yield the same result as
    directly applying :obj:`forward_fn` to :obj:`input_tensors`.
    Args:
        forward_fn (:obj:`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (:obj:`int`):
            The chunk size of a chunked tensor: :obj:`num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (:obj:`int`):
            The dimension over which the :obj:`input_tensors` should be chunked.
        input_tensors (:obj:`Tuple[torch.Tensor]`):
            The input tensors of ``forward_fn`` which will be chunked
    Returns:
        :obj:`torch.Tensor`: A tensor with the same shape as the :obj:`forward_fn` would have given if applied`.
    Examples::
        # rename the usual forward() fn to forward_chunk()
        def forward_chunk(self, hidden_states):
            hidden_states = self.decoder(hidden_states)
            return hidden_states
        # implement a chunked forward function
        def forward(self, hidden_states):
            return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    """

    assert len(input_tensors) > 0, "{} has to be a tuple/list of tensors".format(input_tensors)
    tensor_shape = input_tensors[0].shape[chunk_dim]
    assert all(
        input_tensor.shape[chunk_dim] == tensor_shape for input_tensor in input_tensors
    ), "All input tenors have to be of the same shape"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    assert num_args_in_forward_chunk_fn == len(
        input_tensors
    ), "forward_chunk_fn expects {} arguments, but only {} input tensors are given".format(
        num_args_in_forward_chunk_fn, len(input_tensors)
    )

    if chunk_size > 0:
        assert (
            input_tensors[0].shape[chunk_dim] % chunk_size == 0
        ), "The dimension to be chunked {} has to be a multiple of the chunk size {}".format(
            input_tensors[0].shape[chunk_dim], chunk_size
        )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)



_CONFIG_FOR_DOC = "AlbertConfig"
_TOKENIZER_FOR_DOC = "AlbertTokenizer"


ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "albert-base-v1",
    "albert-large-v1",
    "albert-xlarge-v1",
    "albert-xxlarge-v1",
    "albert-base-v2",
    "albert-large-v2",
    "albert-xlarge-v2",
    "albert-xxlarge-v2",
    # See all ALBERT models at https://huggingface.co/models?filter=albert
]


class AlbertEmbeddings(BertEmbeddings):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__(config)

        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)
        self.LayerNorm = torch.nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)


class AlbertAttention(BertSelfAttention):
    def __init__(self, config):
        super().__init__(config)

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads, self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, input_ids, attention_mask=None, head_mask=None, output_attentions=False, history_states=None):
        if history_states is not None:
            # print("history_states", history_states.shape, "input_ids", input_ids.shape)
            x_states = torch.cat((history_states, input_ids), dim=1)  # history_states：连     input_ids：小Mask        K  连 小 Mask   V   连 小 Mask   Q 小 Mask
            mixed_query_layer = self.query(input_ids)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(input_ids)
            mixed_key_layer = self.key(input_ids)
            mixed_value_layer = self.value(input_ids)


        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.output_dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # self.chunk_size_feed_forward = 768
        self.seq_len_dim = 1
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, history_states=None
    ):
        attention_output = self.attention(hidden_states, attention_mask, head_mask, output_attentions, history_states)

        ffn_output = apply_chunking_to_forward(
            self.ff_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output[0],
        )
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output[0])

        return (hidden_states,) + attention_output[1:]  # add attentions if we output them

    def ff_chunk(self, attention_output):
        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        return ffn_output


class AlbertLayerGroup(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(
        self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, history_states=None
    ):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            layer_output = albert_layer(hidden_states, attention_mask, head_mask[layer_index], output_attentions, history_states=history_states)
            hidden_states = layer_output[0]

            if output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class AlbertTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=False,
        prev_embedding=None,
        prev_encoded_layers=None
    ):

        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = (hidden_states,) if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if prev_embedding is not None and prev_encoded_layers is not None:
            history_states = self.embedding_hidden_mapping_in(prev_embedding)
            for i in range(self.config.num_hidden_layers):
                # Number of layers in a hidden group
                layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

                # Index of the hidden group
                group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

                layer_group_output = self.albert_layer_groups[group_idx](
                    hidden_states,
                    attention_mask,
                    head_mask[group_idx * layers_per_group: (group_idx + 1) * layers_per_group],
                    output_attentions,
                    output_hidden_states,
                    history_states=history_states
                )
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
                hidden_states = layer_group_output[0]

                if output_attentions:
                    all_attentions = all_attentions + layer_group_output[-1]

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
        else:
            for i in range(self.config.num_hidden_layers):
                # Number of layers in a hidden group
                layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

                # Index of the hidden group
                group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

                layer_group_output = self.albert_layer_groups[group_idx](
                    hidden_states,
                    attention_mask,
                    head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                    output_attentions,
                    output_hidden_states,
                )
                hidden_states = layer_group_output[0]

                if output_attentions:
                    all_attentions = all_attentions + layer_group_output[-1]

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)


class AlbertPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = AlbertConfig
    base_model_prefix = "albert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class AlbertModel(AlbertPreTrainedModel):

    config_class = AlbertConfig
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def __init__(self, config):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
        If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
        is a total of 4 different layers.

        These layers are flattened: the indices [0,1] correspond to the two inner groups of the first hidden layer,
        while [2,3] correspond to the two inner groups of the second hidden layer.

        Any layer with in index other than [0,1,2,3] will result in an error.
        See base class PreTrainedModel for more information about head pruning
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        prev_embedding=None,
        prev_encoded_layers=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            prev_embedding=prev_embedding,
            prev_encoded_layers=prev_encoded_layers
        )


        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]



class AlbertForPreTraining(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)
        self.sop_classifier = AlbertSOPHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.predictions.decoder

    def get_input_embeddings(self):
        return self.albert.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sentence_order_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):

        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        total_loss = None
        if labels is not None and sentence_order_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            sentence_order_loss = loss_fct(sop_scores.view(-1, 2), sentence_order_label.view(-1))
            total_loss = masked_lm_loss + sentence_order_loss

        if not return_dict:
            output = (prediction_scores, sop_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return AlbertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            sop_logits=sop_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AlbertForMaskedLM(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.predictions.decoder

    def get_input_embeddings(self):
        return self.albert.embeddings.word_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_outputs = outputs[0]

        prediction_scores = self.predictions(sequence_outputs)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output




class LabelSmoothingLoss(_Loss):
    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0, size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob.type_as(output), reduction='none').view(batch_size, num_pos, -1).sum(2)



class AlbertForSeq2Seq(AlbertPreTrainedModel):
    """refer to BertForPreTraining"""
    def __init__(self, config):
        super(AlbertForSeq2Seq, self).__init__(config)
        # self.albert = AutoModel.from_config(config)
        self.albert = AlbertModel(config)
        # self.cls = BertOnlyMLMHead(config)
        self.predictions = AlbertMLMHead(config)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None
        self.init_weights()


    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, masked_pos=None, masked_weights=None, num_tokens_a=None, num_tokens_b=None):
        # print(attention_mask[0], input_ids.shape, token_type_ids.shape, attention_mask.shape)
        # print(self.bert)
        # sequence_output = self.bert(
        #     input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]  # batch_size * 512 *

        embedding_output = self.albert.embeddings(
            input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
        )

        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        output_attentions = self.config.output_attentions
        output_hidden_states = self.config.output_hidden_states
        return_dict = self.config.use_return_dict

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0



        # print("head_mask", head_mask)
        # input()
        encoder_outputs = self.albert.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        # print(sequence_output.shape)

        def gather_seq_out_by_pos(seq, pos):
            return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

        def gather_seq_out_by_pos_average(seq, pos, mask):
            batch_size, max_token_num = pos.size(0), pos.size(-1)
            pos_vec = torch.gather(seq, 1, pos.view(batch_size, -1).unsqueeze(
                2).expand(-1, -1, seq.size(-1))).view(batch_size, -1, max_token_num, seq.size(-1))
            mask = mask.type_as(pos_vec)
            pos_vec_masked_sum = (
                pos_vec * mask.unsqueeze(3).expand_as(pos_vec)).sum(2)
            return pos_vec_masked_sum / mask.sum(2, keepdim=True).expand_as(pos_vec_masked_sum)

        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        if masked_lm_labels is None:
            if masked_pos is None:
                prediction_scores = self.predictions(sequence_output)
            else:
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores = self.predictions(sequence_output_masked)
            return prediction_scores

        sequence_output_masked = gather_seq_out_by_pos(
            sequence_output, masked_pos)  # sequence_output: batch_size * 512 * 768
        # print("sequence_output_masked: ", sequence_output_masked.shape)
        prediction_scores_masked = self.predictions(sequence_output_masked)
        if self.crit_mask_lm_smoothed:
            masked_lm_loss = self.crit_mask_lm_smoothed(
                F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
        else:
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
        # print(prediction_scores_masked.transpose(1, 2).shape, masked_lm_labels, masked_weights)

        masked_lm_loss = loss_mask_and_normalize(
            masked_lm_loss.float(), masked_weights)
        # print(masked_lm_loss)
        # input()

        return masked_lm_loss


class AlbertModelIncr(AlbertModel):
    def __init__(self, config):
        super(AlbertModelIncr, self).__init__(config)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, output_all_encoded_layers=True, prev_embedding=None,
                prev_encoded_layers=None):
        # print(attention_mask.shape, attention_mask)
        # extended_attention_mask = self.get_extended_attention_mask(
        #     input_ids, token_type_ids, attention_mask)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        # extended_attention_mask e= extended_attention_mask.to(
        #     dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids)

        head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
        output_attentions = self.config.output_attentions
        return_dict = self.config.use_return_dict

        sequence_output, encoded_layers = self.encoder(embedding_output,
                                                       extended_attention_mask,
                                                       head_mask=head_mask,
                                                       output_hidden_states=output_all_encoded_layers,
                                                       output_attentions=output_attentions,
                                                       return_dict=return_dict,
                                                       prev_embedding=prev_embedding,
                                                       prev_encoded_layers=prev_encoded_layers)


        # sequence_output = encoded_layers[-1]
        # print(sequence_output.shape, len(encoded_layers), encoded_layers[-1].shape)
        # input()
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


class AlbertForSeq2SeqDecode(AlbertPreTrainedModel):
    def __init__(self, config, mask_word_id=0,
                 search_beam_size=1, length_penalty=1.0, eos_id=0, sos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, ngram_size=3, min_len=0):
        super(AlbertForSeq2SeqDecode, self).__init__(config)
        self.albert = AlbertModelIncr(config)
        self.predictions = AlbertMLMHead(config)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.mask_word_id = mask_word_id
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        self.min_len = min_len
        self.init_weights()
        # self.tie_weights()

    # def tie_weights(self):
    #     self._tie_or_clone_weights(self.cls.predictions.decoder,
    #                                self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask):
        if self.search_beam_size > 1:
            return self.beam_search(input_ids, token_type_ids, position_ids, attention_mask)

        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids # 32*384
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        # print("curr_ids", curr_ids.shape)
        # print("mask_ids", mask_ids.shape) # 32*1
        # input()
        next_pos = input_length


        num = 0
        while num < 20 and next_pos < output_length:
            num += 1
            curr_length = list(curr_ids.size())[1]   # 384
            # print("################ curr_ids: ", curr_ids, curr_ids.size())

            start_pos = next_pos - curr_length    # 384 - 384 = 0
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)  # 32 * 385

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos+1]  # 32 * 385
            curr_attention_mask = attention_mask[:,
                                                 start_pos:next_pos+1, :next_pos+1] # 32 * 385 * 385
            curr_position_ids = position_ids[:, start_pos:next_pos+1]
            # print("start_pos", start_pos)  # 0 384
            # print("next_pos", next_pos)  # 384 385
            #
            # if prev_embedding is not None:
            #     print("x_input_ids", x_input_ids)
            #     print("prev_embedding", prev_embedding.shape)
            #     print("prev_encoded_layers", len(prev_encoded_layers), prev_encoded_layers[0].shape)
            #     print("curr_token_type_ids", curr_token_type_ids, curr_token_type_ids[0])
            #     print("curr_position_ids", curr_position_ids)
            #     print("curr_attention_mask", curr_attention_mask)
            # input()

            # if prev_encoded_layers is not None:
            #     print("x_input_ids", x_input_ids.shape, "new_embedding", prev_embedding.shape, "new_encoded_layers",
            #           prev_encoded_layers[-1].shape)

            new_embedding, new_encoded_layers, _ = \
                self.albert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                          output_all_encoded_layers=True, prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers)

            new_encoded_layers = new_encoded_layers[1:]

            # print("self.config.num_hidden_layers", self.config.num_hidden_layers, len(new_encoded_layers))
            # input()


            # input()

            # head_mask = self.get_head_mask(None, self.config.num_hidden_layers)
            # output_attentions = self.config.output_attentions
            # return_dict = self.config.use_return_dict
            #
            # sequence_output, encoded_layers = self.albert(x_input_ids,
            #                                                curr_attention_mask,
            #                                                head_mask=head_mask,
            #                                                output_hidden_states=True,
            #                                                output_attentions=output_attentions,
            #                                                return_dict=return_dict,
            #                                                prev_embedding=prev_embedding,
            #                                                prev_encoded_layers=prev_encoded_layers)

            # print("new_encoded_layers", new_encoded_layers[-1].shape)
            # print("")

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            # print("last_hidden", last_hidden.shape)

            prediction_scores = self.predictions(last_hidden)
            _, max_ids = torch.max(prediction_scores, dim=-1)
            output_ids.append(max_ids)

            if prev_embedding is None:
                prev_embedding = new_embedding[:, :-1, :]
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)
            if prev_encoded_layers is None:
                prev_encoded_layers = [x[:, :-1, :]
                                       for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
            curr_ids = max_ids
            next_pos += 1

        return torch.cat(output_ids, dim=1)

    def beam_search(self, input_ids, token_type_ids, position_ids, attention_mask):
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        K = self.search_beam_size
        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                                                 start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            new_embedding, new_encoded_layers, _ = \
                self.albert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                          output_all_encoded_layers=True, prev_embedding=prev_embedding, prev_encoded_layers=prev_encoded_layers)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores = self.predictions(last_hidden)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (next_pos-input_length+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float())
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_embedding is None)

            if prev_embedding is None:
                prev_embedding = first_expand(new_embedding[:, :-1, :])
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                prev_embedding = select_beam_items(
                    prev_embedding, back_ptrs)
            if prev_encoded_layers is None:
                prev_encoded_layers = [first_expand(
                    x[:, :-1, :]) for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
                prev_encoded_layers = [select_beam_items(
                    x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1
            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims
            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, padding_value=0).to(input_ids.device)

        return traces



# config = AlbertConfig.from_pretrained("voidful/albert_chinese_base")
# model = AlbertForSeq2Seq.from_pretrained("voidful/albert_chinese_base", config=config)