import copy
import math

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from torchcrf import CRF
# biobert
from transformers import BertModel, BertPreTrainedModel
# albert
from transformers import AlbertConfig, AlbertModel

from transformers.modeling_outputs import TokenClassifierOutput  ## dict

from hyper_parameters import Hyper


# Attention
class CustomSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        ## Note
        self.num_attention_heads = config.num_attention_heads
        ## Note config.hidden_size birnn_hidden_size 指的是所有head总维数
        ## Todo multi head self-attention
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        ## Note 近似 all_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            ## Note key value
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        # 缩放dot-product 注意力， 为了梯度稳定？
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class CustomSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class CustomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = CustomSelfAttention(config)
        self.output = CustomSelfOutput(config)
        self.pruned_heads = set()

    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return
    #     heads, index = find_pruneable_heads_and_indices(
    #             heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
    #     )
    #
    #     # Prune linear layers
    #     self.self.query = prune_linear_layer(self.self.query, index)
    #     self.self.key = prune_linear_layer(self.self.key, index)
    #     self.self.value = prune_linear_layer(self.self.value, index)
    #     self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
    #
    #     # Update hyper params and store pruned heads
    #     self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
    #     self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# BertModel是BertPreTrainedModel的子类
class BertBilstmCrf(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        # 如果想要自定义初始化， 可以改写这里
        super(BertBilstmCrf, self).__init__(config)
        self.num_labels = config.num_labels

        self.using_attention = Hyper.using_attention
        # Todo 下面两行可以删掉，
        config.word_max_length = 8
        config.word_embedding_size = config.hidden_size

        self.bert = BertModel(config, add_pooling_layer=False)

        # TODO 测试一下dropout
        ## TODO GRU vs LSTM
        ## Note Param
        self.birnn_hidden_size = Hyper.birnn_hidden_state  # 64 128 256 512
        # self.birnn = nn.LSTM(config.hidden_size,  # Note emb = word + char
        #                      self.birnn_hidden_size,
        #                      num_layers=2,  # Note
        #                      bidirectional=True,
        #                      batch_first=True,
        #                      dropout=0.2  # 0.1 0.2 0.3
        #                      )

        self.birnn = nn.GRU(config.hidden_size,  # Note emb = word + char
                            self.birnn_hidden_size,
                            num_layers=2,  # Note
                            bidirectional=True,
                            batch_first=True,
                            dropout=Hyper.birnn_dropout_prob  # 0.1 0.2 0.3
                            )

        # * 2的原因， 使用了双向的信息， 把forward 和 bacKward 做一个拼接，
        out_dim = self.birnn_hidden_size * 2

        ## TODO Attention after LSTM,
        if Hyper.using_attention:
            attentionB_config = copy.deepcopy(config)
            ## Note 多头注意力机制

            attentionB_config.num_attention_heads = 4
            attentionB_config.hidden_size = self.birnn_hidden_size * 2
            self.attentionB = CustomAttention(attentionB_config)

        ## Note attention线性映射维度
        # Token_CLS
        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        # self.hidden2tag = nn.Linear(300, config.num_labels)

        ## inference primary task
        self.crf = CRF(config.num_labels, batch_first=True)

        ##  辅助任务 label = [0, 1, 3] B I O
        self.simpleCrfLinear = nn.Linear(out_dim, 3)
        self.simpleCrf = CRF(3, batch_first=True)
        # sentence_CLS作用不大, 先删掉
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,  #
            head_mask=None,  #
            inputs_embeds=None,  #
            labels=None,
            output_attentions=None,  #
            output_hidden_states=None,  #
            return_dict=None,  #
            tokens=None,  # Note 用于character embedding
            word_ids=None,  # Note 用于
            words=None,
            general_labels=None,
    ):

        emissions, emissionsSimple, outputs = self.embedding2emissions(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,  #
            head_mask,  #
            inputs_embeds,  #
            labels,
            output_attentions,  #
            output_hidden_states,  #
            return_dict,  #
            tokens,  # Note 用于character embedding
            word_ids,  # Note 用于
            words)
        # Todo 多任务
        loss = None  # 主要任务 NER任务
        loss_tokenCLS = None  # 辅助任务1 token分类任务
        loss_simpleNER = None  # 辅助任务2 简单的NER任务
        loss_sentenceCLS = None  # 辅助任务3 句子分类任务
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte())

            # token 分类任务的的损失函数以及损失
            loss_func_tokenCLS = CrossEntropyLoss()
            loss_tokenCLS = loss_func_tokenCLS(emissions.view(-1, self.num_labels), labels.view(-1))

            # simple-NER任务的loss
            loss_simpleNER = -1 * self.simpleCrf(emissionsSimple, general_labels, mask=attention_mask.byte())

            # sentence-CLS, 判别句子是否出现了实体
            loss_sentenceCLS = 0

        logits = emissions
        # Todo 更合理设置多个损失函数的权重
        return TokenClassifierOutput(
            loss=loss + 100 * loss_tokenCLS + 100 * loss_simpleNER + loss_sentenceCLS,
            logits=self.crf.decode(emissions, attention_mask.byte()),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def embedding2emissions(self,
                            input_ids=None,
                            attention_mask=None,
                            token_type_ids=None,
                            position_ids=None,  #
                            head_mask=None,  #
                            inputs_embeds=None,  #
                            labels=None,
                            output_attentions=None,  #
                            output_hidden_states=None,  #
                            return_dict=None,  #
                            tokens=None,  # Note 用于character embedding
                            word_ids=None,  # Note 用于
                            words=None):
        outputs = self.bert(
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
        word_embedding = outputs[0]
        embedding = word_embedding

        birnn_output, _ = self.birnn(embedding)
        output = birnn_output
        if Hyper.using_attention:
            extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask,
                                                                                     attention_mask.shape,
                                                                                     device='cuda')
            output = self.attentionB(birnn_output, attention_mask=extended_attention_mask)[0]

        emissions = self.hidden2tag(output)
        emissionsSimple = self.simpleCrfLinear(output)

        return emissions, emissionsSimple, outputs

    def predict(self, input_ids, token_type_ids=None, attention_mask=None, **keywords):
        emissions, _, _ = self.embedding2emissions(input_ids, token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)
        X = self.crf.decode(emissions, attention_mask.byte())
        return X
