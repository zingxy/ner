import copy
import math

import torch
import torch.utils.checkpoint
from datasets import Dataset
from torch import nn
from torchcrf import CRF
from transformers import AutoTokenizer
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput

from character_model import CharacterModel


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
        ## Note char emb config
        self.using_char_embedding = False
        config.char_embedding_size = 16
        config.chars_max_length = 15
        ## Note Word emb
        self.bert = BertModel(config, add_pooling_layer=False)
        # 是否使用char embedding
        if self.using_char_embedding:
            self.char_model = CharacterModel()
            config.hidden_size += config.char_embedding_size * config.chars_max_length * 2
        # TODO 测试一下dropout
        ## TODO GRU vs LSTM
        ## Note Param
        self.birnn_hidden_size = 256  # 64 128 256 512
        self.birnn = nn.LSTM(config.hidden_size,  # Note emb = word + char
                             self.birnn_hidden_size,
                             num_layers=2,  # Note
                             bidirectional=True,
                             batch_first=True,
                             dropout=0.1  # 0.1 0.2 0.3
                             )

        self.birnn = nn.GRU(config.hidden_size,  # Note emb = word + char
                            self.birnn_hidden_size,
                            num_layers=2,  # Note
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.1  # 0.1 0.2 0.3
                            )

        # * 2的原因， 使用了双向的信息， 把forward 和 bacKward 做一个拼接，
        out_dim = self.birnn_hidden_size * 2
        ## Todo Q K V 维度问题
        ## Note Attention before LSTM
        # attention_config = copy.deepcopy(config)
        # attention_config.num_attention_heads = 1
        # self.attention = CustomAttention(attention_config)

        ## Note Attention after LSTM
        attentionB_config = copy.deepcopy(config)
        ## Note 多头注意力机制
        attentionB_config.num_attention_heads = 4
        attentionB_config.hidden_size = self.birnn_hidden_size * 2
        self.attentionB = CustomAttention(attentionB_config)
        ## Note attention线性映射维度
        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        # self.hidden2tag = nn.Linear(300, config.num_labels)

        ## inference
        self.crf = CRF(config.num_labels, batch_first=True)


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
            tokens=None  # Note 用于character embedding
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # instance of BertModel
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
        ## Note character embedding
        ## char_emb time consuming
        char_embedding = None
        if self.using_char_embedding:
            tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1",
                                                      local_files_only=True,
                                                      do_lower_case=False  ## Note 大小写敏感
                                                      )
            tokens = [tokenizer.convert_ids_to_tokens(input_id) for input_id in input_ids]

            character_batchs: Dataset = self.char_model.seq2char(tokens)
            # character_batchs.set_format(type='torch', columns=['ids', 'mask'])
            char_embedding = self.char_model(character_batchs)

        ## body


        emissions = self.tag_outputs(input_ids,
                                     token_type_ids,
                                     attention_mask,
                                     char_embedding)
        ## Note crf(*)-> log likehood, 极大似然. 因此取负号, 最大化极大似然等价于最小化交叉熵
        ## Note 梯度上升梯度下降
        loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte())
        logits = emissions
        ## Note crf loss
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
        )


    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None, char_embedding=None):
        ## Word embedding
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)
        sequence_output = outputs[0]
        ## Note protrained
        word_embedding = sequence_output
        ## Note embedding = word_emb + char_emb   or   word_emb
        embedding = None
        if self.using_char_embedding:
            embedding = torch.cat((word_embedding, char_embedding), dim=-1)
        else:
            embedding = word_embedding
        ## Note Attention
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(input_mask, input_mask.shape,
                                                                                 device='cuda')
        ## Note with / without attention mask.
        # with
        # attention_output = self.attention(sequence_output, attention_mask=extended_attention_mask)[0]
        # without
        # attention_output = self.attention(sequence_output)[0]
        ## BUG Attention_output
        ## Note Bilstm embedding = word + char
        birnn_output, _ = self.birnn(embedding)

        ### Note AttentionB
        #
        attentionb_output = self.attentionB(birnn_output, attention_mask=extended_attention_mask)[0]

        # emissions = self.hidden2tag(sequence_output)
        # emissions = self.hidden2tag(birnn_output)
        emissions = self.hidden2tag(attentionb_output)
        # crf前的output, 相当于last_hidden_layer
        return emissions


    def predict(self, input_ids, token_type_ids=None, attention_mask=None, **keywords):
        emissions = self.tag_outputs(input_ids, token_type_ids, attention_mask)
        X = self.crf.decode(emissions, attention_mask.byte())
        # todo X.shape [batch , seq_length], 预测的标签
        return X


# if __name__ == '__main__':
#     model_name = "bert-base-cased"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = BertPlusBilstmPlusCrf.from_pretrained(model_name)
