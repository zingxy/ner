from collections import OrderedDict

import torch
import torch.nn as nn
from datasets import Dataset
from torchtext.vocab import vocab


# max_char_length = 15

padding = '<pad>'


class CharacterModel(nn.Module):
    def __init__(self, vocab_size=5, embedding_size=32):
        super(CharacterModel, self).__init__()
        # vocab
        self.vocab = CharacterModel.build_vocab()
        vocab_size = len(self.vocab.get_itos())
        # model component
        self.encoder = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.dropout = nn.Dropout(0.1)
        birnn_hidden_size = 16
        self.birnn = nn.LSTM(embedding_size,
                             birnn_hidden_size,
                             num_layers=2,
                             bidirectional=True,
                             batch_first=True,
                             dropout=0.15)
        birnn_out_size = birnn_hidden_size * 2


    def forward(self, batchs):
        # [batch, seq_length, embedding_dim]
        batch_embedding = []
        for batch in batchs:
            ids = batch['ids']
            ids = torch.tensor(ids)
            ids = ids.to('cuda')
            # batch-of-char,  char_length, embedding
            embeding = self.dropout(self.encoder(ids))
            birnn_output, _ = self.birnn(embeding)
            batch_embedding.append(birnn_output.contiguous().view(birnn_output.shape[0], -1))

            # 所有char embedding 拼接成一个embedding， 后面与word_embedding再拼接
        return torch.cat(batch_embedding).view(
                len(batch_embedding),
                batch_embedding[0].shape[0],
                batch_embedding[0].shape[1])


    @classmethod
    def build_vocab(cls):
        lookup = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        special_map = {
        }
        d = {}
        for k, v in enumerate(lookup):
            d[v] = 1
        d.update(special_map)
        lookup_table = OrderedDict(d)
        v = vocab(lookup_table)
        # <pad>
        v.insert_token(padding, 0)
        # ukn
        v.set_default_index(0)

        return v


    def chars2ids(self, chars):
        return self.vocab.lookup_indices(chars)


    def padding_to_max(self, chars: [], token_idx: int, max_length: int):
        # 填充到最大长度， 当超过是自动截断
        # mask = [1] * len(chars) + [0] * (max_length - len(chars))
        # Note 待优化部分
        # origin = [token_idx] * len(chars) + [None] * (max_length - len(chars))
        chars += [padding] * (max_length -len(chars))
        # for i in range(max_length - len(chars)):
        #     chars.append(padding)

        return {
            # 'chars' : chars,
            'ids'   : self.chars2ids(chars),
            # 'mask'  : mask,
            # 'origin': origin
        }


    def seq2char(self, seqs):
        # char_batch[batch , seq_length, token_length]
        batch = {
            # 'chars' : list(),
            'ids'   : list(),
            # 'mask'  : list(),
            # 'origin': list()
        }
        for tokens in seqs:
            chars = {
                # 'chars' : list(),
                'ids'   : list(),
                # 'mask'  : list(),
                # 'origin': list()
            }
            char_ids = []
            ## Bug
            max_token_length = len(max(tokens, key=lambda item: len(item)))
            ## Param max_token_length
            max_token_length = 15
            # batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
            #                    labels]
            for idx, token in enumerate(tokens):

                char_of_the_token = []

                # # Note 特殊的token
                if token in ['[CLS]', '[PAD]', '[SEP]', '[UNK]']:
                    char_of_the_token.append(token)
                else:
                    if len(token) > max_token_length:
                        char_of_the_token = list(token[:max_token_length])
                    else:
                        char_of_the_token = list(token)
                sample = self.padding_to_max(char_of_the_token,
                                             idx,
                                             max_token_length,
                                             )
                for k in chars:
                    chars[k].append(sample[k])
            for k in batch:
                batch[k].append(chars[k])
        return Dataset.from_dict(batch)


if __name__ == '__main__':
    model = CharacterModel()
    case = torch.tensor([[0, 1, 0, 0, 0]])
    model(case)
