from collections import OrderedDict

import torch
import torch.nn as nn
from datasets import Dataset
from torchtext.vocab import vocab



# max_char_length = 15

padding = '<pad>'


class CharacterModel(nn.Module):
    def __init__(self, embedding_size=16, max_token_length=15):

        super(CharacterModel, self).__init__()
        # vocab
        self.vocab = CharacterModel.build_vocab()
        self.vocab_size = len(self.vocab.get_itos())
        self.max_token_length = max_token_length
        self.embedding_size = embedding_size
        ## Note 固定char embedding
        self.embedding = nn.Embedding(5, embedding_size)
        torch.nn.init.orthogonal_(self.embedding.weight)
        for param in self.embedding.parameters():
            param.requires_grad = False

        birnn_hidden_size = 16
        self.birnn = nn.GRU(self.embedding_size,
                            birnn_hidden_size,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.2)

        birnn_out_size = birnn_hidden_size * 2
        self.norm = nn.LayerNorm([birnn_out_size], eps=1e-12, elementwise_affine=True)


    def forward(self, batchs):
        # [batch, seq_length, embedding_dim]
        batch_embedding = []
        for batch in batchs:
            ids = batch['ids']
            ids = torch.tensor(ids)
            word_onehot = ids
            word_onehot[(word_onehot >= 1) & (word_onehot < 11)] = 1
            word_onehot[(word_onehot >= 11) & (word_onehot < 37)] = 2
            word_onehot[(word_onehot >= 37) & (word_onehot < 63)] = 3
            word_onehot[word_onehot >= 63] = 4
            ids = word_onehot
            ids = ids.to('cuda')    ## Bug 推理的时候不一定再用gpu
            # batch-of-char,  char_length, embedding
            embeding = self.embedding(ids)
            birnn_output, _ = self.birnn(embeding)
            birnn_output = self.norm(birnn_output)
            batch_embedding.append(birnn_output.contiguous().view(birnn_output.shape[0], -1))

            # 所有char embedding 拼接成一个embedding， 后面与word_embedding再拼接
        return torch.cat(batch_embedding).view(
                len(batch_embedding),
                batch_embedding[0].shape[0],
                batch_embedding[0].shape[1])


    @classmethod
    def build_vocab(cls):
        ## cased
        lookup = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        ## uncased
        # lookup = '0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

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


    def chars2ids(self, chars, length=None):
        return self.vocab.lookup_indices(chars)


    def padding_to_max(self, chars: [], token_idx: int, max_length: int):
        # 填充到最大长度， 当超过是自动截断
        # mask = [1] * len(chars) + [0] * (max_length - len(chars))
        # Note 待优化部分
        # origin = [token_idx] * len(chars) + [None] * (max_length - len(chars))
        chars += [padding] * (max_length - len(chars))
        # for i in range(max_length - len(chars)):
        #     chars.append(padding)

        return {
            # 'chars' : chars,
            'ids': self.chars2ids(chars),
            # 'mask'  : mask,
            # 'origin': origin
        }


    def seq2char(self, seqs):
        # char_batch[batch , seq_length, token_length]
        batch = {
            # 'chars' : list(),
            'ids': list(),
            # 'mask'  : list(),
            # 'origin': list()
        }
        for tokens in seqs:
            chars = {
                # 'chars' : list(),
                'ids': list(),
                # 'mask'  : list(),
                # 'origin': list()
            }
            char_ids = []
            ## Bug
            max_token_length = len(max(tokens, key=lambda item: len(item)))
            ## Param max_token_length
            max_token_length = self.max_token_length
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
    pass
