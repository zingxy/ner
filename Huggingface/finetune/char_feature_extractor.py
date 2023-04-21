import os
from collections import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from tqdm import tqdm
import torch.nn.functional as F
from char_alpha import TypingFeature


# from BertBilstmCRF import CustomAttention
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
padding = '<pad>'


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # T
        # 1. Initialize file paths or a list of file names.

        self.path = path
        self.origin = pd.read_csv(self.path, sep='\t', names=['tokens', 'labels'])


    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        k = self.origin['tokens'][index]
        v = 0 if self.origin['labels'][index] == 'no' else 1
        return str(k), int(v)


    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.origin)


class CharacterFeature(nn.Module):
    def __init__(self, embedding_size=16, max_token_length=8):
        super(CharacterFeature, self).__init__()
        # vocab
        self.vocab = CharacterFeature.build_vocab()
        vocab_size = len(self.vocab.get_itos())
        self.max_token_length = max_token_length
        self.embedding_size = embedding_size
        # model component
        # self.embedding = nn.Embedding(vocab_size, self.embedding_size, padding_idx=0)
        type_model = TypingFeature()
        type_model.load_state_dict(torch.load('checkpoint/type.ckpt'))
        for param in type_model.parameters():
            param.requires_grad = False
        self.embedding = type_model.embedding
        # birnn_hidden_size = 16
        # self.birnn = nn.GRU(self.embedding_size,
        #                     birnn_hidden_size,
        #                     num_layers=1,
        #                     bidirectional=True,
        #                     batch_first=True,
        #                     # dropout=0.2
        #                     )
        self.dense = nn.Linear(self.embedding_size, self.embedding_size)
        self.activator = nn.ReLU()
        # birnn_out_size = birnn_hidden_size * 2
        # self.norm = nn.LayerNorm([birnn_out_size], eps=1e-12, elementwise_affine=True)
        # self.hidden2tag = nn.Linear(self.embedding_size * self.max_token_length, 2)
        # self.hidden2tag = nn.Linear(birnn_out_size * max_token_length, 2)
        self.hidden2tag = nn.Linear(self.embedding_size * self.max_token_length, 2)
        # self.hidden2tag = nn.Bilinear


    def forward(self, words, labels):
        word_onehot = self.seq2char(words)
        word_onehot = torch.tensor(word_onehot)
        word_onehot[(word_onehot >= 1) & (word_onehot < 11)] = 1
        word_onehot[(word_onehot >= 11) & (word_onehot < 37)] = 2
        word_onehot[word_onehot >= 37] = 3
        emb = self.embedding(word_onehot.to('cuda'))
        emb = F.normalize(emb, dim=-1)
        emb = self.activator(self.dense(emb))
        emb = emb.reshape(len(labels), -1)
        preds = self.hidden2tag(emb)
        loss = nn.CrossEntropyLoss()(preds, labels.to('cuda'))
        return loss, preds


    @classmethod
    def build_vocab(cls):
        lookup = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        # freq_lookup = '1234605897AGCT.->rpselcndRtaiVSo/IHD(yu)MgL+XbEN_FPWYQfh,' \
        #               'K:[]Bv\'?O*xmjkqwzJUZ!"#$%&;<=@\\^`{|}~'
        # lookup = freq_lookup
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
        chars += [padding] * (max_length - len(chars))
        # for i in range(max_length - len(chars)):
        #     chars.append(padding)

        return self.chars2ids(chars)


    def seq2char(self, words):
        batch = []
        for word in words:
            if len(word) > self.max_token_length:
                word = word[:self.max_token_length]
            word_char_ids = self.padding_to_max(list(word), None, self.max_token_length)
            batch.append(word_char_ids)
        return batch


def doPredict(model, test_loader):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            words, labels = batch
            loss, preds = model(words, labels)
            pred_ids: torch.tensor = preds.argmax(dim=-1)
            results.append((words, labels, pred_ids))
    with open('output/char_feature_extractor.tsv', 'w') as f:
        for (words, labels, preds) in results:
            for word, label, pred in zip(words, labels, preds):
                f.write(word + '\t' + str(label.item()) + '\t' + str(pred.item()) + '\n')


if __name__ == '__main__':
    train_path = '../datasets_/data/TMVAR-CLASS/train.tsv'
    test_path = '../datasets_/data/TMVAR-CLASS/test.tsv'
    train_dataset = CustomDataset(train_path)
    test_dataset = CustomDataset(test_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=256,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=256,
                                              shuffle=False)
    model = CharacterFeature()
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 15
    for epochs in tqdm(range(epochs)):
        for batch in train_loader:
            tokens, labels = batch
            optimizer.zero_grad()
            loss, preds = model(tokens, labels)
            loss.backward()
            optimizer.step()
        print(loss.item())
    doPredict(model, test_loader)
