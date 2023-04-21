import os
import string
import random
from collections import OrderedDict
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from tqdm import tqdm
from typing import List


lookup = '0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
lookup = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

# from BertBilstmCRF import CustomAttention
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
padding = '<pad>'

import csv


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # T
        # 1. Initialize file paths or a list of file names.

        self.path = path
        self.origin = pd.read_csv(self.path, quoting=csv.QUOTE_NONE, sep='\t', names=['tokens', 'labels'])


    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        k = self.origin['tokens'][index]
        v = self.origin['labels'][index]
        return str(k), int(v)


    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.origin)


def dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1 - torch.dot(a, b) / (a.norm() * b.norm())


class TypingFeature(nn.Module):
    def __init__(self, embedding_size=16, max_token_length=8):
        super(TypingFeature, self).__init__()
        # vocab
        self.vocab = TypingFeature.build_vocab()
        vocab_size = len(self.vocab.get_itos())
        self.embedding_size = embedding_size
        self.max_length = max_token_length
        self.embedding = nn.Embedding(5, self.embedding_size, padding_idx=0)

        self.distance = dist
        # self.bn = nn.LayerNorm(self.embedding_size, eps=1e-12, elementwise_affine=True)
        # self.hidden2tag = nn.Linear(self.embedding_size, 4)


    def forward(self, chars: List[List[str]], mode=None):
        ## mode 表示训练还是 预测
        char_onehot = self.chars2ids(list(chars))
        # char_onehot = elf.chars2ids(list(chars))
        #
        emb = None
        loss = None
        if mode is None:  ## predict
            word_onehot = torch.tensor(char_onehot)
            word_onehot[(word_onehot >= 1) & (word_onehot < 11)] = 1
            word_onehot[(word_onehot >= 11) & (word_onehot < 37)] = 2
            word_onehot[(word_onehot >= 37) & (word_onehot < 63)] = 3
            word_onehot[word_onehot >= 63] = 4
            emb = self.embedding(word_onehot)

        if mode is not None:  ## train
            word_onehot = torch.tensor([1, 2, 3, 4])
            emb = self.embedding(word_onehot)
            number_root, small_root, big_root, symbol_root = emb
            distance = self.distance(number_root, small_root) + \
                       self.distance(number_root, small_root) + \
                       self.distance(number_root, big_root) + \
                       self.distance(number_root, symbol_root) + \
                       self.distance(symbol_root, small_root) + \
                       self.distance(symbol_root, big_root) + \
                       self.distance(small_root, big_root)

            loss = 1 / distance.sum()

        return loss, emb


    @classmethod
    def build_vocab(cls):
        special_map = {
        }
        d = {}
        for k, v in enumerate(string.printable):
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


    def padding_to_max(self, chars: [], max_length: int):
        # 填充到最大长度， 当超过是自动截断
        # mask = [1] * len(chars) + [0] * (max_length - len(chars))
        # Note 待优化部分
        chars += [padding] * (max_length - len(chars))

        return self.chars2ids(chars)


    def seq2char(self, words):
        batch = []
        for word in words:
            if len(word) > self.max_token_length:
                word = word[:self.max_token_length]
            word_char_ids = self.padding_to_max(list(word), None, self.max_token_length)
            batch.append(word_char_ids)
        return batch


def performance(preds, refs):
    print('\nAcc', torch.count_nonzero(preds == refs).item() / len(preds) * 100, '\n')


def doPredict(model, test_dataset):
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=256,
                             shuffle=False)
    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            words, labels = batch
            loss, preds, emb = model(words, labels)
            pred_ids: torch.tensor = preds.argmax(dim=-1)
            results.append((words, labels, pred_ids))
    with open('output/char2char.tsv', 'w') as f:
        for (words, labels, preds) in results:
            for word, label, pred in zip(words, labels, preds):
                f.write(word + '\t' + str(label.item()) + '\t' + str(pred.item()) + '\n')
    performance(preds, labels)


def visual():
    test_dataset = CustomDataset('../datasets_/data/TMVAR-CLASS/lambda.txt')
    visual_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=5000,
                                                shuffle=True)

    model.eval()
    all_emb: torch.Tensor = None
    with torch.no_grad():
        for batch in visual_loader:
            words, labels = batch
            loss, emb = model(words)
            break
            # pred_ids: torch.tensor = preds.argmax(dim=-1)
            # if all_emb is not None:
            #     torch.cat((all_emb, emb))
            # else:
            #     all_emb = emb
    # emb = all_emb
    X_embedded = TSNE(n_components=2, n_jobs=40, learning_rate=100).fit_transform(emb.numpy())
    data = X_embedded
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    for i in range(len(labels)):
        plt.annotate(labels[i].item(), xy=(data[i][0], data[i][1]))
    plt.show()
    ## 3d
    X_embedded = TSNE(n_components=3, n_jobs=8, learning_rate=100).fit_transform(emb.numpy())
    data = X_embedded
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels)
    plt.show()


if __name__ == '__main__':
    train_path = '../datasets_/data/TMVAR-CLASS/alpha.txt'
    test_path = '../datasets_/data/TMVAR-CLASS/beta.txt'

    train_dataset = CustomDataset(train_path)
    test_dataset = CustomDataset(test_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=256,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=256,
                                              shuffle=False)
    model = TypingFeature()
    torch.nn.init.orthogonal_(model.embedding.weight)
    # model.load_state_dict(torch.load('checkpoint/type.ckpt'))
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.2)
    visual()
    _, (num, alpha, symbol) = model('1D*')
    print([dist(a, b) for (a, b) in [(num, alpha), (num, symbol), (alpha, symbol)]])

    epochs = 15
    for epochs in tqdm(range(epochs)):
        for batch in train_loader:
            tokens, labels = batch
            optimizer.zero_grad()
            loss, emb = model(tokens, labels)
            loss.backward()
            optimizer.step()
        print('loss:\t', loss.item())

    # model([['a', 'b'], ['c', 'd']])
    visual()
    print([dist(a, b) for (a, b) in [(num, alpha), (num, symbol), (alpha, symbol)]])

