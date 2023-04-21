# @Time 8/20/2021 9:33 PM
# @Author 1067577595@qq.com
import os
import string
import random
from collections import OrderedDict
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchtext.vocab import vocab
from tqdm import tqdm

# 10 26 32
lookup = '0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
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


class CharacterFeature(nn.Module):
    def __init__(self, embedding_size=32, max_token_length=8):
        super(CharacterFeature, self).__init__()
        # vocab
        self.vocab = CharacterFeature.build_vocab()
        vocab_size = len(self.vocab.get_itos())
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size, self.embedding_size, padding_idx=0)
        self.activation1 = nn.ReLU()
        self.dense = nn.Linear(self.embedding_size, self.embedding_size)
        self.activation2 = nn.ReLU()



    def forward(self, chars, labels=None, mode=True):
        if mode:
            chars = lookup
        word_onehot = self.chars2ids(list(chars))
        emb = self.embedding(torch.tensor(word_onehot))
        # Note
        emb = self.dense(self.activation1(emb))
        number_root = torch.mean(emb[:10], dim=0)
        alpha_root = torch.mean(emb[10:36], dim=0)
        symbol_root = torch.mean(emb[36:], dim=0)
        loss = torch.std(emb[:10], dim=0).sum() \
               + torch.std(emb[10:26], dim=0).sum()\
               + torch.std(emb[36:], dim=0).sum()

        root_distance = (number_root - alpha_root) **2\
                        + (symbol_root - alpha_root) **2 \
                        + (number_root - symbol_root) **2

        loss += 1 / root_distance.sum()
        preds = None
        # if labels is not None:
        #     loss = nn.CrossEntropyLoss()(preds, labels)
        return loss, preds, emb


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
                                                batch_size=1500,
                                                shuffle=True)

    model.eval()
    all_emb:torch.Tensor = None
    with torch.no_grad():
        for batch in visual_loader:
            words, labels = batch
            loss, preds, emb = model(words, labels, False)
            break
            # pred_ids: torch.tensor = preds.argmax(dim=-1)
            # if all_emb is not None:
            #     torch.cat((all_emb, emb))
            # else:
            #     all_emb = emb
    # emb = all_emb
    # labels = [0] * 10 + [1] * 26 + [2] * 32
    X_embedded = TSNE(n_components=2, n_jobs=8, learning_rate=100).fit_transform(emb.numpy())
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


def visual3d():
    test_dataset = CustomDataset('../datasets_/data/TMVAR-CLASS/lambda.txt')
    visual_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=2000,
                                                shuffle=True)

    model.eval()
    with torch.no_grad():
        for batch in visual_loader:
            words, labels = batch
            loss, preds, emb = model(words, labels)
            # pred_ids: torch.tensor = preds.argmax(dim=-1)
            break

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
    model = CharacterFeature()
    # model.to('cuda')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.1)
    #
    # nn.init.normal_(model.hidden2tag.weight, 0, 1)
    # nn.init.normal_(model.hidden2tag.bias, 0, 1)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    epochs = 100
    for epochs in tqdm(range(epochs)):
        for batch in train_loader:
            tokens, labels = batch
            optimizer.zero_grad()
            loss, preds, emb = model(tokens, labels)
            loss.backward()
            optimizer.step()
        print('loss:\t',loss.item())
        # doPredict(model, test_dataset)
    # after = list(model.hidden2tag.parameters())
    visual()
    # visual3d()

