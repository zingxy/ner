# @Time 8/23/2021 9:10 PM
# @Author 1067577595@qq.com

from functools import reduce
from typing import Dict, List, Optional

from transformers import AutoTokenizer

from BertBilstmCRF import BertBilstmCrf


# 用于测试
names = [
    "O",
    "B-DNAMutation",
    "I-DNAMutation",
    "B-ProteinMutation",
    "I-ProteinMutation",
    "B-SNP",
    "I-SNP"
]


def calculate_word_label(labels: List[str]):
    ## 决定单词的标签, 第一个非实体标记确定为word标签
    ## Bug 一个单词可能出现两个实体， 连字符， 斜杠
    for i in labels:
        if len(i) > 1:
            return i[2:]
    return 'O'


def process_token(token):
    if token[:2] == '##':
        return token[2:]
    return token


def token2word(token1, token2):
    return token1 + token2

def entity_gen(tokens, word_ids):
    previous_idx = None
    entity_name = ''
    for token, idx in zip(tokens, word_ids):
        if idx != previous_idx and entity_name:
            entity_name += ' ' + token
        else:
            entity_name += token
        previous_idx = idx
    return entity_name

def pipeline(document: str, model=None, tokenizer=None):
    sequence = document
    pre_seq = sequence.split(' ')
    inputs = tokenizer(pre_seq, return_tensors="pt", is_split_into_words=True)
    word_ids = inputs.word_ids()
    tokens = inputs.tokens()
    predictions = model.predict(**inputs)
    spans: List[Dict[str, List[str]]] = []
    group = {}
    for word_idx, token, prediction in zip(word_ids, tokens, predictions[0]):
        ## Bug 连续实体问题
        if names[prediction] == 'O':
            if group:
                spans.append(group)
                group = {}  ###
        else:
            if group:
                # 处理连续实体问题.
                if prediction in [1, 3, 5]:
                    spans.append(group)
                    group = {}
                elif names[prediction][2:] != group['labels'][-1][2:]:
                    spans.append(group)
                    group = {}

            group.setdefault('tokens', [])
            group.setdefault('word_ids', [])
            group.setdefault('labels', [])
            group['tokens'].append(process_token(token))
            group['word_ids'].append(word_idx)
            group['labels'].append(names[prediction])





    entitys: List[Dict[str, str]] = []
    for span in spans:
        entity_name = entity_gen(span['tokens'], span['word_ids'])
        entity_label = span['labels'][-1][2:]
        entitys.append({
            'name' : entity_name,
            'label': entity_label
        })
    return entitys


if __name__ == '__main__':
    pass
