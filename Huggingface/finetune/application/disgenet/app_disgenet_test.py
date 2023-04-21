import re
import string
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List

import pandas as pd
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

from BertBilstmCRF import BertBilstmCrf


@dataclass
class Mutation:
    name: str
    type: str
    start: int
    end: int

    def __repr__(self):
        return self.name


@dataclass
class Token:
    """
    用于模型原始输出
    """
    idx: int
    text: str
    type: str

    def __repr__(self):
        return f"[{self.text}:{self.type})"


# pred2name
idx2name = [
    "O",
    "B-DNAMutation",
    "I-DNAMutation",
    "B-ProteinMutation",
    "I-ProteinMutation",
    "B-SNP",
    "I-SNP"
]
# 训练时使用的tokenizer和模型
# Todo 选择参数
model_path = '../../checkpoint/'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertBilstmCrf.from_pretrained(model_path)
# 使用spacy对文本预处理
preprocessos = spacy.load('en_core_web_sm')


def read_test_data(path: str = ''):
    # 读取文本
    path = './app.test.input.txt'
    test_set: List[str] = []
    # 文章对应的pmid
    pmids: List[str] = []
    with open(path, 'r+', encoding='utf-8') as f:
        lines = f.read().strip().split('\n\n')

        for idx, line in enumerate(lines):
            try:
                pmid, text = line.split('\t', 1)
            except Exception as e:
                print(line)
            test_set.append(text.encode('ascii', 'replace').decode('ascii'))
            pmids.append(pmid)
    return pmids, test_set


def removeTokenizerPrefix(token):
    # ## 开头，说明这个词已经被tokenized
    if token[:2] == '##':
        return token[2:]
    return token


def token2word(token1, token2):
    # 用作reducer
    return token1 + token2


def entity_gen(tokens, word_ids):
    # 将连续tokens 恢复成原来的word(s)
    previous_idx = None
    entity_name = ''
    for token, idx in zip(tokens, word_ids):
        if idx != previous_idx and entity_name:
            entity_name += ' ' + token
        else:
            entity_name += token
        previous_idx = idx
    return entity_name, 0


def group2word(group):
    """
    convert group(tokens) to origin word.
    """

    def label():
        pass

    pass


def tagger(test_set: List[str]):
    # 这里加载需要标注的文本
    # Todo 目前是一篇文章调用一次model, 需要进行处理，使得一次能够处理更多的的文章
    # 预处理可以先进行
    Result: List[List[str]] = []
    with open('./word.txt', mode='wt', encoding='utf8') as f:
        for document in tqdm(test_set):
            doc = preprocessos(document)
            # Using spaCy for segmentation
            sentences: List[str] = [sent.text for sent in doc.sents]
            # Using spaCy for pre tokenize. Alternative
            # wordsList = [[token.text for token in sent if not token.is_space] for sent in doc.sents]
            # Just using blank for tokenize and remove all whitespace tokens.
            wordsList = [[token.strip() for token in sent.split(' ') if token not in string.whitespace] for sent in
                         sentences]

            inputs = tokenizer(wordsList, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)

            predictions = model.predict(**inputs)
            assert len(predictions) == len(wordsList)
            for idx, prediction in enumerate(predictions):
                assert len(inputs.word_ids(idx)) == len(inputs.tokens(idx)) >= len(prediction)
                tokenWithPred: List[Token] = []

                # remove [CLS] and [SEP]
                tempTokens = inputs.tokens(idx)[:len(prediction)][1:-1]
                tempPreds = prediction[1:-1]
                assert len(tempTokens) == len(tempPreds)
                for (token, pred) in zip(tempTokens, tempPreds):
                    tokenWithPred.append(Token(idx=-1, text=removeTokenizerPrefix(token), type=idx2name[pred]))
                words = wordsList[idx]
                assert len(words) <= len(tokenWithPred)
                processedTokenWithPred: List[Token] = []
                for (pos, word) in enumerate(words):
                    span = ''
                    while span != word and tokenWithPred:
                        token: Token = tokenWithPred.pop(0)
                        span = span + token.text
                        token.idx = pos
                        processedTokenWithPred.append(token)
                    # assert word == span

                # assert pos == len(words)
                assert len(tokenWithPred) == 0
                dataframe = pd.DataFrame({
                    "wordIdx": map(lambda token: token.idx, processedTokenWithPred),
                    "token": map(lambda token: token.text, processedTokenWithPred),
                    "pred": map(lambda token: token.type, processedTokenWithPred)
                })

                wordIdxGroup = dataframe.groupby('wordIdx')
                wordIdxGroup.apply(lambda group: f.write(f''))

            # Note 根据label聚集 只筛选实体所在的token
            spans: List[Dict[str, List[str]]] = []
            group = {}
            for idx, prediction in enumerate(predictions):
                for word_idx, token, label_id in zip(inputs.word_ids(idx), inputs.tokens(idx), prediction):

                    # print(f'{token}\t{names[label_id]}')
                    ## Bug 连续实体问题
                    if idx2name[label_id] == 'O':
                        if group:
                            spans.append(group)
                            group = {}  ###
                    else:
                        if group:
                            # 处理连续实体问题.
                            if label_id in [1, 3, 5]:
                                spans.append(group)
                                group = {}
                            elif idx2name[label_id][2:] != group['labels'][-1][2:]:
                                spans.append(group)
                                group = {}

                        group.setdefault('tokens', [])
                        group.setdefault('word_ids', [])
                        group.setdefault('labels', [])
                        group['tokens'].append(removeTokenizerPrefix(token))
                        group['word_ids'].append(word_idx)
                        group['labels'].append(idx2name[label_id])

            entitys: List[Dict[str, str]] = []
            for span in spans:
                entity_name, idx = entity_gen(span['tokens'], span['word_ids'])
                entity_label = span['labels'][-1][2:]
                entitys.append({
                    'name': entity_name,
                    'label': entity_label,
                    'idx': idx,
                })
            # end 根据label聚集

            res = []
            for ent in entitys:
                res.append(ent['name'])
                # print(idx, ' ', ent['name'], '\n')

            Result.append(res)

    return Result


def findAllIdenticalMutation(text: str):
    # Store all mutations
    bucket: List[re.Match] = []

    def mapper(pattern):
        nonlocal text
        nonlocal bucket
        try:
            matches = re.finditer(re.escape(pattern), text)
        except Exception as e:
            print('pattern error')

        else:  # when no exception occur, this block will execute
            # if matches:
            #     # Todo we need save the entity offset information
            #
            #     return [pattern] * len(list(matches))
            # else:
            #     return [pattern]
            # 将匹配到的每一项目都加入到其中
            bucket.extend(matches)

    return mapper, bucket


def removeAlpha(s: str):
    # 可用可不用
    s = s.strip()
    if s.startswith('alpha'):
        s = s[5:]
    if s.endswith("alpha"):
        s = s[:-5]
    return s


def removeOverlap(bucket: List[re.Match]) -> List[Mutation]:
    # 去除位置重叠的实体, 区间合并
    ordered = sorted(bucket, key=lambda match: match.start())
    #
    merged: List[Mutation] = []
    for match in ordered:
        if merged and merged[-1].end >= match.start():
            merged[-1].end = max(merged[-1].end, match.end())
        else:
            merged.append(Mutation(match.group(), '', match.start(), match.end()))

    # print(merged)
    return merged


def appFilters(mutations: List[Mutation]):
    # apply filters to filter nonsense output
    # candidates3 = reduce(lambda arg, fn: filter(fn, arg), fns, candidates2)
    from filters import lengthFilter

    return reduce(lambda muts, f: filter(f, muts), [lengthFilter], mutations)


def save(result: List[List[str]], path: str, pmids, test_set):
    """
    save output
    """
    with open(path, 'w+', encoding='utf-8') as f:
        for pmid, text, candidates in tqdm(zip(pmids, test_set, result)):
            # 剥掉一些字符
            # candidates = map(lambda x: x.strip().strip('[](){}'), candidates)
            candidates = map(lambda x: x.strip(), candidates)
            unique = list(set(candidates))

            mapper, bucket = findAllIdenticalMutation(text)
            for mut in unique:
                mapper(mut)

            mutations = removeOverlap(bucket)
            mutations = appFilters(mutations)

            sep = '\t'
            f.write(f"{pmid}{sep}{sep.join(map(str, mutations))}\n\n")


if __name__ == '__main__':
    src = './app.test.input.txt'
    dest = './app.test.output.txt'

    # 需要预测的文本
    test_set: List[str] = []
    # 文章对应的pmid
    pmids: List[str] = []
    pmids, test_set = read_test_data(src)

    output = tagger(test_set)
    save(output, dest, pmids, test_set)
