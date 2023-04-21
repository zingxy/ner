import re
from functools import reduce
from typing import Dict, List

import spacy
from tqdm import tqdm
from transformers import AutoTokenizer

from BertBilstmCRF import BertBilstmCrf

# from ..filter_fns import produceFilters

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
# 训练时使用的tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained('../../checkpoint_best', )
model = BertBilstmCrf.from_pretrained('../../checkpoint_best')
# 使用spacy对文本预处理
preprocessos = spacy.load('en_core_web_sm')
# 需要预测的文本
Test_Text: List[str] = []
# 文章对应的pmid
Pmids: List[str] = []


def process_token(token):
    # ## 开头，说明这个词已经被tokenized
    # ## 开头，说明这个词已经被tokenized
    if token[:2] == '##':
        return token[2:]
    return token


def token2word(token1, token2):
    # 用作reducer
    return token1 + token2


def read_test_data():
    # 读取文本
    path = './test.txt'
    with open(path, 'r+', encoding='utf-8') as f:
        lines = f.read().strip().split('\n\n')

        for idx, line in enumerate(lines):
            pmid, text = line.split('\t')
            Test_Text.append(text)
            Pmids.append(pmid)


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


def tagger(Result: List[List[str]]):
    # 这里加载需要标注的文本
    # Todo 目前是一篇文章调用一次model, 需要进行处理，使得一次能够处理更多的的文章
    read_test_data()
    for document in tqdm(Test_Text):
        doc = preprocessos(document)
        sentences = [sent.text.split(' ') for sent in doc.sents]

        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
        word_ids = inputs.word_ids()
        tokens = inputs.tokens()

        predictions = model.predict(**inputs)

        ## Note 根据word聚集
        ## Todo 改成groupBy
        words_group = []
        word_spans: Dict[int, list] = {}
        for idx, prediction in enumerate(predictions):
            for word_idx, token, label_id in zip(inputs.word_ids(idx), inputs.tokens(idx), prediction):

                # for word_idx, token, prediction in zip(word_ids, tokens, predictions[0]):
                #         # print((token, model.config.id2label[prediction]))
                if word_idx is not None:
                    word_spans.setdefault(word_idx, {
                        'tokens': [],
                        'labels': []
                    })
                    word_spans[word_idx]['tokens'].append(process_token(token))
                    word_spans[word_idx]['labels'].append(names[label_id])
            words_group.append(word_spans)
            word_spans = {}

        ## Note 根据label聚集 只筛选实体所在的token
        spans: List[Dict[str, List[str]]] = []
        group = {}
        for idx, prediction in enumerate(predictions):
            for word_idx, token, label_id in zip(inputs.word_ids(idx), inputs.tokens(idx), prediction):

                # print(f'{token}\t{names[label_id]}')
                ## Bug 连续实体问题
                if names[label_id] == 'O':
                    if group:
                        spans.append(group)
                        group = {}  ###
                else:
                    if group:
                        # 处理连续实体问题.
                        if label_id in [1, 3, 5]:
                            spans.append(group)
                            group = {}
                        elif names[label_id][2:] != group['labels'][-1][2:]:
                            spans.append(group)
                            group = {}

                    group.setdefault('tokens', [])
                    group.setdefault('word_ids', [])
                    group.setdefault('labels', [])
                    group['tokens'].append(process_token(token))
                    group['word_ids'].append(word_idx)
                    group['labels'].append(names[label_id])

        entitys: List[Dict[str, str]] = []
        for span in spans:
            entity_name, idx = entity_gen(span['tokens'], span['word_ids'])
            entity_label = span['labels'][-1][2:]
            entitys.append({
                'name': entity_name,
                'label': entity_label,
                'idx': idx,
            })
        # Result.append(entitys)
        res = []
        for ent in entitys:
            res.append(ent['name'])
            # print(idx, ' ', ent['name'], '\n')

        Result.append(res)

    return Result


def findAllIdenticalMutation(text):
    # candidates中，每个mutation进行re, 防止漏掉数据
    def mapper(pattern):
        nonlocal text
        try:
            matches = re.findall(re.escape(pattern), text)
        except Exception as e:
            print('pattern error')

        else:  # when no exception occur, this block will execute
            if matches:
                return [pattern] * len(list(matches))
            else:
                return [pattern]

    return mapper


def removeAlpha(s: str):
    # 可用可不用
    s = s.strip()
    if s.startswith('alpha'):
        s = s[5:]
    if s.endswith("alpha"):
        s = s[:-5]
    return s


def removeOverlap():
    # 去除位置重叠的实体
    pass


def save(result: List[List[str]], path: str):
    """
    将结果保存
    """
    with open(path, 'w+', encoding='utf-8') as f:
        for pmid, text, candidates in tqdm(zip(Pmids, Test_Text, result)):
            # 剥掉一些字符
            candidates = map(lambda x: x.strip().strip('[](){}'), candidates)
            unique = list(set(candidates))
            # 应用过滤器
            # Todo 重写过滤规则
            # fns: List = produceFilters()
            # candidates3 = reduce(lambda arg, fn: filter(fn, arg), fns, candidates2)

            mapper = findAllIdenticalMutation(text)
            mutationGroup = map(mapper, unique)

            # reduce, 将2维数组展开成1维
            mutations: List[str]
            mutations = reduce(lambda x, y: x + y, mutationGroup, [])

            # 写入数据
            mutations.insert(0, pmid)

            f.write('\t'.join(mutations) + '\n\n')


if __name__ == '__main__':
    output = tagger([])
    path = './ours.txt'
    save(output, path)
