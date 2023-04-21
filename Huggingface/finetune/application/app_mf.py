# 批量测试文章，一次n篇
import re
import time
from functools import reduce
from typing import Dict, List

import spacy
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
# 训练时使用的tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained('../checkpoint', )
model = BertBilstmCrf.from_pretrained('../checkpoint')
# 使用spacy对文本预处理
preprocessos = spacy.load('en_core_web_sm')
# 需要预测的文本
Test_Text: List[str] = []
# 文章对应的pmid
Pmids: List[str] = []


# ## 开头，说明这个词已经被tokenized
def process_token(token):
    if token[:2] == '##':
        return token[2:]
    return token


# 用作reducer
def token2word(token1, token2):
    return token1 + token2


# 读取文本
def read_test_data():
    path = './input.txt'
    path = 'disgenet/test.txt'
    with open(path, 'r+', encoding='utf-8') as f:
        lines = f.read().split('\n\n')

        for idx, line in enumerate(lines):
            pmid, text = line.split('\t')
            Test_Text.append(text)
            Pmids.append(pmid)


Result: List[List[str]] = []


def main():
    # 这里加载需要查找的文本
    read_test_data()
    for document in Test_Text:
        doc = preprocessos(document)
        sentences = [sent.text.split(' ') for sent in doc.sents]

        pre_seq = None

        # tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))

        t0 = time.time()
        inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
        word_ids = inputs.word_ids()
        tokens = inputs.tokens()

        predictions = model.predict(**inputs)

        t1 = time.time()

        ## Note 根据word聚集
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

        def entity_gen(tokens, word_ids):
            previous_idx = None
            entity_name = ''
            for token, idx in zip(tokens, word_ids):
                if idx != previous_idx and entity_name:
                    entity_name += ' ' + token
                else:
                    entity_name += token
                previous_idx = idx
            return entity_name, 0

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
            print(idx, ' ', ent['name'], '\n')

        Result.append(res)


from application.disgenet.filter_fns import produceFilters

if __name__ == '__main__':
    main()
    path = './output.txt'

    with open(path, 'w+', encoding='utf-8') as f:

        for pmid, text, candidates in zip(Pmids, Test_Text, Result):

            def removeAlpha(s: str):
                s = s.strip()
                if (s.startswith('alpha')):
                    s = s[5:]
                if (s.endswith("alpha")):
                    s = s[:-5]
                return s


            # 应用mapper
            candidates1 = map(lambda x: x.strip('[](){}'), candidates)
            # candidates2 = map(removeAlpha, candidates1)
            candidates3 = candidates2 = candidates1
            # 应用过滤器
            fns: List = produceFilters()
            # candidates3 = reduce(lambda arg, fn: filter(fn, arg), fns, candidates2)

            candidates3 = list(candidates3)


            # candidates中，每个mutation进行re, 防止漏掉数据
            def findAllIdenticalMutation(text):
                def mapper(pattern):
                    nonlocal text
                    try:
                        matches = re.finditer(pattern, text)
                    except Exception as e:
                        print('正则表达式错误', pattern)
                        matches = ['']

                    return [pattern] * len(list(matches))

                return mapper


            mutations: List[str] = []
            mapper = findAllIdenticalMutation(text)
            mutationGroup = map(mapper, set(candidates3))

            # 展开
            mutations: List[str]
            mutations = reduce(lambda x, y: x + y, mutationGroup, [])

            # 写入数据
            mutations.insert(0, pmid)

            f.write('\t'.join(mutations) + '\n\n')
