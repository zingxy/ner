from functools import reduce
from typing import Dict, List

from transformers import AutoTokenizer

from BertBilstmCRF import BertBilstmCrf

import spacy


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


if __name__ == '__main__':
    #
    tokenizer = AutoTokenizer.from_pretrained('checkpoint')
    model = BertBilstmCrf.from_pretrained('checkpoint')
    sequence = '12144785	On one hand, they fold into compact, stable structures; ' \
               'on the other hand, they bind a ligand and catalyze a reaction. To be stable, enzymes fold to maximize ' \
               'favorable interactions, forming a tightly packed hydrophobic core, exposing hydrophilic groups, ' \
               'and optimizing intramolecular hydrogen-bonding. To be functional, enzymes carve out an active site ' \
               'for ligand binding, exposing hydrophobic surface area, clustering like charges, and providing ' \
               'unfulfilled hydrogen bond donors and acceptors. Using AmpC beta-lactamase, an enzyme that is ' \
               'well-characterized structurally and mechanistically, the relationship between enzyme stability and ' \
               'function was investigated by substituting key active-site residues and measuring the changes in ' \
               'stability and activity. Substitutions of catalytic residues Ser64, Lys67, Tyr150, Asn152, and Lys315 ' \
               'decrease the activity of the enzyme by 10(3)-10(5)-fold compared to wild-type. Concomitantly, ' \
               'many of these substitutions increase the stability of the enzyme significantly, by up to 4.7kcal/mol. ' \
               'To determine the structural origins of stabilization, the crystal structures of four mutant enzymes ' \
               'were determined to between 1.90A and 1.50A resolution. These structures revealed several mechanisms ' \
               'by which stability was increased, including mimicry of the substrate by the substituted residue (' \
               'S64D), relief of steric strain (S64G), relief of electrostatic strain (K67Q), and improved polar ' \
               'complementarity (N152H). These results suggest that the preorganization of functionality ' \
               'characteristic of active sites has come at a considerable cost to enzyme stability. In proteins of ' \
               'unknown function, the presence of such destabilized regions may indicate the presence of a binding site.'

pre_seq = sequence.split(' ')
# tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
inputs = tokenizer(pre_seq, return_tensors="pt", is_split_into_words=True)
word_ids = inputs.word_ids()
tokens = inputs.tokens()
predictions = model.predict(**inputs)

word_spans: Dict[int, list] = {}
for word_idx, token, prediction in zip(word_ids, tokens, predictions[0]):
    # print((token, model.config.id2label[prediction]))
    print(f'{token}\t{names[prediction]}')
    if word_idx is not None:
        word_spans.setdefault(word_idx, {
            'tokens': [],
            'labels': []
        })
        word_spans[word_idx]['tokens'].append(process_token(token))
        word_spans[word_idx]['labels'].append(names[prediction])

# for idx, word in word_spans.items():
#     print(f'{idx}\t{word}')

spans: Dict[int, Dict] = {}
for idx, tokens in word_spans.items():
    word = reduce(token2word, tokens['tokens'])
    label = calculate_word_label(tokens['labels'])
    spans[idx] = {
        word: label
    }

for o_word, (idx, word) in zip(pre_seq, spans.items()):
    print(f'{idx}\t{o_word}\t{word}')
