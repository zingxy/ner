import torch
import numpy as np
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

if __name__ == '__main__':
    #
    tokenizer = AutoTokenizer.from_pretrained('crfcheckpoint')
    model = BertBilstmCrf.from_pretrained('crfcheckpoint')
    sequence = 'Mutational analysis of the p63 gene showed a novel heterozygous T>C nucleotide substitution on exon ' \
               '14 (I597T).'
    sequence = 'NM_018718.3(CEP41):c.616C>G (p.Pro206Ala) AND Joubert syndrome'
    sequence = 'We observed a total mutation rate of 21%. We found six mutations on tissue biopsy: Y537S (1), ' \
               'D538G (2), Y537N (1), E380Q (2).'
    # sequence = 'We identified T10191C(P.S45P) in ND3.'
    sequence = 'we investigated whether N372H and another common variant located in the 5-untranslated region (203G > A) of the BRCA2 gene modify breast or ovarian cancer risk in BRCA1 mutation carriers. '
    # BUG
    sequence = 'while the second b-globin gene (inherited paternally) had a 13-bp deletion at nucleotide 90 downstream of the termination codon (CD +90 del 13 bp).'
    # Note 这个可以
    # sequence = 'real-time (RT) PCR. DNA sequencing showed a base pair substitution (c.1706-2 A>T) in the splice acceptor site of LDLR intron 11'
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
    inputs = tokenizer(sequence, return_tensors="pt")
    predictions = model.predict(**inputs)
    for token, prediction in zip(tokens, predictions[0]):
        # print((token, model.config.id2label[prediction]))
        print((token, names[prediction]))
