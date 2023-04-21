from typing import Dict, List

GOLD: Dict[str, List[str]] = {}
TEXT: Dict[str, str] = {}

# 读取mf的gold
def read_gold_data():
    path = './test_gold_std.txt'
    with open(path, 'r+', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.strip().split('\t', 1)
            GOLD[items[0]] = items[1:]

# 读取mf文本text
def read_train_data():
    path = './test_set.mf.txt'
    with open(path, 'r+', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.strip().split('\t', 1)
            text = ''
            if len(items) == 3:
                TEXT[items[0]] = items[1] + '\t' + items[2]
            elif len(items) == 2:
                TEXT[items[0]] = items[1]
            else:
                TEXT[items[0]] = ''

# 根据gold中的pmid排序
def write_var_data():
    path1 = "./vartext.txt"
    with open(path1, 'w+', encoding='utf-8') as f:
        for pmid, mutations in GOLD.items():
            text = TEXT.get(pmid, '')
            if len(mutations) <= 0:
                continue
            print(text)
            f.write(pmid + '\t' + text + "\n\n")


def write_merge_data():
    path = './merge.txt'
    with open(path, 'w+', encoding='utf-8') as f:
        for pmid, mutations in GOLD.items():
            text = TEXT.get(pmid, '')
            if len(mutations) <= 0:
                continue
            print(text)
            f.write(pmid + '\t' + '\t'.join(mutations) + '*********' + text + "\n")


def write_data_groupby_std_pmid():
    path = 'sort.txt'
    with open(path, 'w+', encoding='utf-8') as f:
        for pmid, mutations in GOLD.items():
            text = TEXT.get(pmid, '')
            if len(mutations) <= 0:
                continue
            print(text)
            f.write(pmid + '\t' + text + "\n")

def write_var_gold_data():
    path1 = "./vargold.txt"
    with open(path1, 'w+', encoding='utf-8') as f:
        for pmid, mutations in GOLD.items():
            if len(mutations) <= 0:
                continue
            f.write(pmid + '\t' + '\t'.join(mutations)+'\n\n')

read_gold_data()
read_train_data()

write_var_data()
write_var_gold_data()