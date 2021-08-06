import matplotlib.pyplot as plt
import numpy as np
from datasets import Dataset, load_dataset


def label2id(batch: Dataset) -> dict:
    mapping = ["O",
               "B-DNAMutation",
               "I-DNAMutation",
               "B-ProteinMutation",
               "I-ProteinMutation",
               "B-SNP",
               "I-SNP"]
    batch_labels_str = batch['label_str']
    batch_labels: list[int] = []
    # batch[example1, example2, example3]
    for i, label in enumerate(batch_labels_str):
        # i 指示第几个样本
        # 把单个token的字符label转换成数字
        batch_labels.append(mapping.index(label))
    # 必须返回字典， 返回的数据会添加到每个样本中
    return {
        'label': batch_labels
    }


# data_files
# str: a single string as the path to a single file (considered to constitute the train split by default)
#
# List[str]: a list of strings as paths to a list of files (also considered to constitute the train split by default)
#
# Dict[Union[str, List[str]]]: a dictionary mapping splits names to a single file or a list of files.
if __name__ == '__main__':
    dataset = load_dataset('csv',
                           data_files={
                               'train': '../../data/TMVAR/train.tsv',
                               # 'test':'',
                               # validation: ''
                           },
                           column_names=['token', 'label_str'],
                           delimiter='\t'
                           )

    dataset = dataset.map(label2id, batched=True)
    train_set: Dataset = dataset['train']
    # pie plot
    train_set.set_format('numpy')
    mapping = ["O",
               "B-DNAMutation",
               "I-DNAMutation",
               "B-ProteinMutation",
               "I-ProteinMutation",
               "B-SNP",
               "I-SNP"]
    # len(mapping)
    portion = []
    for i in range(7):
        portion.append(np.count_nonzero(train_set['label'] == i))
    data = {
        'DNAMutation'    : portion[1],  # + portion[2],
        'ProteinMutation': portion[3],  # + portion[4],
        'SNP'            : portion[5],  # + portion[6]
    }
    plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%', shadow=False, startangle=150)
    plt.show()
