# master
import datetime
import logging
import os
from pprint import pprint, pformat
from typing import List, Optional, Union

import torch
from datasets import load_dataset, load_metric
from seqeval.metrics import performance_measure
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import TrainingArguments

from BertBilstmCRF import BertBilstmCrf
from custom_collator import CustomDataCollatorForTokenClassification
from custom_trainer import CustomTrainer, CustomPredictionOutput
from hyper_parameters import Hyper

# Note
DataCollatorForTokenClassification = CustomDataCollatorForTokenClassification

## Note 超参数， 其余默认

# environments
os.environ['HF_DATASETS_OFFLINE'] = '1'
# 0: 16g; 1: 16g; 2:
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# paramters
task = "ner"
# 这里可以切换不同的预训练模型，BERT-like
model_checkpoint = "dmis-lab/biobert-base-cased-v1.1"
model_checkpoint = "dmis-lab/biobert-base-cased-v1.2"
# model_checkpoint = "bert-base-cased"

model_name = model_checkpoint
tokenizer_name = model_checkpoint
# Param Hype paramter
batch_size = Hyper.batch_size
# 截断、填充到 115 120
## Param
max_seq_length = Hyper.max_seq_length  #
# 对齐策略， subword之后， label与源token一致， 否则使用其它标签对齐策略

writer = SummaryWriter('info')
label_all_tokens = True
# Note 标签对齐策略

label_alignment_strategy = {
    'begin': 0
}

## Note tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                          local_files_only=Hyper.local_file_only,
                                          do_lower_case=False  ## Note 大小写敏感
                                          )


## Note Roberta
# tokenizer = AutoTokenizer.from_pretrained('roberta-base', add_prefix_space=True, do_lower_case=False)


def show_dataset(datasets_dict):
    # 数据准备
    logging.debug('show data')
    print(datasets_dict)
    print(datasets_dict['train'][0])
    label_list = datasets_dict['train'].features[f'{task}_tags'].feature.names
    return label_list


def tokenize_and_align_labels(examples) -> dict:
    # 分词并对齐标签
    # examples 是一个batch
    # tokenizing 不padding，
    tokenized_inputs = tokenizer(examples["tokens"],
                                 truncation=True,
                                 is_split_into_words=True,
                                 max_length=max_seq_length)
    # 给tokenized_inputs 添加一个 label
    labels = []
    batch_word_ids = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        # i: 第几个样本
        # @todo batch_index 用来标注是哪一个句子？
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        batch_word_ids.append(word_ids)
        previous_word_idx = None
        previous_token_label = None
        label_ids = []
        entity_header_processed = False
        for word_idx in word_ids:
            if word_idx is None:
                # todo crf Bug label_id 填充问题
                # cls sep 填充0
                label_ids.append(0)
                # softmax
                # label_ids.append(-100)
            ## Note 标签对齐策略
            ## Bug 连续两个实体有问题（B-SNP,B-DNA）
            elif label[word_idx] in [1, 3, 5]:  # 查询当前word_idx所对应的token的标签
                # 是不是实体开头
                if previous_word_idx != word_idx:
                    # 前面那个word也是实体开头， 这个word也是实体开头， 说明遇到了新的实体
                    entity_header_processed = False
                if not entity_header_processed:
                    entity_header_processed = True
                    label_ids.append(label[word_idx])
                # others
                else:
                    label_ids.append(label[word_idx] + 1)
            else:
                label_ids.append(label[word_idx])
                entity_header_processed = False
            # previous_token_label = label[word_idx] if word_idx else None
            previous_word_idx = word_idx
        labels.append(label_ids)
    # 给每一sample增加一个labels

    # todo 去除每个实体标签后的信息e.g. B-DNA -> B, I-DNA -> I
    general_labels: List[List[int]] = []
    for labels_i in labels:
        current_batch_labels: List[int] = []
        for label_id in labels_i:
            if label_id == 0:
                current_batch_labels.append(0)
            elif label_id in [1, 3, 5]:
                current_batch_labels.append(1)
            else:
                current_batch_labels.append(2)
        general_labels.append(current_batch_labels)
    # Note label vs general_label bug
    tokenized_inputs["labels"] = labels

    # Note 冗余信息   general_labels用来寻找边界， 评估的时候用labels
    tokenized_inputs["general_labels"] = general_labels

    # ---begin Todo 用于测试多任务中general_labels, 测试完成需要注释掉这里
    # tokenized_inputs['labels'] = general_labels
    # ---end
    tokenized_inputs["word_ids"] = batch_word_ids
    # Add tokens column.
    tokenized_inputs["tokens"] = [tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i])
                                  for i in range(len(tokenized_inputs['input_ids']))]
    tokenized_inputs["words"] = examples["tokens"]

    return tokenized_inputs


## Bug 原始的predictions被丢弃
def compute_metrics(p):
    model.eval()
    # Todo 这里使用了自定义的eval方式，因为related papers 都是使用macro f1,
    metric = load_metric("seqeval_.py")
    # labels 每个tokens(ids)真实标签 batch * num of tokens
    predictions, labels = p
    pred_ids, input_ids, token_type_ids, attention_mask = predictions
    # pred_ids,  attention_mask = predictions
    predictions = pred_ids
    masks = attention_mask.cpu().numpy()
    padding_ids = input_ids.cpu().numpy()

    true_predictions = [
        [label_list[p] for p in prediction]
        for prediction in predictions
    ]
    # 去掉mask==0, 这些是填充token;
    true_ids = [
        [id for (m, id) in zip(mask, ids) if m != 0]
        for mask, ids in zip(masks, padding_ids)
    ]
    true_labels = [
        [label_list[l] for (m, l) in zip(mask, label) if m != 0]
        for mask, label in zip(masks, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)

    # begin Todo 这里可能有bug, 忘记了用来干什么的了
    matrix = performance_measure(true_predictions, true_labels)
    # end---
    pprint(results)

    ## 用于生成log
    save_entitys('DNA', 'output', true_labels, true_predictions, true_ids)
    save_entitys('Protein', 'output', true_labels, true_predictions, true_ids)
    save_entitys('SNP', 'output', true_labels, true_predictions, true_ids)

    ## 测试集的最终结果
    with open('./output/label.tsv', 'w') as f:
        for x, y, z in zip(true_labels, true_predictions, true_ids):
            for m, n, q in zip(x, y, tokenizer.convert_ids_to_tokens(z)):
                f.write(str(q) + '\t' + str(m) + '\t' + str(n) + '\n')
            f.write('\n')

    return {
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "DNAMutation": results["DNAMutation"],
        ## Todo 使用general_labels没有这些信息，在测试general的时候需要注释下面两行
        "ProteinMutaion": results["ProteinMutation"],
        # "SNP": results["SNP"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def save_entitys(entity_type: str, basedir: Optional[str] = 'output', *info: List[List[Union[int, str]]]):
    entity_type_tags = {
        'SNP': ['B-SNP', 'I-SNP'],
        'DNA': ['B-DNAMutation', 'I-DNAMutation'],
        'Protein': ['B-ProteinMutation', 'I-ProteinMutation'],
    }

    path = os.path.join(basedir, entity_type + '.tsv')

    true_labels, true_predictions, true_ids = info
    with open(path, 'w') as f:
        is_previous_entity = False
        for x, y, z in zip(true_labels, true_predictions, true_ids):
            flag = False
            for m, n, q in zip(x, y, tokenizer.convert_ids_to_tokens(z)):
                if m in entity_type_tags[entity_type] or n in entity_type_tags[entity_type]:  #
                    # m[2:] == 'DNAMutation'
                    flag = True
                    is_previous_entity = True
                    f.write(str(q) + '\t' + str(m) + '\t' + str(n) + '\n')
                # 实体后的第一个O标签 实体分隔
                elif is_previous_entity:
                    is_previous_entity = False
                    f.write('\n')
                else:
                    pass
            # 句子分隔
            if flag:
                f.write('\n')


def performance():
    model.eval()
    metric = load_metric('datasets_/seqeval_.py')
    test_data = trainer.get_test_dataloader(tokenized_datasets[Hyper.test_set])

    true_labels = []
    true_predictions = []
    padding_ids = []
    true_ids = []
    predictions = []
    for batch in tqdm(test_data):  # 111 batchs
        input_ids = batch['input_ids'].to('cuda')
        token_type_ids = batch['token_type_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        masks = attention_mask.cpu().numpy()
        labels = batch['labels'].numpy()
        padding_ids = input_ids.cpu().numpy()
        # 去掉mask==0, 这些是填充token;
        true_labels += [
            [label_list[l] for (m, l) in zip(mask, label) if m != 0]
            for mask, label in zip(masks, labels)
        ]
        true_ids += [
            [id for (m, id) in zip(mask, ids) if m != 0]
            for mask, ids in zip(masks, padding_ids)
        ]
        ## 每个batch的预测值[batch, len(seq)]
        with torch.no_grad():
            predictions += model.predict(input_ids, token_type_ids, attention_mask)
    true_predictions = [
        [label_list[p] for p in prediction]
        for prediction in predictions
    ]
    ## Note Bug: 实体开头字符丢失
    with open('./output/label.tsv', 'w+') as f:
        for x, y, z in zip(true_labels, true_predictions, true_ids):
            for m, n, q in zip(x, y, tokenizer.convert_ids_to_tokens(z)):
                f.write(str(q) + '\t' + str(m) + '\t' + str(n) + '\n')
            f.write('\n')
    results = metric.compute(predictions=true_predictions, references=true_labels)
    pprint.pprint(results)


def save_performance(result: CustomPredictionOutput):
    with open('output/performance.txt', 'a+') as f:
        fmt = f"\n\n{' ' * 20}{'*' * 10}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{'*' * 10}\n"
        f.write(fmt)
        f.write(pformat(Hyper))
        f.write('\n\n')
        f.write(pformat(result.metrics))


if __name__ == "__main__":
    # 数据集加载脚本： train validation(devel) test
    # download_mode = "force_redownload" 如果数据集合发生变化
    datasets = load_dataset("../datasets_/script/tmvar.py",
                            download_mode='force_redownload',
                            ignore_verifications=True)
    if Hyper.do_shuffer:
        datasets = datasets.shuffle()
    label_list = show_dataset(datasets)
    # tokenize_and_align_labels(datasets['train'][:5])
    # subword之后， 对齐token和标签
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    model = BertBilstmCrf.from_pretrained(model_name,
                                          # config=config,
                                          num_labels=len(label_list),
                                          local_files_only=Hyper.local_file_only
                                          )

    args = TrainingArguments(
        'checkpoint',  # output_dir
        logging_dir='log',  # tensorboard data
        evaluation_strategy='epoch',  # epoch steps no
        save_strategy='no',  # 保存策略
        learning_rate=Hyper.lr,  # Param
        per_device_eval_batch_size=batch_size,
        per_device_train_batch_size=batch_size,
        num_train_epochs=Hyper.epochs,  # Param
        weight_decay=Hyper.weight_decay,  # Param
        report_to='tensorboard',
        logging_strategy='epoch',
        # adafactor=True
        # metric_for_best_model='f1',
        # load_best_model_at_end=True
        # do_eval=False
    )
    # @todo 数据对齐
    # crf classifier 填充0
    # softmax classifier 填充-100
    # Bug padding
    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=max_seq_length, label_pad_token_id=0)

    # training loop
    trainer = CustomTrainer(
        model,
        args,
        train_dataset=tokenized_datasets[Hyper.train_set],  ## Note Overfitting
        eval_dataset=tokenized_datasets[Hyper.test_set],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics  # compute_metrics; evaluate
    )
    ## Note Train
    if Hyper.do_train:
        train_result = trainer.train()
    ## Note Eval
    if Hyper.do_predict:
        predict_result = trainer.predict(tokenized_datasets[Hyper.test_set])
        save_performance(predict_result)
