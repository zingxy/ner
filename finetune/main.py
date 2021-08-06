## Attention Perfect.
import os
import pprint

import torch
from datasets import load_dataset, load_metric
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import TrainingArguments

from BertBilstmCRF import BertBilstmCrf
from custom_collator import CustomDataCollatorForTokenClassification
from custom_trainer import CustomTrainer


# Note
DataCollatorForTokenClassification = CustomDataCollatorForTokenClassification


## Note 超参数， 其余默认
class HyperParam:
    def __init__(self):
        # self.model_name = "dmis-lab/biobert-base-cased-v1.1"
        # self.model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        self.batch_size = 16
        self.max_seq_length = 128
        self.birnn_hidden_state = 200
        self.birnn_dropout_prob = 0.1
        self.lr = 2e-5
        self.epochs = 100
        self.weight_decay = 0.01
        self.train_set = 'train'
        self.test_set = 'test'
        self.do_train = True
        self.do_eval = False
        self.online = False  ## 使用remote 或者local


Hyper = HyperParam()

# environments
os.environ['HF_DATASETS_OFFLINE'] = '1'
# 0: 16g; 1: 16g; 2:
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# paramters
task = "ner"
# @Todo 尝试一下pubmedbert
model_checkpoint = "dmis-lab/biobert-base-cased-v1.1"
# model_checkpoint = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
# model_checkpoint = 'dmis-lab/biobert-large-cased-v1.1'
model_name = model_checkpoint
tokenizer_name = model_checkpoint
# Param Hype paramter
batch_size = Hyper.batch_size
# 截断、填充到 115 120
## Param
max_seq_length = Hyper.max_seq_length  #
# 对齐策略， subword之后， label与源token一致， 否则使用其它标签对齐策略


label_all_tokens = True
# Note 标签对齐策略

label_alignment_strategy = {
    'begin': 0
}

# tokenizer， @Note bert和biobert lookup table 一样
# locall or cloud
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,
                                          # local_files_only=True,
                                          do_lower_case=False  ## Note 大小写敏感
                                          )


def show_dataset(datasets_dict):
    # 数据准备
    print('数据展示')
    print(datasets_dict)
    print(datasets_dict['train'][0])
    label_list = datasets_dict['train'].features[f'{task}_tags'].feature.names
    return label_list


def show_tokenizer(datasets_dict):
    # 分词过程
    print('分词过程')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    import transformers
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    # 从句子开始分词
    # tokenized_output = tokenizer('Hello, this is one sentence!')
    # print(tokenized_output)
    # 从words开始分词 已经处理成一个个句子了
    tokenized_output = tokenizer(["Hello", ",", "this", "is", "one", "sentence!"], is_split_into_words=True)
    print(tokenized_output)
    # =============================word vs token====================================
    example = datasets['train'][4]
    # words: 通过空格 标点符号等， 还没subword
    words = example['tokens']
    print(f"{'=' * 20}Words{'=' * 20}:\n{words}")

    tokenized_input = tokenizer(example['tokens'], is_split_into_words=True)
    tokens_ids = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])

    print(f"{'=' * 20}Tokens{'=' * 20}:\n{tokens_ids}")
    print(f"Words长度{len(words)}\tTokens长度{len(tokens_ids)}")
    # 每个tokens 在words列中的索引, 拆分的单词索引一样， 插入的token 【CLS】
    print(tokenized_input.word_ids(), '\t', len(tokenized_input.word_ids()), end='\n')
    # 给每个tokens, 或者每个ids 分配label（类别）, 特殊的tokens [cls] [sep] -100
    # words 和 tokens不等价， tokens 和 ids等价
    print(example)
    word_ids = tokenized_input.word_ids()
    aligned_labels = [-100 if i is None else example[f"{task}_tags"][i] for i in word_ids]
    print(len(aligned_labels), len(tokenized_input["input_ids"]))
    print(aligned_labels)


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
    for i, label in enumerate(examples[f"{task}_tags"]):
        # i: 第几个样本
        # @todo batch_index 用来标注是哪一个句子？
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        entity_header_processed = False
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            # 如果是特殊token, 使得其label为-100， 否则的话使用其源单词的label(还有其它策略)， 交叉熵损失函数忽略-100
            if word_idx is None:
                # todo crf Bug label_id 填充问题
                # cls sep 填充0
                label_ids.append(0)
                # softmax
                # label_ids.append(-100)
            ## Note 标签对齐策略
            elif label[word_idx] in [1, 3, 5]:
                # 是不是实体开头
                # 是实体开头且未被处理
                if not entity_header_processed:
                    entity_header_processed = True
                    label_ids.append(label[word_idx])
                # others
                else:
                    label_ids.append(label[word_idx] + 1)
            else:
                label_ids.append(label[word_idx])
                entity_header_processed = False
            # label_ids.append(label[word_idx] if label_all_tokens else -100)

            previous_word_idx = word_idx

        labels.append(label_ids)
    # 给每一sample增加一个labels
    tokenized_inputs["labels"] = labels

    # Add tokens column.
    tokenized_inputs["tokens"] = [tokenizer.convert_ids_to_tokens(tokenized_inputs['input_ids'][i])
                                  for i in range(len(tokenized_inputs['input_ids']))]
    ## Console validation.
    # x = tokenize_and_align_labels(datasets['train'][:5])

    #
    # for ids, tokens in zip(x['input_ids'], x['tokens']):
    #     print(len(ids), len(tokens))
    # for ids, tokens in zip(x['input_ids'], x['tokens']):
    #     for id, token in zip(ids, tokens):
    #         print(id, '\t', token)
    #
    # for ls, tokens in zip(x['labels'], x['tokens']):
    #     for l, token in zip(ls, tokens):
    #         print(token, '\t', label_list[l])

    # pprint.pprint(tokenized_inputs)
    # 返回的结果会作为新列（column; features）加到数据集
    return tokenized_inputs


## Bug 原始的predictions被丢弃
def compute_metrics(p):
    # return {}  ## if cuda run out
    # # 整个testset 1774 * 115
    model.eval()

    # # @ todo overwrite
    # # p:EvalPrediction(predictions=(input_ids, token_type_ids, attention_mask), label_ids=all_labels)
    # # metric 加载脚本
    metric = load_metric("seqeval_.py")
    # labels 每个tokens(ids)真实标签 batch * num of tokens
    predictions, labels = p
    input_ids, token_type_ids, attention_mask = predictions
    masks = attention_mask.cpu().numpy()
    padding_ids = input_ids.cpu().numpy()
    # todo oom
    with torch.no_grad():
        predictions = model.predict(input_ids, token_type_ids, attention_mask)

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
    pprint.pprint(results)

    ## Note DNA
    with open('./output/dna.tsv', 'w') as f:
        is_previous_entity = False
        for x, y, z in zip(true_labels, true_predictions, true_ids):
            flag = False
            for m, n, q in zip(x, y, tokenizer.convert_ids_to_tokens(z)):
                if m in ['B-DNAMutation', 'I-DNAMutation'] or n in ['B-DNAMutation', 'I-DNAMutation']:  #
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

    ## Note Protein
    with open('./output/protein.tsv', 'w') as f:
        is_previous_entity = False
        for x, y, z in zip(true_labels, true_predictions, true_ids):
            flag = False
            for m, n, q in zip(x, y, tokenizer.convert_ids_to_tokens(z)):
                if m in ['B-ProteinMutation', 'I-ProteinMutation'] or n in ['B-ProteinMutation',
                                                                            'I-ProteinMutation']:  #
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

    ## Note SNP
    with open('./output/snp.tsv', 'w') as f:
        is_previous_entity = False
        for x, y, z in zip(true_labels, true_predictions, true_ids):
            flag = False
            for m, n, q in zip(x, y, tokenizer.convert_ids_to_tokens(z)):
                if m in ['B-SNP', 'I-SNP'] or n in ['B-SNP', 'I-SNP']:  #
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

    with open('./output/label.tsv', 'w') as f:
        for x, y, z in zip(true_labels, true_predictions, true_ids):
            for m, n, q in zip(x, y, tokenizer.convert_ids_to_tokens(z)):
                f.write(str(q) + '\t' + str(m) + '\t' + str(n) + '\n')
            f.write('\n')

    return {
        "precision": results["overall_precision"],
        "recall"   : results["overall_recall"],
        "f1"       : results["overall_f1"],
        "accuracy" : results["overall_accuracy"],
    }
    # return {}


def save_mutation(name: str, path: str):
    """
    :param name:    DNAMutation | ProteinMutation | SNP
    :param path:    where to save.
    :return: None
    """
    pass


def performance():
    model.eval()
    metric = load_metric('seqeval_.py')
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


if __name__ == "__main__":
    # 数据集加载脚本： train validation(devel) test
    # download_mode = "force_redownload" 如果数据集合发生变化
    datasets = load_dataset("../datasets_/script/tmvar.py",
                            # download_mode='force_redownload',
                            ignore_verifications=True)
    label_list = show_dataset(datasets)
    tokenize_and_align_labels(datasets['train'][:5])
    # subword之后， 对齐token和标签
    tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

    # 分类器：crf or softmax
    model = BertBilstmCrf.from_pretrained(model_name,
                                          num_labels=len(label_list),
                                          # local_files_only=True
                                          )
    ## Param
    # model.config.Hyper = HyperParam()
    print(model)
    # 超参数设定 production: epochs=100; dev:epochs:30
    # 其它的使用默认配置
    args = TrainingArguments(
            'checkpoint',  # output_dir
            logging_dir='log',  # tensorboard data
            evaluation_strategy='epoch',  # epoch steps no
            save_strategy='no',  # 保存策略
            learning_rate=Hyper.lr,  # Param
            per_device_eval_batch_size=batch_size,
            per_device_train_batch_size=batch_size,
            num_train_epochs=Hyper.epochs,  # Param
            weight_decay=0.01,  # Param
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
    if Hyper.do_eval:
        eval_result = trainer.evaluate()
    # #
    # metric = load_metric("seqeval_.py")
    # test_data = trainer.get_test_dataloader(tokenized_datasets['test'])
    # batch = next(iter(test_data))
    # input_ids = batch['input_ids'].to('cuda')
    # token_type_ids = batch['token_type_ids'].to('cuda')
    # attention_mask = batch['attention_mask'].to('cuda')
    # model.predict(input_ids, token_type_ids, attention_mask)
    ## Note Performance
    # performance()

    # performance
    # results = metric.compute(predictions=true_predictions, references=true_labels)
    # ## detokenize:
    # with open('./output/label.tsv', 'w+') as f:
    #     for x, y, z in zip(true_labels, true_predictions, input_ids):
    #         for m, n, q in zip(x, y, tokenizer.convert_ids_to_tokens(z, True)):
    #             f.write(str(q) + '\t' + str(m) + '\t' + str(n) + '\n')
    #         f.write('\n')
