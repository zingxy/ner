from dataclasses import dataclass



# 标签映射
idx2name = [
    "O",
    "B-DNAMutation",
    "I-DNAMutation",
    "B-ProteinMutation",
    "I-ProteinMutation",
    "B-SNP",
    "I-SNP"
]


class HyperParamBackup:
    using_attention: bool = True
    augment_data: str = 'No Aug' # 是否使用数据增强
    using_char_embedding: bool = False
    batch_size: int = 16
    max_seq_length: int = 128  ## tokens
    birnn_hidden_state: int = 256
    birnn_dropout_prob: float = 0.25
    lr: float = 2e-5
    epochs: int = 50
    weight_decay: float = 0.01
    train_set: str = 'train'
    test_set: str = 'test'
    devel_set: str = 'validation'
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = True
    do_shuffer: bool = True
    evaluation_strategy: bool = 'epoch'  # no steps
    local_file_only: bool = False
    using_char_embedding: bool = False   # do not use this, unimplemented


@dataclass
class HyperParam:
    using_attention: bool = True
    augment_data: str = 'No Aug' # 是否使用数据增强
    using_char_embedding: bool = False
    batch_size: int = 16
    max_seq_length: int = 128  ## tokens
    birnn_hidden_state: int = 256
    birnn_dropout_prob: float = 0.25
    lr: float = 2e-5
    epochs: int = 50
    weight_decay: float = 0.01
    train_set: str = 'train'
    test_set: str = 'test'
    devel_set: str = 'validation'
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = True
    do_shuffer: bool = True
    evaluation_strategy: bool = 'epoch'  # no steps
    local_file_only: bool = False
    using_char_embedding: bool = False   # do not use this, unimplemented

Hyper = HyperParam(using_attention=True)
