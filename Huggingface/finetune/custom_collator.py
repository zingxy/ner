from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers import DataCollatorForTokenClassification
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class CustomDataCollatorForTokenClassification(DataCollatorForTokenClassification):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`,
        defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding
            index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a
            single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute
            capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100


    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
                features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                # Conversion to tensors will fail if we have labels as they are not of the same length yet.
                return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch
        # note 填充长度, input_ids 使用sequences 最大长度， input_ids长度是相等的，
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
                               labels]
            # note 自定义添加  general_labels
            batch["general_labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in
                               batch['general_labels']]

        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in
                               labels]


        tokens = batch.pop('tokens')
        word_ids = batch.pop('word_ids')
        words = batch.pop('words')
        # general_labels = batch.pop('general_labels')
        # general_labels必须填充成长度一样的, 类似labels一样
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        batch['tokens'] = tokens
        batch['word_ids'] = word_ids
        batch['words'] = words
        # batch['general_labels'] = general_labels
        return batch
