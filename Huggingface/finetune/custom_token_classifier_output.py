# @Time 9/1/2021 12:53 PM
# @Author 1067577595@qq.com
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
from transformers.file_utils import ModelOutput


@dataclass
class TokenClassifierOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    pred_ids:List[List[int]] = None