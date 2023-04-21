import collections
import sys
import time
from logging import StreamHandler
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
from typing import List, TYPE_CHECKING

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import Trainer
from transformers.file_utils import (
    is_datasets_available,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.trainer_callback import (
    DefaultFlowCallback,
    ProgressCallback,
)
from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    nested_concat,
    nested_detach
)
from transformers.trainer_utils import (
    EvalPrediction,
    PredictionOutput,
    denumpify_detensorize,
    speed_metrics,
)
from transformers.utils import logging

# Integrations must be imported before ML frameworks:


_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    pass

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_training_run_on_sagemaker():
    logging.add_handler(StreamHandler(sys.stdout))

if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)


# todo 自定义 overwrite NamedTuple
class CustomPredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    input_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


# 自定义
class CustomTrainer(Trainer):
    ## Note 收集损失
    train_loss = []
    eval_loss = []
    best_recall = None
    best_precision = None
    res = None
    f1 = 0
    recall = 0
    precision = 0

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels:  # Note 如果有标签
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    # 这里会过滤掉None, 和loss（loss已经有了）
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                # TODO: this needs to be fixed and made cleaner later.
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)
        # Note modified by zing. 注释掉下面这样， 以获得crf decode 之后的pre_ids
        # logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    ## @overload
    def prediction_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        if not isinstance(dataloader.dataset, collections.abc.Sized):
            raise ValueError("dataset must implement __len__")
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        if self.args.deepspeed and not self.args.do_train:
            # no harm, but flagging to the user that deepspeed config is ignored for eval
            # flagging only for when --do_train wasn't passed as only then it's redundant
            logger.info("Detected the deepspeed argument but it will not be used for evaluation")

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, half it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)
        # todo **batch info accumulation**
        # todo 修改losses 计算
        losses_host: torch.Tensor = None
        ## Note preds_host
        # preds_host: Union[torch.Tensor, List[torch.Tensor]] = None
        preds_host: List[List[int]] = []
        labels_host: Union[torch.Tensor, List[torch.Tensor]] = None
        # @Todo 自定义添加
        # torch.Size([1774, 115])
        input_ids_host: Union[torch.Tensor, List[torch.Tensor]] = None
        attention_host: Union[torch.Tensor, List[torch.Tensor]] = None
        token_type_host: Union[torch.Tensor, List[torch.Tensor]] = None
        ## Note what is world_size,
        world_size = max(1, self.args.world_size)

        eval_losses_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=batch_size)
        if not prediction_loss_only:
            # The actual number of eval_sample can be greater than num_examples in distributed settings (when we pass
            # a batch size to the sampler)
            make_multiple_of = None
            if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, SequentialDistributedSampler):
                make_multiple_of = dataloader.sampler.batch_size
            preds_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)
            labels_gatherer = DistributedTensorGatherer(world_size, num_examples, make_multiple_of=make_multiple_of)

        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader
        # Loop Body
        for step, inputs in enumerate(dataloader):
            # Note logits=emissions 相当于last_hidden_layer
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # Todo 用与解码   ## Bug 惊天大Bug
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs.setdefault('token_type_ids', None)
            if loss is not None:
                losses = loss.repeat(batch_size)
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            # Note
            if logits is not None:
                # Note logits 返回的是List
                # preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                ## nested_concat做了一样是事情
                for item in logits:
                    preds_host.append(item)
            # 这里的padding_index 用于batch之间的对齐， batch 内对齐已经再分词阶段做过了
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            # if labels is not None:
            #     labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=0)
            # todo 用于解码
            if input_ids is not None:
                # padding_index = 0
                input_ids_host = input_ids if input_ids_host is None else nested_concat(input_ids_host, input_ids,
                                                                                        padding_index=0)
            if attention_mask is not None:
                attention_host = attention_mask if attention_host is None else nested_concat(attention_host,
                                                                                             attention_mask,
                                                                                             padding_index=0)
            if token_type_ids is not None:
                token_type_host = token_type_ids if token_type_host is None else nested_concat(token_type_host,
                                                                                               token_type_ids,
                                                                                               padding_index=0)

            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            ## Note
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
                if not prediction_loss_only:
                    preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
                    labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

                # Set back to None to begin a new accumulation
                input_ids_host, losses_host, preds_host, labels_host = None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        eval_losses_gatherer.add_arrays(self._gather_and_numpify(losses_host, "eval_losses"))
        if not prediction_loss_only:
            ## Note
            # preds_gatherer.add_arrays(self._gather_and_numpify(preds_host, "eval_preds"))
            labels_gatherer.add_arrays(self._gather_and_numpify(labels_host, "eval_label_ids"))

        eval_loss = eval_losses_gatherer.finalize()

        preds = preds_host  ## Note modified
        # preds = preds_gatherer.finalize() if not prediction_loss_only else None
        label_ids = labels_gatherer.finalize() if not prediction_loss_only else None

        if self.compute_metrics is not None and preds is not None and label_ids is not None:
            # todo modify by zing.
            # metrics = self.compute_metrics(EvalPrediction(predictions=preds, label_ids=label_ids))
            ## Note 性能
            metrics = self.compute_metrics(
                # EvalPrediction(predictions=(input_ids_host, token_type_host, attention_host),
                # label_ids=label_ids))
                # EvalPrediction(predictions=(input_ids_host, token_type_host, attention_host),
                # label_ids=label_ids))
                EvalPrediction(predictions=(preds, input_ids_host, token_type_host, attention_host),
                               label_ids=label_ids))

        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if eval_loss is not None:
            metrics[f"{metric_key_prefix}_loss"] = eval_loss.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return CustomPredictionOutput(predictions=preds,
                                      label_ids=label_ids,
                                      input_ids=input_ids_host,
                                      metrics=metrics
                                      )

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        ##
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
        #
        # self.log(output.metrics)
        ## Note Note Note
        # print(output.metrics)
        self.f1 = output.metrics['eval_f1']
        if self.res:
            if self.f1 > self.res['eval_f1']:
                self.res = output.metrics
        else:
            self.res = output.metrics
        self.eval_loss.append(output.metrics['eval_f1'])
        ## Note recall
        self.recall = output.metrics['eval_recall']
        if self.best_recall:
            if self.recall > self.best_recall['eval_recall']:
                self.best_recall = output.metrics
            else:
                pass
        else:
            self.best_recall = output.metrics
        ## Note precision
        self.precision = output.metrics['eval_precision']
        if self.best_precision:
            if self.precision > self.best_precision['eval_precision']:
                self.best_precision = output.metrics
            else:
                pass
        else:
            self.best_precision = output.metrics

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        ## Note Overwrite.
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()

        ## Note Note 注释掉这行
        # elif self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()
        ## Note modify
        self.train_loss.append(loss.item())
        return loss.detach()
