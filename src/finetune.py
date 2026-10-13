"""Seq2Seq fine-tuning of Whisper for Pseudo2Real students."""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .py310_re_lock import install_py310_re_lock

install_py310_re_lock()

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

from .asr_backend import is_qwen3_asr, load_asr_model, load_processor
from .dataset import DataCollatorQwen3ASR, DataCollatorSpeechSeq2SeqWithPadding, L2SpeechDataset
from .progress import hf_trainer_tqdm_disabled

logger = logging.getLogger(__name__)

# Keys the Qwen3-ASR causal LM actually consumes. Extra processor / Trainer
# fields (e.g. mm token type ids, num_items_in_batch leaking into the audio
# encoder) can corrupt Module iteration used by grad clipping.
_QWEN3_ASR_FORWARD_KEYS = (
    "input_ids",
    "attention_mask",
    "input_features",
    "input_features_mask",
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "labels",
    "use_cache",
    "logits_to_keep",
)


_DECAY_NAME_SKIP = re.compile(
    r"bias|layernorm|rmsnorm|(?:^|\.)norm(?:$|\.)|_norm(?:$|\.)",
    re.IGNORECASE,
)


def _iter_qwen_modules(root: torch.nn.Module) -> Iterator[Tuple[str, torch.nn.Module]]:
    """Walk ``_modules`` via ``__dict__`` so a broken ``named_modules`` cannot recurse."""
    seen: set[int] = set()
    stack: List[Tuple[str, torch.nn.Module]] = [("", root)]
    while stack:
        prefix, module = stack.pop()
        if not isinstance(module, torch.nn.Module):
            continue
        identity = id(module)
        if identity in seen:
            continue
        seen.add(identity)
        yield prefix, module
        children = module.__dict__.get("_modules") or {}
        for name, child in children.items():
            if child is None:
                continue
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            stack.append((child_prefix, child))


def _iter_qwen_named_parameters(root: torch.nn.Module) -> List[Tuple[str, torch.nn.Parameter]]:
    """Collect parameters without ``Module.named_parameters``.

    The live walk calls ``isinstance(..., Parameter)`` and unpacks module
    dicts. On Qwen3-ASR that path raises ``NameError: name '__dict__'`` or
    ``ValueError: too many values to unpack (expected 0)`` after the first
    successful pass. The ``_parameters`` dicts themselves stay usable.
    """
    found: List[Tuple[str, torch.nn.Parameter]] = []
    seen: set[int] = set()
    for prefix, module in _iter_qwen_modules(root):
        params = module.__dict__.get("_parameters") or {}
        for name, param in list(params.items()):
            if param is None or id(param) in seen:
                continue
            seen.add(id(param))
            full = f"{prefix}.{name}" if prefix else str(name)
            found.append((full, param))
    return found


def _safe_qwen_train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
    """Set ``training`` without ``Module.__setattr__`` or ``children().train()``."""
    if not isinstance(mode, bool):
        raise ValueError("training mode is expected to be boolean")
    for _, module in _iter_qwen_modules(self):
        object.__setattr__(module, "training", mode)
    return self


def _bind_qwen_walks(model: torch.nn.Module, named_params: List[Tuple[str, torch.nn.Parameter]]) -> None:
    modules = list(_iter_qwen_modules(model))
    params = [param for _, param in named_params]

    def named_modules(_self, memo=None, prefix="", remove_duplicate=True):
        del _self, memo, remove_duplicate
        for name, module in modules:
            if not prefix or name == prefix or name.startswith(prefix + "."):
                yield name, module

    def named_parameters(_self, prefix="", recurse=True, remove_duplicate=True):
        del _self, recurse, remove_duplicate
        for name, param in named_params:
            if not prefix or name == prefix or name.startswith(prefix + "."):
                yield name, param

    def parameters(_self, recurse=True):
        del _self, recurse
        yield from params

    # object.__setattr__ skips Module.__setattr__, whose isinstance(Parameter)
    # check is one of the calls that blows up on this model.
    object.__setattr__(model, "named_modules", types.MethodType(named_modules, model))
    object.__setattr__(model, "named_parameters", types.MethodType(named_parameters, model))
    object.__setattr__(model, "parameters", types.MethodType(parameters, model))
    object.__setattr__(model, "train", types.MethodType(_safe_qwen_train, model))


def _qwen_decay_parameter_names(model: torch.nn.Module) -> List[str]:
    """Same filter as ``Trainer.get_decay_parameter_names``, without ``named_children``."""
    names: List[str] = []
    for prefix, module in _iter_qwen_modules(model):
        skip_module = isinstance(module, torch.nn.LayerNorm)
        params = module.__dict__.get("_parameters") or {}
        for name, param in params.items():
            if param is None:
                continue
            full = f"{prefix}.{name}" if prefix else str(name)
            if skip_module or _DECAY_NAME_SKIP.search(full):
                continue
            names.append(full)
    return names


def _stabilize_qwen3_module_walks(model: torch.nn.Module, *, force: bool = False) -> None:
    """Replace module walks that Qwen3-ASR corrupts.

    Do not snapshot via ``named_modules()`` / ``named_parameters()``. Those are
    the calls that fail, and a snapshot taken through them goes stale when
    ``.to(device)`` replaces Parameter objects.
    """
    if not force and getattr(model, "_l2_module_walks_frozen", False):
        return
    named_params = _iter_qwen_named_parameters(model)
    _bind_qwen_walks(model, named_params)
    object.__setattr__(model, "_l2_module_walks_frozen", True)
    logger.info(
        "Froze Qwen3-ASR module walks (%d modules, %d parameters)",
        sum(1 for _ in _iter_qwen_modules(model)),
        len(named_params),
    )


def _disable_qwen3_gradient_checkpointing(model: torch.nn.Module) -> None:
    """Turn off layer-level and model-level checkpointing.

    Qwen3-ASR audio encoder splits Q/K/V by ``cu_seqlens``. Checkpoint recompute
    (reentrant or not) eventually corrupts that split:

    - ``use_reentrant=False`` → ``TypeError: '_Handle' object is not callable``
    - ``use_reentrant=True`` → ``ValueError: too many values to unpack (expected 0)``
      in ``zip(*splits)``
    """
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        disable()
    if hasattr(model, "gradient_checkpointing"):
        model.gradient_checkpointing = False
    for module in model.modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
    logger.info("Disabled Qwen3-ASR gradient checkpointing (audio cu_seqlens is unsafe with GC)")


class _Qwen3AsrTrainer(Trainer):
    """Keep Qwen3-ASR training off the live ``named_parameters`` / ``train`` walks.

    Transformers 5 calls ``model.train()`` and ``named_parameters()`` on every
    optimizer build and every step. Those walks unpack module state and call
    ``isinstance(..., Parameter)``. On this model they eventually raise
    ``ValueError: too many values to unpack (expected 0)`` or
    ``NameError: name '__dict__'``. Parameter lists are taken from ``_parameters``
    instead, and ``training`` is set with ``object.__setattr__``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clip_param_cache = None
        self._cached_non_embed_params = None
        # Qwen3-ASR.forward has **kwargs, so Trainer would inject num_items_in_batch
        # into the audio encoder. Disable that and let Trainer scale by grad accum.
        self.model_accepts_loss_kwargs = False
        self._freeze_walks()

    def _freeze_walks(self) -> None:
        _stabilize_qwen3_module_walks(self.model)
        wrapped = getattr(self, "model_wrapped", None)
        if wrapped is not None and wrapped is not self.model:
            _stabilize_qwen3_module_walks(wrapped)
        unwrap = getattr(self.accelerator, "unwrap_model", None)
        if unwrap is not None:
            inner = unwrap(self.model)
            if inner is not self.model:
                _stabilize_qwen3_module_walks(inner)

    def get_decay_parameter_names(self, model) -> List[str]:
        return _qwen_decay_parameter_names(model)

    def create_optimizer(self, model=None):
        opt_model = self.model if model is None else model
        # Rebuild after ``.to(device)``, which may have replaced Parameter objects.
        _stabilize_qwen3_module_walks(opt_model, force=True)
        return super().create_optimizer(model)

    def training_step(self, model, inputs, num_items_in_batch=None):
        if isinstance(model, torch.nn.Module):
            object.__setattr__(model, "train", types.MethodType(_safe_qwen_train, model))
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

    def _prepare_for_training(self, *args, **kwargs):
        out = super()._prepare_for_training(*args, **kwargs)
        self._freeze_walks()
        return out

    def floating_point_ops(self, inputs):
        main_input = getattr(self.model, "main_input_name", "input_ids")
        if main_input not in inputs:
            return 0
        if self._cached_non_embed_params is None:
            model = self.accelerator.unwrap_model(self.model)
            try:
                self._cached_non_embed_params = int(model.num_parameters(exclude_embeddings=True))
            except ValueError:
                self._cached_non_embed_params = sum(p.numel() for p in model.parameters())
        return 6 * int(inputs[main_input].numel()) * self._cached_non_embed_params

    def _cached_clip_params(self, model):
        if self._clip_param_cache is None:
            params = [p for p in model.parameters() if p.requires_grad]
            if not params:
                params = list(model.parameters())
            self._clip_param_cache = params
            logger.info("Cached %d Qwen3-ASR parameters for gradient clipping", len(params))
        return self._clip_param_cache

    def _clip_grad_norm(self, model):
        return self.accelerator.clip_grad_norm_(self._cached_clip_params(model), self.args.max_grad_norm)

    def _get_grad_norm(self, model, grad_norm=None):
        if grad_norm is None:
            grad_norm = self.accelerator.clip_grad_norm_(self._cached_clip_params(model), float("inf"))
        dist_type = getattr(self.accelerator, "distributed_type", None)
        if dist_type is not None and str(dist_type).endswith("DEEPSPEED") and hasattr(grad_norm, "item"):
            grad_norm = grad_norm.item()
        return grad_norm

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        filtered = {k: v for k, v in inputs.items() if k in _QWEN3_ASR_FORWARD_KEYS}
        outputs = model(**filtered)
        loss = getattr(outputs, "loss", None)
        if loss is None:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


@dataclass
class FinetuneConfig:
    model_name_or_path: str = "openai/whisper-base"
    train_csv: str = ""
    output_dir: str = "checkpoints/run"
    text_column: str = "text"
    language: str = "en"
    task: str = "transcribe"
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = 4000
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    bf16: bool = False
    logging_steps: int = 50
    save_steps: int = 1000
    save_total_limit: int = 2
    seed: int = 42
    dataloader_num_workers: int = 4
    gradient_checkpointing: bool = True
    eval_csv: Optional[str] = None
    eval_steps: int = 500
    resume_from_checkpoint: Optional[str] = None
    max_grad_norm: float = 1.0
    load_best_model_at_end: bool = False
    metric_for_best_model: str = "eval_loss"
    hypothesis_column: Optional[str] = None
    margin_alpha: float = 0.0
    margin_m: float = 0.0
    margin_reduction: str = "mean"
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    lora_bias: str = "none"
    merge_lora_on_save: bool = True


def _training_arg_extras(args_cls, has_eval: bool, num_workers: int) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    try:
        import inspect

        params = inspect.signature(args_cls.__init__).parameters
        key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    except (TypeError, ValueError):
        params = {}
        key = "evaluation_strategy"
    extra[key] = "steps" if has_eval else "no"
    if "save_strategy" in params:
        extra["save_strategy"] = "steps"
    if "dataloader_pin_memory" in params:
        extra["dataloader_pin_memory"] = torch.cuda.is_available()
    if "dataloader_persistent_workers" in params:
        extra["dataloader_persistent_workers"] = num_workers > 0
    if "dataloader_prefetch_factor" in params and num_workers > 0:
        extra["dataloader_prefetch_factor"] = 4
    return extra


def _trainer_init_params(trainer_cls) -> Dict[str, Any]:
    """Resolve ``Trainer.__init__`` kwargs even when a subclass only has ``*args, **kwargs``."""
    import inspect

    try:
        params = dict(inspect.signature(trainer_cls.__init__).parameters)
    except (TypeError, ValueError):
        params = {}
    if "processing_class" in params or "tokenizer" in params:
        return params
    for base in getattr(trainer_cls, "__mro__", ()):
        if base is trainer_cls:
            continue
        try:
            bparams = dict(inspect.signature(base.__init__).parameters)
        except (TypeError, ValueError):
            continue
        if "processing_class" in bparams or "tokenizer" in bparams:
            return bparams
    return params


def _attach_processor(trainer_kwargs: Dict[str, Any], trainer_cls, processor, *, feature_extractor: bool) -> None:
    payload = processor.feature_extractor if feature_extractor else processor
    tparams = _trainer_init_params(trainer_cls)
    if "processing_class" in tparams:
        trainer_kwargs["processing_class"] = payload
    elif "tokenizer" in tparams:
        trainer_kwargs["tokenizer"] = payload
    else:
        trainer_kwargs["processing_class"] = payload


def _precision_flags(cfg: FinetuneConfig) -> Dict[str, bool]:
    cuda = torch.cuda.is_available()
    use_bf16 = bool(cfg.bf16 and cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(cfg.fp16 and cuda and not use_bf16)
    return {"fp16": use_fp16, "bf16": use_bf16}


def _maybe_apply_lora(model, cfg: FinetuneConfig, *, is_qwen: bool):
    if not cfg.use_lora:
        return model
    from .lora_utils import apply_lora

    logger.info("Wrapping %s with LoRA", cfg.model_name_or_path)
    return apply_lora(
        model,
        is_qwen=is_qwen,
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        bias=cfg.lora_bias,
    )


def _lora_meta(cfg: FinetuneConfig, *, is_qwen: bool) -> Dict[str, Any]:
    from .lora_utils import default_target_modules

    return {
        "use_lora": True,
        "base_model": cfg.model_name_or_path,
        "lora_r": int(cfg.lora_r),
        "lora_alpha": int(cfg.lora_alpha),
        "lora_dropout": float(cfg.lora_dropout),
        "lora_bias": str(cfg.lora_bias),
        "lora_target_modules": list(cfg.lora_target_modules or default_target_modules(is_qwen)),
        "learning_rate": float(cfg.learning_rate),
        "max_steps": int(cfg.max_steps),
    }


def _save_trained_model(trainer, processor, cfg: FinetuneConfig, *, is_qwen: bool) -> None:
    output_dir = Path(cfg.output_dir)
    raw = trainer.model
    unwrap = getattr(getattr(trainer, "accelerator", None), "unwrap_model", None)
    if callable(unwrap):
        raw = unwrap(raw)
    if cfg.use_lora and cfg.merge_lora_on_save and hasattr(raw, "merge_and_unload"):
        from .lora_utils import merge_lora_and_save

        merge_lora_and_save(raw, processor, output_dir, extra_meta=_lora_meta(cfg, is_qwen=is_qwen))
        logger.info("Saved merged LoRA model to %s", output_dir)
        return
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    logger.info("Saved fine-tuned model to %s", output_dir)


def finetune_whisper(cfg: FinetuneConfig) -> str:
    """Fine-tune Whisper or Qwen3-ASR and return the output directory path."""
    if is_qwen3_asr(cfg.model_name_or_path):
        return _finetune_qwen3_asr(cfg)
    return _finetune_whisper(cfg)


def sequence_logprob(logits: torch.Tensor, labels: torch.Tensor, reduction: str) -> torch.Tensor:
    """Per-sequence token log-probability. ``mean`` length-normalizes; ``sum`` is joint log p."""
    if logits.size(1) != labels.size(1):
        length = min(logits.size(1), labels.size(1))
        logits = logits[:, :length]
        labels = labels[:, :length]
    log_probs = logits.float().log_softmax(dim=-1)
    safe_labels = labels.masked_fill(labels.eq(-100), 0).long()
    token_lp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    mask = labels.ne(-100)
    token_lp = token_lp.masked_fill(~mask, 0.0)
    total = token_lp.sum(dim=-1)
    if reduction == "sum":
        return total
    if reduction == "mean":
        lengths = mask.sum(dim=-1).clamp(min=1).to(total.dtype)
        return total / lengths
    raise ValueError(f"margin_reduction must be 'mean' or 'sum', got {reduction!r}")


class _MarginSeq2SeqTrainer(Seq2SeqTrainer):
    """NLL plus a hinge that prefers the surface transcript over the frozen hypothesis.

    L = L_NLL + alpha * max(0, m - s(y|x) + s(yhat|x))
    s is the length-normalized mean token log-probability when margin_reduction='mean'.
    """

    def __init__(self, *args, margin_alpha: float = 1.0, margin_m: float = 0.0, margin_reduction: str = "mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.margin_alpha = float(margin_alpha)
        self.margin_m = float(margin_m)
        self.margin_reduction = str(margin_reduction)
        self._margin_calls = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        hyp_labels = inputs.pop("hyp_labels", None)
        outputs = model(**inputs)
        nll = outputs.loss
        if hyp_labels is None or self.margin_alpha == 0.0:
            return (nll, outputs) if return_outputs else nll

        hyp_labels = hyp_labels.to(device=outputs.logits.device)
        hyp_outputs = model(input_features=inputs["input_features"], labels=hyp_labels)
        s_y = sequence_logprob(outputs.logits, inputs["labels"], self.margin_reduction)
        s_hat = sequence_logprob(hyp_outputs.logits, hyp_labels, self.margin_reduction)
        valid = inputs["labels"].ne(-100).any(dim=-1) & hyp_labels.ne(-100).any(dim=-1)
        hinge = torch.clamp(self.margin_m - s_y + s_hat, min=0.0)
        hinge = torch.where(valid, hinge, torch.zeros_like(hinge))
        denom = valid.sum().clamp(min=1).to(hinge.dtype)
        margin = hinge.sum() / denom
        loss = nll + (self.margin_alpha * margin).to(dtype=nll.dtype)

        self._margin_calls += 1
        logging_steps = max(int(getattr(self.args, "logging_steps", 50) or 50), 1)
        if self._margin_calls == 1 or self._margin_calls % logging_steps == 0:
            active = (hinge > 0).float()
            active_rate = (active * valid.float()).sum() / denom
            logger.info(
                "margin step_calls=%d nll=%.4f margin=%.4f active=%.3f s_y=%.4f s_hat=%.4f",
                self._margin_calls,
                float(nll.detach()),
                float(margin.detach()),
                float(active_rate.detach()),
                float(s_y[valid].mean().detach()) if bool(valid.any()) else float("nan"),
                float(s_hat[valid].mean().detach()) if bool(valid.any()) else float("nan"),
            )
        return (loss, outputs) if return_outputs else loss


def _finetune_whisper(cfg: FinetuneConfig) -> str:
    if not cfg.train_csv:
        raise ValueError("train_csv is required")
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Fine-tuning %s -> %s (text_column=%s)", cfg.model_name_or_path, output_dir, cfg.text_column)

    processor = WhisperProcessor.from_pretrained(cfg.model_name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_name_or_path)

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=cfg.language, task=cfg.task)
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.forced_decoder_ids = forced_decoder_ids
    model.config.suppress_tokens = []
    model.generation_config.language = cfg.language
    model.generation_config.task = cfg.task

    use_margin = bool(cfg.hypothesis_column) and float(cfg.margin_alpha) != 0.0
    # Two teacher-forced forwards share the encoder graph. Checkpointing frees
    # that graph on the first backward, so the hypothesis term cannot backprop.
    if cfg.gradient_checkpointing and not use_margin:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    model = _maybe_apply_lora(model, cfg, is_qwen=False)
    if cfg.use_lora and cfg.gradient_checkpointing and not use_margin and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    max_label_tokens = getattr(model.config, "max_target_positions", None)
    train_dataset = L2SpeechDataset(
        cfg.train_csv,
        processor=processor,
        text_column=cfg.text_column,
        hypothesis_column=cfg.hypothesis_column,
        max_label_tokens=max_label_tokens,
        skipped_rows_path=str(output_dir / "skipped_train_rows.csv"),
        language=cfg.language,
    )
    eval_dataset = None
    if cfg.eval_csv:
        eval_dataset = L2SpeechDataset(
            cfg.eval_csv,
            processor=processor,
            text_column="text",
            max_label_tokens=max_label_tokens,
            skipped_rows_path=str(output_dir / "skipped_eval_rows.csv"),
            language=cfg.language,
        )

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    eval_kwargs = _training_arg_extras(
        Seq2SeqTrainingArguments, eval_dataset is not None, cfg.dataloader_num_workers
    )
    precision = _precision_flags(cfg)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=int(
            cfg.per_device_eval_batch_size
            if cfg.per_device_eval_batch_size is not None
            else cfg.per_device_train_batch_size
        ),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_steps=cfg.eval_steps if eval_dataset is not None else None,
        predict_with_generate=False,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        max_grad_norm=float(cfg.max_grad_norm),
        load_best_model_at_end=bool(cfg.load_best_model_at_end) and eval_dataset is not None,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=False,
        push_to_hub=False,
        disable_tqdm=hf_trainer_tqdm_disabled(),
        **precision,
        **eval_kwargs,
    )

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    _attach_processor(trainer_kwargs, Seq2SeqTrainer, processor, feature_extractor=True)
    if use_margin:
        if cfg.margin_reduction not in {"mean", "sum"}:
            raise ValueError(f"margin_reduction must be 'mean' or 'sum', got {cfg.margin_reduction!r}")
        logger.info(
            "Margin loss alpha=%s m=%s reduction=%s hypothesis_column=%s",
            cfg.margin_alpha,
            cfg.margin_m,
            cfg.margin_reduction,
            cfg.hypothesis_column,
        )
        trainer = _MarginSeq2SeqTrainer(
            margin_alpha=cfg.margin_alpha,
            margin_m=cfg.margin_m,
            margin_reduction=cfg.margin_reduction,
            **trainer_kwargs,
        )
        (output_dir / "margin_loss.json").write_text(
            json.dumps(
                {
                    "alpha": cfg.margin_alpha,
                    "m": cfg.margin_m,
                    "reduction": cfg.margin_reduction,
                    "hypothesis_column": cfg.hypothesis_column,
                    "formula": "L = L_NLL + alpha * max(0, m - s(y|x) + s(yhat|x))",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        trainer = Seq2SeqTrainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    _save_trained_model(trainer, processor, cfg, is_qwen=False)
    return str(output_dir)


def _qwen_training_ready(output_dir: Path, max_steps: int) -> bool:
    """True when a Qwen worker already wrote a finished checkpoint before dying."""
    has_weights = (
        (output_dir / "model.safetensors").is_file()
        or (output_dir / "pytorch_model.bin").is_file()
        or (output_dir / "model.safetensors.index.json").is_file()
    )
    checkpoints = [path for path in output_dir.glob("checkpoint-*") if path.is_dir()]

    def step_of(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    latest = max(checkpoints, key=step_of) if checkpoints else None
    if has_weights:
        if latest is None:
            return True
        state_path = latest / "trainer_state.json"
        if not state_path.is_file():
            return True
        step = json.loads(state_path.read_text(encoding="utf-8")).get("global_step")
        return step is None or int(step) >= int(max_steps)
    if latest is None:
        return False
    state_path = latest / "trainer_state.json"
    if not state_path.is_file():
        return False
    step = json.loads(state_path.read_text(encoding="utf-8")).get("global_step")
    return step is not None and int(step) >= int(max_steps)


def _spawn_qwen_finetune(cfg: FinetuneConfig) -> str:
    """Run one Qwen3-ASR fine-tune in a fresh interpreter.

    A finished Trainer leaves Accelerator / CUDA state in-process. The next
    level then dies inside ``create_optimizer`` before the first step.
    """
    fd, path = tempfile.mkstemp(prefix="qwen-finetune-", suffix=".pkl")
    os.close(fd)
    cfg_path = Path(path)
    cfg_path.write_bytes(pickle.dumps(cfg))
    env = os.environ.copy()
    env["L2_QWEN_FINETUNE_WORKER"] = "1"
    env["L2_QWEN_FINETUNE_CFG"] = str(cfg_path)
    env["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    logger.info("Qwen3-ASR fine-tune in a fresh process -> %s", cfg.output_dir)
    try:
        try:
            subprocess.run(
                [sys.executable, "-c", "from src.finetune import _run_qwen_finetune_worker; _run_qwen_finetune_worker()"],
                check=True,
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            # CUDA teardown can SIGSEGV after safetensors has already been written.
            if exc.returncode == -11 and _qwen_training_ready(Path(cfg.output_dir), cfg.max_steps):
                logger.warning(
                    "Qwen worker SIGSEGV after a completed checkpoint at %s; continuing",
                    cfg.output_dir,
                )
            else:
                raise
    finally:
        cfg_path.unlink(missing_ok=True)
    return str(cfg.output_dir)


def _run_qwen_finetune_worker() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    cfg = pickle.loads(Path(os.environ["L2_QWEN_FINETUNE_CFG"]).read_bytes())
    _finetune_qwen3_asr(cfg)
    # Skip interpreter and CUDA destructor SIGSEGV after a successful save.
    logging.shutdown()
    os._exit(0)


def _finetune_qwen3_asr(cfg: FinetuneConfig) -> str:
    if os.environ.get("L2_QWEN_FINETUNE_WORKER") != "1":
        return _spawn_qwen_finetune(cfg)
    if cfg.dataloader_num_workers:
        logger.info(
            "Qwen3-ASR fine-tune forces dataloader_num_workers=0 (was %d) so CUDA shutdown does not SIGSEGV",
            cfg.dataloader_num_workers,
        )
        cfg.dataloader_num_workers = 0
    if not cfg.train_csv:
        raise ValueError("train_csv is required")
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Fine-tuning Qwen3-ASR %s -> %s (text_column=%s)", cfg.model_name_or_path, output_dir, cfg.text_column)

    processor = load_processor(cfg.model_name_or_path)
    model = load_asr_model(cfg.model_name_or_path)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if cfg.gradient_checkpointing:
        logger.warning(
            "Ignoring gradient_checkpointing for Qwen3-ASR: audio-encoder cu_seqlens "
            "crashes under both reentrant and non-reentrant checkpointing."
        )
    _disable_qwen3_gradient_checkpointing(model)
    model = _maybe_apply_lora(model, cfg, is_qwen=True)
    _stabilize_qwen3_module_walks(model)

    train_dataset = L2SpeechDataset(
        cfg.train_csv,
        processor=processor,
        text_column=cfg.text_column,
        max_label_tokens=512,
        skipped_rows_path=str(output_dir / "skipped_train_rows.csv"),
        language=cfg.language,
    )
    eval_dataset = None
    if cfg.eval_csv:
        eval_dataset = L2SpeechDataset(
            cfg.eval_csv,
            processor=processor,
            text_column="text",
            max_label_tokens=512,
            skipped_rows_path=str(output_dir / "skipped_eval_rows.csv"),
            language=cfg.language,
        )

    collator = DataCollatorQwen3ASR(processor=processor, language=cfg.language)
    eval_kwargs = _training_arg_extras(TrainingArguments, eval_dataset is not None, cfg.dataloader_num_workers)
    precision = _precision_flags(cfg)
    try:
        import inspect

        ta_params = inspect.signature(TrainingArguments.__init__).parameters
        if "gradient_checkpointing" in ta_params:
            eval_kwargs["gradient_checkpointing"] = False
        if "gradient_checkpointing_kwargs" in ta_params:
            eval_kwargs.pop("gradient_checkpointing_kwargs", None)
        if "max_grad_norm" in ta_params:
            eval_kwargs.setdefault("max_grad_norm", 1.0)
    except (TypeError, ValueError):
        pass
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=int(
            cfg.per_device_eval_batch_size
            if cfg.per_device_eval_batch_size is not None
            else cfg.per_device_train_batch_size
        ),
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_steps=cfg.warmup_steps,
        max_steps=cfg.max_steps,
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        eval_steps=cfg.eval_steps if eval_dataset is not None else None,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_num_workers,
        seed=cfg.seed,
        load_best_model_at_end=False,
        greater_is_better=False,
        push_to_hub=False,
        disable_tqdm=hf_trainer_tqdm_disabled(),
        **precision,
        **eval_kwargs,
    )

    trainer_kwargs = dict(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    _attach_processor(trainer_kwargs, _Qwen3AsrTrainer, processor, feature_extractor=False)
    trainer = _Qwen3AsrTrainer(**trainer_kwargs)

    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)
    _save_trained_model(trainer, processor, cfg, is_qwen=True)
    return str(output_dir)


def finetune_from_dict(cfg_dict: Dict[str, Any]) -> str:
    known = {f.name for f in FinetuneConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in cfg_dict.items() if k in known}
    return finetune_whisper(FinetuneConfig(**filtered))
