"""LoRA wrap / merge helpers for Whisper and Qwen3-ASR."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_WHISPER_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
DEFAULT_QWEN_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc1",
    "fc2",
    "linear_1",
    "linear_2",
]


def default_target_modules(is_qwen: bool) -> list:
    return list(DEFAULT_QWEN_TARGETS if is_qwen else DEFAULT_WHISPER_TARGETS)


def apply_lora(
    model,
    *,
    is_qwen: bool,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
    bias: str = "none",
):
    """Wrap ``model`` with PEFT LoRA. Trainable params stay the adapter tensors."""
    from peft import LoraConfig, get_peft_model

    targets = list(target_modules) if target_modules else default_target_modules(is_qwen)
    # Do not set task_type. SEQ_2_SEQ_LM / CAUSAL_LM wrap PeftModelFor* classes
    # that re-bind input_ids and crash WhisperDecoder / Qwen3-ASR.
    lora_cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=targets,
        bias=str(bias or "none"),
    )
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA r=%s alpha=%s dropout=%s bias=%s targets=%s trainable=%.2fM / %.2fM (%.3f%%)",
        r,
        alpha,
        dropout,
        bias,
        targets,
        trainable / 1e6,
        total / 1e6,
        100.0 * trainable / max(total, 1),
    )
    print_fn = getattr(model, "print_trainable_parameters", None)
    if callable(print_fn):
        print_fn()
    return model


def merge_lora_and_save(model, processor, output_dir: Path, extra_meta: Optional[dict] = None) -> Any:
    """Merge LoRA into base weights so later decode can ``from_pretrained`` the dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not hasattr(model, "merge_and_unload"):
        model.save_pretrained(str(output_dir))
        processor.save_pretrained(str(output_dir))
        return model
    logger.info("Merging LoRA adapters into base weights -> %s", output_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
        leftover = output_dir / name
        if leftover.is_file():
            leftover.unlink()
    meta = {"merged": True}
    if extra_meta:
        meta.update(extra_meta)
    (output_dir / "lora_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return merged
