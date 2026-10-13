#!/usr/bin/env python3
"""Fine-tune Whisper-base or Qwen3-ASR on SANDI train-asr (official SpWER later)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.finetune import finetune_from_dict  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/baseline/sandi_full_ft_spwer.yaml"))
    parser.add_argument("--resume_from_checkpoint", default="")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    payload = {
        "model_name_or_path": cfg["student_model"],
        "train_csv": str(PROJECT_ROOT / cfg["train_csv"]),
        "output_dir": str(PROJECT_ROOT / cfg["output_dir"]),
        "eval_csv": str(PROJECT_ROOT / cfg["eval_csv"]) if cfg.get("eval_csv") else None,
        "language": cfg.get("language", "en"),
        "task": cfg.get("task", "transcribe"),
        "learning_rate": float(cfg.get("learning_rate", 1e-5)),
        "weight_decay": float(cfg.get("weight_decay", 0.01)),
        "warmup_steps": int(cfg.get("warmup_steps", 500)),
        "max_steps": int(cfg.get("max_steps", 4000)),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 12)),
        "per_device_eval_batch_size": int(
            cfg.get("per_device_eval_batch_size", cfg.get("eval_batch_size", cfg.get("per_device_train_batch_size", 12)))
        ),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "fp16": bool(cfg.get("fp16", True)),
        "bf16": bool(cfg.get("bf16", False)),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", True)),
        "seed": int(cfg.get("seed", 42)),
        "dataloader_num_workers": int(cfg.get("num_workers", 16)),
        "save_steps": int(cfg.get("save_steps", 250)),
        "eval_steps": int(cfg.get("eval_steps", 250)),
        "save_total_limit": int(cfg.get("save_total_limit", 12)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "load_best_model_at_end": bool(cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": str(cfg.get("metric_for_best_model", "eval_loss")),
        "use_lora": bool(cfg.get("use_lora", False)),
        "lora_r": int(cfg.get("lora_r", 16)),
        "lora_alpha": int(cfg.get("lora_alpha", 32)),
        "lora_dropout": float(cfg.get("lora_dropout", 0.05)),
        "lora_bias": str(cfg.get("lora_bias", "none")),
        "merge_lora_on_save": bool(cfg.get("merge_lora_on_save", True)),
    }
    if cfg.get("lora_target_modules"):
        payload["lora_target_modules"] = list(cfg["lora_target_modules"])
    if args.resume_from_checkpoint:
        payload["resume_from_checkpoint"] = args.resume_from_checkpoint
    print(f"Fine-tune -> {payload['output_dir']}", flush=True)
    finetune_from_dict(payload)


if __name__ == "__main__":
    main()
