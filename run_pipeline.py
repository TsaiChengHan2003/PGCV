#!/usr/bin/env python3
"""End-to-end Pseudo2Real pipeline for L2 ASR.

Stages
------
See config-dependent stage titles at runtime.

Example
-------
python run_pipeline.py --config configs/teemi_config.yaml --stage all
python run_pipeline.py --config configs/l2arctic_config.yaml --stage 5,6,7
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
import yaml
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.asr_backend import load_asr_model, load_processor as load_asr_processor  # noqa: E402
from src.audio_utils import ensure_device  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_eval_dataloader,
    evaluate_model,
    write_eval_detailed_json,
    write_eval_summary_md,
)
from src.finetune import FinetuneConfig, finetune_whisper  # noqa: E402
from src.progress import configure_experiment_progress, progress_bar, stage_progress  # noqa: E402
from src.pseudo_label import generate_pseudo_labels  # noqa: E402
from src.task_vector import (  # noqa: E402
    apply_task_vector,
    calculate_task_vector,
    grid_search_best_lambda,
    load_tau,
)
from src.verbatim_diag import PROFICIENCY_ORDER  # noqa: E402

logger = logging.getLogger("run_pipeline")


def is_main_process() -> bool:
    return True


def resolve_runtime_device(device: str) -> str:
    return device


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config: {path}")
    return cfg


def resolve_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p


def parse_stages(stage_arg: str, available: Optional[Set[int]] = None) -> Set[int]:
    available = set(available or {1, 2, 3, 4, 5, 6, 7})
    stage_arg = stage_arg.strip().lower()
    if stage_arg in {"all", "*"}:
        return available
    stages: Set[int] = set()
    for part in stage_arg.split(","):
        part = part.strip()
        if not part:
            continue
        stages.add(int(part))
    invalid = stages - available
    if invalid:
        raise ValueError(f"Unsupported stage(s) {sorted(invalid)}; available={sorted(available)}")
    return stages


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    checkpoints: List[Path] = []
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir():
            try:
                int(path.name.split("-")[-1])
            except ValueError:
                continue
            checkpoints.append(path)
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))


def _checkpoint_global_step(checkpoint_dir: Path) -> Optional[int]:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        return None
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    step = state.get("global_step")
    return int(step) if step is not None else None


def _training_completed(output_dir: Path, max_steps: int) -> bool:
    latest = _latest_checkpoint(output_dir)
    if latest is None:
        return False
    step = _checkpoint_global_step(latest)
    return step is not None and step >= max_steps


def get_experiment_type(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("experiment_type", "pseudo2real_standard"))


def get_display_name(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("display_name", cfg.get("experiment_name", "experiment")))


def get_prompt_transfer_split_groups(cfg: Dict[str, Any]) -> Dict[str, List[Tuple[str, str]]]:
    raw = cfg.get("prompt_transfer_split_groups")
    if raw is None:
        return {
            "dev": [("dev", "Overall"), ("dev_a02", "A02"), ("dev_b02", "B02"), ("dev_c02", "C02")],
            "test": [("test", "Overall"), ("test_a02", "A02"), ("test_b02", "B02"), ("test_c02", "C02")],
        }
    groups: Dict[str, List[Tuple[str, str]]] = {}
    for section_name, items in raw.items():
        pairs: List[Tuple[str, str]] = []
        for item in items:
            if isinstance(item, dict):
                key = str(item["key"])
                label = str(item.get("label", key))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                key = str(item[0])
                label = str(item[1])
            else:
                raise ValueError(f"Unsupported split group entry for {section_name}: {item!r}")
            pairs.append((key, label))
        groups[str(section_name)] = pairs
    return groups


def get_prompt_transfer_eval_splits(cfg: Dict[str, Any], section_name: str) -> List[str]:
    return [split_key for split_key, _ in get_prompt_transfer_split_groups(cfg).get(section_name, [])]


def get_available_stages(cfg: Dict[str, Any]) -> Set[int]:
    exp_type = get_experiment_type(cfg)
    if exp_type == "supervised_prompt_transfer":
        return {1, 2, 3, 4}
    if exp_type in {"pseudo2real_standard", "pseudo2real_strong_teacher"}:
        return {1, 2, 3, 4, 5, 6, 7}
    if exp_type == "supervised_target_real":
        return {1, 6, 7}
    raise ValueError(f"Unsupported experiment_type={exp_type}")


def get_stage_titles(cfg: Dict[str, Any]) -> Dict[int, str]:
    exp_type = get_experiment_type(cfg)
    if exp_type == "supervised_prompt_transfer":
        return {
            1: "Fine-tune the student model on train.csv with dev.csv validation",
            2: "Evaluate pretrained and fine-tuned models on dev splits",
            3: "Evaluate pretrained and fine-tuned models on test splits",
            4: "Write combined baseline report",
        }
    if exp_type == "pseudo2real_standard":
        return {
            1: "Generate base-teacher pseudo-labels (source/target train)",
            2: "Fine-tune theta_s_real on source GT",
            3: "Fine-tune theta_s_pseudo / theta_t_pseudo",
            4: "Compute correction vector tau",
            5: "Grid-search best_lambda on source_dev",
            6: "Apply tau and evaluate on target_test",
            7: "Write evaluation summaries",
        }
    if exp_type == "pseudo2real_strong_teacher":
        return {
            1: "Fine-tune theta_s_real strong teacher on source GT",
            2: "Generate pseudo-labels with theta_s_real",
            3: "Fine-tune theta_s_pseudo / theta_t_pseudo",
            4: "Compute correction vector tau",
            5: "Grid-search best_lambda on source_dev",
            6: "Apply tau and evaluate on target_test",
            7: "Write evaluation summaries",
        }
    if exp_type == "supervised_target_real":
        return {
            1: "Fine-tune theta_t_real on target GT",
            6: "Evaluate pretrained and theta_t_real on target_test",
            7: "Write evaluation summaries",
        }
    raise ValueError(f"Unsupported experiment_type={exp_type}")


def log_experiment_identity(cfg: Dict[str, Any], stages: Set[int]) -> None:
    logger.info("Experiment name: %s", get_display_name(cfg))
    logger.info("Experiment type: %s", get_experiment_type(cfg))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logger.warning(
            "Single-process launch sees %d visible GPUs. To force true single-GPU execution, "
            "set CUDA_VISIBLE_DEVICES to one GPU (for example via run_single_gpu.sh).",
            torch.cuda.device_count(),
        )
    cfg_path = cfg.get("__config_path__")
    if cfg_path:
        logger.info("Config path: %s", cfg_path)
    logger.info("Selected stages: %s", sorted(stages))
    logger.info("Data dir: %s", resolve_path(cfg["data_dir"]))
    logger.info("Checkpoint dir: %s", resolve_path(cfg["checkpoint_dir"]))
    if "vector_path" in cfg:
        logger.info("Vector path: %s", resolve_path(cfg["vector_path"]))
    logger.info("Results dir: %s", resolve_path(cfg["results_dir"]))


def _stage5_output_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    results_dir = resolve_path(cfg["results_dir"])
    ckpts = cfg["checkpoints"]
    exp_type = get_experiment_type(cfg)
    if exp_type == "supervised_target_real":
        corrected_dir = resolve_path(ckpts["target_real"])
        variants = {
            "theta_pre": cfg["student_model"],
            "theta_t_real": str(resolve_path(ckpts["target_real"])),
        }
    else:
        corrected_dir = resolve_path(
            ckpts.get("target_corrected", Path(ckpts["target_pseudo"]).parent / "whisper_target_corrected")
        )
        variants = {
            "theta_pre": cfg["student_model"],
            "theta_s_real": str(resolve_path(ckpts["source_real"])),
            "theta_t_pseudo": str(resolve_path(ckpts["target_pseudo"])),
            "theta_t_corrected": str(corrected_dir),
        }
    return {
        "corrected_dir": corrected_dir,
        "summary_path": results_dir / "stage5_summary_tmp.json",
        "comparison_path": results_dir / "predictions_comparison.csv",
        "prediction_paths": [results_dir / f"predictions_{name}.csv" for name in variants],
        "variants": variants,
    }


def _stage5_outputs_exist(cfg: Dict[str, Any]) -> bool:
    paths = _stage5_output_paths(cfg)
    return (
        paths["corrected_dir"].is_dir()
        and paths["summary_path"].is_file()
        and paths["comparison_path"].is_file()
        and all(path.is_file() for path in paths["prediction_paths"])
    )


def _stage6_outputs_exist(cfg: Dict[str, Any]) -> bool:
    results_dir = resolve_path(cfg["results_dir"])
    return (
        (results_dir / "target_eval_detailed.json").is_file()
        and (results_dir / "target_eval_summary.md").is_file()
    )


def _log_training_action(name: str, output_dir: Path, max_steps: int) -> None:
    latest_ckpt = _latest_checkpoint(output_dir)
    if _training_completed(output_dir, max_steps=max_steps):
        logger.info("  [skip] %s already completed at %s", name, latest_ckpt)
    elif latest_ckpt is not None:
        logger.info("  [resume] %s from %s", name, latest_ckpt)
    else:
        logger.info("  [run] %s from scratch", name)


def log_execution_plan(cfg: Dict[str, Any], stages: Set[int]) -> None:
    exp_type = get_experiment_type(cfg)
    splits = cfg.get("splits", {})
    ckpts = cfg.get("checkpoints", {})
    max_steps = int(cfg.get("max_steps", 4000))
    stage_titles = get_stage_titles(cfg)

    logger.info("Execution plan:")
    for stage_id in sorted(stages):
        logger.info("  Stage %d: %s", stage_id, stage_titles[stage_id])

    if exp_type == "supervised_prompt_transfer":
        if 1 in stages:
            _log_training_action("theta_baseline", resolve_path(ckpts["baseline"]), max_steps)
        if 2 in stages:
            logger.info("    Stage 2 [run]: dev eval -> %s", resolve_path(cfg["results_dir"]) / "dev_eval_summary.md")
        if 3 in stages:
            logger.info("    Stage 3 [run]: test eval -> %s", resolve_path(cfg["results_dir"]) / "test_eval_summary.md")
        if 4 in stages:
            logger.info("    Stage 4 [run]: report -> %s", resolve_path(cfg["results_dir"]) / "experiment_report.md")
        return

    if exp_type == "pseudo2real_standard":
        if 1 in stages:
            for src_key, dst_key in [("source_train", "source_train_pseudo"), ("target_train", "target_train_pseudo")]:
                out_csv = resolve_path(splits[dst_key])
                action = "skip" if out_csv.is_file() else "run"
                logger.info("    Stage 1 [%s]: %s -> %s", action, src_key, out_csv)
        if 2 in stages:
            _log_training_action("theta_s_real", resolve_path(ckpts["source_real"]), max_steps)
        if 3 in stages:
            _log_training_action("theta_s_pseudo", resolve_path(ckpts["source_pseudo"]), max_steps)
            _log_training_action("theta_t_pseudo", resolve_path(ckpts["target_pseudo"]), max_steps)
    elif exp_type == "pseudo2real_strong_teacher":
        if 1 in stages:
            _log_training_action("theta_s_real", resolve_path(ckpts["source_real"]), max_steps)
        if 2 in stages:
            for src_key, dst_key in [("source_train", "source_train_pseudo"), ("target_train", "target_train_pseudo")]:
                out_csv = resolve_path(splits[dst_key])
                action = "skip" if out_csv.is_file() else "run"
                logger.info("    Stage 2 [%s]: %s -> %s", action, src_key, out_csv)
        if 3 in stages:
            _log_training_action("theta_s_pseudo", resolve_path(ckpts["source_pseudo"]), max_steps)
            _log_training_action("theta_t_pseudo", resolve_path(ckpts["target_pseudo"]), max_steps)
    elif exp_type == "supervised_target_real":
        if 1 in stages:
            _log_training_action("theta_t_real", resolve_path(ckpts["target_real"]), max_steps)

    if 4 in stages and "vector_path" in cfg:
        vector_path = resolve_path(cfg["vector_path"])
        action = "skip" if vector_path.is_file() else "run"
        logger.info("    Stage 4 [%s]: tau -> %s", action, vector_path)

    if 5 in stages:
        best_path = resolve_path(cfg["results_dir"]) / "best_lambda.json"
        action = "skip" if best_path.is_file() else "run"
        logger.info("    Stage 5 [%s]: best_lambda -> %s", action, best_path)

    if 6 in stages:
        paths = _stage5_output_paths(cfg)
        action = "skip" if _stage5_outputs_exist(cfg) else "run"
        logger.info("    Stage 6 [%s]: eval outputs -> %s", action, paths["summary_path"])

    if 7 in stages:
        summary_md = resolve_path(cfg["results_dir"]) / "target_eval_summary.md"
        action = "skip" if _stage6_outputs_exist(cfg) else "run"
        logger.info("    Stage 7 [%s]: report -> %s", action, summary_md)


def stage1_pseudo_labels(cfg: Dict[str, Any], device: str, teacher_model: Optional[str] = None) -> None:
    splits = cfg["splits"]
    teacher = teacher_model or cfg["teacher_model"]
    batch_size = int(cfg.get("pseudo_batch_size", 8))

    jobs = [
        ("source_train", "source_train_pseudo"),
        ("target_train", "target_train_pseudo"),
    ]
    for src_key, dst_key in jobs:
        in_csv = resolve_path(splits[src_key])
        out_csv = resolve_path(splits[dst_key])
        if not in_csv.is_file():
            raise FileNotFoundError(
                f"Missing {in_csv}. Run scripts/prepare_*.py first."
            )
        if out_csv.is_file():
            logger.info("[Stage 1] Skip %s; pseudo labels already exist at %s", src_key, out_csv)
            continue
        logger.info("[Stage 1] Pseudo-label %s -> %s using teacher=%s", in_csv, out_csv, teacher)
        generate_pseudo_labels(
            model_name=teacher,
            input_csv=str(in_csv),
            output_csv=str(out_csv),
            batch_size=batch_size,
            device=device,
            language=cfg.get("language", "en"),
            task=cfg.get("task", "transcribe"),
            num_workers=int(cfg.get("num_workers", 0)),
        )


def _ft_common(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(
        model_name_or_path=cfg["student_model"],
        language=cfg.get("language", "en"),
        task=cfg.get("task", "transcribe"),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        warmup_steps=int(cfg.get("warmup_steps", 500)),
        max_steps=int(cfg.get("max_steps", 4000)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(cfg.get("eval_batch_size", cfg.get("per_device_train_batch_size", 8))),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 2)),
        fp16=bool(cfg.get("fp16", True)),
        bf16=bool(cfg.get("bf16", False)),
        seed=int(cfg.get("seed", 42)),
        dataloader_num_workers=int(cfg.get("num_workers", 4)),
        use_lora=bool(cfg.get("use_lora", False)),
        lora_r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        lora_target_modules=list(cfg["lora_target_modules"]) if cfg.get("lora_target_modules") else None,
        lora_bias=str(cfg.get("lora_bias", "none")),
        merge_lora_on_save=bool(cfg.get("merge_lora_on_save", True)),
    )


def _run_finetune_jobs(cfg: Dict[str, Any], jobs: List[Dict[str, Any]], desc: str) -> None:
    common = _ft_common(cfg)
    max_steps = int(common["max_steps"])
    for job in progress_bar(jobs, total=len(jobs), desc=desc, unit="model"):
        logger.info("Fine-tune %s", job["name"])
        if not job["train_csv"].is_file():
            raise FileNotFoundError(job["train_csv"])
        latest_ckpt = _latest_checkpoint(job["output_dir"])
        if _training_completed(job["output_dir"], max_steps=max_steps):
            logger.info(
                "Skip %s; found completed checkpoint at %s",
                job["name"],
                latest_ckpt,
            )
            continue
        ft_cfg = FinetuneConfig(
            **common,
            train_csv=str(job["train_csv"]),
            text_column=job["text_column"],
            output_dir=str(job["output_dir"]),
            eval_csv=str(job["eval_csv"]) if job["eval_csv"] else None,
            resume_from_checkpoint=str(latest_ckpt) if latest_ckpt else None,
        )
        if latest_ckpt is not None:
            logger.info("Resume %s from %s", job["name"], latest_ckpt)
        finetune_whisper(ft_cfg)


def stage2_train_source_real(cfg: Dict[str, Any]) -> None:
    splits = cfg["splits"]
    ckpts = cfg["checkpoints"]
    jobs = [
        {
            "name": "theta_s_real",
            "train_csv": resolve_path(splits["source_train"]),
            "text_column": "text",
            "output_dir": resolve_path(ckpts["source_real"]),
            "eval_csv": resolve_path(splits["source_dev"]),
        }
    ]
    _run_finetune_jobs(cfg, jobs, desc="fine-tune source_real")


def stage3_train_pseudo_students(cfg: Dict[str, Any]) -> None:
    splits = cfg["splits"]
    ckpts = cfg["checkpoints"]
    jobs = [
        {
            "name": "theta_s_pseudo",
            "train_csv": resolve_path(splits["source_train_pseudo"]),
            "text_column": "pseudo_text",
            "output_dir": resolve_path(ckpts["source_pseudo"]),
            "eval_csv": resolve_path(splits["source_dev"]),
        },
        {
            "name": "theta_t_pseudo",
            "train_csv": resolve_path(splits["target_train_pseudo"]),
            "text_column": "pseudo_text",
            "output_dir": resolve_path(ckpts["target_pseudo"]),
            "eval_csv": None,
        },
    ]
    _run_finetune_jobs(cfg, jobs, desc="fine-tune pseudo students")


def stage1_train_target_real(cfg: Dict[str, Any]) -> None:
    splits = cfg["splits"]
    ckpts = cfg["checkpoints"]
    jobs = [
        {
            "name": "theta_t_real",
            "train_csv": resolve_path(splits["target_train"]),
            "text_column": "text",
            "output_dir": resolve_path(ckpts["target_real"]),
            "eval_csv": resolve_path(splits["target_dev"]),
        }
    ]
    _run_finetune_jobs(cfg, jobs, desc="fine-tune target_real")


def stage1_train_prompt_transfer(cfg: Dict[str, Any]) -> None:
    splits = cfg["splits"]
    ckpts = cfg["checkpoints"]
    jobs = [
        {
            "name": "theta_baseline",
            "train_csv": resolve_path(splits["train"]),
            "text_column": "text",
            "output_dir": resolve_path(ckpts["baseline"]),
            "eval_csv": resolve_path(splits["dev"]),
        }
    ]
    _run_finetune_jobs(cfg, jobs, desc="fine-tune baseline")


def _prompt_transfer_variants(cfg: Dict[str, Any]) -> Dict[str, str]:
    ckpts = cfg["checkpoints"]
    return {
        "theta_pre": cfg["student_model"],
        "theta_ft": str(resolve_path(ckpts["baseline"])),
    }


def _write_prompt_transfer_split_outputs(
    summary: Dict[str, Dict[str, float]],
    comparison_frames: List[pd.DataFrame],
    results_dir: Path,
    split_label: str,
    display_name: str,
) -> None:
    detailed_path = results_dir / f"{split_label}_eval_detailed.json"
    cmp_path = results_dir / f"predictions_{split_label}_comparison.csv"
    write_eval_detailed_json(summary, str(detailed_path))
    merged = comparison_frames[0]
    for frame in comparison_frames[1:]:
        merged = merged.merge(frame.drop(columns=["reference", "reference_norm"]), on="audio_path", how="outer")
    merged = merged.rename(
        columns={
            "hyp_theta_pre": "pretrained",
            "hyp_theta_ft": "finetuned",
            "reference": "GT",
        }
    )
    merged.to_csv(cmp_path, index=False)
    logger.info("Wrote %s", cmp_path)


def _metrics_from_counts(total_h: float, total_s: float, total_d: float, total_i: float, n_utterances: int) -> Dict[str, float]:
    n_ref_total = total_h + total_s + total_d
    if n_ref_total == 0:
        wer = 0.0 if total_i == 0 else 100.0
        sub = del_ = 0.0
        ins = 100.0 if total_i > 0 else 0.0
    else:
        wer = 100.0 * (total_s + total_d + total_i) / n_ref_total
        sub = 100.0 * total_s / n_ref_total
        del_ = 100.0 * total_d / n_ref_total
        ins = 100.0 * total_i / n_ref_total
    return {
        "wer": float(wer),
        "sub": float(sub),
        "ins": float(ins),
        "del": float(del_),
        "hits": float(total_h),
        "substitutions": float(total_s),
        "deletions": float(total_d),
        "insertions": float(total_i),
        "n_ref_words": float(n_ref_total),
        "n_utterances": float(n_utterances),
    }


def _grouped_metrics_from_prediction_csv(prediction_csv: Path, metadata_csv: Path, group_col: str) -> Dict[str, Dict[str, float]]:
    pred_df = pd.read_csv(prediction_csv)
    meta_df = pd.read_csv(metadata_csv)
    if group_col not in meta_df.columns:
        return {}
    meta_df = meta_df[["audio_path", group_col]].drop_duplicates()
    merged = pred_df.merge(meta_df, on="audio_path", how="left")
    group_values = merged[group_col].fillna("").astype(str).str.strip()
    valid_mask = merged[group_col].notna() & group_values.ne("") & group_values.ne("No answer")
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        logger.warning(
            "Dropping %d rows without valid %s from grouped report for %s",
            dropped,
            group_col,
            prediction_csv.name,
        )
    merged = merged.loc[valid_mask].copy()
    if merged.empty:
        return {}
    merged[group_col] = merged[group_col].astype(str)
    grouped: Dict[str, Dict[str, float]] = {}
    for group_name, sub in merged.groupby(group_col):
        grouped[str(group_name)] = _metrics_from_counts(
            total_h=float(sub["hits"].sum()),
            total_s=float(sub["substitutions"].sum()),
            total_d=float(sub["deletions"].sum()),
            total_i=float(sub["insertions"].sum()),
            n_utterances=int(len(sub)),
        )
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _write_grouped_metrics_md(
    grouped_summary: Dict[str, Dict[str, Dict[str, float]]],
    overall_summary: Dict[str, Dict[str, float]],
    output_path: Path,
    title: str,
    group_label: str,
) -> None:
    lines = [
        f"# {title}",
        "",
        "## Overall Average",
        "",
        "| Model | WER (%) | Sub (%) | Ins (%) | Del (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name, metrics in overall_summary.items():
        lines.append(
            f"| {model_name} | {metrics.get('wer', float('nan')):.2f} | "
            f"{metrics.get('sub', float('nan')):.2f} | "
            f"{metrics.get('ins', float('nan')):.2f} | "
            f"{metrics.get('del', float('nan')):.2f} |"
        )
    for model_name, groups in grouped_summary.items():
        lines.extend(
            [
                "",
                f"## {model_name}",
                "",
                f"| {group_label} | WER (%) | Sub (%) | Ins (%) | Del (%) | Utterances |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group_name, metrics in groups.items():
            lines.append(
                f"| {group_name} | {metrics.get('wer', float('nan')):.2f} | "
                f"{metrics.get('sub', float('nan')):.2f} | "
                f"{metrics.get('ins', float('nan')):.2f} | "
                f"{metrics.get('del', float('nan')):.2f} | "
                f"{int(metrics.get('n_utterances', 0))} |"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output_path)


def _write_prompt_transfer_grouped_outputs(
    summary: Dict[str, Dict[str, float]],
    results_dir: Path,
    split_label: str,
    metadata_csv: Path,
    group_col: str,
    display_name: str,
) -> None:
    grouped_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_name in summary:
        pred_path = results_dir / f"predictions_{split_label}_{model_name}.csv"
        grouped_summary[model_name] = _grouped_metrics_from_prediction_csv(pred_path, metadata_csv, group_col)
    json_path = results_dir / f"{split_label}_by_{group_col}.json"
    json_path.write_text(json.dumps(grouped_summary, indent=2), encoding="utf-8")


def _evaluate_prompt_transfer_split(cfg: Dict[str, Any], split_key: str, device: str) -> Dict[str, Dict[str, float]]:
    results_dir = resolve_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    split_csv = resolve_path(cfg["splits"][split_key])
    detailed_path = results_dir / f"{split_key}_eval_detailed.json"
    cmp_path = results_dir / f"predictions_{split_key}_comparison.csv"
    grouped_path = results_dir / f"{split_key}_by_proficiency_level.json"
    variants = _prompt_transfer_variants(cfg)
    prediction_paths = [results_dir / f"predictions_{split_key}_{name}.csv" for name in variants]
    if (
        detailed_path.is_file()
        and cmp_path.is_file()
        and grouped_path.is_file()
        and all(path.is_file() for path in prediction_paths)
    ):
        logger.info("[Eval %s] Skip; found cached detailed/grouped outputs", split_key)
        with open(detailed_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if all(path.is_file() for path in prediction_paths):
        logger.info("[Eval %s] Rebuild summary from cached prediction CSVs (skip decode)", split_key)
        summary: Dict[str, Dict[str, float]] = {}
        pred_frames: List[pd.DataFrame] = []
        for name in variants:
            out_csv = results_dir / f"predictions_{split_key}_{name}.csv"
            df = pd.read_csv(out_csv)
            summary[name] = _metrics_from_counts(
                total_h=float(df["hits"].sum()),
                total_s=float(df["substitutions"].sum()),
                total_d=float(df["deletions"].sum()),
                total_i=float(df["insertions"].sum()),
                n_utterances=int(len(df)),
            )
            renamed = df.rename(
                columns={
                    "hypothesis": f"hyp_{name}",
                    "hypothesis_norm": f"hyp_norm_{name}",
                }
            )
            pred_frames.append(
                renamed[["audio_path", "reference", "reference_norm", f"hyp_{name}", f"hyp_norm_{name}"]]
            )
        _write_prompt_transfer_split_outputs(summary, pred_frames, results_dir, split_key, get_display_name(cfg))
        _write_prompt_transfer_grouped_outputs(
            summary=summary,
            results_dir=results_dir,
            split_label=split_key,
            metadata_csv=split_csv,
            group_col="proficiency_level",
            display_name=get_display_name(cfg),
        )
        return summary

    batch_size = int(cfg.get("eval_batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))
    lang = cfg.get("language", "en")
    task = cfg.get("task", "transcribe")
    device_t = ensure_device(device)
    summary: Dict[str, Dict[str, float]] = {}
    pred_frames: List[pd.DataFrame] = []

    for name, model_path in progress_bar(
        list(variants.items()),
        total=len(variants),
        desc=f"eval {split_key}",
        unit="model",
    ):
        logger.info("[Eval %s] Evaluate %s", split_key, name)
        model = load_asr_model(model_path, device=device_t)
        processor = load_asr_processor(model_path)
        loader = build_eval_dataloader(
            str(split_csv),
            processor=processor,
            batch_size=batch_size,
            num_workers=num_workers,
            language=lang,
        )
        out_csv = results_dir / f"predictions_{split_key}_{name}.csv"
        metrics = evaluate_model(
            model,
            loader,
            processor,
            output_csv_path=str(out_csv),
            language=lang,
            task=task,
            desc=f"{split_key}:{name}",
            max_new_tokens=int(cfg.get("max_new_tokens", 256)),
        )
        summary[name] = metrics
        df = pd.read_csv(out_csv)
        df = df.rename(
            columns={
                "hypothesis": f"hyp_{name}",
                "hypothesis_norm": f"hyp_norm_{name}",
            }
        )
        pred_frames.append(df[["audio_path", "reference", "reference_norm", f"hyp_{name}", f"hyp_norm_{name}"]])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_prompt_transfer_split_outputs(summary, pred_frames, results_dir, split_key, get_display_name(cfg))
    _write_prompt_transfer_grouped_outputs(
        summary=summary,
        results_dir=results_dir,
        split_label=split_key,
        metadata_csv=split_csv,
        group_col="proficiency_level",
        display_name=get_display_name(cfg),
    )
    return summary


def stage2_eval_prompt_transfer_dev(cfg: Dict[str, Any], device: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    outputs: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split_key in get_prompt_transfer_eval_splits(cfg, "dev"):
        outputs[split_key] = _evaluate_prompt_transfer_split(cfg, split_key, device=device)
    return outputs


def stage3_eval_prompt_transfer_test(cfg: Dict[str, Any], device: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    outputs: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split_key in get_prompt_transfer_eval_splits(cfg, "test"):
        outputs[split_key] = _evaluate_prompt_transfer_split(cfg, split_key, device=device)
    return outputs


def stage4_prompt_transfer_report(cfg: Dict[str, Any]) -> None:
    results_dir = resolve_path(cfg["results_dir"])
    report_path = results_dir / "experiment_report.md"
    for stale_path in sorted(results_dir.glob("*_eval_summary.md")):
        stale_path.unlink(missing_ok=True)
    for stale_path in sorted(results_dir.glob("*_by_proficiency_level.md")):
        stale_path.unlink(missing_ok=True)

    split_groups = get_prompt_transfer_split_groups(cfg)
    proficiency_order = {**PROFICIENCY_ORDER, "No answer": 99}

    def _load_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_proficiency_table(split_key: str) -> List[str]:
        lines = [
            "| Proficiency | Pretrained WER | Finetuned WER | Gain | Pretrained Sub | Pretrained Ins | Pretrained Del | Finetuned Sub | Finetuned Ins | Finetuned Del | Utterances |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        grouped_path = results_dir / f"{split_key}_by_proficiency_level.json"
        detailed_path = results_dir / f"{split_key}_eval_detailed.json"
        if not grouped_path.is_file() or not detailed_path.is_file():
            return lines
        grouped = _load_json(grouped_path)
        overall = _load_json(detailed_path)
        pre_groups = grouped.get("theta_pre", {})
        ft_groups = grouped.get("theta_ft", {})
        prof_levels = sorted(
            set(pre_groups) | set(ft_groups),
            key=lambda level: (proficiency_order.get(level, 999), level),
        )
        kept_utterances = int(
            sum(int(pre_groups[level].get("n_utterances", 0)) for level in prof_levels)
        )
        total_utterances = int(overall["theta_pre"].get("n_utterances", 0))
        omitted = total_utterances - kept_utterances
        for level in prof_levels:
            pre = pre_groups.get(level, {})
            ft = ft_groups.get(level, {})
            pre_wer = float(pre.get("wer", float("nan")))
            ft_wer = float(ft.get("wer", float("nan")))
            pre_sub = float(pre.get("sub", float("nan")))
            pre_ins = float(pre.get("ins", float("nan")))
            pre_del = float(pre.get("del", float("nan")))
            ft_sub = float(ft.get("sub", float("nan")))
            ft_ins = float(ft.get("ins", float("nan")))
            ft_del = float(ft.get("del", float("nan")))
            n_utt = int(ft.get("n_utterances", pre.get("n_utterances", 0)))
            lines.append(
                f"| {level} | {pre_wer:.2f} | {ft_wer:.2f} | {pre_wer - ft_wer:.2f} | "
                f"{pre_sub:.2f} | {pre_ins:.2f} | {pre_del:.2f} | "
                f"{ft_sub:.2f} | {ft_ins:.2f} | {ft_del:.2f} | {n_utt} |"
            )
        lines.append(
            f"| Overall | {overall['theta_pre']['wer']:.2f} | {overall['theta_ft']['wer']:.2f} | "
            f"{overall['theta_pre']['wer'] - overall['theta_ft']['wer']:.2f} | "
            f"{overall['theta_pre']['sub']:.2f} | {overall['theta_pre']['ins']:.2f} | {overall['theta_pre']['del']:.2f} | "
            f"{overall['theta_ft']['sub']:.2f} | {overall['theta_ft']['ins']:.2f} | {overall['theta_ft']['del']:.2f} | "
            f"{int(overall['theta_ft'].get('n_utterances', 0))} |"
        )
        if omitted > 0:
            lines.append(
                f"| omitted (missing proficiency) | - | - | - | - | - | - | - | - | - | {omitted} |"
            )
        return lines

    scope_lines = list(
        cfg.get(
            "report_scope_lines",
            [
                "- Train booklets: A01 / B01 / C01",
                "- Eval booklets: A02 / B02 / C02",
                "- Split unit: global speaker id",
                "- Human-readable output is consolidated into this single markdown report.",
            ],
        )
    )

    sections: List[str] = [f"# Experiment Report ({get_display_name(cfg)})", "", "## Scope", "", *scope_lines, ""]
    if split_groups.get("dev"):
        sections.extend(["## DEV", ""])
        for split_key, label in split_groups["dev"]:
            sections.extend(
                [
                    f"### {label}",
                    "",
                    *_build_proficiency_table(split_key),
                    "",
                ]
            )
    if split_groups.get("test"):
        sections.extend(["## TEST", ""])
        for split_key, label in split_groups["test"]:
            sections.extend(
                [
                    f"### {label}",
                    "",
                    *_build_proficiency_table(split_key),
                    "",
                ]
            )
    report_path.write_text("\n".join(sections), encoding="utf-8")
    logger.info("Wrote consolidated report %s", report_path)


def stage3_tau(cfg: Dict[str, Any]) -> None:
    ckpts = cfg["checkpoints"]
    vector_path = resolve_path(cfg["vector_path"])
    if vector_path.is_file():
        logger.info("[Stage 3] Skip; tau already exists at %s", vector_path)
        return
    logger.info("[Stage 3] Compute tau -> %s", vector_path)
    calculate_task_vector(
        real_ckpt_dir=str(resolve_path(ckpts["source_real"])),
        pseudo_ckpt_dir=str(resolve_path(ckpts["source_pseudo"])),
        save_path=str(vector_path),
    )


def stage4_lambda_search(cfg: Dict[str, Any], device: str) -> float:
    results_dir = resolve_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpts = cfg["checkpoints"]
    splits = cfg["splits"]
    lambda_list = cfg.get("lambda_list", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    best_json_path = results_dir / "best_lambda.json"

    if best_json_path.is_file():
        with open(best_json_path, "r", encoding="utf-8") as f:
            best = json.load(f)
        best_lambda = float(best["best_lambda"])
        logger.info("[Stage 4] Skip; reusing best_lambda=%.3f from %s", best_lambda, best_json_path)
        return best_lambda

    device_t = ensure_device(device)
    model = WhisperForConditionalGeneration.from_pretrained(
        str(resolve_path(ckpts["target_pseudo"]))
    ).to(device_t)
    processor = WhisperProcessor.from_pretrained(str(resolve_path(ckpts["target_pseudo"])))
    tau = load_tau(str(resolve_path(cfg["vector_path"])))

    dev_loader = build_eval_dataloader(
        str(resolve_path(splits["source_dev"])),
        processor=processor,
        batch_size=int(cfg.get("eval_batch_size", 8)),
        num_workers=int(cfg.get("num_workers", 2)),
    )

    def eval_fn(m, loader):
        return evaluate_model(
            m,
            loader,
            processor,
            output_csv_path=None,
            language=cfg.get("language", "en"),
            task=cfg.get("task", "transcribe"),
            desc="source_dev",
        )

    logger.info("[Stage 4] Lambda grid search on SOURCE DEV only")
    best_lambda, _ = grid_search_best_lambda(
        model=model,
        tau=tau,
        dev_loader=dev_loader,
        eval_fn=eval_fn,
        lambda_list=lambda_list,
        curve_csv_path=str(results_dir / "lambda_search_curve.csv"),
        best_json_path=str(best_json_path),
    )
    return best_lambda


def _load_best_lambda(cfg: Dict[str, Any]) -> float:
    path = resolve_path(cfg["results_dir"]) / "best_lambda.json"
    with open(path, "r", encoding="utf-8") as f:
        return float(json.load(f)["best_lambda"])


def stage5_apply_and_eval(cfg: Dict[str, Any], device: str, best_lambda: Optional[float] = None) -> Dict[str, Dict[str, float]]:
    if best_lambda is None:
        best_lambda = _load_best_lambda(cfg)

    results_dir = resolve_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpts = cfg["checkpoints"]
    splits = cfg["splits"]
    stage5_paths = _stage5_output_paths(cfg)
    corrected_dir = stage5_paths["corrected_dir"]
    variants = stage5_paths["variants"]
    stage5_summary_path = stage5_paths["summary_path"]
    cmp_path = stage5_paths["comparison_path"]
    if _stage5_outputs_exist(cfg):
        logger.info("[Stage 5] Skip; found existing evaluation outputs in %s", results_dir)
        with open(stage5_summary_path, "r", encoding="utf-8") as f:
            return json.load(f)

    device_t = ensure_device(device)
    target_test = str(resolve_path(splits["target_test"]))
    batch_size = int(cfg.get("eval_batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))
    lang = cfg.get("language", "en")
    task = cfg.get("task", "transcribe")
    tau = load_tau(str(resolve_path(cfg["vector_path"])))

    # Build corrected target model and save
    t_pseudo = WhisperForConditionalGeneration.from_pretrained(
        str(resolve_path(ckpts["target_pseudo"]))
    )
    t_corrected = copy.deepcopy(t_pseudo)
    apply_task_vector(t_corrected, tau, best_lambda)
    corrected_dir.mkdir(parents=True, exist_ok=True)
    t_corrected.save_pretrained(str(corrected_dir))
    WhisperProcessor.from_pretrained(str(resolve_path(ckpts["target_pseudo"]))).save_pretrained(
        str(corrected_dir)
    )
    logger.info("[Stage 5] Saved theta_t_corrected (lambda=%.3f) -> %s", best_lambda, corrected_dir)

    summary: Dict[str, Dict[str, float]] = {}
    pred_frames: List[pd.DataFrame] = []

    for name, model_path in progress_bar(
        list(variants.items()),
        total=len(variants),
        desc="eval target_test",
        unit="model",
    ):
        logger.info("[Stage 5] Evaluate %s on target_test", name)
        model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device_t)
        processor = WhisperProcessor.from_pretrained(model_path)
        loader = build_eval_dataloader(
            target_test,
            processor=processor,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        out_csv = results_dir / f"predictions_{name}.csv"
        metrics = evaluate_model(
            model,
            loader,
            processor,
            output_csv_path=str(out_csv),
            language=lang,
            task=task,
            desc=name,
        )
        summary[name] = metrics
        df = pd.read_csv(out_csv)
        df = df.rename(
            columns={
                "hypothesis": f"hyp_{name}",
                "hypothesis_norm": f"hyp_norm_{name}",
            }
        )
        keep = ["audio_path", "reference", "reference_norm", f"hyp_{name}", f"hyp_norm_{name}"]
        pred_frames.append(df[keep])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Merge sentence-level comparison
    merged = pred_frames[0]
    for frame in pred_frames[1:]:
        merged = merged.merge(frame.drop(columns=["reference", "reference_norm"]), on="audio_path", how="outer")
    # Friendly aliases for paper table
    rename = {
        "hyp_theta_pre": "pretrained",
        "hyp_theta_t_pseudo": "pseudo",
        "hyp_theta_t_corrected": "corrected",
        "hyp_theta_s_real": "source_real",
        "reference": "GT",
    }
    merged = merged.rename(columns={k: v for k, v in rename.items() if k in merged.columns})
    merged.to_csv(cmp_path, index=False)
    logger.info("Wrote %s", cmp_path)

    with open(stage5_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def stage6_eval_supervised_target_real(cfg: Dict[str, Any], device: str) -> Dict[str, Dict[str, float]]:
    results_dir = resolve_path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    stage6_paths = _stage5_output_paths(cfg)
    if _stage5_outputs_exist(cfg):
        logger.info("[Stage 6] Skip; found existing evaluation outputs in %s", results_dir)
        with open(stage6_paths["summary_path"], "r", encoding="utf-8") as f:
            return json.load(f)

    splits = cfg["splits"]
    target_test = str(resolve_path(splits["target_test"]))
    batch_size = int(cfg.get("eval_batch_size", 8))
    num_workers = int(cfg.get("num_workers", 2))
    lang = cfg.get("language", "en")
    task = cfg.get("task", "transcribe")
    device_t = ensure_device(device)

    summary: Dict[str, Dict[str, float]] = {}
    pred_frames: List[pd.DataFrame] = []
    for name, model_path in progress_bar(
        list(stage6_paths["variants"].items()),
        total=len(stage6_paths["variants"]),
        desc="eval target_test",
        unit="model",
    ):
        logger.info("[Stage 6] Evaluate %s on target_test", name)
        model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device_t)
        processor = WhisperProcessor.from_pretrained(model_path)
        loader = build_eval_dataloader(
            target_test,
            processor=processor,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        out_csv = results_dir / f"predictions_{name}.csv"
        metrics = evaluate_model(
            model,
            loader,
            processor,
            output_csv_path=str(out_csv),
            language=lang,
            task=task,
            desc=name,
        )
        summary[name] = metrics
        df = pd.read_csv(out_csv)
        df = df.rename(
            columns={
                "hypothesis": f"hyp_{name}",
                "hypothesis_norm": f"hyp_norm_{name}",
            }
        )
        keep = ["audio_path", "reference", "reference_norm", f"hyp_{name}", f"hyp_norm_{name}"]
        pred_frames.append(df[keep])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged = pred_frames[0]
    for frame in pred_frames[1:]:
        merged = merged.merge(frame.drop(columns=["reference", "reference_norm"]), on="audio_path", how="outer")
    merged = merged.rename(
        columns={
            "hyp_theta_pre": "pretrained",
            "hyp_theta_t_real": "target_real",
            "reference": "GT",
        }
    )
    merged.to_csv(stage6_paths["comparison_path"], index=False)
    logger.info("Wrote %s", stage6_paths["comparison_path"])

    with open(stage6_paths["summary_path"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def stage6_reports(cfg: Dict[str, Any], summary: Optional[Dict[str, Dict[str, float]]] = None) -> None:
    results_dir = resolve_path(cfg["results_dir"])
    detailed_path = results_dir / "target_eval_detailed.json"
    summary_md_path = results_dir / "target_eval_summary.md"
    if _stage6_outputs_exist(cfg):
        logger.info("[Stage 6] Skip; reports already exist in %s", results_dir)
        return
    if summary is None:
        tmp = results_dir / "stage5_summary_tmp.json"
        src = tmp if tmp.is_file() else detailed_path
        with open(src, "r", encoding="utf-8") as f:
            summary = json.load(f)

    write_eval_detailed_json(summary, str(detailed_path))
    write_eval_summary_md(
        summary,
        str(summary_md_path),
        title=f"Target Evaluation Summary ({get_display_name(cfg)})",
    )

    # Attach best lambda note
    best_path = results_dir / "best_lambda.json"
    if best_path.is_file():
        with open(best_path, "r", encoding="utf-8") as f:
            best = json.load(f)
        with open(summary_md_path, "a", encoding="utf-8") as f:
            f.write("\n## Lambda selection (source_dev only)\n\n")
            f.write(f"- best_lambda: **{best.get('best_lambda')}**\n")
            f.write(f"- best_wer on source_dev: {best.get('best_wer')}\n")
            f.write("- zero_target_leakage: true\n")


def run(cfg: Dict[str, Any], stages: Set[int], device: str) -> None:
    exp_type = get_experiment_type(cfg)
    device = resolve_runtime_device(device)
    best_lambda: Optional[float] = None
    summary: Optional[Dict[str, Dict[str, float]]] = None
    stage_titles = get_stage_titles(cfg)
    log_experiment_identity(cfg, stages)
    log_execution_plan(cfg, stages)

    with stage_progress(sorted(stages), desc=get_display_name(cfg)) as sp:
        if exp_type == "supervised_prompt_transfer":
            if 1 in stages:
                sp.begin(1, stage_titles[1])
                stage1_train_prompt_transfer(cfg)
                sp.end(1)
            if 2 in stages:
                sp.begin(2, stage_titles[2])
                stage2_eval_prompt_transfer_dev(cfg, device=device)
                sp.end(2)
            if 3 in stages:
                sp.begin(3, stage_titles[3])
                stage3_eval_prompt_transfer_test(cfg, device=device)
                sp.end(3)
            if 4 in stages:
                sp.begin(4, stage_titles[4])
                stage4_prompt_transfer_report(cfg)
                sp.end(4)
        elif exp_type == "pseudo2real_standard":
            if 1 in stages:
                sp.begin(1, stage_titles[1])
                stage1_pseudo_labels(cfg, device=device)
                sp.end(1)
            if 2 in stages:
                sp.begin(2, stage_titles[2])
                stage2_train_source_real(cfg)
                sp.end(2)
            if 3 in stages:
                sp.begin(3, stage_titles[3])
                stage3_train_pseudo_students(cfg)
                sp.end(3)
            if 4 in stages:
                sp.begin(4, stage_titles[4])
                stage3_tau(cfg)
                sp.end(4)
            if 5 in stages:
                sp.begin(5, stage_titles[5])
                best_lambda = stage4_lambda_search(cfg, device=device)
                sp.end(5)
            if 6 in stages:
                sp.begin(6, stage_titles[6])
                summary = stage5_apply_and_eval(cfg, device=device, best_lambda=best_lambda)
                sp.end(6)
            if 7 in stages:
                sp.begin(7, stage_titles[7])
                stage6_reports(cfg, summary=summary)
                sp.end(7)
        elif exp_type == "pseudo2real_strong_teacher":
            if 1 in stages:
                sp.begin(1, stage_titles[1])
                stage2_train_source_real(cfg)
                sp.end(1)
            if 2 in stages:
                sp.begin(2, stage_titles[2])
                teacher = str(resolve_path(cfg["checkpoints"]["source_real"]))
                stage1_pseudo_labels(cfg, device=device, teacher_model=teacher)
                sp.end(2)
            if 3 in stages:
                sp.begin(3, stage_titles[3])
                stage3_train_pseudo_students(cfg)
                sp.end(3)
            if 4 in stages:
                sp.begin(4, stage_titles[4])
                stage3_tau(cfg)
                sp.end(4)
            if 5 in stages:
                sp.begin(5, stage_titles[5])
                best_lambda = stage4_lambda_search(cfg, device=device)
                sp.end(5)
            if 6 in stages:
                sp.begin(6, stage_titles[6])
                summary = stage5_apply_and_eval(cfg, device=device, best_lambda=best_lambda)
                sp.end(6)
            if 7 in stages:
                sp.begin(7, stage_titles[7])
                stage6_reports(cfg, summary=summary)
                sp.end(7)
        elif exp_type == "supervised_target_real":
            if 1 in stages:
                sp.begin(1, stage_titles[1])
                stage1_train_target_real(cfg)
                sp.end(1)
            if 6 in stages:
                sp.begin(6, stage_titles[6])
                summary = stage6_eval_supervised_target_real(cfg, device=device)
                sp.end(6)
            if 7 in stages:
                sp.begin(7, stage_titles[7])
                stage6_reports(cfg, summary=summary)
                sp.end(7)
        else:
            raise ValueError(f"Unsupported experiment_type={exp_type}")

    logger.info("Pipeline finished stages=%s", sorted(stages))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pseudo2Real L2 ASR pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        help="Stage id(s): experiment-dependent, comma-separated, or 'all'",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable all tqdm progress bars (for clean log files)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    # academic default: force bars even under non-TTY (Cursor / tee / nohup)
    progress_disable = bool(args.no_progress)
    configure_experiment_progress(force=not progress_disable, disable=progress_disable)
    # Re-apply tqdm-safe handlers after basicConfig.
    configure_experiment_progress(force=not progress_disable, disable=progress_disable)

    cfg_path = resolve_path(args.config) if not Path(args.config).is_absolute() else Path(args.config)
    cfg = load_config(str(cfg_path))
    cfg["__config_path__"] = str(cfg_path)
    stages = parse_stages(args.stage, available=get_available_stages(cfg))
    logger.info("Config=%s stages=%s device=%s", cfg_path, sorted(stages), args.device)
    run(cfg, stages=stages, device=args.device)


if __name__ == "__main__":
    main()
