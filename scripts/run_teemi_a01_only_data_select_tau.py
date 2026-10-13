#!/usr/bin/env python3
"""A01-only confidence-bucket data selection + signed correction vectors.

Stages:
1. Score A01 train with θ_now (pretrained) for confidence × correctness.
2. Materialize 2×2 buckets, error union, drop-easy union, and a size-matched random control.
3. Fine-tune subset GT models (and a confident-error hypothesis model for τ_bad).
4. Compute τ = θ_S − θ_now and pairwise cosine.
5. Select λ on A01 dev (zero test leakage).
6. Evaluate on A01 internal test and A02 external test; write the report.

Do not use A01 test or A02 for selection. Existing baseline / tau artifacts are left untouched.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr_backend import family_slug, load_asr_model, load_processor as load_asr_processor  # noqa: E402
from src.audio_utils import ensure_device  # noqa: E402
from src.evaluate import (  # noqa: E402
    build_eval_dataloader,
    evaluate_model,
    score_model_with_confidence,
    write_eval_detailed_json,
)
from src.finetune import FinetuneConfig, finetune_whisper  # noqa: E402
from src.progress import configure_experiment_progress, progress_bar, stage_progress  # noqa: E402
from src.task_vector import apply_task_vector, calculate_task_vector, load_tau  # noqa: E402
from src.verbatim_diag import proficiency_sort_key  # noqa: E402

logger = logging.getLogger("teemi_a01_data_select_tau")

BUCKET_NAMES = ("conf_correct", "conf_error", "unc_error", "unc_correct")


@dataclass(frozen=True)
class TrainJob:
    name: str
    subset: str
    text_column: str
    reuse_cfg_key: Optional[str] = None
    use_full_max_steps: bool = False


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    terms: Tuple[Tuple[str, float], ...]
    combo: bool = False
    fixed_lambdas: Optional[Tuple[float, ...]] = None
    load_checkpoint: bool = False


TRAIN_JOBS: Tuple[TrainJob, ...] = (
    TrainJob("all_gt", "all", "text", reuse_cfg_key="all_gt_ckpt", use_full_max_steps=True),
    TrainJob("error_gt", "error", "text"),
    TrainJob("conf_error_gt", "conf_error", "text"),
    TrainJob("unc_correct_gt", "unc_correct", "text"),
    TrainJob("error_union_unc_correct_gt", "error_union_unc_correct", "text"),
    TrainJob("random_gt", "random_error_size", "text"),
    TrainJob("conf_error_hyp", "conf_error", "hypothesis"),
)

METHODS: Tuple[Method, ...] = (
    Method("theta_pre", "Pretrained (no tau)", ()),
    Method("all_gt_add", "All-data GT tau (+)", (("all_gt", 1.0),)),
    Method("error_gt_add", "Error-only GT tau (+)", (("error_gt", 1.0),)),
    Method("conf_error_gt_add", "Confident-error GT tau (+)", (("conf_error_gt", 1.0),)),
    Method("error_union_unc_correct_gt_add", "Error ∪ unconfident-correct GT tau (+)", (("error_union_unc_correct_gt", 1.0),)),
    Method("random_gt_add", "Random subset GT tau (+)", (("random_gt", 1.0),)),
    Method("conf_error_hyp_sub", "Confident-error hyp tau (−)", (("conf_error_hyp", -1.0),)),
    Method(
        "combo_error_plus_unc_correct",
        "Error GT (+) + unconfident-correct GT (+)",
        (("error_gt", 1.0), ("unc_correct_gt", 1.0)),
        combo=True,
    ),
)

# Decode θ_S itself (λ=1). Not included unless listed in enabled_methods.
FULL_FT_METHODS: Tuple[Method, ...] = (
    Method(
        "all_gt_ft",
        "All-data full FT (λ=1)",
        (("all_gt", 1.0),),
        fixed_lambdas=(1.0,),
        load_checkpoint=True,
    ),
    Method(
        "error_gt_ft",
        "Error-only full FT (λ=1)",
        (("error_gt", 1.0),),
        fixed_lambdas=(1.0,),
        load_checkpoint=True,
    ),
)

METHOD_CATALOG: Tuple[Method, ...] = METHODS + FULL_FT_METHODS
FT_CURVE_SIBLING = {
    "all_gt_ft": "all_gt_add",
    "error_gt_ft": "error_gt_add",
}


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_config(config_path: str) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict config, got {type(cfg)}")
    return cfg


def parse_stages(raw: str) -> Set[int]:
    available = {1, 2, 3, 4, 5, 6}
    text = str(raw).strip().lower()
    if text == "all":
        return available
    stages: Set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        stage = int(part)
        if stage not in available:
            raise ValueError(f"Unsupported stage {stage}; choose from {sorted(available)} or 'all'")
        stages.add(stage)
    return stages


def stage_titles(cfg: Optional[Dict[str, Any]] = None) -> Dict[int, str]:
    titles = {
        1: "Score A01 train confidence × correctness",
        2: "Materialize selection buckets and random control",
        3: "Fine-tune subset models",
        4: "Compute signed taus and cosine",
        5: "Select lambda on A01 dev",
        6: "Evaluate A01 test + A02 external and write report",
    }
    extra = (cfg or {}).get("stage_titles") or {}
    for key, value in extra.items():
        titles[int(key)] = str(value)
    return titles


def write_pipeline_flow(cfg: Dict[str, Any]) -> None:
    lines = cfg.get("pipeline_flow_lines")
    if not lines:
        return
    path = results_dir(cfg) / "pipeline_flow.md"
    path.write_text("\n".join(str(x) for x in lines) + "\n", encoding="utf-8")
    logger.info("Wrote %s", path)


def enabled_train_jobs(cfg: Dict[str, Any]) -> Tuple[TrainJob, ...]:
    names = cfg.get("enabled_train_jobs")
    if not names:
        return TRAIN_JOBS
    allow = {str(x) for x in names}
    jobs = tuple(job for job in TRAIN_JOBS if job.name in allow)
    unknown = allow - {job.name for job in TRAIN_JOBS}
    if unknown:
        raise ValueError(f"Unknown enabled_train_jobs: {sorted(unknown)}")
    return jobs


def configured_methods(cfg: Dict[str, Any]) -> Tuple[Method, ...]:
    names = cfg.get("enabled_methods")
    if not names:
        return METHODS
    allow = [str(x) for x in names]
    by_name = {method.name: method for method in METHOD_CATALOG}
    unknown = [name for name in allow if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown enabled_methods: {unknown}")
    return tuple(by_name[name] for name in allow)


def family_root(cfg: Dict[str, Any], key: str) -> Path:
    return resolve_repo_path(cfg[key]) / family_slug(cfg["student_model"])


def split_csv(cfg: Dict[str, Any], split_key: str) -> Path:
    return resolve_repo_path(cfg["splits"][split_key])


def scored_train_path(cfg: Dict[str, Any]) -> Path:
    return family_root(cfg, "data_dir") / "train_scored.csv"


def subset_csv_path(cfg: Dict[str, Any], name: str) -> Path:
    return family_root(cfg, "data_dir") / f"train_{name}.csv"


def _reuse_ckpt_path(cfg: Dict[str, Any], job: TrainJob) -> Optional[Path]:
    """Return a configured reuse directory only when the key is non-empty and exists.

    An unset/empty ``all_gt_ckpt`` must not resolve ``""`` to the repo root
    (that path always exists and would skip training).
    """
    if not job.reuse_cfg_key:
        return None
    raw = cfg.get(job.reuse_cfg_key)
    if raw is None or str(raw).strip() == "":
        return None
    path = resolve_repo_path(str(raw).strip())
    return path if path.exists() else None


def job_ckpt_dir(cfg: Dict[str, Any], job: TrainJob) -> Path:
    reuse = _reuse_ckpt_path(cfg, job)
    if reuse is not None:
        return reuse
    return family_root(cfg, "checkpoint_dir") / job.name


def tau_path(cfg: Dict[str, Any], job_name: str) -> Path:
    return family_root(cfg, "vector_dir") / f"tau_{job_name}.pt"


def results_dir(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "results_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def eval_splits_from_cfg(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in cfg.get("eval_splits", []):
        pairs.append((str(item["key"]), str(item.get("label", item["key"]))))
    return pairs


def row_count(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    return len(pd.read_csv(csv_path))


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.is_dir():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))


def _checkpoint_global_step(checkpoint_dir: Path) -> Optional[int]:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        return None
    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = state.get("global_step")
    return int(step) if step is not None else None


def training_completed(output_dir: Path, max_steps: int) -> bool:
    has_weights = (
        (output_dir / "model.safetensors").is_file()
        or (output_dir / "pytorch_model.bin").is_file()
        or (output_dir / "model.safetensors.index.json").is_file()
    )
    if has_weights:
        latest = _latest_checkpoint(output_dir)
        if latest is None:
            return True
        step = _checkpoint_global_step(latest)
        return step is None or step >= max_steps
    latest = _latest_checkpoint(output_dir)
    if latest is None:
        return False
    step = _checkpoint_global_step(latest)
    return step is not None and step >= max_steps


def resolve_weight_dir(output_dir: Path) -> Path:
    if (
        (output_dir / "model.safetensors").is_file()
        or (output_dir / "pytorch_model.bin").is_file()
        or (output_dir / "model.safetensors.index.json").is_file()
    ):
        return output_dir
    latest = _latest_checkpoint(output_dir)
    if latest is not None:
        return latest
    if (output_dir / "config.json").is_file():
        return output_dir
    raise FileNotFoundError(f"Could not locate evaluable weights under {output_dir}")


def load_processor(cfg: Dict[str, Any]):
    return load_asr_processor(cfg["student_model"])


def load_base_model(cfg: Dict[str, Any], device: torch.device):
    return load_asr_model(cfg.get("base_model", cfg["student_model"]), device=device)


def decode_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "language": str(cfg.get("language", "en")),
        "task": str(cfg.get("task", "transcribe")),
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
    }


def min_subset_utterances(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("min_subset_utterances", 16))


def subset_usable(cfg: Dict[str, Any], name: str) -> bool:
    return row_count(subset_csv_path(cfg, name)) >= min_subset_utterances(cfg)


def _metrics_from_counts(
    total_h: float,
    total_s: float,
    total_d: float,
    total_i: float,
    n_utterances: int,
) -> Dict[str, float]:
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


def _metrics_from_pred_df(pred_df: pd.DataFrame) -> Dict[str, float]:
    return _metrics_from_counts(
        total_h=float(pred_df["hits"].sum()),
        total_s=float(pred_df["substitutions"].sum()),
        total_d=float(pred_df["deletions"].sum()),
        total_i=float(pred_df["insertions"].sum()),
        n_utterances=int(len(pred_df)),
    )


def job_by_name(name: str) -> TrainJob:
    for job in TRAIN_JOBS:
        if job.name == name:
            return job
    raise KeyError(name)


def job_has_weights(cfg: Dict[str, Any], job: TrainJob) -> bool:
    try:
        resolve_weight_dir(job_ckpt_dir(cfg, job))
        return True
    except FileNotFoundError:
        return False


def available_methods(cfg: Dict[str, Any], *, require_tau: bool = False) -> List[Method]:
    usable: List[Method] = []
    for method in configured_methods(cfg):
        if not method.terms:
            usable.append(method)
            continue
        ok = True
        for job_name, _sign in method.terms:
            job = job_by_name(job_name)
            if method.load_checkpoint:
                if not job_has_weights(cfg, job):
                    ok = False
                    break
                continue
            if require_tau and not tau_path(cfg, job_name).is_file():
                ok = False
                break
            if not require_tau and not job_has_weights(cfg, job) and not subset_usable(cfg, job.subset):
                ok = False
                break
        if ok:
            usable.append(method)
        else:
            logger.warning("Skip method %s; required subset/tau is missing", method.name)
    return usable


def score_train(cfg: Dict[str, Any], device: str) -> None:
    out_csv = scored_train_path(cfg)
    if out_csv.is_file():
        logger.info("[Stage 1] Skip; found %s", out_csv)
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    train_path = split_csv(cfg, "train")
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    model = load_base_model(cfg, device_t)
    loader = build_eval_dataloader(
        str(train_path),
        processor=processor,
        batch_size=int(cfg.get("eval_batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 2)),
        language=str(cfg.get("language", "en")),
    )
    scored = score_model_with_confidence(
        model,
        loader,
        processor,
        desc=str(cfg.get("score_train_desc", "score A01 train")),
        **decode_kwargs(cfg),
    )
    train_df = pd.read_csv(train_path)
    merged = train_df.merge(scored, on="audio_path", how="inner")
    dropped = len(train_df) - len(merged)
    if dropped:
        logger.warning("Dropped %d train rows that were not scored", dropped)
    merged.to_csv(out_csv, index=False)
    logger.info("[Stage 1] Wrote %s (%d rows)", out_csv, len(merged))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _assign_buckets(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, float]:
    col = str(cfg.get("confidence_column", "mean_token_logprob"))
    quantile = float(cfg.get("confidence_quantile", 0.5))
    threshold = float(df[col].quantile(quantile))
    confident = df[col] >= threshold
    correct = df["is_correct"].astype(bool)
    out = df.copy()
    out["confidence_threshold"] = threshold
    out["is_confident"] = confident
    out["bucket"] = "unc_error"
    out.loc[confident & correct, "bucket"] = "conf_correct"
    out.loc[confident & ~correct, "bucket"] = "conf_error"
    out.loc[~confident & correct, "bucket"] = "unc_correct"
    out.loc[~confident & ~correct, "bucket"] = "unc_error"
    return out, threshold


def _write_subset(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def partition_buckets(cfg: Dict[str, Any]) -> None:
    data_root = family_root(cfg, "data_dir")
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "bucket_manifest.json"
    scored_path = scored_train_path(cfg)
    if not scored_path.is_file():
        raise FileNotFoundError(f"Run stage 1 first: missing {scored_path}")
    df = pd.read_csv(scored_path)
    df, threshold = _assign_buckets(df, cfg)
    _write_subset(df, scored_path)

    buckets = {name: df[df["bucket"] == name].copy() for name in BUCKET_NAMES}
    error = pd.concat([buckets["conf_error"], buckets["unc_error"]], ignore_index=True)
    union = pd.concat([error, buckets["unc_correct"]], ignore_index=True)
    seed = int(cfg.get("seed", 42))
    n_error = len(error)
    random_df = df.sample(n=min(n_error, len(df)), random_state=seed).copy() if n_error else df.iloc[0:0].copy()

    subsets = {
        "conf_correct": buckets["conf_correct"],
        "conf_error": buckets["conf_error"],
        "unc_error": buckets["unc_error"],
        "unc_correct": buckets["unc_correct"],
        "error": error,
        "error_union_unc_correct": union,
        "all": df,
        "random_error_size": random_df,
    }
    stats: List[Dict[str, Any]] = []
    for name, sub in subsets.items():
        path = subset_csv_path(cfg, name)
        _write_subset(sub, path)
        by_level = (
            sub["proficiency_level"].astype(str).value_counts().to_dict()
            if "proficiency_level" in sub.columns and len(sub)
            else {}
        )
        stats.append(
            {
                "subset": name,
                "utterances": int(len(sub)),
                "speakers": int(sub["global_speaker_id"].nunique()) if "global_speaker_id" in sub.columns and len(sub) else 0,
                "correct": int(sub["is_correct"].sum()) if "is_correct" in sub.columns and len(sub) else 0,
                "by_proficiency": {str(k): int(v) for k, v in by_level.items()},
            }
        )
        logger.info("Subset %-28s %d utt", name, len(sub))

    payload = {
        "confidence_column": str(cfg.get("confidence_column", "mean_token_logprob")),
        "confidence_quantile": float(cfg.get("confidence_quantile", 0.5)),
        "confidence_threshold": threshold,
        "n_train_scored": int(len(df)),
        "subsets": stats,
        "selection_split": "train",
        "zero_test_leakage": True,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_df = pd.DataFrame([{k: v for k, v in row.items() if k != "by_proficiency"} for row in stats])
    stats_df.to_csv(results_dir(cfg) / "bucket_counts.csv", index=False)
    _write_bucket_manifest_md(cfg, payload)
    logger.info("[Stage 2] threshold=%.4f -> %s", threshold, manifest_path)


def _write_bucket_manifest_md(cfg: Dict[str, Any], payload: Dict[str, Any]) -> None:
    lines = [
        f"# {cfg.get('display_name', 'Data-Select Tau')} Bucket Manifest",
        "",
        f"- Selection split: **{cfg.get('selection_split_label', 'A01 train only')}**",
        f"- Confidence: `{payload['confidence_column']}` quantile={payload['confidence_quantile']}",
        f"- Threshold: **{payload['confidence_threshold']:.4f}**",
        f"- Scored train utterances: **{payload['n_train_scored']}**",
        "",
        "| Subset | Utterances | Speakers | Correct |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["subsets"]:
        lines.append(
            f"| {row['subset']} | {row['utterances']} | {row['speakers']} | {row['correct']} |"
        )
    lines.extend(
        [
            "",
            "- `conf_correct`: drop (do not train).",
            "- `conf_error` / `unc_error`: train; `τ_fix` is added.",
            "- `unc_correct`: optional stabilizer; added in union / combo methods.",
            "- `random_error_size`: size-matched control vs error-only.",
            "",
        ]
    )
    path = results_dir(cfg) / "bucket_manifest.md"
    path.write_text("\n".join(lines), encoding="utf-8")


def _job_max_steps(cfg: Dict[str, Any], job: TrainJob) -> int:
    if job.use_full_max_steps:
        return int(cfg.get("max_steps", 4000))
    return int(cfg.get("subset_max_steps", cfg.get("max_steps", 2000)))


def _job_warmup_steps(cfg: Dict[str, Any], job: TrainJob) -> int:
    if job.use_full_max_steps:
        return int(cfg.get("warmup_steps", 500))
    return int(cfg.get("subset_warmup_steps", min(200, _job_max_steps(cfg, job) // 10)))


def train_subset_models(cfg: Dict[str, Any]) -> None:
    jobs = enabled_train_jobs(cfg)
    for job in progress_bar(jobs, total=len(jobs), desc="subset finetune", unit="job"):
        csv_path = subset_csv_path(cfg, job.subset)
        n = row_count(csv_path)
        reuse = _reuse_ckpt_path(cfg, job)
        if n < min_subset_utterances(cfg) and reuse is None:
            logger.warning("Skip train %s; only %d rows in %s", job.name, n, csv_path)
            continue
        out_dir = job_ckpt_dir(cfg, job)
        max_steps = _job_max_steps(cfg, job)
        if reuse is not None:
            logger.info("Reuse %s checkpoint %s", job.name, reuse)
            continue
        if training_completed(out_dir, max_steps):
            logger.info("Skip train %s; completed at %s", job.name, out_dir)
            continue
        if job.text_column not in pd.read_csv(csv_path, nrows=1).columns:
            raise KeyError(f"{csv_path} missing text column {job.text_column}")
        resume = _latest_checkpoint(out_dir)
        logger.info("Fine-tune %s on %s (%d utt) -> %s", job.name, job.subset, n, out_dir)
        finetune_whisper(
            FinetuneConfig(
                model_name_or_path=str(cfg["student_model"]),
                train_csv=str(csv_path),
                output_dir=str(out_dir),
                text_column=job.text_column,
                language=str(cfg.get("language", "en")),
                task=str(cfg.get("task", "transcribe")),
                learning_rate=float(cfg.get("learning_rate", 1e-5)),
                weight_decay=float(cfg.get("weight_decay", 0.01)),
                warmup_steps=_job_warmup_steps(cfg, job),
                max_steps=max_steps,
                per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 12)),
                per_device_eval_batch_size=int(cfg.get("eval_batch_size", cfg.get("per_device_train_batch_size", 12))),
                gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
                fp16=bool(cfg.get("fp16", True)),
                bf16=bool(cfg.get("bf16", False)),
                seed=int(cfg.get("seed", 42)),
                dataloader_num_workers=int(cfg.get("num_workers", 4)),
                resume_from_checkpoint=str(resume) if resume else None,
            )
        )


def compute_taus(cfg: Dict[str, Any]) -> None:
    base_src = str(cfg.get("base_model", cfg["student_model"]))
    for job in enabled_train_jobs(cfg):
        if not subset_usable(cfg, job.subset) and _reuse_ckpt_path(cfg, job) is None:
            continue
        out = tau_path(cfg, job.name)
        if out.is_file():
            logger.info("Skip tau %s; found %s", job.name, out)
            continue
        ckpt = job_ckpt_dir(cfg, job)
        logger.info("Resolve weights for %s -> %s", job.name, ckpt)
        weight_dir = resolve_weight_dir(ckpt)
        out.parent.mkdir(parents=True, exist_ok=True)
        logger.info("tau_%s = %s - %s", job.name, weight_dir, base_src)
        calculate_task_vector(str(weight_dir), base_src, str(out))
    _analyze_tau_cosine(cfg)


def _dot_tau(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, tensor in a.items():
        if key not in b:
            continue
        if tensor.shape != b[key].shape:
            continue
        total += float(torch.dot(tensor.float().view(-1), b[key].float().view(-1)))
    return total


def _norm_tau(tau: Dict[str, torch.Tensor]) -> float:
    return _dot_tau(tau, tau) ** 0.5


def _analyze_tau_cosine(cfg: Dict[str, Any]) -> None:
    names = [job.name for job in enabled_train_jobs(cfg) if tau_path(cfg, job.name).is_file()]
    if len(names) < 2:
        logger.warning("Need at least two taus for cosine analysis")
        return
    taus = {name: load_tau(str(tau_path(cfg, name))) for name in names}
    norms = {name: _norm_tau(tau) for name, tau in taus.items()}
    rows: List[Dict[str, Any]] = []
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for i, ni in enumerate(names):
        for nj in names[i:]:
            denom = norms[ni] * norms[nj]
            cos = 0.0 if denom == 0 else _dot_tau(taus[ni], taus[nj]) / denom
            matrix.loc[ni, nj] = cos
            matrix.loc[nj, ni] = cos
            if ni != nj:
                rows.append({"tau_a": ni, "tau_b": nj, "cosine": float(cos)})
    out_dir = results_dir(cfg)
    matrix.to_csv(out_dir / "tau_cosine_similarity.csv")
    pd.DataFrame(rows).to_csv(out_dir / "tau_cosine_pairs.csv", index=False)
    logger.info("Wrote tau cosine tables -> %s", out_dir)


def _apply_method(
    model,
    cfg: Dict[str, Any],
    method: Method,
    lambdas: Sequence[float],
    tau_cache: Dict[str, Dict[str, torch.Tensor]],
) -> None:
    if len(method.terms) != len(lambdas):
        raise ValueError(f"{method.name} expected {len(method.terms)} lambdas, got {len(lambdas)}")
    for (job_name, sign), lam in zip(method.terms, lambdas):
        if job_name not in tau_cache:
            tau_cache[job_name] = load_tau(str(tau_path(cfg, job_name)))
        apply_task_vector(model, tau_cache[job_name], float(sign) * float(lam))


def _ft_checkpoint_dir(cfg: Dict[str, Any], method: Method) -> Path:
    if not method.load_checkpoint or not method.terms:
        raise ValueError(f"{method.name} is not a checkpoint-backed full-FT method")
    job = job_by_name(method.terms[0][0])
    return resolve_weight_dir(job_ckpt_dir(cfg, job))


def load_eval_model(
    cfg: Dict[str, Any],
    method: Method,
    device: torch.device,
    lambdas: Sequence[float],
    tau_cache: Dict[str, Dict[str, torch.Tensor]],
):
    if method.load_checkpoint:
        weight_dir = _ft_checkpoint_dir(cfg, method)
        logger.info("Load full-FT weights %s -> %s", method.name, weight_dir)
        return load_asr_model(str(weight_dir), device=device)
    model = load_base_model(cfg, device)
    if method.terms:
        _apply_method(model, cfg, method, lambdas, tau_cache)
    return model


def _lambda_payload_path(cfg: Dict[str, Any], method_name: str) -> Path:
    return results_dir(cfg) / f"best_lambda_{method_name}.json"


def _load_best_lambdas(cfg: Dict[str, Any], method: Method) -> List[float]:
    if not method.terms:
        return []
    if method.fixed_lambdas is not None:
        return [float(x) for x in method.fixed_lambdas]
    path = _lambda_payload_path(cfg, method.name)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if method.combo:
        return [float(payload["lambda_fix"]), float(payload["lambda_unc"])]
    return [float(payload["best_lambda"])]


def _payload_from_sibling_curve(cfg: Dict[str, Any], method: Method) -> Optional[Dict[str, Any]]:
    sibling = FT_CURVE_SIBLING.get(method.name)
    if sibling is None or method.fixed_lambdas is None:
        return None
    path = results_dir(cfg) / f"lambda_curve_{sibling}.csv"
    if not path.is_file():
        return None
    curve = pd.read_csv(path)
    if "lambda" not in curve.columns:
        return None
    target = float(method.fixed_lambdas[0])
    hit = curve[(curve["lambda"].astype(float) - target).abs() < 1e-9]
    if hit.empty:
        return None
    row = hit.iloc[0]
    return {
        "method": method.name,
        "best_lambda": target,
        "signed_lambda": float(row.get("signed_lambda", target)),
        "best_wer": float(row["wer"]),
        "selection_set": str(cfg.get("lambda_selection_set", "source_dev")),
        "zero_target_leakage": True,
        "fixed_lambda": True,
        "source_curve": path.name,
        **{k: float(row[k]) for k in ("sub", "ins", "del") if k in row.index},
    }


def select_lambdas(cfg: Dict[str, Any], device: str) -> None:
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    model = load_base_model(cfg, device_t)
    base_state = copy.deepcopy(model.state_dict())
    dev_loader = build_eval_dataloader(
        str(split_csv(cfg, "dev")),
        processor=processor,
        batch_size=int(cfg.get("eval_batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 2)),
        language=str(cfg.get("language", "en")),
    )
    lambda_list = [float(x) for x in cfg.get("lambda_list", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])]
    combo_fix = [float(x) for x in cfg.get("combo_lambda_fix", [0.0, 0.3, 0.5, 0.7])]
    combo_unc = [float(x) for x in cfg.get("combo_lambda_unc", [0.0, 0.2, 0.4])]
    eval_kw = decode_kwargs(cfg)

    def eval_dev(desc: str) -> Dict[str, float]:
        return evaluate_model(model, dev_loader, processor, desc=desc, **eval_kw)

    tau_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    for method in available_methods(cfg, require_tau=True):
        out_json = _lambda_payload_path(cfg, method.name)
        if out_json.is_file():
            logger.info("Skip lambda search %s; found %s", method.name, out_json)
            continue
        if not method.terms:
            payload = {"method": method.name, "best_lambda": 0.0, "best_wer": None, "selection_set": "source_dev"}
            model.load_state_dict(base_state, strict=True)
            metrics = eval_dev("dev:theta_pre")
            payload["best_wer"] = float(metrics["wer"])
            payload.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            continue

        if method.fixed_lambdas is not None:
            payload = _payload_from_sibling_curve(cfg, method)
            if payload is None:
                lambdas = [float(x) for x in method.fixed_lambdas]
                if method.load_checkpoint:
                    ft_model = load_asr_model(str(_ft_checkpoint_dir(cfg, method)), device=device_t)
                    metrics = evaluate_model(ft_model, dev_loader, processor, desc=f"dev:{method.name}:ft", **eval_kw)
                    del ft_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    model.load_state_dict(base_state, strict=True)
                    _apply_method(model, cfg, method, lambdas, tau_cache)
                    metrics = eval_dev(f"dev:{method.name}:fixed")
                payload = {
                    "method": method.name,
                    "best_lambda": lambdas[0],
                    "signed_lambda": float(method.terms[0][1]) * lambdas[0],
                    "best_wer": float(metrics["wer"]),
                    "selection_set": str(cfg.get("lambda_selection_set", "source_dev")),
                    "zero_target_leakage": True,
                    "fixed_lambda": True,
                    **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                }
            out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            logger.info("%s fixed λ=%s -> WER=%.2f", method.name, method.fixed_lambdas, payload["best_wer"])
            continue

        rows: List[Dict[str, Any]] = []
        if method.combo:
            grid = [(lf, lu) for lf in combo_fix for lu in combo_unc]
            for lf, lu in progress_bar(grid, total=len(grid), desc=f"lambda {method.name}", unit="λ"):
                model.load_state_dict(base_state, strict=True)
                _apply_method(model, cfg, method, (lf, lu), tau_cache)
                metrics = eval_dev(f"dev:{method.name}:fix{lf}:unc{lu}")
                row = {"lambda_fix": lf, "lambda_unc": lu, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}}
                rows.append(row)
            curve = pd.DataFrame(rows)
            best = curve.loc[curve["wer"].idxmin()]
            payload = {
                "method": method.name,
                "lambda_fix": float(best["lambda_fix"]),
                "lambda_unc": float(best["lambda_unc"]),
                "best_lambda": float(best["lambda_fix"]),
                "best_wer": float(best["wer"]),
                "selection_set": "source_dev",
                "zero_target_leakage": True,
            }
        else:
            sign = method.terms[0][1]
            for lam in progress_bar(lambda_list, total=len(lambda_list), desc=f"lambda {method.name}", unit="λ"):
                model.load_state_dict(base_state, strict=True)
                _apply_method(model, cfg, method, (lam,), tau_cache)
                metrics = eval_dev(f"dev:{method.name}:{sign * lam:.2f}")
                row = {"lambda": lam, "signed_lambda": sign * lam, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}}
                rows.append(row)
            curve = pd.DataFrame(rows)
            best = curve.loc[curve["wer"].idxmin()]
            payload = {
                "method": method.name,
                "best_lambda": float(best["lambda"]),
                "signed_lambda": float(best["signed_lambda"]),
                "best_wer": float(best["wer"]),
                "lambda_list": lambda_list,
                "selection_set": "source_dev",
                "zero_target_leakage": True,
            }
        curve_path = results_dir(cfg) / f"lambda_curve_{method.name}.csv"
        curve.to_csv(curve_path, index=False)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("%s best %s -> WER=%.2f", method.name, payload, payload["best_wer"])

    model.load_state_dict(base_state, strict=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prediction_csv(cfg: Dict[str, Any], split_key: str, method_name: str) -> Path:
    return results_dir(cfg) / f"predictions_{split_key}_{method_name}.csv"


def evaluate_methods(cfg: Dict[str, Any], device: str) -> None:
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    eval_kw = decode_kwargs(cfg)
    batch_size = int(cfg.get("eval_batch_size", 4))
    num_workers = int(cfg.get("num_workers", 2))
    methods = available_methods(cfg, require_tau=True)
    split_summaries: Dict[str, Dict[str, Dict[str, float]]] = {key: {} for key, _ in eval_splits_from_cfg(cfg)}
    tau_cache: Dict[str, Dict[str, torch.Tensor]] = {}

    for method in methods:
        lambdas = _load_best_lambdas(cfg, method) if method.terms else []
        pending = False
        for split_key, _ in eval_splits_from_cfg(cfg):
            out_csv = _prediction_csv(cfg, split_key, method.name)
            if out_csv.is_file():
                split_summaries[split_key][method.name] = _metrics_from_pred_df(pd.read_csv(out_csv))
            else:
                pending = True
        if not pending:
            logger.info("Skip eval %s; predictions already exist", method.name)
            continue

        model = load_eval_model(cfg, method, device_t, lambdas, tau_cache)
        for split_key, label in eval_splits_from_cfg(cfg):
            out_csv = _prediction_csv(cfg, split_key, method.name)
            if out_csv.is_file():
                split_summaries[split_key][method.name] = _metrics_from_pred_df(pd.read_csv(out_csv))
                continue
            loader = build_eval_dataloader(
                str(split_csv(cfg, split_key)),
                processor=processor,
                batch_size=batch_size,
                num_workers=num_workers,
                language=str(cfg.get("language", "en")),
            )
            metrics = evaluate_model(
                model,
                loader,
                processor,
                output_csv_path=str(out_csv),
                desc=f"{split_key}:{method.name}",
                **eval_kw,
            )
            split_summaries[split_key][method.name] = metrics
            logger.info("%s / %s WER=%.2f", label, method.name, metrics["wer"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for split_key, _ in eval_splits_from_cfg(cfg):
        write_eval_detailed_json(split_summaries[split_key], str(results_dir(cfg) / f"{split_key}_eval_detailed.json"))
    _write_experiment_report(cfg, split_summaries)


def _by_proficiency_metrics(cfg: Dict[str, Any], split_key: str, method_name: str) -> Dict[str, Dict[str, float]]:
    pred_path = _prediction_csv(cfg, split_key, method_name)
    if not pred_path.is_file():
        return {}
    pred = pd.read_csv(pred_path)
    meta = pd.read_csv(split_csv(cfg, split_key))
    cols = [c for c in ["audio_path", "proficiency_level", "cefr"] if c in meta.columns]
    merged = pred.merge(meta[cols], on="audio_path", how="left")
    level_col = "proficiency_level" if "proficiency_level" in merged.columns else ("cefr" if "cefr" in merged.columns else None)
    if not level_col:
        return {}
    grouped: Dict[str, Dict[str, float]] = {}
    valid = merged[merged[level_col].notna() & (merged[level_col].astype(str).str.strip() != "")]
    for level, sub in valid.groupby(valid[level_col].astype(str).str.strip()):
        grouped[str(level)] = _metrics_from_pred_df(sub)
    return grouped


def _write_experiment_report(cfg: Dict[str, Any], split_summaries: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    methods = available_methods(cfg, require_tau=True)
    scope = cfg.get("report_scope") or [
        "Booklet: A01 train → select buckets; A01 dev → λ; A01 test + A02 external → report.",
        "`τ_fix = θ_GT_S − θ_now` is **added**. `τ_bad` from confident-error hypotheses is **subtracted**.",
        "`conf_correct` is dropped and is never used for training.",
        "A01 test and A02 are not used for selection or λ search.",
        f"θ_now / student: `{cfg.get('base_model', cfg['student_model'])}`",
    ]
    lambda_label = cfg.get("lambda_selection_label", "A01 dev")
    lines = [
        f"# Experiment Report ({cfg.get('display_name', 'A01-only Data-Select Tau')})",
        "",
        "## Scope",
        "",
    ]
    lines.extend(f"- {item}" for item in scope)
    lines.extend(
        [
            "",
            f"## Selected lambdas ({lambda_label})",
            "",
            "| Method | λ | Dev WER |",
            "|---|---:|---:|",
        ]
    )
    for method in methods:
        path = _lambda_payload_path(cfg, method.name)
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if method.combo:
            lam = f"fix={payload.get('lambda_fix')}, unc={payload.get('lambda_unc')}"
        else:
            lam = str(payload.get("signed_lambda", payload.get("best_lambda", 0.0)))
        lines.append(f"| {method.label} | {lam} | {float(payload.get('best_wer', float('nan'))):.2f} |")
    lines.append("")

    for split_key, label in eval_splits_from_cfg(cfg):
        lines.extend(
            [
                f"## {label}",
                "",
                "| Method | WER (%) | Sub (%) | Ins (%) | Del (%) | Gain vs Pretrained |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        pre = split_summaries.get(split_key, {}).get("theta_pre", {})
        pre_wer = float(pre.get("wer", float("nan")))
        for method in methods:
            metrics = split_summaries.get(split_key, {}).get(method.name, {})
            if not metrics:
                continue
            wer = float(metrics.get("wer", float("nan")))
            gain = pre_wer - wer
            lines.append(
                f"| {method.label} | {wer:.2f} | {float(metrics.get('sub', float('nan'))):.2f} | "
                f"{float(metrics.get('ins', float('nan'))):.2f} | {float(metrics.get('del', float('nan'))):.2f} | "
                f"{gain:.2f} |"
            )
        lines.append("")
        lines.extend([f"### {label} {cfg.get('proficiency_section_suffix', 'by proficiency')}", ""])
        header = "| Proficiency | n | " + " | ".join(m.label for m in methods) + " |"
        align = "|---|" + "---:|" + "|".join("---:" for _ in methods) + "|"
        lines.extend([header, align])
        by_method = {m.name: _by_proficiency_metrics(cfg, split_key, m.name) for m in methods}
        levels = sorted(
            {lvl for grouped in by_method.values() for lvl in grouped},
            key=proficiency_sort_key,
        )
        for level in levels:
            n_val = None
            for method in methods:
                metrics = by_method.get(method.name, {}).get(level)
                if metrics and n_val is None:
                    n_val = int(metrics.get("n_utterances", 0))
            cells = [level, str(n_val) if n_val is not None else "—"]
            for method in methods:
                metrics = by_method.get(method.name, {}).get(level)
                cells.append(f"{metrics['wer']:.2f}" if metrics else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    cosine_path = results_dir(cfg) / "tau_cosine_similarity.csv"
    if cosine_path.is_file():
        lines.extend(["## Tau cosine similarity", "", f"See `{cosine_path.name}`.", ""])
    report_path = results_dir(cfg) / "experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report -> %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="A01-only data-select signed tau pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    configure_experiment_progress(force=not args.no_progress, disable=bool(args.no_progress))

    cfg = load_config(args.config)
    write_pipeline_flow(cfg)
    stages = parse_stages(args.stage)
    titles = stage_titles(cfg)
    logger.info("Config=%s stages=%s device=%s", args.config, sorted(stages), args.device)

    with stage_progress(sorted(stages), desc=cfg.get("display_name", "A01-only Data-Select Tau")) as sp:
        if 1 in stages:
            sp.begin(1, titles[1])
            score_train(cfg, device=args.device)
            sp.end(1)
        if 2 in stages:
            sp.begin(2, titles[2])
            partition_buckets(cfg)
            sp.end(2)
        if 3 in stages:
            sp.begin(3, titles[3])
            train_subset_models(cfg)
            sp.end(3)
        if 4 in stages:
            sp.begin(4, titles[4])
            compute_taus(cfg)
            sp.end(4)
        if 5 in stages:
            sp.begin(5, titles[5])
            select_lambdas(cfg, device=args.device)
            sp.end(5)
        if 6 in stages:
            sp.begin(6, titles[6])
            evaluate_methods(cfg, device=args.device)
            sp.end(6)


if __name__ == "__main__":
    main()
