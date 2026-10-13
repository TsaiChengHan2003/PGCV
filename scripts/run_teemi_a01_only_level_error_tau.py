#!/usr/bin/env python3
"""Per-level error-GT τ, cross-apply matrix, and adjacent simplex mix.

Stages:
1. Reuse / score A01 train (confidence × correctness) and write the data-flow note.
2. Materialize per-level error-GT train CSVs and per-level eval slices.
3. Fine-tune one error-GT model per level (same recipe as global data-select).
4. Compute τ_L = θ_{error,L} − θ_now and cosine vs global / neighbors.
5. Decode the donor bank on A01 dev; select λ; decode test / A02 at chosen λ;
   write the cross-apply matrices.
6. For each focus level L, search α in span{τ_{L-1}, τ_L, τ_{L+1}} on that
   level's A01 dev, then evaluate the winner on that level's test / A02.
7. Write the experiment report.

Gold CEFR only slices buckets. No NN gate. A01 test and A02 are held out of
training, λ search, and α search.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import shutil
import sys
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
)
from src.finetune import FinetuneConfig, finetune_whisper  # noqa: E402
from src.progress import configure_experiment_progress, progress_bar, stage_progress  # noqa: E402
from src.task_vector import apply_task_vector, calculate_task_vector, load_tau, mix_task_vectors  # noqa: E402

logger = logging.getLogger("teemi_a01_level_error_tau")

GLOBAL_NAME = "global"
PRETRAINED_NAME = "theta_pre"


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
    available = {1, 2, 3, 4, 5, 6, 7}
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


def stage_titles() -> Dict[int, str]:
    return {
        1: "Reuse or score A01 train and write data flow",
        2: "Materialize per-level error-GT subsets",
        3: "Fine-tune per-level error-GT models",
        4: "Compute per-level taus and cosine",
        5: "Cross-apply bank, select lambda, decode eval",
        6: "Adjacent simplex mix on focus levels",
        7: "Write experiment report",
    }


def family_root(cfg: Dict[str, Any], key: str) -> Path:
    return resolve_repo_path(cfg[key]) / family_slug(cfg["student_model"])


def split_csv(cfg: Dict[str, Any], split_key: str) -> Path:
    return resolve_repo_path(cfg["splits"][split_key])


def levels_from_cfg(cfg: Dict[str, Any]) -> List[str]:
    return [str(x) for x in cfg["levels"]]


def slugify_level(level: str) -> str:
    text = level.lower().replace("+", "_plus")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def data_root(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "data_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "results_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def ckpt_root(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "checkpoint_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def vector_root(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "vector_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def scored_train_path(cfg: Dict[str, Any]) -> Path:
    return data_root(cfg) / "train_scored.csv"


def level_train_csv(cfg: Dict[str, Any], level: str) -> Path:
    return data_root(cfg) / "levels" / slugify_level(level) / "train_error.csv"


def level_split_csv(cfg: Dict[str, Any], split_key: str, level: str) -> Path:
    return data_root(cfg) / "levels" / slugify_level(level) / f"{split_key}.csv"


def level_ckpt_dir(cfg: Dict[str, Any], level: str) -> Path:
    return ckpt_root(cfg) / slugify_level(level)


def level_tau_path(cfg: Dict[str, Any], level: str) -> Path:
    return vector_root(cfg) / f"tau_{slugify_level(level)}.pt"


def bank_pred_path(cfg: Dict[str, Any], split_key: str, donor: str, lam: float) -> Path:
    return data_root(cfg) / "_bank" / f"pred_{split_key}_{donor}_lam{lam:.2f}.csv"


def adjacent_pred_path(cfg: Dict[str, Any], split_key: str, level: str) -> Path:
    return results_dir(cfg) / f"predictions_{split_key}_{slugify_level(level)}_adjacent.csv"


def eval_splits_from_cfg(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in cfg.get("eval_splits", []):
        pairs.append((str(item["key"]), str(item.get("label", item["key"]))))
    return pairs


def lambda_list(cfg: Dict[str, Any]) -> List[float]:
    return [float(x) for x in cfg.get("lambda_list", [0.0, 0.5, 0.7, 1.0])]


def adjacent_lambda_list(cfg: Dict[str, Any]) -> List[float]:
    return [float(x) for x in cfg.get("adjacent_lambda_list", [0.5, 0.7, 1.0])]


def min_subset_utterances(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("min_subset_utterances", 16))


def decode_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "language": str(cfg.get("language", "en")),
        "task": str(cfg.get("task", "transcribe")),
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
    }


def load_processor(cfg: Dict[str, Any]):
    return load_asr_processor(cfg["student_model"])


def load_base_model(cfg: Dict[str, Any], device: torch.device):
    return load_asr_model(cfg.get("base_model", cfg["student_model"]), device=device)


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


def metrics_from_pred_df(pred_df: pd.DataFrame) -> Dict[str, float]:
    if pred_df is None or len(pred_df) == 0:
        return _metrics_from_counts(0, 0, 0, 0, 0)
    return _metrics_from_counts(
        total_h=float(pred_df["hits"].sum()),
        total_s=float(pred_df["substitutions"].sum()),
        total_d=float(pred_df["deletions"].sum()),
        total_i=float(pred_df["insertions"].sum()),
        n_utterances=int(len(pred_df)),
    )


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def donor_label(name: str, levels: Sequence[str]) -> str:
    if name == PRETRAINED_NAME:
        return "Pretrained"
    if name == GLOBAL_NAME:
        return "Global error-GT"
    for level in levels:
        if slugify_level(level) == name or level == name:
            return f"τ_{level}"
    return name


def neighbors_of(level: str, levels: Sequence[str]) -> List[str]:
    ordered = [lv for lv in levels]
    if level not in ordered:
        return [level]
    idx = ordered.index(level)
    names = []
    if idx > 0:
        names.append(ordered[idx - 1])
    names.append(level)
    if idx + 1 < len(ordered):
        names.append(ordered[idx + 1])
    return names


def _simplex_grid(n_weights: int, step: float) -> List[Tuple[float, ...]]:
    if n_weights <= 0:
        return []
    units = max(int(round(1.0 / float(step))), 1)
    grid: List[Tuple[float, ...]] = []

    def rec(remain: int, left: int, acc: List[int]) -> None:
        if left == 1:
            grid.append(tuple(x / units for x in acc + [remain]))
            return
        for k in range(remain + 1):
            rec(remain - k, left - 1, acc + [k])

    rec(units, n_weights, [])
    return grid


def attach_level(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["audio_path", "proficiency_level"] if c in meta.columns]
    merged = pred.merge(meta[cols], on="audio_path", how="left")
    if "proficiency_level" in merged.columns:
        merged["proficiency_level"] = merged["proficiency_level"].astype(str).str.strip()
    return merged


def slice_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if "proficiency_level" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["proficiency_level"].astype(str).str.strip() == str(level)].copy()


def write_pipeline_flow(cfg: Dict[str, Any]) -> None:
    levels = ", ".join(levels_from_cfg(cfg))
    focus = cfg.get("focus_levels") or "all eligible"
    lines = [
        "# Per-level Error-GT τ：資料流說明",
        "",
        "本實驗**不訓練 NN gate**。Whisper 只在各等級的 error-GT 子集上 fine-tune；",
        "交叉套用與相鄰混合都在凍結的 τ 上做。",
        "",
        "## 要回答的問題",
        "",
        "1. 用和 global 相同的 error-GT 食譜訓出來的 `τ_L`，在金標等級 L 上是否贏過 global τ？",
        "2. 交叉矩陣 `WER(τ_j → 等級 i)` 是不是帶狀／相鄰較好？",
        "3. 固定評定等級 L 時，相鄰凸包 `α_{-1}τ_{L-1}+α_0τ_L+α_{+1}τ_{L+1}` 能否贏過 `τ_L` 或 global？",
        "",
        "若 (3) 沒贏，就不必上 local gate。",
        "",
        "## 設定",
        "",
        f"- 等級：{levels}",
        f"- 相鄰搜尋焦點：{focus}",
        "- 選句：A01 train 的 `conf_error ∪ unc_error`，再依 `proficiency_level` 切開。",
        "- λ / α 只在對應等級的 **A01 dev** 上選。A01 test 與 A02 完全 hold-out。",
        "",
        "## Stage",
        "",
        "```",
        "reuse data-select train_scored（或現場 score）",
        "    ├─[2] 各等級 error-GT CSV",
        "    ├─[3] 各等級 fine-tune",
        "    ├─[4] τ_L = θ_{error,L} − θ_now；與 global / 鄰居算 cosine",
        "    ├─[5] donor bank decode → 交叉矩陣（target-calibrated λ 與 donor λ）",
        "    ├─[6] 焦點等級相鄰 simplex 搜 α、λ",
        "    └─[7] report",
        "```",
        "",
        "## 輸出",
        "",
        f"`{cfg.get('results_dir')}/<model_family>/`",
        "",
        "- `pipeline_flow.md`：本說明",
        "- `level_error_manifest.md`：各等級 error 句數",
        "- `tau_cosine_similarity.csv`",
        "- `cross_apply_{target_calibrated,donor_matched}_<split>.csv` / `.md`",
        "- `adjacent_mix_*.json`",
        "- `experiment_report.md`",
        "",
    ]
    path = results_dir(cfg) / "pipeline_flow.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def _assign_buckets(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    if "bucket" in df.columns and df["bucket"].notna().any():
        return df
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
    return out


def prepare_scored_train(cfg: Dict[str, Any], device: str) -> None:
    dest = scored_train_path(cfg)
    reuse = cfg.get("reuse_scored_train")
    if dest.is_file():
        logger.info("[Stage 1] Skip; found %s", dest)
        return
    if reuse:
        src = resolve_repo_path(str(reuse))
        if src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if bool(cfg.get("refresh_reused_scored_metadata", False)):
                scored = pd.read_csv(src)
                train = pd.read_csv(split_csv(cfg, "train"))
                score_columns = [
                    column
                    for column in scored.columns
                    if column == "audio_path" or column not in train.columns
                ]
                if "audio_path" not in score_columns:
                    raise KeyError(f"Missing audio_path in reused scored train: {src}")
                merged = train.merge(
                    scored[score_columns],
                    on="audio_path",
                    how="inner",
                    validate="one_to_one",
                )
                if len(merged) != len(train):
                    raise ValueError(
                        f"Reused scores cover {len(merged)}/{len(train)} rows from {split_csv(cfg, 'train')}"
                    )
                merged.to_csv(dest, index=False)
                logger.info(
                    "[Stage 1] Reused scores from %s with refreshed metadata -> %s",
                    src,
                    dest,
                )
            else:
                shutil.copy2(src, dest)
                logger.info("[Stage 1] Reused scored train %s -> %s", src, dest)
            return
        logger.warning("reuse_scored_train missing at %s; scoring instead", src)
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    model = load_base_model(cfg, device_t)
    train_path = split_csv(cfg, "train")
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
        desc="score A01 train",
        **decode_kwargs(cfg),
    )
    train_df = pd.read_csv(train_path)
    merged = train_df.merge(scored, on="audio_path", how="inner")
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(dest, index=False)
    logger.info("[Stage 1] Wrote %s (%d rows)", dest, len(merged))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def materialize_level_subsets(cfg: Dict[str, Any]) -> None:
    scored = pd.read_csv(scored_train_path(cfg))
    scored = _assign_buckets(scored, cfg)
    error_buckets = {str(x) for x in cfg.get("error_buckets", ["conf_error", "unc_error"])}
    error = scored[scored["bucket"].astype(str).isin(error_buckets)].copy()
    frames = {split: pd.read_csv(split_csv(cfg, split)) for split in cfg["splits"]}
    rows: List[Dict[str, Any]] = []
    for level in levels_from_cfg(cfg):
        level_dir = data_root(cfg) / "levels" / slugify_level(level)
        level_dir.mkdir(parents=True, exist_ok=True)
        sub_err = error[error["proficiency_level"].astype(str).str.strip() == level].copy()
        sort_cols = [c for c in ["booklet", "global_speaker_id", "utt_id"] if c in sub_err.columns]
        if sort_cols:
            sub_err = sub_err.sort_values(sort_cols)
        sub_err.to_csv(level_train_csv(cfg, level), index=False)
        rec: Dict[str, Any] = {
            "level": level,
            "train_error_utterances": int(len(sub_err)),
            "train_error_speakers": int(sub_err["global_speaker_id"].nunique()) if len(sub_err) and "global_speaker_id" in sub_err.columns else 0,
        }
        for split_name, df in frames.items():
            part = df[df["proficiency_level"].astype(str).str.strip() == level].copy()
            if sort_cols:
                present = [c for c in sort_cols if c in part.columns]
                if present:
                    part = part.sort_values(present)
            part.to_csv(level_split_csv(cfg, split_name, level), index=False)
            rec[f"{split_name}_utterances"] = int(len(part))
        rows.append(rec)
        logger.info("Level %s error-GT train %d utt", level, rec["train_error_utterances"])
    manifest = pd.DataFrame(rows)
    out_csv = results_dir(cfg) / "level_error_manifest.csv"
    manifest.to_csv(out_csv, index=False)
    lines = [
        f"# {cfg.get('display_name', 'Per-level Error-GT')} Manifest",
        "",
        f"- Error buckets: `{', '.join(sorted(error_buckets))}`",
        f"- Scored train utterances: **{len(scored)}**; error utterances: **{len(error)}**",
        "",
        "| " + " | ".join(manifest.columns) + " |",
        "|" + "|".join("---" if c == "level" else "---:" for c in manifest.columns) + "|",
    ]
    for rec in rows:
        vals = []
        for col in manifest.columns:
            val = rec[col]
            vals.append(str(val) if col == "level" else str(int(val)))
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    (results_dir(cfg) / "level_error_manifest.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("[Stage 2] Wrote %s", out_csv)


def train_level_models(cfg: Dict[str, Any]) -> None:
    max_steps = int(cfg.get("max_steps", 2000))
    for level in progress_bar(levels_from_cfg(cfg), total=len(levels_from_cfg(cfg)), desc="level finetune", unit="L"):
        csv_path = level_train_csv(cfg, level)
        n = row_count(csv_path)
        if n < min_subset_utterances(cfg):
            logger.warning("Skip train %s; only %d error-GT rows", level, n)
            continue
        out_dir = level_ckpt_dir(cfg, level)
        if training_completed(out_dir, max_steps):
            logger.info("Skip train %s; completed at %s", level, out_dir)
            continue
        resume = _latest_checkpoint(out_dir)
        logger.info("Fine-tune error-GT %s (%d utt) -> %s", level, n, out_dir)
        finetune_whisper(
            FinetuneConfig(
                model_name_or_path=str(cfg["student_model"]),
                train_csv=str(csv_path),
                output_dir=str(out_dir),
                text_column="text",
                language=str(cfg.get("language", "en")),
                task=str(cfg.get("task", "transcribe")),
                learning_rate=float(cfg.get("learning_rate", 1e-5)),
                weight_decay=float(cfg.get("weight_decay", 0.01)),
                warmup_steps=int(cfg.get("warmup_steps", 200)),
                max_steps=max_steps,
                per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 12)),
                gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
                fp16=bool(cfg.get("fp16", True)),
                bf16=bool(cfg.get("bf16", False)),
                seed=int(cfg.get("seed", 42)),
                dataloader_num_workers=int(cfg.get("num_workers", 4)),
                resume_from_checkpoint=str(resume) if resume else None,
            )
        )


def _dot_tau(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for key, tensor in a.items():
        if key not in b or tensor.shape != b[key].shape:
            continue
        total += float(torch.dot(tensor.float().view(-1), b[key].float().view(-1)))
    return total


def _norm_tau(tau: Dict[str, torch.Tensor]) -> float:
    return _dot_tau(tau, tau) ** 0.5


def compute_taus(cfg: Dict[str, Any]) -> None:
    base_src = str(cfg.get("base_model", cfg["student_model"]))
    for level in levels_from_cfg(cfg):
        csv_path = level_train_csv(cfg, level)
        if row_count(csv_path) < min_subset_utterances(cfg):
            continue
        out = level_tau_path(cfg, level)
        if out.is_file():
            logger.info("Skip tau %s; found %s", level, out)
            continue
        weight_dir = resolve_weight_dir(level_ckpt_dir(cfg, level))
        out.parent.mkdir(parents=True, exist_ok=True)
        logger.info("tau_%s = %s - %s", level, weight_dir, base_src)
        calculate_task_vector(str(weight_dir), base_src, str(out))
    _analyze_tau_cosine(cfg)


def _analyze_tau_cosine(cfg: Dict[str, Any]) -> None:
    names: List[str] = []
    taus: Dict[str, Dict[str, torch.Tensor]] = {}
    if cfg.get("global_tau"):
        gpath = resolve_repo_path(str(cfg["global_tau"]))
        if gpath.is_file():
            names.append(GLOBAL_NAME)
            taus[GLOBAL_NAME] = load_tau(str(gpath))
    for level in levels_from_cfg(cfg):
        path = level_tau_path(cfg, level)
        if path.is_file():
            names.append(slugify_level(level))
            taus[slugify_level(level)] = load_tau(str(path))
    if len(names) < 2:
        logger.warning("Need at least two taus for cosine analysis")
        return
    norms = {name: _norm_tau(tau) for name, tau in taus.items()}
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    pairs: List[Dict[str, Any]] = []
    for i, ni in enumerate(names):
        for nj in names[i:]:
            denom = norms[ni] * norms[nj]
            cos = 0.0 if denom == 0 else _dot_tau(taus[ni], taus[nj]) / denom
            matrix.loc[ni, nj] = cos
            matrix.loc[nj, ni] = cos
            if ni != nj:
                pairs.append({"tau_a": ni, "tau_b": nj, "cosine": float(cos)})
    out_dir = results_dir(cfg)
    matrix.to_csv(out_dir / "tau_cosine_similarity.csv")
    pd.DataFrame(pairs).to_csv(out_dir / "tau_cosine_pairs.csv", index=False)
    logger.info("Wrote tau cosine tables -> %s", out_dir)


def _load_donor_tau(cfg: Dict[str, Any], donor: str, cache: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if donor == PRETRAINED_NAME:
        raise ValueError("pretrained donor has no tau")
    if donor in cache:
        return cache[donor]
    if donor == GLOBAL_NAME:
        path = resolve_repo_path(str(cfg["global_tau"]))
    else:
        path = level_tau_path(cfg, donor)
        if not path.is_file():
            path = vector_root(cfg) / f"tau_{donor}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing tau for donor {donor}: {path}")
    cache[donor] = load_tau(str(path))
    return cache[donor]


def _decode_csv(cfg: Dict[str, Any], model, processor, csv_path: Path, dest: Path, desc: str) -> Dict[str, float]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    loader = build_eval_dataloader(
        str(csv_path),
        processor=processor,
        batch_size=int(cfg.get("eval_batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 2)),
        language=str(cfg.get("language", "en")),
    )
    return evaluate_model(model, loader, processor, output_csv_path=str(dest), desc=desc, **decode_kwargs(cfg))


def _available_level_donors(cfg: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for level in levels_from_cfg(cfg):
        if level_tau_path(cfg, level).is_file():
            names.append(slugify_level(level))
    return names


def _all_donors(cfg: Dict[str, Any]) -> List[str]:
    names = [PRETRAINED_NAME]
    if cfg.get("global_tau") and resolve_repo_path(str(cfg["global_tau"])).is_file():
        names.append(GLOBAL_NAME)
    names.extend(_available_level_donors(cfg))
    return names


def _apply_donor(cfg: Dict[str, Any], model, donor: str, lam: float, cache: Dict[str, Dict[str, torch.Tensor]]) -> None:
    if donor == PRETRAINED_NAME or abs(float(lam)) < 1e-12:
        return
    apply_task_vector(model, _load_donor_tau(cfg, donor, cache), float(lam))


def decode_dev_bank(cfg: Dict[str, Any], device: str) -> None:
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    model = load_base_model(cfg, device_t)
    base_state = copy.deepcopy(model.state_dict())
    tau_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    donors = _all_donors(cfg)
    lambdas = lambda_list(cfg)
    jobs: List[Tuple[str, float]] = [(PRETRAINED_NAME, 0.0)]
    for donor in donors:
        if donor == PRETRAINED_NAME:
            continue
        for lam in lambdas:
            jobs.append((donor, float(lam)))
    csv_path = split_csv(cfg, "dev")
    for donor, lam in progress_bar(jobs, total=len(jobs), desc="dev bank", unit="τλ"):
        dest = bank_pred_path(cfg, "dev", donor, lam)
        if dest.is_file():
            continue
        model.load_state_dict(base_state, strict=True)
        _apply_donor(cfg, model, donor, lam, tau_cache)
        _decode_csv(cfg, model, processor, csv_path, dest, desc=f"dev:{donor}:λ{lam:.2f}")
    model.load_state_dict(base_state, strict=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _wer_for_level(pred_path: Path, meta: pd.DataFrame, level: Optional[str]) -> Dict[str, float]:
    if not pred_path.is_file():
        return {}
    pred = attach_level(pd.read_csv(pred_path), meta)
    if level is None:
        return metrics_from_pred_df(pred)
    return metrics_from_pred_df(slice_level(pred, level))


def select_cross_lambdas(cfg: Dict[str, Any]) -> Dict[str, Any]:
    levels = levels_from_cfg(cfg)
    donors = _all_donors(cfg)
    lambdas = lambda_list(cfg)
    meta = pd.read_csv(split_csv(cfg, "dev"))
    records: List[Dict[str, Any]] = []
    selection: Dict[str, Any] = {"target_calibrated": {}, "donor_matched": {}, "lambda_list": lambdas}

    for donor in donors:
        donor_lams = [0.0] if donor == PRETRAINED_NAME else lambdas
        best_overall = None
        for lam in donor_lams:
            metrics = _wer_for_level(bank_pred_path(cfg, "dev", donor, lam), meta, None)
            if not metrics:
                continue
            rec = {"eval_level": "ALL", "donor": donor, "lambda": float(lam), **metrics, "protocol": "overall"}
            records.append(rec)
            if best_overall is None or metrics["wer"] < best_overall["wer"]:
                best_overall = {"lambda": float(lam), "wer": float(metrics["wer"])}
        if best_overall is not None:
            selection["donor_matched"][donor] = {**best_overall, "scope": "full_dev"}
        for level in levels:
            best = None
            for lam in donor_lams:
                metrics = _wer_for_level(bank_pred_path(cfg, "dev", donor, lam), meta, level)
                if not metrics or metrics["n_utterances"] <= 0:
                    continue
                rec = {"eval_level": level, "donor": donor, "lambda": float(lam), **metrics, "protocol": "per_level"}
                records.append(rec)
                if best is None or metrics["wer"] < best["wer"]:
                    best = {"lambda": float(lam), "wer": float(metrics["wer"]), "n_utterances": float(metrics["n_utterances"])}
            if best is not None:
                selection["target_calibrated"].setdefault(level, {})[donor] = best

    # Donor-matched per level: λ of τ_L on its own bucket (or global's published λ if provided).
    for level in levels:
        donor = slugify_level(level)
        if donor in selection["target_calibrated"].get(level, {}):
            selection["donor_matched"][donor] = {
                **selection["target_calibrated"][level][donor],
                "scope": f"level:{level}",
            }
    if GLOBAL_NAME in _all_donors(cfg):
        gjson = cfg.get("global_lambda_json")
        published = None
        if gjson:
            gpath = resolve_repo_path(str(gjson))
            if gpath.is_file():
                payload = json.loads(gpath.read_text(encoding="utf-8"))
                published = float(payload.get("signed_lambda", payload.get("best_lambda", 1.0)))
        overall = selection["donor_matched"].get(GLOBAL_NAME, {})
        selection["donor_matched"][GLOBAL_NAME] = {
            "lambda": published if published is not None else float(overall.get("lambda", 1.0)),
            "wer": overall.get("wer"),
            "scope": "published" if published is not None else "full_dev",
        }

    curve_path = results_dir(cfg) / "cross_apply_dev_curve.csv"
    pd.DataFrame(records).to_csv(curve_path, index=False)
    dump_json(results_dir(cfg) / "cross_lambda_selection.json", selection)
    logger.info("[Stage 5] Wrote lambda selection -> %s", results_dir(cfg) / "cross_lambda_selection.json")
    return selection


def _needed_eval_jobs(cfg: Dict[str, Any], selection: Dict[str, Any]) -> List[Tuple[str, float]]:
    jobs: Set[Tuple[str, float]] = {(PRETRAINED_NAME, 0.0)}
    for donor, payload in selection.get("donor_matched", {}).items():
        jobs.add((donor, float(payload["lambda"])))
    for _level, donors in selection.get("target_calibrated", {}).items():
        for donor, payload in donors.items():
            jobs.add((donor, float(payload["lambda"])))
    return sorted(jobs, key=lambda x: (x[0], x[1]))


def decode_eval_bank(cfg: Dict[str, Any], device: str, selection: Dict[str, Any]) -> None:
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    model = load_base_model(cfg, device_t)
    base_state = copy.deepcopy(model.state_dict())
    tau_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    jobs = _needed_eval_jobs(cfg, selection)
    for split_key, _ in eval_splits_from_cfg(cfg):
        csv_path = split_csv(cfg, split_key)
        for donor, lam in progress_bar(jobs, total=len(jobs), desc=f"eval bank {split_key}", unit="τλ"):
            dest = bank_pred_path(cfg, split_key, donor, lam)
            if dest.is_file():
                continue
            model.load_state_dict(base_state, strict=True)
            _apply_donor(cfg, model, donor, lam, tau_cache)
            _decode_csv(cfg, model, processor, csv_path, dest, desc=f"{split_key}:{donor}:λ{lam:.2f}")
    model.load_state_dict(base_state, strict=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _matrix_rows(cfg: Dict[str, Any], split_key: str, selection: Dict[str, Any], protocol: str) -> pd.DataFrame:
    levels = levels_from_cfg(cfg)
    donors = _all_donors(cfg)
    meta = pd.read_csv(split_csv(cfg, split_key))
    rows: List[Dict[str, Any]] = []
    for level in levels:
        for donor in donors:
            if protocol == "target_calibrated":
                payload = selection.get("target_calibrated", {}).get(level, {}).get(donor)
                if not payload:
                    continue
                lam = float(payload["lambda"])
            elif protocol == "donor_matched":
                payload = selection.get("donor_matched", {}).get(donor)
                if not payload:
                    continue
                lam = float(payload["lambda"])
            else:
                raise ValueError(protocol)
            metrics = _wer_for_level(bank_pred_path(cfg, split_key, donor, lam), meta, level)
            if not metrics:
                continue
            rows.append(
                {
                    "split": split_key,
                    "protocol": protocol,
                    "eval_level": level,
                    "donor": donor,
                    "lambda": lam,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def write_cross_matrices(cfg: Dict[str, Any], selection: Dict[str, Any]) -> None:
    levels = levels_from_cfg(cfg)
    splits = [("dev", "A01 Internal Dev"), *eval_splits_from_cfg(cfg)]
    all_rows: List[pd.DataFrame] = []
    for split_key, label in splits:
        for protocol in ("target_calibrated", "donor_matched"):
            df = _matrix_rows(cfg, split_key, selection, protocol)
            if df.empty:
                continue
            all_rows.append(df)
            wide = df.pivot_table(index="eval_level", columns="donor", values="wer", aggfunc="first")
            ordered_idx = [lv for lv in levels if lv in wide.index]
            ordered_col = [c for c in _all_donors(cfg) if c in wide.columns]
            wide = wide.reindex(index=ordered_idx, columns=ordered_col)
            csv_path = results_dir(cfg) / f"cross_apply_{protocol}_{split_key}.csv"
            wide.to_csv(csv_path)
            md = [
                f"# Cross-apply WER ({label}, {protocol})",
                "",
                "Rows = gold eval level. Columns = donor τ. Values = WER (%).",
                "",
                "| eval \\ donor | " + " | ".join(donor_label(c, levels) for c in wide.columns) + " |",
                "|---|" + "|".join("---:" for _ in wide.columns) + "|",
            ]
            for level, row in wide.iterrows():
                cells = [str(level)]
                for donor in wide.columns:
                    val = row[donor]
                    cells.append("—" if pd.isna(val) else f"{float(val):.2f}")
                md.append("| " + " | ".join(cells) + " |")
            md.append("")
            (results_dir(cfg) / f"cross_apply_{protocol}_{split_key}.md").write_text("\n".join(md), encoding="utf-8")
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(results_dir(cfg) / "cross_apply_long.csv", index=False)


def run_cross_apply(cfg: Dict[str, Any], device: str) -> None:
    decode_dev_bank(cfg, device)
    selection = select_cross_lambdas(cfg)
    decode_eval_bank(cfg, device, selection)
    write_cross_matrices(cfg, selection)


def _focus_levels(cfg: Dict[str, Any]) -> List[str]:
    raw = cfg.get("focus_levels")
    levels = levels_from_cfg(cfg)
    if not raw:
        return levels
    wanted = [str(x) for x in raw]
    return [lv for lv in levels if lv in wanted]


def _available_neighbors(cfg: Dict[str, Any], level: str) -> List[str]:
    names: List[str] = []
    for nb in neighbors_of(level, levels_from_cfg(cfg)):
        if level_tau_path(cfg, nb).is_file():
            names.append(nb)
    return names


def search_adjacent_mixes(cfg: Dict[str, Any], device: str) -> None:
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    model = load_base_model(cfg, device_t)
    base_state = copy.deepcopy(model.state_dict())
    tau_cache: Dict[str, Dict[str, torch.Tensor]] = {}
    step = float(cfg.get("adjacent_alpha_step", 0.25))
    min_dev = int(cfg.get("min_level_dev_utterances", 16))
    lambdas = adjacent_lambda_list(cfg)
    summaries: List[Dict[str, Any]] = []

    for level in _focus_levels(cfg):
        neighbors = _available_neighbors(cfg, level)
        dev_csv = level_split_csv(cfg, "dev", level)
        n_dev = row_count(dev_csv)
        if len(neighbors) < 2:
            logger.warning("Skip adjacent mix %s; need ≥2 neighbor taus, got %s", level, neighbors)
            continue
        if n_dev < min_dev:
            logger.warning("Skip adjacent mix %s; only %d dev utt", level, n_dev)
            continue
        grid = _simplex_grid(len(neighbors), step)
        curve_rows: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        jobs = [(alpha, lam) for alpha in grid for lam in lambdas]
        cache_dir = data_root(cfg) / "_adjacent" / slugify_level(level)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for alpha, lam in progress_bar(jobs, total=len(jobs), desc=f"adjacent {level}", unit="αλ"):
            alpha_tag = "_".join(f"{a:.2f}" for a in alpha)
            dest = cache_dir / f"dev_lam{lam:.2f}_a{alpha_tag}.csv"
            if dest.is_file():
                metrics = metrics_from_pred_df(pd.read_csv(dest))
            else:
                model.load_state_dict(base_state, strict=True)
                taus = [_load_donor_tau(cfg, slugify_level(nb), tau_cache) for nb in neighbors]
                mixed = mix_task_vectors(taus, list(alpha))
                apply_task_vector(model, mixed, float(lam))
                metrics = _decode_csv(cfg, model, processor, dev_csv, dest, desc=f"adj-dev:{level}")
            row = {
                "level": level,
                "lambda": float(lam),
                "wer": float(metrics["wer"]),
                **{f"alpha_{slugify_level(nb)}": float(a) for nb, a in zip(neighbors, alpha)},
            }
            curve_rows.append(row)
            if best is None or metrics["wer"] < float(best["dev_wer"]):
                best = {
                    "level": level,
                    "neighbors": neighbors,
                    "alpha": {slugify_level(nb): float(a) for nb, a in zip(neighbors, alpha)},
                    "lambda": float(lam),
                    "dev_wer": float(metrics["wer"]),
                    "dev_utterances": n_dev,
                }
        pd.DataFrame(curve_rows).to_csv(results_dir(cfg) / f"adjacent_curve_{slugify_level(level)}.csv", index=False)
        if best is None:
            continue
        dump_json(results_dir(cfg) / f"best_adjacent_{slugify_level(level)}.json", best)
        # Eval the winner on this level's held-out slices.
        taus = [_load_donor_tau(cfg, slugify_level(nb), tau_cache) for nb in neighbors]
        alpha_vals = [best["alpha"][slugify_level(nb)] for nb in neighbors]
        mixed = mix_task_vectors(taus, alpha_vals)
        split_metrics: Dict[str, Dict[str, float]] = {"dev": {"wer": best["dev_wer"]}}
        for split_key, _ in eval_splits_from_cfg(cfg):
            dest = adjacent_pred_path(cfg, split_key, level)
            csv_path = level_split_csv(cfg, split_key, level)
            if dest.is_file():
                metrics = metrics_from_pred_df(pd.read_csv(dest))
            else:
                model.load_state_dict(base_state, strict=True)
                apply_task_vector(model, mixed, float(best["lambda"]))
                metrics = _decode_csv(cfg, model, processor, csv_path, dest, desc=f"adj-{split_key}:{level}")
            split_metrics[split_key] = metrics
            best[f"{split_key}_wer"] = float(metrics["wer"])
            best[f"{split_key}_sub"] = float(metrics["sub"])
            best[f"{split_key}_ins"] = float(metrics["ins"])
            best[f"{split_key}_del"] = float(metrics["del"])
            best[f"{split_key}_utterances"] = float(metrics["n_utterances"])
        dump_json(results_dir(cfg) / f"best_adjacent_{slugify_level(level)}.json", best)
        summaries.append(best)
        logger.info(
            "Adjacent %s λ=%.2f α=%s dev WER=%.2f",
            level,
            best["lambda"],
            best["alpha"],
            best["dev_wer"],
        )

    if summaries:
        dump_json(results_dir(cfg) / "adjacent_mix_summary.json", {"levels": summaries})
    model.load_state_dict(base_state, strict=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _lookup_cross_wer(long_df: pd.DataFrame, split_key: str, protocol: str, level: str, donor: str) -> Optional[float]:
    if long_df.empty:
        return None
    sub = long_df[
        (long_df["split"] == split_key)
        & (long_df["protocol"] == protocol)
        & (long_df["eval_level"] == level)
        & (long_df["donor"] == donor)
    ]
    if sub.empty:
        return None
    return float(sub.iloc[0]["wer"])


def write_experiment_report(cfg: Dict[str, Any]) -> None:
    levels = levels_from_cfg(cfg)
    long_path = results_dir(cfg) / "cross_apply_long.csv"
    long_df = pd.read_csv(long_path) if long_path.is_file() else pd.DataFrame()
    selection_path = results_dir(cfg) / "cross_lambda_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {}
    lines = [
        f"# Experiment Report ({cfg.get('display_name', 'Per-level Error-GT Tau')})",
        "",
        "## Scope",
        "",
        "- Per-level `τ_L` is trained on that level's A01-train **error-GT** subset (same recipe as global data-select).",
        "- Cross-apply: gold eval level *i* vs donor τ *j*. λ is never taken from A01 test or A02.",
        "- `target_calibrated`: λ_{i←j} chosen on **level-i A01 dev**.",
        "- `donor_matched`: each donor keeps the λ chosen on its own bucket / published global λ.",
        "- Adjacent mix: for a fixed assessment level L, search α on the neighbor simplex using **level-L A01 dev**.",
        f"- θ_now / student: `{cfg.get('base_model', cfg['student_model'])}`",
        "",
        "## How to read the matrices",
        "",
        "- If **Global** wins every row: local L2 directions are not buying WER; keep global + data-select.",
        "- If the **diagonal** (`τ_i` on level *i*) wins: matched specialists help, neighbors are unnecessary.",
        "- If a **neighbor** of *i* beats `τ_i`: the assessment label is not the best correction space — this is the gate hypothesis.",
        "- Adjacent mix should beat both `τ_L` and Global on that row before a local NN router is worth training.",
        "",
    ]

    manifest_md = results_dir(cfg) / "level_error_manifest.md"
    if manifest_md.is_file():
        lines.extend(["## Level error-GT counts", "", manifest_md.read_text(encoding="utf-8"), ""])

    cosine_path = results_dir(cfg) / "tau_cosine_similarity.csv"
    if cosine_path.is_file():
        lines.extend(["## Tau cosine similarity", "", f"See `{cosine_path.name}`.", ""])

    donors = _all_donors(cfg)
    for split_key, label in [("dev", "A01 Internal Dev"), *eval_splits_from_cfg(cfg)]:
        for protocol, proto_label in (
            ("target_calibrated", "target-calibrated λ (chosen on that level's A01 dev)"),
            ("donor_matched", "donor-matched λ"),
        ):
            lines.extend(
                [
                    f"## {label} — {proto_label}",
                    "",
                    "| Proficiency | " + " | ".join(donor_label(d, levels) for d in donors) + " | Best donor | Adjacent mix |",
                    "|---|" + "|".join("---:" for _ in donors) + "|---|---:|",
                ]
            )
            for level in levels:
                cells = [level]
                best_name = None
                best_wer = None
                for donor in donors:
                    wer = _lookup_cross_wer(long_df, split_key, protocol, level, donor)
                    cells.append("—" if wer is None else f"{wer:.2f}")
                    if wer is not None and (best_wer is None or wer < best_wer):
                        best_wer = wer
                        best_name = donor_label(donor, levels)
                adj_path = results_dir(cfg) / f"best_adjacent_{slugify_level(level)}.json"
                adj_cell = "—"
                if adj_path.is_file():
                    payload = json.loads(adj_path.read_text(encoding="utf-8"))
                    key = "dev_wer" if split_key == "dev" else f"{split_key}_wer"
                    if key in payload:
                        adj_cell = f"{float(payload[key]):.2f}"
                cells.append(best_name or "—")
                cells.append(adj_cell)
                lines.append("| " + " | ".join(cells) + " |")
            lines.append("")

    lines.extend(["## Adjacent mix (focus levels)", ""])
    focus = _focus_levels(cfg)
    if not focus:
        lines.append("No focus levels configured.")
    else:
        lines.extend(
            [
                "| Level | Neighbors | α | λ | Dev WER | A01 Test WER | A02 WER |",
                "|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for level in focus:
            path = results_dir(cfg) / f"best_adjacent_{slugify_level(level)}.json"
            if not path.is_file():
                lines.append(f"| {level} | — | — | — | — | — | — |")
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            alpha = payload.get("alpha", {})
            alpha_s = ", ".join(f"{k}={v:.2f}" for k, v in alpha.items())
            neighbors = ", ".join(payload.get("neighbors", []))
            test_wer = payload.get("test_wer", float("nan"))
            a02_wer = payload.get("test_a02_external_wer", float("nan"))
            lines.append(
                f"| {level} | {neighbors} | {alpha_s} | {float(payload.get('lambda', float('nan'))):.2f} | "
                f"{float(payload.get('dev_wer', float('nan'))):.2f} | "
                f"{float(test_wer):.2f} | {float(a02_wer):.2f} |"
            )
        lines.append("")

    if selection.get("donor_matched"):
        lines.extend(["## Selected donor λ", "", "| Donor | λ | Scope | Dev WER |", "|---|---:|---|---:|"])
        for donor, payload in selection["donor_matched"].items():
            wer = payload.get("wer")
            wer_s = "—" if wer is None else f"{float(wer):.2f}"
            lines.append(
                f"| {donor_label(donor, levels)} | {float(payload.get('lambda', float('nan'))):.2f} | "
                f"{payload.get('scope', '')} | {wer_s} |"
            )
        lines.append("")

    report_path = results_dir(cfg) / "experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report -> %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-level error-GT tau, cross-apply, adjacent mix")
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
    stages = parse_stages(args.stage)
    titles = stage_titles()
    logger.info("Config=%s stages=%s device=%s", args.config, sorted(stages), args.device)

    with stage_progress(sorted(stages), desc=cfg.get("display_name", "Per-level Error-GT Tau")) as sp:
        if 1 in stages:
            sp.begin(1, titles[1])
            write_pipeline_flow(cfg)
            prepare_scored_train(cfg, device=args.device)
            sp.end(1)
        if 2 in stages:
            sp.begin(2, titles[2])
            materialize_level_subsets(cfg)
            sp.end(2)
        if 3 in stages:
            sp.begin(3, titles[3])
            train_level_models(cfg)
            sp.end(3)
        if 4 in stages:
            sp.begin(4, titles[4])
            compute_taus(cfg)
            sp.end(4)
        if 5 in stages:
            sp.begin(5, titles[5])
            run_cross_apply(cfg, device=args.device)
            sp.end(5)
        if 6 in stages:
            sp.begin(6, titles[6])
            search_adjacent_mixes(cfg, device=args.device)
            sp.end(6)
        if 7 in stages:
            sp.begin(7, titles[7])
            write_experiment_report(cfg)
            sp.end(7)


if __name__ == "__main__":
    main()
