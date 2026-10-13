#!/usr/bin/env python3
"""Smoothed proficiency correction vectors on frozen per-level τ.

No new Whisper fine-tune. Reuses error-GT τ_g and per-level error-GT
checkpoints (τ_n = θ_{error,n} − θ_now).

    τ̃ = β Σ_n α_n τ_n + (1 − β) τ_g
    θ' = θ_now + λ τ̃

α_n comes from cosine attention over train CEFR prototype embeddings.
Gold CEFR is not an inference input.

Stages:
1. Materialize τ_n, residuals, speaker attention α.
2. Select β, λ on A01 dev.
3. Evaluate A01 test + A02; write the report.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr_backend import family_slug, load_asr_model, load_processor as load_asr_processor  # noqa: E402
from src.audio_utils import ensure_device  # noqa: E402
from src.evaluate import build_eval_dataloader, evaluate_model, write_eval_detailed_json  # noqa: E402
from src.progress import configure_experiment_progress, progress_bar, stage_progress  # noqa: E402
from src.prototype_gate import (  # noqa: E402
    PrototypeCrossAttnGate,
    build_gate,
    gate_predictions,
    save_gate,
    train_gate_ce,
)
from src.task_vector import (  # noqa: E402
    apply_task_vector,
    calculate_task_vector,
    load_tau,
    mix_task_vectors,
    orthogonal_residual,
    project_tau_onto_basis,
    svd_subspace_from_taus,
    tau_dot,
    tau_norm_sq,
)

logger = logging.getLogger("teemi_a01_smooth_tau")

SPEAKER_COL = "global_speaker_id"
PROFICIENCY_ORDER = {
    "pre-A": 0,
    "A1": 1,
    "A1A2": 2,
    "A2": 3,
    "A2B1": 4,
    "B1": 5,
    "B1B2": 6,
    "B2": 7,
}

METHOD_ORDER = [
    "theta_pre",
    "theta_ft",
    "global_error_add",
    "uniform_smooth",
    "attn_hard_smooth",
    "attn_soft_smooth",
    "pgcv",
    "residual_soft",
    "svd1_soft",
    "svd2_soft",
]
METHOD_LABEL = {
    "theta_pre": "Pretrained (no tau)",
    "theta_ft": "Full fine-tuning",
    "global_error_add": "Global error-GT τ",
    "uniform_smooth": "Uniform soup: β (1/N) Σ τ_n + (1−β) τ_g",
    "attn_hard_smooth": "Hard attn: β τ_{k*} + (1−β) τ_g",
    "attn_soft_smooth": "Soft attn: β Σ α_n τ_n + (1−β) τ_g",
    "pgcv": "PGCV",
    "residual_soft": "Residual: τ_g + β Σ α_n τ_n^⊥",
    "svd1_soft": "SVD-1: τ_g + β Σ α_n τ̃_n^(1)",
    "svd2_soft": "SVD-2: τ_g + β Σ α_n τ̃_n^(2)",
}

DEFAULT_BETA_LIST = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_LAMBDA_LIST = [0.25, 0.5, 0.7, 0.75, 1.0]
PRED_COLUMNS = ("audio_path", "hits", "substitutions", "deletions", "insertions")
LEARNED_ATTENTION_MODES = {"learned", "learned_cefr", "cefr", "cefr_supervised"}


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
    available = {1, 2, 3}
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
        1: "Compute τ_n, residuals, cosine attention α",
        2: "Select β / λ on A01 dev",
        3: "Evaluate A01 test + A02 and write report",
    }


def family_root(cfg: Dict[str, Any], key: str) -> Path:
    return resolve_repo_path(cfg[key]) / family_slug(cfg["student_model"])


def split_csv(cfg: Dict[str, Any], split_key: str) -> Path:
    return resolve_repo_path(cfg["splits"][split_key])


def data_root(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "data_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "results_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def vector_root(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "vector_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_root(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "checkpoint_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def embedding_dir(cfg: Dict[str, Any]) -> Path:
    raw = cfg.get("reuse_embedding_dir")
    if raw:
        path = resolve_repo_path(str(raw))
        if path.is_dir():
            return path
        logger.warning("reuse_embedding_dir missing (%s)", path)
    return data_root(cfg)


def level_ckpt_root(cfg: Dict[str, Any]) -> Path:
    return family_root(cfg, "level_checkpoint_dir")


def level_vector_fallback_root(cfg: Dict[str, Any]) -> Path:
    return family_root(cfg, "level_vector_dir")


def slugify_level(level: str) -> str:
    text = level.lower().replace("+", "_plus")
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def cfg_levels(cfg: Dict[str, Any]) -> List[str]:
    return [str(x) for x in cfg.get("levels", list(PROFICIENCY_ORDER))]


def gate_id_column(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("gate_id_column", SPEAKER_COL))


def search_methods(cfg: Dict[str, Any]) -> List[str]:
    default = ["uniform_smooth", "attn_hard_smooth", "attn_soft_smooth", "residual_soft"]
    return [str(x) for x in cfg.get("search_methods", default)]


def report_methods(cfg: Dict[str, Any]) -> List[str]:
    default = ["theta_pre", "global_error_add", *search_methods(cfg)]
    return [str(x) for x in cfg.get("report_methods", default)]


def float_list(cfg: Dict[str, Any], key: str, default: Sequence[float]) -> List[float]:
    return [float(x) for x in cfg.get(key, default)]


def eval_splits_from_cfg(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    return [(str(item["key"]), str(item.get("label", item["key"]))) for item in cfg.get("eval_splits", [])]


def all_split_keys(cfg: Dict[str, Any]) -> List[str]:
    keys = ["train", "dev"]
    for key, _ in eval_splits_from_cfg(cfg):
        if key not in keys:
            keys.append(key)
    return keys


def decode_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "language": str(cfg.get("language", "en")),
        "task": str(cfg.get("task", "transcribe")),
        "max_new_tokens": int(cfg.get("max_new_tokens", 256)),
    }


def global_tau_path(cfg: Dict[str, Any]) -> Path:
    return resolve_repo_path(str(cfg["global_tau"]))


def level_tau_path(cfg: Dict[str, Any], level: str) -> Path:
    return vector_root(cfg) / f"tau_{slugify_level(level)}.pt"


def residual_tau_path(cfg: Dict[str, Any], level: str) -> Path:
    return vector_root(cfg) / f"tau_{slugify_level(level)}_perp_global.pt"


def svd_tau_path(cfg: Dict[str, Any], level: str, rank: int) -> Path:
    return vector_root(cfg) / f"tau_{slugify_level(level)}_svd{int(rank)}.pt"


def svd_rank_of_method(method: str) -> Optional[int]:
    if str(method).startswith("svd1"):
        return 1
    if str(method).startswith("svd2"):
        return 2
    return None


def needed_svd_ranks(cfg: Dict[str, Any]) -> List[int]:
    ranks = set()
    for method in [*search_methods(cfg), *report_methods(cfg)]:
        rank = svd_rank_of_method(method)
        if rank is not None:
            ranks.add(rank)
    return sorted(ranks)


def family_reuse_root(cfg: Dict[str, Any], key: str) -> Optional[Path]:
    raw = cfg.get(key)
    if not raw:
        return None
    return resolve_repo_path(str(raw)) / family_slug(cfg["student_model"])


def speaker_emb_path(cfg: Dict[str, Any], split_key: str) -> Path:
    return embedding_dir(cfg) / f"speaker_embeddings_{split_key}.npz"


def require_speaker_embeddings(cfg: Dict[str, Any]) -> None:
    missing = [speaker_emb_path(cfg, key) for key in all_split_keys(cfg) if not speaker_emb_path(cfg, key).is_file()]
    if not missing:
        return
    listed = "\n  ".join(str(path) for path in missing)
    raise FileNotFoundError(
        "Missing speaker embeddings:\n  "
        f"{listed}\n"
        "Extract them first (cached files are skipped):\n"
        "  bash run_teemi_a01_only_pgcv_qwen3_asr_0.6b.sh embeddings 2 0\n"
        "  bash run_teemi_a01_only_pgcv_qwen3_asr_0.6b.sh pgcv all 0"
    )


def assignment_path(cfg: Dict[str, Any], split_key: str) -> Path:
    return data_root(cfg) / f"assignment_{split_key}.csv"


def _lambda_json(cfg: Dict[str, Any], method: str) -> Path:
    return results_dir(cfg) / f"best_{method}.json"


def _prediction_csv(cfg: Dict[str, Any], split_key: str, method: str) -> Path:
    return results_dir(cfg) / f"predictions_{split_key}_{method}.csv"


def _pred_is_complete(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 32:
        return False
    try:
        frame = pd.read_csv(path, usecols=list(PRED_COLUMNS))
    except Exception:
        return False
    return len(frame) > 0 and set(PRED_COLUMNS).issubset(frame.columns)


def _pred_row_count(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(pd.read_csv(path, usecols=["audio_path"]))
    except Exception:
        return 0


def _dev_combo_dest(tmp_root: Path, method: str, beta: float, lam: float) -> Path:
    return tmp_root / f"dev_{method}_b{beta:.2f}_l{lam:.2f}.csv"


def _eval_combo_dest(tmp_root: Path, split_key: str, method: str, beta: float, lam: float) -> Path:
    return tmp_root / f"{split_key}_{method}_b{beta:.2f}_l{lam:.2f}.csv"


def _stage1_complete(cfg: Dict[str, Any]) -> bool:
    if not global_tau_path(cfg).is_file():
        return False
    for level in cfg_levels(cfg):
        if not level_tau_path(cfg, level).is_file():
            return False
        if not residual_tau_path(cfg, level).is_file():
            return False
        for rank in needed_svd_ranks(cfg):
            if not svd_tau_path(cfg, level, rank).is_file():
                return False
    attention_mode = str(cfg.get("attention_mode", "cosine")).strip().lower()
    if attention_mode in LEARNED_ATTENTION_MODES and not (checkpoint_root(cfg) / "cefr_gate.pt").is_file():
        return False
    return all(assignment_path(cfg, split_key).is_file() for split_key in all_split_keys(cfg))


def _seed_eval_from_canonical(
    cfg: Dict[str, Any],
    split_key: str,
    method: str,
    beta: float,
    lam: float,
    tagged: Path,
) -> None:
    if _pred_is_complete(tagged):
        return
    canonical = _prediction_csv(cfg, split_key, method)
    summary_path = results_dir(cfg) / "eval_summary.json"
    if not _pred_is_complete(canonical) or not summary_path.is_file():
        return
    try:
        spec = ((json.loads(summary_path.read_text(encoding="utf-8")).get(split_key) or {}).get(method) or {})
        prev_beta = float(spec.get("beta", float("nan")))
        prev_lam = float(spec.get("lambda", float("nan")))
    except Exception:
        return
    if not (np.isclose(prev_beta, beta) and np.isclose(prev_lam, lam)):
        return
    tagged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(canonical, tagged)
    logger.info("skip eval %s/%s β=%.2f λ=%.2f (reuse %s)", split_key, method, beta, lam, canonical.name)


def _latest_checkpoint(output_dir: Path) -> Optional[Path]:
    if not output_dir.is_dir():
        return None
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))


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


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def published_global_lambda(cfg: Dict[str, Any]) -> float:
    if "published_global_lambda" in cfg:
        return float(cfg["published_global_lambda"])
    raw = cfg.get("published_global_lambda_json")
    if not raw:
        return 1.0
    path = resolve_repo_path(str(raw))
    if not path.is_file():
        return 1.0
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload.get("signed_lambda", payload.get("best_lambda", 1.0)))


def cosine_tau(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    denom = (tau_norm_sq(a) ** 0.5) * (tau_norm_sq(b) ** 0.5)
    if denom <= 1e-12:
        return 0.0
    return float(tau_dot(a, b) / denom)


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


def _fmt(value: Optional[float], digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.clip(exp.sum(axis=axis, keepdims=True), 1e-12, None)


def _entropy(alpha: np.ndarray) -> np.ndarray:
    clipped = np.clip(alpha, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def load_spk_emb(path: Path) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=True)
    speakers = payload["speaker"].tolist()
    matrix = np.asarray(payload["matrix"], dtype=np.float32)
    n_utts = payload["n_utterances"].tolist()
    levels = payload["proficiency_level"].tolist() if "proficiency_level" in payload.files else [""] * len(speakers)
    return pd.DataFrame(
        {
            SPEAKER_COL: [str(s) for s in speakers],
            "n_utterances": [int(n) for n in n_utts],
            "embedding": list(matrix),
            "proficiency_level": [str(lv) for lv in levels],
        }
    )


def write_pipeline_flow(cfg: Dict[str, Any]) -> None:
    names = ", ".join(cfg_levels(cfg))
    attention_mode = str(cfg.get("attention_mode", "cosine")).strip().lower()
    learned = attention_mode in {"learned", "learned_cefr", "cefr", "cefr_supervised"}
    alpha_formula = "α = softmax(g_φ(x))" if learned else "α_n = softmax( cos(x, p_n) / T )"
    attention_text = (
        "Train CEFR supervision learns a small gate; gold CEFR is not a test-time input."
        if learned
        else "Gold CEFR is not an inference input; it only constructs train-side prototype keys."
    )
    lines = [
        "# PGCV：資料流說明",
        "",
        "本實驗凍結 ASR backbone、global τ_g 與各程度 error-GT τ_n。",
        attention_text,
        "",
        "## 公式",
        "",
        "```",
        alpha_formula,
        "τ̃  = β Σ_n α_n τ_n + (1 − β) τ_g",
        "θ'  = θ_now + λ τ̃",
        "```",
        "",
        "對照：",
        "",
        "- uniform：α_n = 1/N（沒有 router）",
        "- hard：k* = argmax α，只用 τ_{k*}",
        "- residual：θ' = θ_now + λ (τ_g + β Σ_n α_n τ_n^⊥)",
        "- SVD-k：先對 {τ_n^⊥} 做 Gram SVD，再用前 k 軸重建殘差後做 soft mix",
        "",
        f"程度銀行：{names}",
        f"Attention mode = {attention_mode}",
        "",
        f"β、λ 只在 {cfg.get('selection_split_label', 'A01 dev')} 上選；evaluation splits 不參與。",
        "成功標準：任一 smoothing 方法在 A01 test **且** A02 擊敗 global error-GT。",
        "沒贏就不進 methodology。",
        "",
        "## Stage",
        "",
        "```",
        "既有 speaker embeddings（speaker_cluster_tau）",
        "既有 per-level error-GT checkpoints",
        f"    ├─[1] τ_n = θ_n − θ_now、殘差、{attention_mode} attention α",
        "    ├─[2] A01 dev 搜 β、λ",
        "    └─[3] A01 test / A02 + report",
        "```",
        "",
    ]
    path = results_dir(cfg) / "pipeline_flow.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


def _ensure_level_tau(cfg: Dict[str, Any], level: str) -> Path:
    dest = level_tau_path(cfg, level)
    if dest.is_file():
        return dest
    fallback = level_vector_fallback_root(cfg) / f"tau_{slugify_level(level)}.pt"
    if fallback.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback, dest)
        logger.info("Copied existing τ_%s %s -> %s", level, fallback, dest)
        return dest
    ckpt_dir = level_ckpt_root(cfg) / slugify_level(level)
    weight_dir = resolve_weight_dir(ckpt_dir)
    base_src = str(cfg.get("base_model", cfg["student_model"]))
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("tau_%s = %s - %s", level, weight_dir, base_src)
    calculate_task_vector(str(weight_dir), base_src, str(dest))
    return dest


def _stratified_train_val_indices(
    labels: np.ndarray,
    val_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    train_idx: List[int] = []
    val_idx: List[int] = []
    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        n_val = 0
        if len(indices) > 1 and float(val_fraction) > 0:
            n_val = max(1, int(round(len(indices) * float(val_fraction))))
            n_val = min(n_val, len(indices) - 1)
        val_idx.extend(indices[:n_val].tolist())
        train_idx.extend(indices[n_val:].tolist())
    return np.asarray(sorted(train_idx), dtype=np.int64), np.asarray(sorted(val_idx), dtype=np.int64)


def _copy_if_needed(src: Path, dest: Path) -> bool:
    if not src.is_file():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.resolve() == src.resolve():
        return True
    shutil.copy2(src, dest)
    logger.info("Copied %s -> %s", src, dest)
    return True


def _reuse_pgcv_attention(cfg: Dict[str, Any]) -> None:
    assign_root = family_reuse_root(cfg, "reuse_assignment_dir")
    if assign_root is not None:
        for split_key in all_split_keys(cfg):
            _copy_if_needed(assign_root / f"assignment_{split_key}.csv", assignment_path(cfg, split_key))
        _copy_if_needed(assign_root / "cefr_prototypes.npz", data_root(cfg) / "cefr_prototypes.npz")
    gate_root = family_reuse_root(cfg, "reuse_gate_dir")
    if gate_root is not None:
        _copy_if_needed(gate_root / "cefr_gate.pt", checkpoint_root(cfg) / "cefr_gate.pt")


def _materialize_svd_bank(cfg: Dict[str, Any], perps: Sequence[Dict[str, torch.Tensor]]) -> None:
    ranks = needed_svd_ranks(cfg)
    if not ranks:
        return
    levels = cfg_levels(cfg)
    max_rank = max(ranks)
    basis, eigenvalues, loadings = svd_subspace_from_taus(list(perps), max_rank)
    if not basis:
        raise RuntimeError("SVD of residual τ bank produced an empty basis")
    if loadings and loadings[0] and (loadings[-1][0] < loadings[0][0]):
        basis[0] = mix_task_vectors([basis[0]], [-1.0])
        for row in loadings:
            if row:
                row[0] = -float(row[0])
        logger.info("Flipped SVD u_1 so higher CEFR has a larger PC1 loading")
    total = sum(max(float(v), 0.0) for v in eigenvalues) or 1.0
    geo_rows: List[Dict[str, Any]] = []
    for rank in ranks:
        rank_basis = basis[: min(rank, len(basis))]
        for level, residual in zip(levels, perps):
            projected = project_tau_onto_basis(residual, rank_basis)
            dest = svd_tau_path(cfg, level, rank)
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save(projected, dest)
        logger.info("Wrote SVD-%d residual bank (%d axes) under %s", rank, len(rank_basis), vector_root(cfg))
    for idx, level in enumerate(levels):
        row: Dict[str, Any] = {"level": level}
        for axis, value in enumerate(loadings[idx] if idx < len(loadings) else []):
            row[f"pc{axis + 1}"] = float(value)
        geo_rows.append(row)
    dump_json(
        results_dir(cfg) / "svd_geometry.json",
        {
            "eigenvalues": [float(v) for v in eigenvalues],
            "explained_energy": [float(v) / total for v in eigenvalues],
            "n_axes": len(basis),
            "loadings": geo_rows,
        },
    )
    md = [
        "# SVD residual geometry",
        "",
        "Gram SVD of `{τ_n^⊥}` after removing the global error-GT direction.",
        "",
        "| Axis | Eigenvalue | Energy share |",
        "|---|---:|---:|",
    ]
    for axis, value in enumerate(eigenvalues):
        md.append(f"| PC{axis + 1} | {float(value):.4f} | {float(value) / total:.4f} |")
    md.extend(["", "| Level | " + " | ".join(f"PC{i + 1}" for i in range(len(basis))) + " |", "|---|" + "|".join("---:" for _ in basis) + "|"])
    for row in geo_rows:
        cells = [row["level"]] + [_fmt(row.get(f"pc{i + 1}"), 4) for i in range(len(basis))]
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    (results_dir(cfg) / "svd_geometry.md").write_text("\n".join(md), encoding="utf-8")


def prepare_vectors_and_attention(cfg: Dict[str, Any], device: str = "cuda") -> None:
    write_pipeline_flow(cfg)
    _reuse_pgcv_attention(cfg)
    if _stage1_complete(cfg):
        logger.info(
            "Stage 1 skip: τ bank, gate, and assignment tables already exist under %s",
            data_root(cfg),
        )
        return
    require_speaker_embeddings(cfg)
    tau_g = load_tau(str(global_tau_path(cfg)))
    g_norm = tau_norm_sq(tau_g) ** 0.5
    rows: List[Dict[str, Any]] = []
    levels = cfg_levels(cfg)
    for level in progress_bar(levels, total=len(levels), desc="τ_n", unit="level"):
        tau_path = _ensure_level_tau(cfg, level)
        tau_n = load_tau(str(tau_path))
        residual = orthogonal_residual(tau_n, tau_g)
        res_path = residual_tau_path(cfg, level)
        torch.save(residual, res_path)
        n_norm = tau_norm_sq(tau_n) ** 0.5
        r_norm = tau_norm_sq(residual) ** 0.5
        cos = cosine_tau(tau_n, tau_g)
        rel = 0.0 if n_norm <= 1e-12 else r_norm / n_norm
        rows.append(
            {
                "level": level,
                "slug": slugify_level(level),
                "tau_path": str(tau_path),
                "residual_path": str(res_path),
                "n_tensors": len(tau_n),
                "l2_norm": float(n_norm),
                "cosine_vs_global": float(cos),
                "residual_l2_norm": float(r_norm),
                "residual_relative_norm": float(rel),
            }
        )
        logger.info(
            "τ_%s ‖τ‖=%.3f cos(τ,τ_g)=%.3f ‖τ^⊥‖/‖τ‖=%.3f",
            level,
            n_norm,
            cos,
            rel,
        )
    geo = {
        "global_tau": str(global_tau_path(cfg)),
        "global_l2_norm": float(g_norm),
        "published_global_lambda": published_global_lambda(cfg),
        "levels": rows,
    }
    dump_json(results_dir(cfg) / "tau_geometry.json", geo)
    geo_md = [
        "# Smooth-Tau Geometry",
        "",
        f"- Global τ: `{global_tau_path(cfg)}`",
        f"- ‖τ_g‖₂ = **{g_norm:.4f}**",
        f"- Published global λ = **{published_global_lambda(cfg):.2f}**",
        "",
        "| Level | cos(τ_n, τ_g) | ‖τ_n^⊥‖ / ‖τ_n‖ | ‖τ_n‖ |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        geo_md.append(
            f"| {row['level']} | {row['cosine_vs_global']:.4f} | "
            f"{row['residual_relative_norm']:.4f} | {row['l2_norm']:.4f} |"
        )
    geo_md.append("")
    (results_dir(cfg) / "tau_geometry.md").write_text("\n".join(geo_md), encoding="utf-8")

    if needed_svd_ranks(cfg):
        perps_for_svd = [load_tau(str(residual_tau_path(cfg, level))) for level in levels]
        _materialize_svd_bank(cfg, perps_for_svd)

    attention_ready = all(assignment_path(cfg, split_key).is_file() for split_key in all_split_keys(cfg))
    attention_mode = str(cfg.get("attention_mode", "cosine")).strip().lower()
    if attention_mode in LEARNED_ATTENTION_MODES:
        attention_ready = attention_ready and (checkpoint_root(cfg) / "cefr_gate.pt").is_file()
    if attention_ready:
        logger.info("Stage 1 attention reuse: assignment tables already exist")
        return

    train_spk = load_spk_emb(speaker_emb_path(cfg, "train"))
    train_x = _l2_normalize(np.stack(train_spk["embedding"].to_list(), axis=0).astype(np.float32))
    prototypes: List[np.ndarray] = []
    proto_counts: List[int] = []
    for level in levels:
        mask = train_spk["proficiency_level"].astype(str) == level
        count = int(mask.sum())
        proto_counts.append(count)
        if count == 0:
            vec = train_x.mean(axis=0)
            logger.warning("No train speakers for %s; using global mean embedding", level)
        else:
            vec = train_x[mask.to_numpy()].mean(axis=0)
        prototypes.append(vec)
    proto = _l2_normalize(np.stack(prototypes, axis=0).astype(np.float32))
    np.savez(
        data_root(cfg) / "cefr_prototypes.npz",
        levels=np.array(levels),
        matrix=proto,
        n_train_speakers=np.array(proto_counts),
    )

    attention_mode = str(cfg.get("attention_mode", "cosine")).strip().lower()
    learned_modes = {"learned", "learned_cefr", "cefr", "cefr_supervised"}
    gate = None
    gate_stats: Dict[str, Any] = {}
    if attention_mode in learned_modes:
        gate_cfg = cfg.get("gate") or {}
        label_to_id = {level: idx for idx, level in enumerate(levels)}
        raw_labels = train_spk["proficiency_level"].astype(str).to_numpy()
        unknown = sorted(set(raw_labels) - set(label_to_id))
        if unknown:
            raise ValueError(f"Unknown train proficiency labels: {unknown}")
        train_y = np.asarray([label_to_id[label] for label in raw_labels], dtype=np.int64)
        fit_idx, val_idx = _stratified_train_val_indices(
            train_y,
            val_fraction=float(gate_cfg.get("val_fraction", 0.2)),
            seed=int(cfg.get("seed", 42)),
        )
        if not len(fit_idx):
            raise ValueError("CEFR gate received no training speakers")
        kind = str(gate_cfg.get("kind", "crossattn"))
        gate = build_gate(
            kind,
            in_dim=int(train_x.shape[1]),
            n_prototypes=len(levels),
            d_model=int(gate_cfg.get("d_model", 64)),
            dropout=float(gate_cfg.get("dropout", 0.3)),
        )
        if isinstance(gate, PrototypeCrossAttnGate):
            init_proto: List[np.ndarray] = []
            global_mean = train_x[fit_idx].mean(axis=0)
            for level_id in range(len(levels)):
                mask = train_y[fit_idx] == level_id
                vec = train_x[fit_idx][mask].mean(axis=0) if np.any(mask) else global_mean
                init_proto.append(vec)
            gate.init_prototypes(_l2_normalize(np.stack(init_proto).astype(np.float32)))

        class_weights = None
        if bool(gate_cfg.get("balanced_classes", True)):
            counts = np.bincount(train_y[fit_idx], minlength=len(levels)).astype(np.float32)
            class_weights = 1.0 / np.sqrt(np.clip(counts, 1.0, None))
            class_weights = class_weights / class_weights.mean()
        gate_stats = train_gate_ce(
            gate,
            train_x[fit_idx],
            train_y[fit_idx],
            val_x=train_x[val_idx] if len(val_idx) else None,
            val_y=train_y[val_idx] if len(val_idx) else None,
            class_weights=class_weights,
            epochs=int(gate_cfg.get("epochs", 200)),
            batch_size=int(gate_cfg.get("batch_size", 64)),
            lr=float(gate_cfg.get("lr", 1e-3)),
            weight_decay=float(gate_cfg.get("weight_decay", 0.05)),
            patience=int(gate_cfg.get("patience", 20)),
            seed=int(cfg.get("seed", 42)),
            device=ensure_device(device),
        )
        gate_stats.update(
            {
                "attention_mode": attention_mode,
                "gate_kind": kind,
                "n_fit_speakers": int(len(fit_idx)),
                "n_val_speakers": int(len(val_idx)),
                "class_weights": class_weights.tolist() if class_weights is not None else None,
            }
        )
        save_gate(
            gate,
            checkpoint_root(cfg) / "cefr_gate.pt",
            extra={key: value for key, value in gate_stats.items() if key != "history"},
        )
        dump_json(results_dir(cfg) / "gate_training.json", gate_stats)
    elif attention_mode != "cosine":
        raise ValueError(f"Unknown attention_mode={attention_mode!r}")

    temperature = float(cfg.get("attn_temperature", 0.10))
    assign_stats: List[Dict[str, Any]] = []
    for split_key in all_split_keys(cfg):
        spk = load_spk_emb(speaker_emb_path(cfg, split_key))
        x = _l2_normalize(np.stack(spk["embedding"].to_list(), axis=0).astype(np.float32))
        if gate is None:
            cosine = x @ proto.T
            alpha = _softmax(cosine / max(temperature, 1e-8), axis=1)
            hard = np.argmax(alpha, axis=1)
        else:
            alpha, hard = gate_predictions(
                gate,
                x,
                batch_size=int((cfg.get("gate") or {}).get("eval_batch_size", 256)),
                device=ensure_device(device),
            )
        entropy = _entropy(alpha)
        frame = pd.DataFrame(
            {
                SPEAKER_COL: spk[SPEAKER_COL].astype(str).tolist(),
                "n_utterances": spk["n_utterances"].tolist(),
                "proficiency_level": spk["proficiency_level"].astype(str).tolist(),
                "hard_id": hard.astype(int),
                "hard_level": [levels[int(i)] for i in hard],
                "entropy": entropy.astype(float),
            }
        )
        for i, level in enumerate(levels):
            frame[f"alpha_{slugify_level(level)}"] = alpha[:, i]
        dest = assignment_path(cfg, split_key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, index=False)
        gold = frame["proficiency_level"].astype(str)
        match = float((gold == frame["hard_level"]).mean()) if len(frame) else float("nan")
        assign_stats.append(
            {
                "split": split_key,
                "n_speakers": int(len(frame)),
                "mean_entropy": float(np.mean(entropy)) if len(frame) else float("nan"),
                "hard_vs_gold": match,
            }
        )
        logger.info(
            "[Stage 1] %s speakers=%d meanH(α)=%.3f hard=gold %.3f",
            split_key,
            len(frame),
            float(np.mean(entropy)) if len(frame) else float("nan"),
            match,
        )
    dump_json(
        results_dir(cfg) / "attention_stats.json",
        {
            "attention_mode": attention_mode,
            "temperature": temperature if gate is None else None,
            "splits": assign_stats,
        },
    )


def alpha_columns(cfg: Dict[str, Any]) -> List[str]:
    return [f"alpha_{slugify_level(level)}" for level in cfg_levels(cfg)]


def load_assignment(cfg: Dict[str, Any], split_key: str) -> pd.DataFrame:
    path = assignment_path(cfg, split_key)
    if not path.is_file():
        raise FileNotFoundError(f"Missing assignment table {path}; run stage 1")
    return pd.read_csv(path)


def group_key_from_alpha(values: Sequence[float], ndigits: int) -> Tuple[float, ...]:
    return tuple(round(float(x), ndigits) for x in values)


def _write_speaker_csv(
    src_csv: Path,
    speakers: Sequence[str],
    dest: Path,
    id_column: str = SPEAKER_COL,
) -> int:
    df = pd.read_csv(src_csv)
    if id_column not in df.columns:
        raise KeyError(f"Missing gate id column {id_column!r} in {src_csv}")
    sub = df[df[id_column].astype(str).isin(set(map(str, speakers)))]
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(dest, index=False)
    return len(sub)


def _decode_csv(
    cfg: Dict[str, Any],
    model,
    processor,
    csv_path: Path,
    dest: Path,
    desc: str,
) -> None:
    if _pred_is_complete(dest):
        return
    n_rows = 0
    if csv_path.is_file():
        n_rows = len(pd.read_csv(csv_path, usecols=[0]))
    if n_rows == 0:
        pd.DataFrame(columns=["audio_path", "hits", "substitutions", "deletions", "insertions"]).to_csv(
            dest, index=False
        )
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    workers = int(cfg.get("num_workers", 2))
    if n_rows < 32:
        workers = 0
    loader = build_eval_dataloader(
        str(csv_path),
        processor=processor,
        batch_size=int(cfg.get("eval_batch_size", 4)),
        num_workers=workers,
        language=str(cfg.get("language", "en")),
    )
    evaluate_model(model, loader, processor, output_csv_path=str(dest), desc=desc, **decode_kwargs(cfg))


def _restore_apply(model, base_state, tau: Optional[Dict[str, torch.Tensor]], lam: float) -> None:
    model.load_state_dict(base_state, strict=True)
    if tau is not None and abs(lam) > 1e-12:
        apply_task_vector(model, tau, float(lam), log=False)


def _load_tau_bank(
    cfg: Dict[str, Any],
) -> Tuple[
    Dict[str, torch.Tensor],
    List[Dict[str, torch.Tensor]],
    List[Dict[str, torch.Tensor]],
    Dict[int, List[Dict[str, torch.Tensor]]],
]:
    tau_g = load_tau(str(global_tau_path(cfg)))
    taus = [load_tau(str(level_tau_path(cfg, level))) for level in cfg_levels(cfg)]
    perps = [load_tau(str(residual_tau_path(cfg, level))) for level in cfg_levels(cfg)]
    svd: Dict[int, List[Dict[str, torch.Tensor]]] = {}
    for rank in needed_svd_ranks(cfg):
        svd[rank] = [load_tau(str(svd_tau_path(cfg, level, rank))) for level in cfg_levels(cfg)]
    return tau_g, taus, perps, svd


def mix_smooth(
    tau_g: Dict[str, torch.Tensor],
    taus: Sequence[Dict[str, torch.Tensor]],
    alpha: Sequence[float],
    beta: float,
) -> Dict[str, torch.Tensor]:
    specialist = mix_task_vectors(list(taus), [float(a) for a in alpha])
    b = float(beta)
    if b <= 1e-12:
        return tau_g
    if b >= 1.0 - 1e-12:
        return specialist
    return mix_task_vectors([specialist, tau_g], [b, 1.0 - b])


def mix_residual(
    tau_g: Dict[str, torch.Tensor],
    perps: Sequence[Dict[str, torch.Tensor]],
    alpha: Sequence[float],
    beta: float,
) -> Dict[str, torch.Tensor]:
    if float(beta) <= 1e-12:
        return tau_g
    residual = mix_task_vectors(list(perps), [float(a) for a in alpha])
    return mix_task_vectors([tau_g, residual], [1.0, float(beta)])


def _maybe_copy_reuse_pred(cfg: Dict[str, Any], split_key: str, method: str, dest: Path) -> bool:
    reuse = cfg.get("reuse_predictions") or {}
    raw = (reuse.get(split_key) or {}).get(method)
    if not raw:
        return False
    src = resolve_repo_path(str(raw))
    if not src.is_file():
        logger.warning("reuse_predictions missing %s", src)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != src.resolve():
        shutil.copy2(src, dest)
    logger.info("Reuse predictions %s -> %s", src, dest)
    return True


def uniform_alpha(n: int) -> List[float]:
    return [1.0 / float(n)] * n


def _one_hot(index: int, n: int) -> List[float]:
    out = [0.0] * n
    out[int(index)] = 1.0
    return out


def _renorm(values: Sequence[float]) -> List[float]:
    arr = np.clip(np.asarray(values, dtype=np.float64), 0.0, None)
    total = float(arr.sum())
    if total <= 1e-12:
        return uniform_alpha(len(arr))
    return (arr / total).astype(float).tolist()


def _groups_for_method(
    method: str,
    assign: pd.DataFrame,
    cfg: Dict[str, Any],
) -> List[Tuple[str, List[str], List[float]]]:
    levels = cfg_levels(cfg)
    n = len(levels)
    ndigits = int(cfg.get("alpha_round_digits", 2))
    cols = alpha_columns(cfg)
    groups: Dict[Tuple[Any, ...], List[str]] = {}
    alpha_sum: Dict[Tuple[Any, ...], np.ndarray] = {}
    if method == "uniform_smooth":
        key = ("uniform",)
        groups[key] = assign[SPEAKER_COL].astype(str).tolist()
        alpha_sum[key] = np.asarray(uniform_alpha(n), dtype=np.float64)
    elif method == "attn_hard_smooth":
        for _, row in assign.iterrows():
            k = int(row["hard_id"])
            key = ("hard", k)
            groups.setdefault(key, []).append(str(row[SPEAKER_COL]))
            alpha_sum[key] = np.asarray(_one_hot(k, n), dtype=np.float64)
    else:
        for _, row in assign.iterrows():
            values = [float(row[c]) for c in cols]
            key = ("soft",) + group_key_from_alpha(values, ndigits)
            groups.setdefault(key, []).append(str(row[SPEAKER_COL]))
            vec = np.asarray(values, dtype=np.float64)
            if key not in alpha_sum:
                alpha_sum[key] = vec
            else:
                alpha_sum[key] = alpha_sum[key] + vec
        for key, speakers in groups.items():
            alpha_sum[key] = alpha_sum[key] / max(len(speakers), 1)
    return [(repr(k), groups[k], _renorm(alpha_sum[k])) for k in groups]


def decode_method_split(
    cfg: Dict[str, Any],
    model,
    processor,
    base_state,
    tau_g: Dict[str, torch.Tensor],
    taus: Sequence[Dict[str, torch.Tensor]],
    perps: Sequence[Dict[str, torch.Tensor]],
    split_key: str,
    method: str,
    beta: float,
    lam: float,
    dest: Path,
    tmp_root: Path,
    svd: Optional[Dict[int, List[Dict[str, torch.Tensor]]]] = None,
) -> pd.DataFrame:
    if _pred_is_complete(dest):
        logger.info("skip decode %s", dest.name)
        return pd.read_csv(dest)
    if method in {"theta_pre", "theta_ft", "global_error_add"} and _maybe_copy_reuse_pred(
        cfg, split_key, method, dest
    ):
        return pd.read_csv(dest)

    csv_path = split_csv(cfg, split_key)
    if method == "theta_ft":
        raise FileNotFoundError(
            f"Full-FT predictions for {split_key} were not supplied in reuse_predictions"
        )
    if method == "theta_pre":
        _restore_apply(model, base_state, None, 0.0)
        _decode_csv(cfg, model, processor, csv_path, dest, desc=f"{split_key}:theta_pre")
        return pd.read_csv(dest)
    if method == "global_error_add" or abs(float(beta)) <= 1e-12:
        _restore_apply(model, base_state, tau_g, lam)
        _decode_csv(cfg, model, processor, csv_path, dest, desc=f"{split_key}:global:{lam:.2f}")
        return pd.read_csv(dest)

    assign = load_assignment(cfg, split_key)
    frames: List[pd.DataFrame] = []
    groups = _groups_for_method(method, assign, cfg)
    svd_rank = svd_rank_of_method(method)
    if svd_rank is not None:
        mixer = mix_residual
        bank = (svd or {}).get(svd_rank) or []
        if len(bank) != len(cfg_levels(cfg)):
            raise FileNotFoundError(f"Missing SVD-{svd_rank} residual bank; run stage 1")
    elif method == "residual_soft":
        mixer = mix_residual
        bank = perps
    else:
        mixer = mix_smooth
        bank = taus
    expected = 0
    for gidx, (key, speakers, alpha) in enumerate(progress_bar(groups, total=len(groups), desc=f"{method} {split_key}", unit="grp")):
        part = tmp_root / f"{split_key}_{method}_b{beta:.2f}_l{lam:.2f}_g{gidx}.csv"
        # β/λ in the name so the four part scripts do not overwrite one shared list.
        src = tmp_root / f"{split_key}_{method}_b{beta:.2f}_l{lam:.2f}_g{gidx}_utts.csv"
        n = _write_speaker_csv(csv_path, speakers, src, id_column=gate_id_column(cfg))
        if n == 0:
            continue
        expected += n
        if _pred_row_count(part) != n:
            if part.is_file():
                part.unlink()
        elif _pred_is_complete(part):
            frames.append(pd.read_csv(part))
            continue
        mixed = mixer(tau_g, bank, alpha, beta)
        _restore_apply(model, base_state, mixed, lam)
        _decode_csv(cfg, model, processor, src, part, desc=f"{split_key}:{method}:g{gidx}")
        got = _pred_row_count(part)
        if got != n:
            raise RuntimeError(
                f"{part.name} has {got} predictions for {n} utterances. "
                "Refusing a partial dev set so WER stays comparable to the full 8636-utterance runs. "
                f"Check that every path in {src.name} exists under /share/corpus."
            )
        frames.append(pd.read_csv(part))
    if not frames:
        empty = pd.DataFrame(columns=["audio_path", "hits", "substitutions", "deletions", "insertions"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(dest, index=False)
        return empty
    out = pd.concat(frames, ignore_index=True)
    if len(out) != expected:
        raise RuntimeError(
            f"{dest.name} would have {len(out)} rows, expected {expected}. "
            "Not writing this cell, so it cannot be scored against the full dev set."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return out


def _best_row(curve: pd.DataFrame) -> pd.Series:
    ordered = curve.sort_values(["wer", "beta", "lambda"], ascending=[True, True, False])
    return ordered.iloc[0]


def _search_grid_lines(curve_path: Path, method: str) -> List[str]:
    if not curve_path.is_file():
        return []
    curve = pd.read_csv(curve_path)
    if curve.empty or "beta" not in curve.columns or "lambda" not in curve.columns:
        return []
    betas = sorted({float(v) for v in curve["beta"].tolist()})
    lambdas = sorted({float(v) for v in curve["lambda"].tolist()})
    header = "| β \\ λ | " + " | ".join(f"{lam:.2f}" for lam in lambdas) + " |"
    align = "|---|" + "|".join("---:" for _ in lambdas) + "|"
    lines = [
        f"## Search grid ({METHOD_LABEL.get(method, method)})",
        "",
        header,
        align,
    ]
    for beta in betas:
        cells = [f"{beta:.2f}"]
        for lam in lambdas:
            match = curve.loc[
                np.isclose(curve["beta"], beta) & np.isclose(curve["lambda"], lam)
            ]
            cells.append(_fmt(float(match.iloc[0]["wer"])) if len(match) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines



def select_coefficients(
    cfg: Dict[str, Any],
    device: str,
    only_lambdas: Optional[Sequence[float]] = None,
    only_betas: Optional[Sequence[float]] = None,
    write_selection: bool = True,
    global_only: bool = False,
) -> None:
    betas = float_list(cfg, "beta_list", DEFAULT_BETA_LIST)
    lambdas = float_list(cfg, "lambda_list", DEFAULT_LAMBDA_LIST)
    if only_lambdas is not None:
        wanted = [float(x) for x in only_lambdas]
        lambdas = [lam for lam in lambdas if any(np.isclose(lam, item) for item in wanted)]
        missing = [item for item in wanted if not any(np.isclose(lam, item) for lam in lambdas)]
        if missing:
            raise ValueError(f"lambda {missing} is not in config lambda_list")
    if only_betas is not None:
        wanted_beta = [float(x) for x in only_betas]
        betas = [beta for beta in betas if any(np.isclose(beta, item) for item in wanted_beta)]
        missing_beta = [item for item in wanted_beta if not any(np.isclose(beta, item) for beta in betas)]
        if missing_beta:
            raise ValueError(f"beta {missing_beta} is not in config beta_list")
    run_global = global_only or only_betas is None
    run_search = (not global_only) and (only_betas is None or len(betas) > 0)
    tmp_root = data_root(cfg) / "_lambda_dev"
    tmp_root.mkdir(parents=True, exist_ok=True)
    reuse_dev = family_reuse_root(cfg, "reuse_dev_global_dir")
    if reuse_dev is not None:
        src_dev = reuse_dev / "_lambda_dev"
        if src_dev.is_dir():
            for src in src_dev.glob("dev_global_l*.csv"):
                _copy_if_needed(src, tmp_root / src.name)
    published = published_global_lambda(cfg)
    device_t: Optional[str] = None
    processor = None
    model = None
    base_state = None
    tau_g: Optional[Dict[str, torch.Tensor]] = None
    taus: Optional[List[Dict[str, torch.Tensor]]] = None
    perps: Optional[List[Dict[str, torch.Tensor]]] = None
    svd: Optional[Dict[int, List[Dict[str, torch.Tensor]]]] = None

    def ensure_decode_model() -> None:
        nonlocal device_t, processor, model, base_state, tau_g, taus, perps, svd
        if model is not None:
            return
        device_t = ensure_device(device)
        processor = load_asr_processor(cfg["student_model"])
        model = load_asr_model(cfg.get("base_model", cfg["student_model"]), device=device_t)
        base_state = copy.deepcopy(model.state_dict())
        tau_g, taus, perps, svd = _load_tau_bank(cfg)

    def decode_or_reuse(split_key: str, method: str, beta: float, lam: float, dest: Path) -> pd.DataFrame:
        if _pred_is_complete(dest):
            logger.info("skip %s β=%.2f λ=%.2f (%s)", method, beta, lam, dest.name)
            return pd.read_csv(dest)
        ensure_decode_model()
        return decode_method_split(
            cfg, model, processor, base_state, tau_g, taus, perps, split_key, method, beta, lam, dest, tmp_root, svd=svd
        )

    if run_global:
        global_rows = []
        for lam in progress_bar(lambdas, total=len(lambdas), desc="λ global", unit="λ"):
            dest = tmp_root / f"dev_global_l{lam:.2f}.csv"
            pred = decode_or_reuse("dev", "global_error_add", 0.0, lam, dest)
            metrics = metrics_from_pred_df(pred)
            global_rows.append({"method": "global_error_add", "beta": 0.0, "lambda": lam, **metrics})
            logger.info("global λ=%.2f dev WER=%.2f", lam, metrics["wer"])
        if write_selection:
            global_curve = pd.DataFrame(global_rows)
            global_curve.to_csv(results_dir(cfg) / "lambda_curve_global_error_add.csv", index=False)
            g_best = _best_row(global_curve)
            dump_json(
                _lambda_json(cfg, "global_error_add"),
                {
                    "method": "global_error_add",
                    "beta": 0.0,
                    "best_lambda": float(g_best["lambda"]),
                    "best_wer": float(g_best["wer"]),
                    "published_lambda": published,
                    "selection_set": "source_dev",
                },
            )

    if not run_search:
        if model is not None:
            model.load_state_dict(base_state, strict=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return

    for method in search_methods(cfg):
        rows = []
        for beta in betas:
            for lam in lambdas:
                dest = _dev_combo_dest(tmp_root, method, beta, lam)
                if abs(beta) <= 1e-12:
                    src = tmp_root / f"dev_global_l{lam:.2f}.csv"
                    if _pred_is_complete(src) and not _pred_is_complete(dest):
                        shutil.copy2(src, dest)
                pred = decode_or_reuse("dev", method, beta, lam, dest)
                metrics = metrics_from_pred_df(pred)
                rows.append({"method": method, "beta": beta, "lambda": lam, **metrics})
                logger.info("%s β=%.2f λ=%.2f dev WER=%.2f", method, beta, lam, metrics["wer"])
        if not write_selection:
            continue
        curve = pd.DataFrame(rows)
        curve.to_csv(results_dir(cfg) / f"lambda_curve_{method}.csv", index=False)
        best = _best_row(curve)
        dump_json(
            _lambda_json(cfg, method),
            {
                "method": method,
                "beta": float(best["beta"]),
                "best_lambda": float(best["lambda"]),
                "best_wer": float(best["wer"]),
                "selection_set": "source_dev",
            },
        )

    if model is not None:
        model.load_state_dict(base_state, strict=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_best(cfg: Dict[str, Any], method: str) -> Dict[str, Any]:
    path = _lambda_json(cfg, method)
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected coefficients {path}; run stage 2")
    return json.loads(path.read_text(encoding="utf-8"))


def wer_by_level(pred: pd.DataFrame, meta: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    cols = ["audio_path"]
    if "proficiency_level" in meta.columns:
        cols.append("proficiency_level")
    merged = pred.merge(meta[cols], on="audio_path", how="inner")
    out: Dict[str, Dict[str, float]] = {"__overall__": metrics_from_pred_df(merged)}
    if "proficiency_level" not in merged.columns:
        return out
    for level, sub in merged.groupby("proficiency_level", sort=False):
        out[str(level)] = metrics_from_pred_df(sub)
    return out


def evaluate_and_report(cfg: Dict[str, Any], device: str) -> None:
    tmp_root = data_root(cfg) / "_eval"
    tmp_root.mkdir(parents=True, exist_ok=True)
    selected: Dict[str, Dict[str, Any]] = {}
    selected["theta_pre"] = {"method": "theta_pre", "beta": 0.0, "best_lambda": 0.0}
    if "theta_ft" in report_methods(cfg):
        selected["theta_ft"] = {"method": "theta_ft", "beta": 0.0, "best_lambda": 0.0}
    selected["global_error_add"] = load_best(cfg, "global_error_add")
    selected["global_error_add"]["best_lambda"] = published_global_lambda(cfg)
    for method in search_methods(cfg):
        selected[method] = load_best(cfg, method)

    device_t: Optional[str] = None
    processor = None
    model = None
    base_state = None
    tau_g: Optional[Dict[str, torch.Tensor]] = None
    taus: Optional[List[Dict[str, torch.Tensor]]] = None
    perps: Optional[List[Dict[str, torch.Tensor]]] = None
    svd: Optional[Dict[int, List[Dict[str, torch.Tensor]]]] = None

    def ensure_decode_model() -> None:
        nonlocal device_t, processor, model, base_state, tau_g, taus, perps, svd
        if model is not None:
            return
        device_t = ensure_device(device)
        processor = load_asr_processor(cfg["student_model"])
        model = load_asr_model(cfg.get("base_model", cfg["student_model"]), device=device_t)
        base_state = copy.deepcopy(model.state_dict())
        tau_g, taus, perps, svd = _load_tau_bank(cfg)

    summaries: Dict[str, Dict[str, Dict[str, Any]]] = {}
    detailed: Dict[str, Dict[str, float]] = {}
    for split_key, label in eval_splits_from_cfg(cfg):
        meta = pd.read_csv(split_csv(cfg, split_key))
        summaries[split_key] = {"label": label}
        for method in report_methods(cfg):
            spec = selected[method]
            beta = float(spec.get("beta", 0.0))
            lam = float(spec.get("best_lambda", 0.0))
            tagged = _eval_combo_dest(tmp_root, split_key, method, beta, lam)
            canonical = _prediction_csv(cfg, split_key, method)
            _seed_eval_from_canonical(cfg, split_key, method, beta, lam, tagged)
            if _pred_is_complete(tagged):
                logger.info("skip eval %s/%s β=%.2f λ=%.2f", split_key, method, beta, lam)
                pred = pd.read_csv(tagged)
            else:
                ensure_decode_model()
                pred = decode_method_split(
                    cfg,
                    model,
                    processor,
                    base_state,
                    tau_g,
                    taus,
                    perps,
                    split_key,
                    method,
                    beta,
                    lam,
                    tagged,
                    tmp_root,
                    svd=svd,
                )
            canonical.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tagged, canonical)
            by_level = wer_by_level(pred, meta)
            summaries[split_key][method] = {"metrics": by_level["__overall__"], "by_level": by_level, "beta": beta, "lambda": lam}
            detailed[f"{split_key}:{method}"] = by_level["__overall__"]

    write_eval_detailed_json(detailed, str(results_dir(cfg) / "eval_detailed.json"))
    dump_json(results_dir(cfg) / "eval_summary.json", summaries)
    _write_report(cfg, selected, summaries)

    if model is not None:
        model.load_state_dict(base_state, strict=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _write_report(
    cfg: Dict[str, Any],
    selected: Dict[str, Dict[str, Any]],
    summaries: Dict[str, Dict[str, Any]],
) -> None:
    levels = cfg_levels(cfg)
    attention_mode = str(cfg.get("attention_mode", "cosine")).strip().lower()
    learned = attention_mode in {"learned", "learned_cefr", "cefr", "cefr_supervised"}
    attention_scope = (
        "A small gate is trained with train-speaker CEFR labels; gold CEFR is **not** a test input."
        if learned
        else "Cosine attention uses train CEFR prototype embeddings; gold CEFR is **not** a test input."
    )
    alpha_formula = "α = softmax(g_φ(x))" if learned else "α_n = softmax(cos(x, p_n) / T)"
    svd_run = bool(needed_svd_ranks(cfg))
    formula_lines = (
        [
            alpha_formula,
            "τ̃_n = τ_n − proj_{τ_g}(τ_n)",
            "{u_j} = SVD of {τ̃_n}",
            "τ̃_n^(k) = Σ_{j=1}^{k} <τ̃_n, u_j> u_j",
            "θ'  = θ_now + λ (τ_g + β Σ_n α_n τ̃_n^(k))",
        ]
        if svd_run
        else [
            alpha_formula,
            "τ̃  = β Σ_n α_n τ_n + (1 − β) τ_g",
            "θ'  = θ_now + λ τ̃",
        ]
    )
    lines = [
        f"# Experiment Report ({cfg.get('display_name', 'TEEMI A01-only PGCV')})",
        "",
        "## Scope",
        "",
        "- Frozen global error-GT `τ_g` and per-level error-GT `τ_n` (`τ = θ − θ_now`).",
        f"- {attention_scope}",
        f"- `β`, `λ` selected on A01 dev only. Backbone: `{cfg['student_model']}`.",
        "",
        "## Formula",
        "",
        "```",
        *formula_lines,
        "```",
        "",
        f"- Attention mode = {attention_mode}",
        f"- Levels: {', '.join(levels)}",
        "",
        f"## Selected coefficients ({cfg.get('selection_split_label', 'A01 dev')})",
        "",
        "| Method | β | λ | Dev WER |",
        "|---|---:|---:|---:|",
    ]
    methods = report_methods(cfg)
    for method in methods:
        spec = selected.get(method) or {}
        if method in {"theta_pre", "theta_ft"}:
            lines.append(f"| {METHOD_LABEL[method]} | — | 0.00 | — |")
            continue
        curve_path = results_dir(cfg) / f"lambda_curve_{method}.csv"
        dev_wer = spec.get("best_wer")
        if method == "global_error_add" and curve_path.is_file():
            curve = pd.read_csv(curve_path)
            pub = published_global_lambda(cfg)
            match = curve.loc[np.isclose(curve["lambda"], pub)]
            if len(match):
                dev_wer = float(match.iloc[0]["wer"])
        lines.append(
            f"| {METHOD_LABEL[method]} | {_fmt(spec.get('beta'))} | {_fmt(spec.get('best_lambda'))} | {_fmt(dev_wer)} |"
        )
    lines.append("")
    for method in search_methods(cfg):
        lines.extend(_search_grid_lines(results_dir(cfg) / f"lambda_curve_{method}.csv", method))
    

    global_test = None
    for split_key, _label in eval_splits_from_cfg(cfg):
        block = summaries.get(split_key) or {}
        label = str(block.get("label", split_key))
        lines.extend(
            [
                f"## {label}",
                "",
                "| Method | WER | Sub | Del | Ins | Δ vs global |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        g = ((block.get("global_error_add") or {}).get("metrics") or {}).get("wer")
        if split_key == "test":
            global_test = g
        for method in methods:
            mets = (block.get(method) or {}).get("metrics") or {}
            wer = mets.get("wer")
            delta = "—"
            if g is not None and wer is not None and method != "global_error_add":
                delta = f"{float(g) - float(wer):+.2f}"
            elif method == "global_error_add":
                delta = "0.00"
            lines.append(
                f"| {METHOD_LABEL[method]} | {_fmt(wer)} | {_fmt(mets.get('sub'))} | "
                f"{_fmt(mets.get('del'))} | {_fmt(mets.get('ins'))} | {delta} |"
            )
        lines.extend(["", f"### {label} by proficiency", ""])
        header = "| Proficiency | " + " | ".join(METHOD_LABEL[m] for m in methods) + " |"
        align = "|---|" + "|".join("---:" for _ in methods) + "|"
        lines.extend([header, align])
        seen = []
        for method in methods:
            by_level = (block.get(method) or {}).get("by_level") or {}
            for level in by_level:
                if level != "__overall__" and level not in seen:
                    seen.append(level)
        ordered = [lv for lv in levels if lv in seen] + [lv for lv in seen if lv not in levels]
        for level in ordered:
            cells = [level]
            for method in methods:
                wer = ((block.get(method) or {}).get("by_level") or {}).get(level, {}).get("wer")
                cells.append(_fmt(wer))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    winners = []
    for method in search_methods(cfg):
        comparisons = []
        for split_key, _ in eval_splits_from_cfg(cfg):
            block = summaries.get(split_key) or {}
            global_wer = ((block.get("global_error_add") or {}).get("metrics") or {}).get("wer")
            method_wer = ((block.get(method) or {}).get("metrics") or {}).get("wer")
            if global_wer is not None and method_wer is not None:
                comparisons.append(float(method_wer) < float(global_wer))
        if comparisons and all(comparisons):
            winners.append(method)

    lines.extend(
        [
            "## Verdict",
            "",
        ]
    )
    if winners:
        names = ", ".join(METHOD_LABEL[m] for m in winners)
        lines.append(
            f"- **Keep considering for methodology**: {names} beat the global correction on every evaluation split."
        )
    else:
        lines.append(
            "- **Do not write into methodology.** No smoothing variant beat global error-GT on both A01 test and A02."
        )
        if global_test is not None:
            lines.append(f"- Global error-GT A01 test WER remains the reference ({_fmt(global_test)}).")
    lines.extend(
        [
            "",
            "- Positive Δ vs global means the new method is better (lower WER).",
            "- `β*=0` means the search collapsed back to global τ_g.",
            "",
        ]
    )
    path = results_dir(cfg) / "experiment_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report -> %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoothed proficiency correction vectors")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default="all")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--only-lambdas", type=str, default="", help="Comma-separated λ subset for stage 2")
    parser.add_argument("--only-betas", type=str, default="", help="Comma-separated β subset for stage 2 search")
    parser.add_argument("--global-only", action="store_true", help="Stage 2 decodes global λ only")
    parser.add_argument("--decode-only", action="store_true", help="Write prediction CSVs without selection JSON")
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

    with stage_progress(sorted(stages), desc=cfg.get("display_name", "Smoothed Proficiency Tau")) as sp:
        if 1 in stages:
            sp.begin(1, titles[1])
            prepare_vectors_and_attention(cfg, device=args.device)
            sp.end(1)
        if 2 in stages:
            sp.begin(2, titles[2])
            only_lambdas = [float(x) for x in args.only_lambdas.split(",") if x.strip()] or None
            only_betas = [float(x) for x in args.only_betas.split(",") if x.strip()] or None
            select_coefficients(
                cfg,
                device=args.device,
                only_lambdas=only_lambdas,
                only_betas=only_betas,
                write_selection=not args.decode_only,
                global_only=bool(args.global_only),
            )
            sp.end(2)
        if 3 in stages:
            sp.begin(3, titles[3])
            evaluate_and_report(cfg, device=args.device)
            sp.end(3)


if __name__ == "__main__":
    main()
