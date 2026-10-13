#!/usr/bin/env python3
"""A01-only speaker-cluster gate + per-prototype error-GT correction vectors.

Data / leakage
--------------
- Fit the gate (k-means centroids) on **A01 train speakers only**.
- Train each cluster tau on **that cluster's A01 train error utterances** (GT text).
- Select each lambda on **A01 dev speakers assigned by the train centroids**.
- A01 test and A02 never enter fit / train / lambda search.

Inference
---------
Speaker embedding -> nearest train centroid k -> decode with
``θ_now + λ_k * τ_k``.

Stages
------
1. Score A01 train with θ_now (correctness).
2. Extract encoder embeddings; pool to speaker vectors.
3. K-means on train speakers; assign every split by nearest centroid.
4. Materialize global + per-cluster error-GT subsets.
5. Fine-tune subset models.
6. Compute τ = θ_S − θ_now.
7. Select λ on A01 dev (global and per-cluster).
8. Evaluate gated / ablated methods; write the report.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
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
from src.speaker_gate import (  # noqa: E402
    SPEAKER_COL,
    assign_nearest,
    embedding_matrix,
    extract_utterance_embeddings,
    kmeans_fit,
    pool_speaker_embeddings,
    reorder_centroids,
    reorder_clusters_by_error_rate,
)
from src.task_vector import apply_task_vector, calculate_task_vector, load_tau  # noqa: E402

logger = logging.getLogger("teemi_a01_speaker_cluster_tau")

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

METHOD_LABELS = {
    "theta_pre": "Pretrained (no tau)",
    "global_error_add": "Global error-GT tau (one τ, one λ)",
    "cluster_tau_cluster_lambda": "Gated: τ_k + λ_k (speaker k-means)",
    "global_tau_cluster_lambda": "Ablation: one τ, λ_k per cluster",
    "cluster_tau_global_lambda": "Ablation: τ_k, one shared λ",
    "random_cluster_tau": "Control: random cluster + (τ_k, λ_k)",
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
    available = {1, 2, 3, 4, 5, 6, 7, 8}
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
        1: "Score A01 train correctness",
        2: "Extract speaker embeddings",
        3: "Cluster train speakers and assign all splits",
        4: "Materialize per-cluster error-GT subsets",
        5: "Fine-tune global + cluster error models",
        6: "Compute taus",
        7: "Select lambdas on A01 dev",
        8: "Evaluate gated methods and write report",
    }


def family_root(cfg: Dict[str, Any], key: str) -> Path:
    return resolve_repo_path(cfg[key]) / family_slug(cfg["student_model"])


def split_csv(cfg: Dict[str, Any], split_key: str) -> Path:
    return resolve_repo_path(cfg["splits"][split_key])


def n_clusters(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("n_clusters", 3))


def cluster_ids(cfg: Dict[str, Any]) -> List[int]:
    return list(range(n_clusters(cfg)))


def data_root(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "data_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(cfg: Dict[str, Any]) -> Path:
    path = family_root(cfg, "results_dir")
    path.mkdir(parents=True, exist_ok=True)
    return path


def scored_train_path(cfg: Dict[str, Any]) -> Path:
    return data_root(cfg) / "train_scored.csv"


def subset_csv_path(cfg: Dict[str, Any], name: str) -> Path:
    return data_root(cfg) / f"train_{name}.csv"


def job_ckpt_dir(cfg: Dict[str, Any], job_name: str) -> Path:
    if job_name == "error_gt_global" and cfg.get("reuse_global_error_ckpt"):
        reuse = resolve_repo_path(str(cfg["reuse_global_error_ckpt"]))
        if reuse.exists():
            return reuse
    return family_root(cfg, "checkpoint_dir") / job_name


def tau_path(cfg: Dict[str, Any], job_name: str) -> Path:
    return family_root(cfg, "vector_dir") / f"tau_{job_name}.pt"


def centroids_path(cfg: Dict[str, Any]) -> Path:
    return data_root(cfg) / "cluster_centroids.npy"


def assignment_path(cfg: Dict[str, Any], split_key: str) -> Path:
    return data_root(cfg) / f"speaker_assignment_{split_key}.csv"


def eval_splits_from_cfg(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for item in cfg.get("eval_splits", []):
        pairs.append((str(item["key"]), str(item.get("label", item["key"]))))
    return pairs


def all_split_keys(cfg: Dict[str, Any]) -> List[str]:
    keys = ["train"]
    for key, _ in eval_splits_from_cfg(cfg):
        if key not in keys:
            keys.append(key)
    if "dev" in cfg.get("splits", {}) and "dev" not in keys:
        keys.append("dev")
    return keys


def row_count(csv_path: Path) -> int:
    if not csv_path.is_file():
        return 0
    return len(pd.read_csv(csv_path))


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


def _metrics_from_pred_df(pred_df: pd.DataFrame) -> Dict[str, float]:
    return _metrics_from_counts(
        total_h=float(pred_df["hits"].sum()),
        total_s=float(pred_df["substitutions"].sum()),
        total_d=float(pred_df["deletions"].sum()),
        total_i=float(pred_df["insertions"].sum()),
        n_utterances=int(len(pred_df)),
    )


def cluster_job_name(cluster_id: int) -> str:
    return f"error_gt_c{cluster_id}"


def cluster_subset_name(cluster_id: int) -> str:
    return f"error_c{cluster_id}"


def write_pipeline_flow(cfg: Dict[str, Any]) -> None:
    k = n_clusters(cfg)
    text = f"""# Speaker-Cluster Gate Tau：資料流說明

本實驗**不看 gold CEFR 來決定套哪條 τ**。熟練度標籤只出現在報告的事後交叉表，用來檢查分群有沒有對上 L2 程度。

## 兩個變因

1. **哪一種 L2 speaker prototype**：train speaker 的 encoder embedding 做 k-means（k={k}）。
2. **該 prototype 該修正多少**：每群自己的 `λ_k`，只在 A01 dev 上選。

推論：

```
θ'(speaker) = θ_now + λ_k * τ_k
k = nearest_centroid(mean_pool(encoder(speaker utterances)))
```

## 為什麼 gate 做在 speaker

TEEMI 的 split 單位是 `global_speaker_id`。熟練度是語者特質，不是單句特質。同一 speaker 的所有句子共用同一個 k、同一條 τ、同一個 λ。

## 零洩漏

| 步驟 | 用到的資料 | 不用的資料 |
|---|---|---|
| 抽 embedding、k-means 擬合 centroid | **A01 train speakers** | dev / test / A02 |
| 每群 error-GT fine-tune → `τ_k` | 該群 **train** 且 pretrained **聽錯** 的句子 + GT 稿 | 正確句、dev/test |
| 選 `λ` / `λ_k` | **A01 dev**（用 train centroid **指定**群，不重擬合） | test / A02 |
| 報告 | A01 test、A02 external | 不回寫選模 |

Test / A02 的 speaker **從未出現在 train**，所以 gate 一定是「把新 speaker 丟到最近的 train centroid」，不是查表。

## Stage 資料流

```
A01 train.csv
    │
    ├─[1] θ_now greedy decode
    │      → train_scored.csv   (is_correct, hypothesis, mean_token_logprob)
    │
    ├─[2] encoder mean-pool per utterance
    │      → speaker mean-pool + L2 normalize
    │      對 train / dev / test / A02 都抽 embedding
    │      （擬合只用 train；其他 split 只是準備好向量）
    │
    └─[3] k-means(train speakers) → centroids.npy
           cluster 0 = train 上平均錯誤率最低（較接近母語）
           cluster {k - 1} = 平均錯誤率最高（較重的 L2 偏誤）
           每個 split：nearest centroid → speaker_assignment_<split>.csv

train_scored.csv ⋈ train assignment
    │
    ├─[4] error = is_correct == False
    │      train_error_global.csv          全部聽錯句（global 對照）
    │      train_error_c0.csv … c{k - 1}   該群 speaker 的聽錯句
    │
    ├─[5] fine-tune Whisper/Qwen on GT `text`
    │      error_gt_global, error_gt_c0, …
    │
    └─[6] τ = θ_finetune − θ_now
           tau_error_gt_global.pt, tau_error_gt_c0.pt, …

A01 dev.csv ⋈ speaker_assignment_dev.csv
    │
    └─[7] 在 dev 上選 λ（仍是 greedy WER）
           global_error_add          一條 τ_global，一個 λ
           cluster_tau_cluster_lambda  每群 τ_k，每群自己的 λ_k   ← 主方法
           global_tau_cluster_lambda   共用 τ_global，每群不同 λ_k
           cluster_tau_global_lambda   每群 τ_k，共用一個 λ

A01 test / A02
    │
    └─[8] 用 assignment 分群後套對應 (τ, λ) decode
           random_cluster_tau：打亂 speaker 的 k，但沿用主方法的 (τ_k, λ_k)
```

## 輸出根目錄

`{cfg.get("results_dir")}/<model_family>/`

- `cluster_manifest.md`：每群人數、事後 CEFR 分布、error 句數
- `pipeline_flow.md`：本說明
- `experiment_report.md`：WER 與選到的 λ
- `predictions_<split>_<method>.csv`：句級結果
"""
    path = results_dir(cfg) / "pipeline_flow.md"
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", path)


def score_train(cfg: Dict[str, Any], device: str) -> None:
    out_csv = scored_train_path(cfg)
    if out_csv.is_file():
        logger.info("[Stage 1] Skip; found %s", out_csv)
        return
    reuse = cfg.get("reuse_scored_train")
    if reuse:
        src = resolve_repo_path(str(reuse))
        if src.is_file():
            shutil.copy2(src, out_csv)
            logger.info("[Stage 1] Reused scored train %s -> %s", src, out_csv)
            return
        logger.warning("reuse_scored_train missing (%s); scoring from scratch", src)
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
        desc="score A01 train",
        **decode_kwargs(cfg),
    )
    train_df = pd.read_csv(train_path)
    merged = train_df.merge(scored, on="audio_path", how="inner")
    merged.to_csv(out_csv, index=False)
    logger.info("[Stage 1] Wrote %s (%d rows)", out_csv, len(merged))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _utt_emb_path(cfg: Dict[str, Any], split_key: str) -> Path:
    return data_root(cfg) / f"utt_embeddings_{split_key}.npz"


def _spk_emb_path(cfg: Dict[str, Any], split_key: str) -> Path:
    return data_root(cfg) / f"speaker_embeddings_{split_key}.npz"


def _save_utt_emb(path: Path, df: pd.DataFrame) -> None:
    matrix = np.stack(df["embedding"].to_list(), axis=0)
    np.savez_compressed(path, audio_path=np.asarray(df["audio_path"].astype(str)), matrix=matrix)


def _load_utt_emb(path: Path) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=True)
    return pd.DataFrame(
        {
            "audio_path": payload["audio_path"].tolist(),
            "embedding": list(payload["matrix"]),
        }
    )


def _save_spk_emb(path: Path, df: pd.DataFrame) -> None:
    extra = {}
    if "proficiency_level" in df.columns:
        extra["proficiency_level"] = np.asarray(df["proficiency_level"].astype(str))
    np.savez_compressed(
        path,
        speaker=np.asarray(df[SPEAKER_COL].astype(str)),
        n_utterances=np.asarray(df["n_utterances"], dtype=np.int32),
        matrix=embedding_matrix(df),
        **extra,
    )


def _load_spk_emb(path: Path) -> pd.DataFrame:
    payload = np.load(path, allow_pickle=True)
    rows = []
    speakers = payload["speaker"].tolist()
    matrix = payload["matrix"]
    n_utts = payload["n_utterances"].tolist()
    levels = payload["proficiency_level"].tolist() if "proficiency_level" in payload.files else [""] * len(speakers)
    for speaker, n_utt, vec, level in zip(speakers, n_utts, matrix, levels):
        rows.append(
            {
                SPEAKER_COL: str(speaker),
                "n_utterances": int(n_utt),
                "embedding": vec.astype(np.float32),
                "proficiency_level": str(level),
            }
        )
    return pd.DataFrame(rows)


def extract_split_embeddings(cfg: Dict[str, Any], device: str) -> None:
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    model = load_base_model(cfg, device_t)
    for split_key in all_split_keys(cfg):
        utt_path = _utt_emb_path(cfg, split_key)
        spk_path = _spk_emb_path(cfg, split_key)
        csv_path = split_csv(cfg, split_key)
        if not csv_path.is_file():
            logger.warning("Skip embeddings for missing split %s", csv_path)
            continue
        if utt_path.is_file() and spk_path.is_file():
            logger.info("[Stage 2] Skip %s embeddings; found cache", split_key)
            continue
        loader = build_eval_dataloader(
            str(csv_path),
            processor=processor,
            batch_size=int(cfg.get("eval_batch_size", 4)),
            num_workers=int(cfg.get("num_workers", 2)),
            language=str(cfg.get("language", "en")),
        )
        utt_df = extract_utterance_embeddings(model, loader, processor, desc=f"embed {split_key}")
        meta = pd.read_csv(csv_path)
        gate_id_col = str(cfg.get("gate_id_column", SPEAKER_COL))
        spk_df = pool_speaker_embeddings(utt_df, meta, speaker_col=gate_id_col)
        if gate_id_col != SPEAKER_COL:
            spk_df = spk_df.rename(columns={gate_id_col: SPEAKER_COL})
        _save_utt_emb(utt_path, utt_df)
        _save_spk_emb(spk_path, spk_df)
        logger.info("[Stage 2] %s: %d utt / %d speakers", split_key, len(utt_df), len(spk_df))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _speaker_error_rates(cfg: Dict[str, Any]) -> Dict[str, float]:
    scored = pd.read_csv(scored_train_path(cfg))
    rates: Dict[str, float] = {}
    for speaker, sub in scored.groupby(SPEAKER_COL):
        correct = sub["is_correct"].astype(bool)
        rates[str(speaker)] = float((~correct).mean()) if len(sub) else float("nan")
    return rates


def _assignment_frame(
    speaker_df: pd.DataFrame,
    labels: np.ndarray,
    distances: np.ndarray,
) -> pd.DataFrame:
    out = speaker_df[[SPEAKER_COL, "n_utterances"]].copy()
    if "proficiency_level" in speaker_df.columns:
        out["proficiency_level"] = speaker_df["proficiency_level"].astype(str)
    out["cluster_id"] = labels.astype(int)
    out["centroid_distance"] = distances.astype(float)
    return out


def cluster_and_assign(cfg: Dict[str, Any]) -> None:
    cent_path = centroids_path(cfg)
    train_assign = assignment_path(cfg, "train")
    train_spk = _load_spk_emb(_spk_emb_path(cfg, "train"))
    x_train = embedding_matrix(train_spk)
    if cent_path.is_file() and train_assign.is_file():
        logger.info("[Stage 3] Skip k-means; found centroids")
        centroids = np.load(cent_path)
    else:
        raw_centroids, raw_labels = kmeans_fit(
            x_train,
            n_clusters=n_clusters(cfg),
            seed=int(cfg.get("seed", 42)),
            n_iter=int(cfg.get("kmeans_iters", 100)),
            n_init=int(cfg.get("kmeans_n_init", 8)),
        )
        new_labels, order = reorder_clusters_by_error_rate(
            raw_labels, train_spk, _speaker_error_rates(cfg)
        )
        centroids = reorder_centroids(raw_centroids, order)
        np.save(cent_path, centroids)
        distances = assign_nearest(x_train, centroids)[1]
        _assignment_frame(train_spk, new_labels, distances).to_csv(train_assign, index=False)
        logger.info("[Stage 3] Fitted k=%d on %d train speakers", len(centroids), len(train_spk))

    centroids = np.load(cent_path)
    for split_key in all_split_keys(cfg):
        out = assignment_path(cfg, split_key)
        if out.is_file():
            continue
        spk_path = _spk_emb_path(cfg, split_key)
        if not spk_path.is_file():
            continue
        spk_df = _load_spk_emb(spk_path)
        labels, dists = assign_nearest(embedding_matrix(spk_df), centroids)
        _assignment_frame(spk_df, labels, dists).to_csv(out, index=False)
        logger.info("[Stage 3] Assigned %s (%d speakers)", split_key, len(spk_df))
    _write_cluster_manifest(cfg)


def _write_cluster_manifest(cfg: Dict[str, Any]) -> None:
    scored = pd.read_csv(scored_train_path(cfg))
    assign = pd.read_csv(assignment_path(cfg, "train"))
    merged = scored.merge(assign[[SPEAKER_COL, "cluster_id"]], on=SPEAKER_COL, how="left")
    lines = [
        f"# {cfg.get('display_name', 'Speaker-Cluster Tau')} Manifest",
        "",
        f"- k-means on **A01 train speakers only**, k={n_clusters(cfg)}",
        "- Cluster 0 = lowest mean train error rate (post-hoc sort; not a CEFR label)",
        "- Gold `proficiency_level` is **not** used by the gate",
        "",
        "| Cluster | Train speakers | Train utt | Error utt | Mean speaker error rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for cluster in cluster_ids(cfg):
        sub = merged[merged["cluster_id"] == cluster]
        speakers = assign[assign["cluster_id"] == cluster]
        n_err = int((~sub["is_correct"].astype(bool)).sum()) if len(sub) else 0
        rates = []
        for speaker in speakers[SPEAKER_COL].astype(str):
            utt = scored[scored[SPEAKER_COL].astype(str) == speaker]
            if len(utt):
                rates.append(float((~utt["is_correct"].astype(bool)).mean()))
        mean_err = float(np.mean(rates)) if rates else float("nan")
        lines.append(
            f"| c{cluster} | {len(speakers)} | {len(sub)} | {n_err} | {mean_err:.3f} |"
        )
    lines.extend(["", "### Post-hoc CEFR of train speakers (not used by gate)", ""])
    if "proficiency_level" in assign.columns:
        ct = pd.crosstab(assign["cluster_id"], assign["proficiency_level"].astype(str))
        levels = [c for c in PROFICIENCY_ORDER if c in ct.columns] + [
            c for c in ct.columns if c not in PROFICIENCY_ORDER
        ]
        header = "| Cluster | " + " | ".join(levels) + " |"
        align = "|---|" + "|".join("---:" for _ in levels) + "|"
        lines.extend([header, align])
        for cluster in cluster_ids(cfg):
            cells = [f"c{cluster}"]
            for level in levels:
                cells.append(str(int(ct.loc[cluster, level])) if cluster in ct.index and level in ct.columns else "0")
            lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path = results_dir(cfg) / "cluster_manifest.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    assign.to_csv(results_dir(cfg) / "speaker_assignment_train.csv", index=False)
    logger.info("Wrote %s", path)


def materialize_subsets(cfg: Dict[str, Any]) -> None:
    scored = pd.read_csv(scored_train_path(cfg))
    assign = pd.read_csv(assignment_path(cfg, "train"))
    merged = scored.merge(assign[[SPEAKER_COL, "cluster_id"]], on=SPEAKER_COL, how="left")
    error = merged[~merged["is_correct"].astype(bool)].copy()
    subset_csv_path(cfg, "error_global").parent.mkdir(parents=True, exist_ok=True)
    error.to_csv(subset_csv_path(cfg, "error_global"), index=False)
    logger.info("Subset error_global %d utt", len(error))
    for cluster in cluster_ids(cfg):
        sub = error[error["cluster_id"] == cluster].copy()
        path = subset_csv_path(cfg, cluster_subset_name(cluster))
        sub.to_csv(path, index=False)
        logger.info("Subset %s %d utt", cluster_subset_name(cluster), len(sub))


def _job_max_steps(cfg: Dict[str, Any], use_full: bool) -> int:
    if use_full:
        return int(cfg.get("max_steps", 4000))
    return int(cfg.get("subset_max_steps", 2000))


def _job_warmup_steps(cfg: Dict[str, Any], use_full: bool, max_steps: int) -> int:
    if use_full:
        return int(cfg.get("warmup_steps", 500))
    return int(cfg.get("subset_warmup_steps", min(200, max_steps // 10)))


def _finetune_job(cfg: Dict[str, Any], job_name: str, subset_name: str, use_full: bool = False) -> None:
    csv_path = subset_csv_path(cfg, subset_name)
    n = row_count(csv_path)
    out_dir = job_ckpt_dir(cfg, job_name)
    max_steps = _job_max_steps(cfg, use_full)
    if job_name == "error_gt_global" and cfg.get("reuse_global_error_ckpt"):
        reuse = resolve_repo_path(str(cfg["reuse_global_error_ckpt"]))
        if reuse.exists():
            logger.info("Reuse global error-GT checkpoint %s", reuse)
            return
    if n < min_subset_utterances(cfg):
        logger.warning("Skip train %s; only %d rows in %s", job_name, n, csv_path)
        return
    if training_completed(out_dir, max_steps):
        logger.info("Skip train %s; completed at %s", job_name, out_dir)
        return
    resume = _latest_checkpoint(out_dir)
    logger.info("Fine-tune %s on %s (%d utt) -> %s", job_name, subset_name, n, out_dir)
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
            warmup_steps=_job_warmup_steps(cfg, use_full, max_steps),
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


def train_models(cfg: Dict[str, Any]) -> None:
    jobs = [("error_gt_global", "error_global", False)]
    for cluster in cluster_ids(cfg):
        jobs.append((cluster_job_name(cluster), cluster_subset_name(cluster), False))
    for job_name, subset_name, use_full in progress_bar(jobs, total=len(jobs), desc="cluster finetune", unit="job"):
        _finetune_job(cfg, job_name, subset_name, use_full=use_full)


def _job_names(cfg: Dict[str, Any]) -> List[str]:
    names = ["error_gt_global"]
    names.extend(cluster_job_name(c) for c in cluster_ids(cfg))
    return names


def compute_taus(cfg: Dict[str, Any]) -> None:
    base_src = str(cfg.get("base_model", cfg["student_model"]))
    for job_name in _job_names(cfg):
        out = tau_path(cfg, job_name)
        if out.is_file():
            logger.info("Skip tau %s; found %s", job_name, out)
            continue
        try:
            weight_dir = resolve_weight_dir(job_ckpt_dir(cfg, job_name))
        except FileNotFoundError:
            logger.warning("Skip tau %s; no weights", job_name)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        logger.info("tau_%s = %s - %s", job_name, weight_dir, base_src)
        calculate_task_vector(str(weight_dir), base_src, str(out))


def _lambda_json(cfg: Dict[str, Any], method: str) -> Path:
    return results_dir(cfg) / f"best_lambda_{method}.json"


def _speakers_in_cluster(cfg: Dict[str, Any], split_key: str, cluster_id: int) -> List[str]:
    assign = pd.read_csv(assignment_path(cfg, split_key))
    mask = assign["cluster_id"].astype(int) == int(cluster_id)
    return assign.loc[mask, SPEAKER_COL].astype(str).tolist()


def _write_speaker_csv(src_csv: Path, speakers: Sequence[str], dest: Path) -> int:
    df = pd.read_csv(src_csv)
    sub = df[df[SPEAKER_COL].astype(str).isin(set(map(str, speakers)))]
    dest.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(dest, index=False)
    return len(sub)


def _eval_on_csv(model, processor, csv_path: Path, cfg: Dict[str, Any], desc: str) -> Dict[str, float]:
    if not csv_path.is_file() or row_count(csv_path) == 0:
        return _metrics_from_counts(0, 0, 0, 0, 0)
    loader = build_eval_dataloader(
        str(csv_path),
        processor=processor,
        batch_size=int(cfg.get("eval_batch_size", 4)),
        num_workers=int(cfg.get("num_workers", 2)),
        language=str(cfg.get("language", "en")),
    )
    return evaluate_model(model, loader, processor, desc=desc, **decode_kwargs(cfg))


def _load_tau_if_exists(cfg: Dict[str, Any], job_name: str) -> Optional[Dict[str, torch.Tensor]]:
    path = tau_path(cfg, job_name)
    if not path.is_file():
        return None
    return load_tau(str(path))


def select_lambdas(cfg: Dict[str, Any], device: str) -> None:
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    model = load_base_model(cfg, device_t)
    base_state = copy.deepcopy(model.state_dict())
    lambda_list = [float(x) for x in cfg.get("lambda_list", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])]
    dev_csv = split_csv(cfg, "dev")
    tmp_root = data_root(cfg) / "_lambda_dev"
    min_dev = int(cfg.get("min_cluster_dev_utterances", 16))
    tau_global = _load_tau_if_exists(cfg, "error_gt_global")
    tau_k = {c: _load_tau_if_exists(cfg, cluster_job_name(c)) for c in cluster_ids(cfg)}

    def restore_apply(tau: Optional[Dict[str, torch.Tensor]], lam: float) -> None:
        model.load_state_dict(base_state, strict=True)
        if tau is not None and abs(lam) > 1e-12:
            apply_task_vector(model, tau, float(lam))

    # global one-τ one-λ
    global_json = _lambda_json(cfg, "global_error_add")
    if not global_json.is_file() and tau_global is not None:
        rows = []
        for lam in progress_bar(lambda_list, total=len(lambda_list), desc="λ global", unit="λ"):
            restore_apply(tau_global, lam)
            metrics = _eval_on_csv(model, processor, dev_csv, cfg, desc=f"dev:global:{lam}")
            rows.append({"lambda": lam, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}})
        curve = pd.DataFrame(rows)
        curve.to_csv(results_dir(cfg) / "lambda_curve_global_error_add.csv", index=False)
        best = curve.loc[curve["wer"].idxmin()]
        global_json.write_text(
            json.dumps(
                {
                    "method": "global_error_add",
                    "best_lambda": float(best["lambda"]),
                    "best_wer": float(best["wer"]),
                    "lambda_list": lambda_list,
                    "selection_set": "source_dev",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    elif tau_global is None:
        logger.warning("No global error tau; skip global lambda search")

    global_lam = 0.7
    if global_json.is_file():
        global_lam = float(json.loads(global_json.read_text(encoding="utf-8"))["best_lambda"])

    # per-cluster λ with cluster τ_k  (main)
    main_json = _lambda_json(cfg, "cluster_tau_cluster_lambda")
    if not main_json.is_file():
        per_cluster: Dict[str, Any] = {}
        for cluster in cluster_ids(cfg):
            speakers = _speakers_in_cluster(cfg, "dev", cluster)
            n = _write_speaker_csv(dev_csv, speakers, tmp_root / f"dev_c{cluster}.csv")
            tau = tau_k.get(cluster)
            if tau is None or n < min_dev:
                logger.warning("Cluster %d dev utt=%d; fall back to global λ=%.2f", cluster, n, global_lam)
                per_cluster[str(cluster)] = {
                    "best_lambda": global_lam,
                    "best_wer": None,
                    "n_dev_utterances": n,
                    "fallback": True,
                }
                continue
            rows = []
            for lam in progress_bar(lambda_list, total=len(lambda_list), desc=f"λ c{cluster} τ_k", unit="λ"):
                restore_apply(tau, lam)
                metrics = _eval_on_csv(model, processor, tmp_root / f"dev_c{cluster}.csv", cfg, desc=f"dev:c{cluster}:{lam}")
                rows.append({"lambda": lam, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}})
            curve = pd.DataFrame(rows)
            curve.to_csv(results_dir(cfg) / f"lambda_curve_cluster_tau_c{cluster}.csv", index=False)
            best = curve.loc[curve["wer"].idxmin()]
            per_cluster[str(cluster)] = {
                "best_lambda": float(best["lambda"]),
                "best_wer": float(best["wer"]),
                "n_dev_utterances": n,
                "fallback": False,
            }
        main_json.write_text(
            json.dumps({"method": "cluster_tau_cluster_lambda", "per_cluster": per_cluster, "selection_set": "source_dev"}, indent=2),
            encoding="utf-8",
        )

    # one global τ, λ_k per cluster
    dose_json = _lambda_json(cfg, "global_tau_cluster_lambda")
    if not dose_json.is_file() and tau_global is not None:
        per_cluster = {}
        for cluster in cluster_ids(cfg):
            speakers = _speakers_in_cluster(cfg, "dev", cluster)
            n = _write_speaker_csv(dev_csv, speakers, tmp_root / f"dev_c{cluster}.csv")
            if n < min_dev:
                per_cluster[str(cluster)] = {"best_lambda": global_lam, "best_wer": None, "n_dev_utterances": n, "fallback": True}
                continue
            rows = []
            for lam in progress_bar(lambda_list, total=len(lambda_list), desc=f"λ c{cluster} τ_g", unit="λ"):
                restore_apply(tau_global, lam)
                metrics = _eval_on_csv(model, processor, tmp_root / f"dev_c{cluster}.csv", cfg, desc=f"dev:gτ:c{cluster}:{lam}")
                rows.append({"lambda": lam, **{k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}})
            curve = pd.DataFrame(rows)
            curve.to_csv(results_dir(cfg) / f"lambda_curve_global_tau_c{cluster}.csv", index=False)
            best = curve.loc[curve["wer"].idxmin()]
            per_cluster[str(cluster)] = {
                "best_lambda": float(best["lambda"]),
                "best_wer": float(best["wer"]),
                "n_dev_utterances": n,
                "fallback": False,
            }
        dose_json.write_text(
            json.dumps({"method": "global_tau_cluster_lambda", "per_cluster": per_cluster, "selection_set": "source_dev"}, indent=2),
            encoding="utf-8",
        )

    # τ_k with one shared λ on full dev
    shared_json = _lambda_json(cfg, "cluster_tau_global_lambda")
    if not shared_json.is_file():
        rows = []
        for lam in progress_bar(lambda_list, total=len(lambda_list), desc="λ shared τ_k", unit="λ"):
            parts = []
            for cluster in cluster_ids(cfg):
                speakers = _speakers_in_cluster(cfg, "dev", cluster)
                n = _write_speaker_csv(dev_csv, speakers, tmp_root / f"dev_c{cluster}.csv")
                if n == 0 or tau_k.get(cluster) is None:
                    continue
                restore_apply(tau_k[cluster], lam)
                metrics = _eval_on_csv(model, processor, tmp_root / f"dev_c{cluster}.csv", cfg, desc=f"dev:shared:c{cluster}:{lam}")
                parts.append(metrics)
            pooled = _pool_metric_list(parts)
            rows.append({"lambda": lam, **pooled})
        curve = pd.DataFrame(rows)
        if len(curve) and "wer" in curve.columns:
            curve.to_csv(results_dir(cfg) / "lambda_curve_cluster_tau_global_lambda.csv", index=False)
            best = curve.loc[curve["wer"].idxmin()]
            shared_json.write_text(
                json.dumps(
                    {
                        "method": "cluster_tau_global_lambda",
                        "best_lambda": float(best["lambda"]),
                        "best_wer": float(best["wer"]),
                        "lambda_list": lambda_list,
                        "selection_set": "source_dev",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    model.load_state_dict(base_state, strict=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _pool_metric_list(items: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not items:
        return _metrics_from_counts(0, 0, 0, 0, 0)
    return _metrics_from_counts(
        total_h=sum(float(m.get("hits", 0.0)) for m in items),
        total_s=sum(float(m.get("substitutions", 0.0)) for m in items),
        total_d=sum(float(m.get("deletions", 0.0)) for m in items),
        total_i=sum(float(m.get("insertions", 0.0)) for m in items),
        n_utterances=int(sum(float(m.get("n_utterances", 0.0)) for m in items)),
    )


def _prediction_csv(cfg: Dict[str, Any], split_key: str, method: str) -> Path:
    return results_dir(cfg) / f"predictions_{split_key}_{method}.csv"


def _cluster_lambdas(cfg: Dict[str, Any], method: str) -> Dict[int, float]:
    path = _lambda_json(cfg, method)
    if not path.is_file():
        return {c: 0.0 for c in cluster_ids(cfg)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "per_cluster" in payload:
        return {int(k): float(v["best_lambda"]) for k, v in payload["per_cluster"].items()}
    lam = float(payload.get("best_lambda", 0.0))
    return {c: lam for c in cluster_ids(cfg)}


def _eval_gated_split(
    cfg: Dict[str, Any],
    model,
    processor,
    base_state,
    split_key: str,
    method: str,
    tau_for_cluster: Dict[int, Optional[Dict[str, torch.Tensor]]],
    lambda_for_cluster: Dict[int, float],
    assignment_override: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    out_csv = _prediction_csv(cfg, split_key, method)
    if out_csv.is_file():
        return _metrics_from_pred_df(pd.read_csv(out_csv))
    assign = assignment_override if assignment_override is not None else pd.read_csv(assignment_path(cfg, split_key))
    tmp_root = data_root(cfg) / f"_eval_{split_key}_{method}"
    frames: List[pd.DataFrame] = []
    for cluster in cluster_ids(cfg):
        speakers = assign.loc[assign["cluster_id"].astype(int) == cluster, SPEAKER_COL].astype(str).tolist()
        n = _write_speaker_csv(split_csv(cfg, split_key), speakers, tmp_root / f"c{cluster}.csv")
        if n == 0:
            continue
        model.load_state_dict(base_state, strict=True)
        tau = tau_for_cluster.get(cluster)
        lam = float(lambda_for_cluster.get(cluster, 0.0))
        if tau is not None and abs(lam) > 1e-12:
            apply_task_vector(model, tau, lam)
        pred_path = tmp_root / f"pred_c{cluster}.csv"
        loader = build_eval_dataloader(
            str(tmp_root / f"c{cluster}.csv"),
            processor=processor,
            batch_size=int(cfg.get("eval_batch_size", 4)),
            num_workers=int(cfg.get("num_workers", 2)),
            language=str(cfg.get("language", "en")),
        )
        evaluate_model(
            model,
            loader,
            processor,
            output_csv_path=str(pred_path),
            desc=f"{split_key}:{method}:c{cluster}",
            **decode_kwargs(cfg),
        )
        if pred_path.is_file():
            frames.append(pd.read_csv(pred_path))
    if not frames:
        empty = pd.DataFrame(columns=["audio_path", "hits", "substitutions", "deletions", "insertions"])
        empty.to_csv(out_csv, index=False)
        return _metrics_from_counts(0, 0, 0, 0, 0)
    pred = pd.concat(frames, ignore_index=True)
    pred.to_csv(out_csv, index=False)
    return _metrics_from_pred_df(pred)


def evaluate_methods(cfg: Dict[str, Any], device: str) -> None:
    device_t = ensure_device(device)
    processor = load_processor(cfg)
    model = load_base_model(cfg, device_t)
    base_state = copy.deepcopy(model.state_dict())
    tau_global = _load_tau_if_exists(cfg, "error_gt_global")
    tau_k = {c: _load_tau_if_exists(cfg, cluster_job_name(c)) for c in cluster_ids(cfg)}
    lam_main = _cluster_lambdas(cfg, "cluster_tau_cluster_lambda")
    lam_dose = _cluster_lambdas(cfg, "global_tau_cluster_lambda")
    lam_shared = _cluster_lambdas(cfg, "cluster_tau_global_lambda")
    global_payload = _lambda_json(cfg, "global_error_add")
    global_lam = float(json.loads(global_payload.read_text(encoding="utf-8"))["best_lambda"]) if global_payload.is_file() else 0.0

    methods = {
        "theta_pre": (
            {c: None for c in cluster_ids(cfg)},
            {c: 0.0 for c in cluster_ids(cfg)},
        ),
        "global_error_add": (
            {c: tau_global for c in cluster_ids(cfg)},
            {c: global_lam for c in cluster_ids(cfg)},
        ),
        "cluster_tau_cluster_lambda": (tau_k, lam_main),
        "global_tau_cluster_lambda": (
            {c: tau_global for c in cluster_ids(cfg)},
            lam_dose,
        ),
        "cluster_tau_global_lambda": (tau_k, lam_shared),
    }

    split_summaries: Dict[str, Dict[str, Dict[str, float]]] = {key: {} for key, _ in eval_splits_from_cfg(cfg)}
    for split_key, label in eval_splits_from_cfg(cfg):
        for method, (taus, lams) in methods.items():
            metrics = _eval_gated_split(cfg, model, processor, base_state, split_key, method, taus, lams)
            split_summaries[split_key][method] = metrics
            logger.info("%s / %s WER=%.2f", label, method, metrics["wer"])

        # random cluster control using main (τ_k, λ_k)
        rng = np.random.RandomState(int(cfg.get("seed", 42)))
        assign = pd.read_csv(assignment_path(cfg, split_key))
        shuffled = assign.copy()
        labels = shuffled["cluster_id"].to_numpy().copy()
        rng.shuffle(labels)
        shuffled["cluster_id"] = labels
        metrics = _eval_gated_split(
            cfg,
            model,
            processor,
            base_state,
            split_key,
            "random_cluster_tau",
            tau_k,
            lam_main,
            assignment_override=shuffled,
        )
        split_summaries[split_key]["random_cluster_tau"] = metrics
        logger.info("%s / random_cluster_tau WER=%.2f", label, metrics["wer"])

    model.load_state_dict(base_state, strict=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for split_key, _ in eval_splits_from_cfg(cfg):
        write_eval_detailed_json(split_summaries[split_key], str(results_dir(cfg) / f"{split_key}_eval_detailed.json"))
    _write_experiment_report(cfg, split_summaries)


def _by_proficiency(cfg: Dict[str, Any], split_key: str, method: str) -> Dict[str, Dict[str, float]]:
    pred_path = _prediction_csv(cfg, split_key, method)
    if not pred_path.is_file():
        return {}
    pred = pd.read_csv(pred_path)
    meta = pd.read_csv(split_csv(cfg, split_key))
    cols = [c for c in ["audio_path", "proficiency_level"] if c in meta.columns]
    merged = pred.merge(meta[cols], on="audio_path", how="left")
    if "proficiency_level" not in merged.columns:
        return {}
    grouped: Dict[str, Dict[str, float]] = {}
    valid = merged[merged["proficiency_level"].notna() & (merged["proficiency_level"].astype(str).str.strip() != "")]
    for level, sub in valid.groupby(valid["proficiency_level"].astype(str).str.strip()):
        grouped[str(level)] = _metrics_from_pred_df(sub)
    return grouped


def _write_experiment_report(cfg: Dict[str, Any], split_summaries: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    methods = list(METHOD_LABELS.keys())
    lines = [
        f"# Experiment Report ({cfg.get('display_name', 'Speaker-Cluster Tau')})",
        "",
        "## Scope",
        "",
        "- Gate: k-means on **A01 train speaker** encoder embeddings.",
        "- τ_k: error-only GT fine-tune within cluster k, minus θ_now.",
        "- λ / λ_k selected on A01 dev only. Test and A02 are held out.",
        "- Gold CEFR is not an input to the gate.",
        f"- k={n_clusters(cfg)}; θ_now=`{cfg.get('base_model', cfg['student_model'])}`",
        "",
        "See `pipeline_flow.md` and `cluster_manifest.md`.",
        "",
        "## Selected lambdas (A01 dev)",
        "",
        "| Method | λ |",
        "|---|---|",
    ]
    for method in methods:
        path = _lambda_json(cfg, method)
        if method == "theta_pre":
            lines.append("| Pretrained (no tau) | 0.0 |")
            continue
        if method == "random_cluster_tau":
            lines.append("| Control: random cluster | uses gated λ_k |")
            continue
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "per_cluster" in payload:
            bits = [f"c{k}={v['best_lambda']}" for k, v in sorted(payload["per_cluster"].items(), key=lambda kv: int(kv[0]))]
            lines.append(f"| {METHOD_LABELS[method]} | {', '.join(bits)} |")
        else:
            lines.append(f"| {METHOD_LABELS[method]} | {payload.get('best_lambda')} |")
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
        pre_wer = float(split_summaries.get(split_key, {}).get("theta_pre", {}).get("wer", float("nan")))
        for method in methods:
            metrics = split_summaries.get(split_key, {}).get(method, {})
            if not metrics:
                continue
            wer = float(metrics.get("wer", float("nan")))
            lines.append(
                f"| {METHOD_LABELS[method]} | {wer:.2f} | {float(metrics.get('sub', float('nan'))):.2f} | "
                f"{float(metrics.get('ins', float('nan'))):.2f} | {float(metrics.get('del', float('nan'))):.2f} | "
                f"{pre_wer - wer:.2f} |"
            )
        lines.append("")
        lines.extend([f"### {label} by proficiency (gold, post-hoc)", ""])
        header = "| Proficiency | " + " | ".join(METHOD_LABELS[m] for m in methods) + " |"
        align = "|---|" + "|".join("---:" for _ in methods) + "|"
        lines.extend([header, align])
        by_method = {m: _by_proficiency(cfg, split_key, m) for m in methods}
        levels = sorted({lvl for grouped in by_method.values() for lvl in grouped}, key=lambda x: (PROFICIENCY_ORDER.get(x, 999), x))
        for level in levels:
            cells = [level]
            for method in methods:
                metrics = by_method.get(method, {}).get(level)
                cells.append(f"{metrics['wer']:.2f}" if metrics else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    report_path = results_dir(cfg) / "experiment_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report -> %s", report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="A01-only speaker-cluster gated tau")
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
    write_pipeline_flow(cfg)
    logger.info("Config=%s stages=%s device=%s", args.config, sorted(stages), args.device)

    with stage_progress(sorted(stages), desc=cfg.get("display_name", "Speaker-Cluster Tau")) as sp:
        if 1 in stages:
            sp.begin(1, titles[1])
            score_train(cfg, device=args.device)
            sp.end(1)
        if 2 in stages:
            sp.begin(2, titles[2])
            extract_split_embeddings(cfg, device=args.device)
            sp.end(2)
        if 3 in stages:
            sp.begin(3, titles[3])
            cluster_and_assign(cfg)
            sp.end(3)
        if 4 in stages:
            sp.begin(4, titles[4])
            materialize_subsets(cfg)
            sp.end(4)
        if 5 in stages:
            sp.begin(5, titles[5])
            train_models(cfg)
            sp.end(5)
        if 6 in stages:
            sp.begin(6, titles[6])
            compute_taus(cfg)
            sp.end(6)
        if 7 in stages:
            sp.begin(7, titles[7])
            select_lambdas(cfg, device=args.device)
            sp.end(7)
        if 8 in stages:
            sp.begin(8, titles[8])
            evaluate_methods(cfg, device=args.device)
            sp.end(8)


if __name__ == "__main__":
    main()
