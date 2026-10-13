#!/usr/bin/env python3
"""Diagnostic analysis of soft proficiency-affinity composition on TEEMI A01 test.

Does not train, and does not write into existing checkpoint or result directories.
tau in the paper is stored as ``lambda`` in the existing experiment JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import run_teemi_a01_only_smooth_tau as pipe  # noqa: E402

OUT_ROOT = PROJECT_ROOT / "analysis"
LEVELS = ["pre-A", "A1", "A1A2", "A2", "A2B1", "B1", "B1B2", "B2"]
DISPLAY = {
    "pre-A": "pre-A",
    "A1": "A1",
    "A1A2": "A1+",
    "A2": "A2",
    "A2B1": "A2+",
    "B1": "B1",
    "B1B2": "B1+",
    "B2": "B2",
}
DISPLAY_ORDER = [DISPLAY[level] for level in LEVELS]
ALPHA_SRC = [f"alpha_{pipe.slugify_level(level)}" for level in LEVELS]
ALPHA_OUT = [f"alpha_{DISPLAY[level]}" for level in LEVELS]

# Paper notation tau. Existing configs store it as best_lambda.
EXPECTED = {
    "whisper_base": {
        "config": "configs/teemi_a01_only_pgcv_whisper_base.yaml",
        "panel": "(a) Whisper-base",
        "label": "Whisper-base",
        "beta": 0.25,
        "tau": 1.0,
        "soft_wer": 12.03,
        "global_wer": 11.97,
    },
    "qwen3_asr_0_6b": {
        "config": "configs/teemi_a01_only_pgcv_qwen3_asr_0.6b.yaml",
        "panel": "(b) Qwen3-ASR-0.6B",
        "label": "Qwen3-ASR-0.6B",
        "beta": 0.25,
        "tau": 0.7,
        "soft_wer": 8.43,
        "global_wer": 8.57,
    },
}
CONDITIONS = [
    ("global", "Global"),
    ("hard_affinity", "Hard affinity"),
    ("oracle_hard", "Oracle hard proficiency"),
    ("soft_pgcv", "Soft PGCV"),
]


def entropy_rows(alpha: np.ndarray) -> np.ndarray:
    clipped = np.clip(alpha, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1)


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in nats (natural log), not the sqrt distance."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def adjacent(a: str, b: str) -> bool:
    ia = LEVELS.index(a)
    ib = LEVELS.index(b)
    return abs(ia - ib) == 1


def load_assignment(slug: str) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "teemi" / "a01_only" / "pgcv" / slug / "assignment_test.csv"
    frame = pd.read_csv(path)
    missing = [c for c in ["global_speaker_id", "n_utterances", "proficiency_level", "hard_id", "hard_level", *ALPHA_SRC] if c not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing columns {missing}")
    alpha = frame[ALPHA_SRC].to_numpy(dtype=np.float64)
    argmax = alpha.argmax(axis=1)
    if not np.array_equal(argmax, frame["hard_id"].to_numpy(dtype=int)):
        raise RuntimeError(f"{slug}: saved hard_id is not argmax(alpha_s)")
    return frame


def speaker_table(frame: pd.DataFrame) -> pd.DataFrame:
    alpha = frame[ALPHA_SRC].to_numpy(dtype=np.float64)
    argmax = alpha.argmax(axis=1)
    rows = {
        "speaker_id": frame["global_speaker_id"].astype(str),
        "gold_proficiency": frame["proficiency_level"].map(DISPLAY),
        "argmax_affinity_level": [DISPLAY[LEVELS[i]] for i in argmax],
        "max_affinity": alpha.max(axis=1),
        "affinity_entropy": entropy_rows(alpha),
        "num_utterances": frame["n_utterances"].astype(int),
    }
    out = pd.DataFrame(rows)
    for src, dst in zip(ALPHA_SRC, ALPHA_OUT):
        out[dst] = frame[src].to_numpy(dtype=np.float64)
    ordered = [
        "speaker_id",
        "gold_proficiency",
        *ALPHA_OUT,
        "argmax_affinity_level",
        "max_affinity",
        "affinity_entropy",
        "num_utterances",
    ]
    return out[ordered]


def affinity_matrix(speakers: pd.DataFrame) -> pd.DataFrame:
    matrix = []
    for gold in DISPLAY_ORDER:
        sub = speakers.loc[speakers["gold_proficiency"] == gold, ALPHA_OUT]
        if len(sub) == 0:
            matrix.append([np.nan] * len(DISPLAY_ORDER))
        else:
            matrix.append(sub.mean(axis=0).to_numpy(dtype=np.float64))
    return pd.DataFrame(matrix, index=DISPLAY_ORDER, columns=DISPLAY_ORDER)


def plot_heatmap(matrix: pd.DataFrame, panel: str, dest_stem: Path) -> None:
    values = matrix.to_numpy(dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=max(0.5, float(np.nanmax(values))), aspect="equal")
    ax.set_xticks(range(len(DISPLAY_ORDER)))
    ax.set_yticks(range(len(DISPLAY_ORDER)))
    ax.set_xticklabels(DISPLAY_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(DISPLAY_ORDER)
    ax.set_xlabel("Proficiency-specific correction")
    ax.set_ylabel("Gold proficiency")
    ax.set_title(panel)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val >= 0.45 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest_stem.with_suffix(".png"), dpi=300)
    fig.savefig(dest_stem.with_suffix(".pdf"))
    plt.close(fig)


def within_level(speakers: pd.DataFrame) -> pd.DataFrame:
    alpha = speakers[ALPHA_OUT].to_numpy(dtype=np.float64)
    records = []
    for gold_disp, gold_code in zip(DISPLAY_ORDER, LEVELS):
        mask = speakers["gold_proficiency"].to_numpy() == gold_disp
        block = alpha[mask]
        if len(block) == 0:
            continue
        mean_vec = block.mean(axis=0)
        std_vec = block.std(axis=0, ddof=0)
        argmax = block.argmax(axis=1)
        gold_idx = LEVELS.index(gold_code)
        match = argmax == gold_idx
        neighbor = np.array([abs(int(i) - gold_idx) == 1 for i in argmax])
        js = [js_divergence(row, mean_vec) for row in block]
        rec = {
            "gold_proficiency": gold_disp,
            "n_speakers": int(len(block)),
            "mean_entropy": float(entropy_rows(block).mean()),
            "median_entropy": float(np.median(entropy_rows(block))),
            "mean_max_affinity": float(block.max(axis=1).mean()),
            "pct_argmax_equals_gold": float(100.0 * match.mean()),
            "pct_argmax_adjacent": float(100.0 * neighbor.mean()),
            "mean_js_to_group_mean_nats": float(np.mean(js)),
            "median_js_to_group_mean_nats": float(np.median(js)),
        }
        for name, mu, sd in zip(DISPLAY_ORDER, mean_vec, std_vec):
            rec[f"mean_alpha_{name}"] = float(mu)
            rec[f"std_alpha_{name}"] = float(sd)
        records.append(rec)
    return pd.DataFrame(records)


def score_predictions(pred_path: Path, meta: pd.DataFrame) -> Dict[str, float]:
    pred = pd.read_csv(pred_path)
    scored = pipe.wer_by_level(pred, meta)
    out = {DISPLAY.get(level, level): float(scored[level]["wer"]) for level in LEVELS if level in scored}
    out["overall"] = float(scored["__overall__"]["wer"])
    missing = [DISPLAY[level] for level in LEVELS if DISPLAY[level] not in out]
    if missing:
        raise RuntimeError(f"{pred_path} missing proficiency bands {missing}")
    return out


def speaker_wer(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    cols = ["audio_path", "global_speaker_id"]
    merged = pred.merge(meta[cols], on="audio_path", how="inner")
    rows = {}
    for speaker, sub in merged.groupby("global_speaker_id"):
        metrics = pipe.metrics_from_pred_df(sub)
        rows[str(speaker)] = metrics["wer"]
    return pd.Series(rows, name="wer")


def one_hot_assignment(base: pd.DataFrame, index: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    alpha = np.zeros((len(out), len(LEVELS)), dtype=np.float64)
    alpha[np.arange(len(out)), index] = 1.0
    for col, values in zip(ALPHA_SRC, alpha.T):
        out[col] = values
    out["hard_id"] = index.astype(int)
    out["hard_level"] = [LEVELS[i] for i in index]
    return out


def decode_condition(cfg, model, processor, base_state, tau_g, taus, perps, assignment: pd.DataFrame, dest: Path, tmp_root: Path) -> None:
    """Decode with the existing PGCV mixer. Assignment is a temporary one-hot table."""
    work_cfg = dict(cfg)
    work_root = tmp_root / "assignment_root"
    slug = pipe.family_slug(cfg["student_model"])
    assign_dir = work_root / slug
    assign_dir.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(assign_dir / "assignment_test.csv", index=False)
    work_cfg["data_dir"] = str(work_root.resolve())
    last_error = None
    for attempt in range(3):
        try:
            pipe.decode_method_split(
                work_cfg,
                model,
                processor,
                base_state,
                tau_g,
                taus,
                perps,
                "test",
                "attn_hard_smooth",
                float(cfg["_analysis_beta"]),
                float(cfg["_analysis_tau"]),
                dest,
                tmp_root,
            )
            last_error = None
            break
        except RuntimeError as exc:
            last_error = exc
            message = str(exc)
            if "predictions for" not in message or attempt == 2:
                raise
            # Transient NAS/dataloader shortfalls: delete the short group file and retry.
            for part in tmp_root.glob("test_attn_hard_smooth_*.csv"):
                if part.name.endswith("_utts.csv"):
                    continue
                utts = part.with_name(part.name.replace(".csv", "_utts.csv"))
                if not utts.is_file():
                    continue
                if pipe._pred_row_count(part) != pipe._pred_row_count(utts):
                    part.unlink(missing_ok=True)
    if last_error is not None:
        raise last_error


def ensure_model(cfg, device: str):
    device_t = pipe.ensure_device(device)
    processor = pipe.load_asr_processor(cfg["student_model"])
    model = pipe.load_asr_model(cfg.get("base_model", cfg["student_model"]), device=device_t)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    # Hard/oracle use mix_smooth, so residual perps are unused. Skip ~24GB NAS reads.
    print(f"Loading Delta_global from {pipe.global_tau_path(cfg)}", flush=True)
    tau_g = pipe.load_tau(str(pipe.global_tau_path(cfg)))
    taus = []
    for level in pipe.cfg_levels(cfg):
        path = pipe.level_tau_path(cfg, level)
        print(f"Loading Delta_{level} from {path}", flush=True)
        taus.append(pipe.load_tau(str(path)))
    perps: List[Dict[str, torch.Tensor]] = []
    return model, processor, base_state, tau_g, taus, perps


def describe_matrix(matrix: pd.DataFrame) -> str:
    values = matrix.to_numpy(dtype=np.float64)
    n = values.shape[0]
    diag = float(np.mean(np.diag(values)))
    band_mass = float(
        np.mean(
            [
                np.sum([values[i, j] for j in range(n) if abs(i - j) <= 1])
                for i in range(n)
            ]
        )
    )
    far_mass = 1.0 - band_mass
    col_mean = values.mean(axis=0)
    top_col = DISPLAY_ORDER[int(np.argmax(col_mean))]
    row_entropy = float(np.mean(entropy_rows(values)))
    logn = float(np.log(n))
    if col_mean.max() >= 0.40 and (col_mean.max() - np.partition(col_mean, -2)[-2]) >= 0.18:
        pattern = "collapsed toward one proficiency-specific correction"
    elif diag >= 0.50 and band_mass >= 0.75:
        pattern = "close to a hard diagonal"
    elif band_mass >= 0.55:
        pattern = "an ordered soft/banded structure around gold and neighboring levels"
    elif row_entropy >= 0.80 * logn or far_mass >= 0.55:
        pattern = "diffuse across proficiency-specific corrections"
    else:
        pattern = "neither a hard diagonal nor a fully diffuse assignment"
    return (
        f"{pattern}. Mean diagonal mass is {diag:.2f}; mean mass on the gold level and its "
        f"immediate neighbors is {band_mass:.2f}; mean row entropy is {row_entropy:.2f} nats "
        f"(uniform entropy is {logn:.2f}); the largest column mean belongs to {top_col} "
        f"({col_mean.max():.2f})."
    )


def run_backbone(slug: str, device: str, skip_decode: bool) -> Dict:
    spec = EXPECTED[slug]
    cfg = pipe.load_config(str(PROJECT_ROOT / spec["config"]))
    best_pgcv = json.loads((pipe.results_dir(cfg) / "best_pgcv.json").read_text())
    best_global = json.loads((pipe.results_dir(cfg) / "best_global_error_add.json").read_text())
    published_tau = float(pipe.published_global_lambda(cfg))
    pgcv_beta = float(best_pgcv["beta"])
    pgcv_tau = float(best_pgcv["best_lambda"])
    global_tau = float(best_global.get("published_lambda", best_global["best_lambda"]))
    if not np.isclose(pgcv_beta, spec["beta"]) or not np.isclose(pgcv_tau, spec["tau"]):
        raise SystemExit(
            f"{slug}: config selection beta={pgcv_beta}, tau(lambda)={pgcv_tau} "
            f"differs from requested beta={spec['beta']}, tau={spec['tau']}"
        )
    if not np.isclose(published_tau, global_tau):
        raise SystemExit(f"{slug}: published global tau {published_tau} != recorded {global_tau}")

    out_dir = OUT_ROOT / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    assignment = load_assignment(slug)
    speakers = speaker_table(assignment)
    speakers_out = speakers.copy()
    speakers_out.insert(0, "backbone", spec["label"])
    speakers.to_csv(out_dir / "speaker_affinities.csv", index=False)
    speakers_out.to_csv(out_dir / "speaker_affinities_with_backbone.csv", index=False)
    matrix = affinity_matrix(speakers)
    matrix.to_csv(out_dir / "affinity_matrix.csv", index_label="gold_proficiency")
    plot_heatmap(matrix, spec["panel"], out_dir / "affinity_heatmap")
    plot_heatmap(matrix, spec["panel"], OUT_ROOT / f"affinity_heatmap_{slug}")
    variation = within_level(speakers)
    variation.to_csv(out_dir / "within_level_variation.csv", index=False)

    meta = pd.read_csv(pipe.split_csv(cfg, "test"))
    soft_path = pipe.results_dir(cfg) / "predictions_test_pgcv.csv"
    global_path = pipe.results_dir(cfg) / "predictions_test_global_error_add.csv"
    soft_scores = score_predictions(soft_path, meta)
    global_scores = score_predictions(global_path, meta)
    if abs(soft_scores["overall"] - spec["soft_wer"]) > 0.1:
        raise SystemExit(
            f"{slug}: Soft PGCV overall WER {soft_scores['overall']:.4f} differs from "
            f"existing result {spec['soft_wer']} by more than 0.1. Stopping before ablation interpretation."
        )

    hard_path = out_dir / "predictions_hard_affinity.csv"
    oracle_path = out_dir / "predictions_oracle_hard.csv"
    if not skip_decode:
        cfg["_analysis_beta"] = pgcv_beta
        cfg["_analysis_tau"] = pgcv_tau
        model, processor, base_state, tau_g, taus, perps = ensure_model(cfg, device)
        tmp = out_dir / "_decode_tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        hard_assign = one_hot_assignment(assignment, assignment["hard_id"].to_numpy(dtype=int))
        gold_index = assignment["proficiency_level"].map({level: i for i, level in enumerate(LEVELS)}).to_numpy(dtype=int)
        if np.isnan(gold_index).any():
            raise RuntimeError(f"{slug}: gold proficiency is not in the fixed level order")
        oracle_assign = one_hot_assignment(assignment, gold_index.astype(int))
        print(f"{slug}: decoding Hard affinity -> {hard_path}", flush=True)
        decode_condition(cfg, model, processor, base_state, tau_g, taus, perps, hard_assign, hard_path, tmp / "hard")
        print(f"{slug}: decoding Oracle hard -> {oracle_path}", flush=True)
        decode_condition(cfg, model, processor, base_state, tau_g, taus, perps, oracle_assign, oracle_path, tmp / "oracle")
        del model, base_state, tau_g, taus, perps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not hard_path.is_file() or not oracle_path.is_file():
        if skip_decode:
            summary = {
                "slug": slug,
                "label": spec["label"],
                "panel": spec["panel"],
                "beta": pgcv_beta,
                "tau": pgcv_tau,
                "global_tau": global_tau,
                "n_speakers": int(len(speakers)),
                "n_per_level": speakers["gold_proficiency"].value_counts().reindex(DISPLAY_ORDER).fillna(0).astype(int).to_dict(),
                "mean_entropy": float(speakers["affinity_entropy"].mean()),
                "median_entropy": float(speakers["affinity_entropy"].median()),
                "pct_argmax_gold": float(
                    100.0
                    * (speakers["argmax_affinity_level"] == speakers["gold_proficiency"]).mean()
                ),
                "pct_argmax_adjacent": float(
                    100.0
                    * np.mean(
                        [
                            adjacent(
                                {DISPLAY[k]: k for k in LEVELS}[a],
                                {DISPLAY[k]: k for k in LEVELS}[b],
                            )
                            for a, b in zip(
                                speakers["argmax_affinity_level"],
                                speakers["gold_proficiency"],
                            )
                        ]
                    )
                ),
                "matrix_description": describe_matrix(matrix),
                "mean_js": float(variation["mean_js_to_group_mean_nats"].mean()),
                "wer": {
                    "Global": global_scores,
                    "Soft PGCV": soft_scores,
                },
                "decode_complete": False,
                "reproduced_soft": soft_scores["overall"],
                "reproduced_global": global_scores["overall"],
            }
            (out_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
        raise SystemExit(f"{slug}: hard/oracle predictions missing. Re-run without --skip-decode.")

    hard_scores = score_predictions(hard_path, meta)
    oracle_scores = score_predictions(oracle_path, meta)
    by_method = {
        "Global": global_scores,
        "Hard affinity": hard_scores,
        "Oracle hard proficiency": oracle_scores,
        "Soft PGCV": soft_scores,
    }
    method_rows = [{"method": name, "overall_wer": scores["overall"]} for name, scores in by_method.items()]
    pd.DataFrame(method_rows).to_csv(out_dir / "method_wer.csv", index=False)
    strat_rows = []
    for name, scores in by_method.items():
        row = {"method": name, **{level: scores[level] for level in DISPLAY_ORDER}, "overall": scores["overall"]}
        strat_rows.append(row)
    pd.DataFrame(strat_rows).to_csv(out_dir / "proficiency_wer.csv", index=False)

    soft_spk = speaker_wer(pd.read_csv(soft_path), meta)
    hard_spk = speaker_wer(pd.read_csv(hard_path), meta)
    joined = speakers.merge(
        pd.DataFrame({"speaker_id": soft_spk.index, "wer_soft": soft_spk.values}),
        on="speaker_id",
        how="inner",
    ).merge(
        pd.DataFrame({"speaker_id": hard_spk.index, "wer_hard": hard_spk.values}),
        on="speaker_id",
        how="inner",
    )
    joined["soft_gain"] = joined["wer_hard"] - joined["wer_soft"]
    joined.to_csv(out_dir / "speaker_soft_gain.csv", index=False)
    rho, pvalue = spearmanr(joined["affinity_entropy"], joined["soft_gain"])
    plot_scatter(joined, spec["panel"], out_dir / "entropy_soft_gain", float(rho), float(pvalue))

    alpha = speakers[ALPHA_OUT].to_numpy(dtype=np.float64)
    gold_codes = speakers["gold_proficiency"].map({DISPLAY[k]: k for k in LEVELS}).to_numpy()
    argmax_codes = speakers["argmax_affinity_level"].map({DISPLAY[k]: k for k in LEVELS}).to_numpy()
    match = argmax_codes == gold_codes
    neighbor = np.array([adjacent(a, b) for a, b in zip(argmax_codes, gold_codes)])
    ent = speakers["affinity_entropy"].to_numpy()
    summary = {
        "slug": slug,
        "label": spec["label"],
        "panel": spec["panel"],
        "beta": pgcv_beta,
        "tau": pgcv_tau,
        "global_tau": global_tau,
        "global_tau_equals_pgcv_tau": bool(np.isclose(global_tau, pgcv_tau)),
        "n_speakers": int(len(speakers)),
        "n_per_level": speakers["gold_proficiency"].value_counts().reindex(DISPLAY_ORDER).fillna(0).astype(int).to_dict(),
        "mean_entropy": float(ent.mean()),
        "median_entropy": float(np.median(ent)),
        "pct_argmax_gold": float(100.0 * match.mean()),
        "pct_argmax_adjacent": float(100.0 * neighbor.mean()),
        "pct_argmax_gold_or_adjacent": float(100.0 * (match | neighbor).mean()),
        "matrix_description": describe_matrix(matrix),
        "mean_js": float(variation["mean_js_to_group_mean_nats"].mean()),
        "wer": {name: scores["overall"] for name, scores in by_method.items()},
        "stratified": by_method,
        "spearman_rho": float(rho),
        "spearman_p": float(pvalue),
        "n_scatter": int(len(joined)),
        "decode_complete": True,
        "reproduced_soft": soft_scores["overall"],
        "reproduced_global": global_scores["overall"],
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def plot_scatter(frame: pd.DataFrame, panel: str, dest_stem: Path, rho: float, pvalue: float) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.scatter(frame["affinity_entropy"], frame["soft_gain"], s=18, c="#1f4e79", alpha=0.75, edgecolors="none")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"Affinity entropy $H(\alpha_s)$ (nats)")
    ax.set_ylabel(r"soft_gain$_s$ = WER$_{hard}$ $-$ WER$_{soft}$")
    ax.set_title(f"{panel}\nSpearman $\\rho$={rho:.2f}, p={pvalue:.3g}, n={len(frame)}")
    fig.tight_layout()
    fig.savefig(dest_stem.with_suffix(".png"), dpi=300)
    fig.savefig(dest_stem.with_suffix(".pdf"))
    plt.close(fig)


def _wer_value(row: Dict, method: str) -> Optional[float]:
    scores = row.get("wer", {}).get(method)
    if scores is None:
        return None
    if isinstance(scores, dict):
        return float(scores["overall"])
    return float(scores)


def _relation(soft: float, other: float) -> str:
    if soft < other - 1e-9:
        return f"lower than ({soft:.2f} vs {other:.2f})"
    if soft > other + 1e-9:
        return f"higher than ({soft:.2f} vs {other:.2f}); no advantage is observed"
    return f"the same as ({soft:.2f})"


def write_requested_tables(rows: Sequence[Dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    affinity_frames = []
    for row in rows:
        path = OUT_ROOT / row["slug"] / "speaker_affinities_with_backbone.csv"
        if path.is_file():
            affinity_frames.append(pd.read_csv(path))
    if affinity_frames:
        pd.concat(affinity_frames, ignore_index=True).to_csv(
            OUT_ROOT / "speaker_affinities.csv", index=False
        )

    comparison_rows = []
    for row in rows:
        if not row.get("decode_complete"):
            continue
        for key, label in CONDITIONS:
            scores = row["stratified"][label]
            rec = {
                "backbone": row["label"],
                "method": label,
                "beta": row["beta"],
                "tau": row["tau"],
                "overall_wer": scores["overall"],
            }
            for level in DISPLAY_ORDER:
                rec[f"wer_{level}"] = scores[level]
            comparison_rows.append(rec)
    if comparison_rows:
        pd.DataFrame(comparison_rows).to_csv(
            OUT_ROOT / "soft_hard_oracle_global_wer_comparison.csv", index=False
        )


def write_summary(rows: Sequence[Dict]) -> None:
    write_requested_tables(rows)
    decode_ready = all(row.get("decode_complete") for row in rows)
    lines = [
        "# PGCV soft proficiency composition analysis (TEEMI A01 test)",
        "",
        "Gold proficiency is used only for analysis. It is not an input to the gate.",
        "Notation follows the paper: `tau` is correction strength and `beta` mixes `Delta_global` with `Delta_prof,s`.",
        "Existing experiment files store `tau` as `best_lambda`. Selected values are Whisper-base `beta=0.25`, `tau=1.0` and Qwen3-ASR-0.6B `beta=0.25`, `tau=0.7`.",
        "Soft PGCV and Global WER are taken from the existing prediction files. Hard affinity and oracle hard are decoded with the same mixer, same `beta`/`tau`, and the existing correction bank.",
        "",
        "Sanity check against the requested Soft PGCV overall WER:",
    ]
    for row in rows:
        lines.append(
            f"- {row['label']}: reproduced Soft PGCV = {row['reproduced_soft']:.2f} "
            f"(target {EXPECTED[row['slug']]['soft_wer']:.2f}); "
            f"reproduced Global = {row['reproduced_global']:.2f}."
        )
    lines += ["", "## 1. Does affinity show an ordered proficiency structure?", ""]
    for row in rows:
        lines.append(
            f"- {row['label']}: {row['matrix_description']} "
            f"argmax equals gold for {row['pct_argmax_gold']:.1f}% of speakers; "
            f"argmax is adjacent for {row['pct_argmax_adjacent']:.1f}%. "
            f"Mean affinity entropy is {row['mean_entropy']:.3f} nats."
        )
    if not decode_ready:
        lines += [
            "",
            "## 2–4. Soft vs hard / oracle / global",
            "",
            "Hard affinity and oracle-hard predictions are not yet available. "
            "Affinity heatmaps and `speaker_affinities.csv` are written; WER comparison is withheld until decode finishes.",
            "",
        ]
        (OUT_ROOT / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
        return

    lines += ["", "## 2. Is soft composition better than hard affinity?", ""]
    for row in rows:
        soft = _wer_value(row, "Soft PGCV")
        hard = _wer_value(row, "Hard affinity")
        lines.append(f"- {row['label']}: Soft PGCV overall WER is {_relation(soft, hard)} hard affinity.")
    lines += ["", "## 3. Is soft better than oracle gold-level selection?", ""]
    for row in rows:
        soft = _wer_value(row, "Soft PGCV")
        oracle = _wer_value(row, "Oracle hard proficiency")
        lines.append(
            f"- {row['label']}: Soft PGCV overall WER is {_relation(soft, oracle)} oracle hard proficiency."
        )
    lines += ["", "## 4. Is soft better than global?", ""]
    for row in rows:
        soft = _wer_value(row, "Soft PGCV")
        glob = _wer_value(row, "Global")
        lines.append(f"- {row['label']}: Soft PGCV overall WER is {_relation(soft, glob)} Global.")
    lines += ["", "## 5. Are Whisper and Qwen consistent?", ""]
    if len(rows) < 2:
        lines.append("Only one backbone is available in this run; consistency is not evaluated.")
    else:
        same = []
        different = []
        for label in ["Hard affinity", "Oracle hard proficiency", "Global"]:
            w_soft = _wer_value(rows[0], "Soft PGCV")
            q_soft = _wer_value(rows[1], "Soft PGCV")
            w_other = _wer_value(rows[0], label)
            q_other = _wer_value(rows[1], label)
            w_cmp = np.sign(w_soft - w_other)
            q_cmp = np.sign(q_soft - q_other)
            if w_cmp == q_cmp:
                same.append(label)
            else:
                different.append(label)
        if same:
            lines.append("The overall Soft-vs-control direction is the same on both backbones for: " + ", ".join(same) + ".")
        if different:
            lines.append("The overall Soft-vs-control direction differs between backbones for: " + ", ".join(different) + ".")
        if not different:
            lines.append("On overall WER, the two backbones agree on every Soft-vs-control comparison reported here.")
        else:
            lines.append("The two backbones are not fully consistent on overall WER ordering.")
    lines += ["", "## Overall WER", "", "| method | Whisper-base | Qwen3-ASR-0.6B |", "| --- | ---: | ---: |"]
    by_slug = {row["slug"]: row for row in rows}
    for _, label in CONDITIONS:
        w = _wer_value(by_slug["whisper_base"], label) if "whisper_base" in by_slug else None
        q = _wer_value(by_slug["qwen3_asr_0_6b"], label) if "qwen3_asr_0_6b" in by_slug else None
        w_txt = f"{w:.2f}" if w is not None else "—"
        q_txt = f"{q:.2f}" if q is not None else "—"
        lines.append(f"| {label} | {w_txt} | {q_txt} |")
    lines += ["", "## Speaker variation within the same gold proficiency", ""]
    for row in rows:
        var_path = OUT_ROOT / row["slug"] / "within_level_variation.csv"
        if not var_path.is_file():
            continue
        var = pd.read_csv(var_path)
        ent = var["mean_entropy"].to_numpy(dtype=np.float64)
        js = var["mean_js_to_group_mean_nats"].to_numpy(dtype=np.float64)
        lines.append(
            f"- {row['label']}: within-gold mean entropy ranges from {ent.min():.2f} to {ent.max():.2f} nats; "
            f"mean JS to the group-mean affinity is {js.mean():.3f} nats. "
            "Speakers who share the same gold label still differ in `alpha_s`; the spread is larger at mid levels than at B2."
        )
    lines += [
        "",
        "Per-level WER is in `soft_hard_oracle_global_wer_comparison.csv`.",
        "Within-gold speaker variation is in each backbone's `within_level_variation.csv`.",
        "",
    ]
    (OUT_ROOT / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_ROOT / "comparison" / "analysis_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "comparison" / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _compare_line(index: int, baseline: str, method: str, row: Dict) -> str:
    a = row["wer"][method]
    b = row["wer"][baseline]
    if a < b - 1e-9:
        relation = f"lower than {baseline} ({a:.2f} vs {b:.2f})"
    elif a > b + 1e-9:
        relation = f"higher than {baseline} ({a:.2f} vs {b:.2f}). No advantage over this control is observed"
    else:
        relation = f"the same as {baseline} ({a:.2f})"
    return f"{index}. Soft PGCV overall WER is {relation}."


def _load_saved_summary(slug: str) -> Optional[Dict]:
    path = OUT_ROOT / slug / "diagnostics.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("decode_complete", bool(payload.get("stratified")))
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--only", default="", help="whisper_base or qwen3_asr_0_6b")
    args = parser.parse_args()
    slugs = [args.only] if args.only else list(EXPECTED)
    rows = [run_backbone(slug, args.device, args.skip_decode) for slug in slugs]
    have = {row["slug"] for row in rows}
    for slug in EXPECTED:
        if slug in have:
            continue
        saved = _load_saved_summary(slug)
        if saved:
            rows.append(saved)
    rows.sort(key=lambda row: list(EXPECTED).index(row["slug"]))
    if rows:
        write_summary(rows)
    if len(rows) != len(EXPECTED):
        print("Partial backbone run; combined tables use only the completed models.")


if __name__ == "__main__":
    main()
