#!/usr/bin/env python3
"""Official S&I SpWER for a PGCV configuration selected on dev segment WER."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from decode_sandi_qwen_ctm import decode as decode_qwen  # noqa: E402
from decode_sandi_whisper_longform import decode as decode_whisper  # noqa: E402
from run_teemi_a01_only_smooth_tau import (  # noqa: E402
    _groups_for_method,
    _load_tau_bank,
    cfg_levels,
    family_slug,
    gate_id_column,
    load_assignment,
    load_best,
    load_config,
    mix_smooth,
    resolve_repo_path,
    results_dir,
)
from score_sandi_spwer import score_ctm  # noqa: E402
from src.asr_backend import is_qwen3_asr  # noqa: E402


CORPUS_DEFAULT = Path("/share/corpus/sandi2025-challenge")


def merge_ctms(paths: Sequence[Path], destination: Path) -> Path:
    lines: List[str] = []
    for path in paths:
        if path.is_file():
            lines.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    lines.sort(key=lambda line: (line.split()[0], float(line.split()[2])))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return destination


def subset_reference_files(
    corpus: Path,
    split: str,
    level: str,
    file_ids: Sequence[str],
    output_dir: Path,
) -> tuple[Path, Path]:
    wanted = {str(file_id) for file_id in file_ids}
    source_flist = corpus / "reference-materials" / "flists.flac" / f"{split}-asr.tsv"
    source_stm = corpus / "reference-materials" / "stms" / f"{split}-asr.stm"
    flist = output_dir / f"{split}-{level}.tsv"
    stm = output_dir / f"{split}-{level}.stm"
    flist_lines = [
        line
        for line in source_flist.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.split()[0] in wanted
    ]
    stm_lines = [
        line
        for line in source_stm.read_text(encoding="utf-8").splitlines()
        if line.strip() and (line.startswith(";;") or line.split()[0] in wanted)
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    flist.write_text("\n".join(flist_lines) + "\n", encoding="utf-8")
    stm.write_text("\n".join(stm_lines) + "\n", encoding="utf-8")
    return flist, stm


def file_level_map(cfg: Dict[str, Any], split: str) -> Dict[str, str]:
    frame = pd.read_csv(resolve_repo_path(str(cfg["splits"][split])))
    mapping: Dict[str, str] = {}
    for file_id, part in frame.groupby("file_id", sort=False):
        labels = [str(value) for value in part["proficiency_level"].dropna()]
        if labels:
            mapping[str(file_id)] = Counter(labels).most_common(1)[0][0]
    return mapping


def expand_gate_ids_to_file_ids(
    cfg: Dict[str, Any],
    split: str,
    gate_ids: Sequence[str],
) -> List[str]:
    id_column = gate_id_column(cfg)
    if id_column == "file_id":
        return [str(value) for value in gate_ids]
    frame = pd.read_csv(resolve_repo_path(str(cfg["splits"][split])))
    required = {id_column, "file_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns needed for official decode: {missing}")
    wanted = {str(value) for value in gate_ids}
    selected = frame.loc[frame[id_column].astype(str).isin(wanted), "file_id"]
    return list(dict.fromkeys(selected.astype(str).tolist()))


def decode_groups(
    cfg: Dict[str, Any],
    split: str,
    corpus: Path,
    device: str,
    force: bool,
) -> tuple[Path, Dict[str, Any]]:
    selected = load_best(cfg, "pgcv")
    beta = float(selected["beta"])
    lam = float(selected["best_lambda"])
    tau_g, taus, _, _ = _load_tau_bank(cfg)
    assignment = load_assignment(cfg, split)
    groups = _groups_for_method("pgcv", assignment, cfg)
    root = results_dir(cfg) / "official_spwer" / split
    tau_dir = root / "mixed_taus"
    group_ctms: List[Path] = []
    qwen = is_qwen3_asr(str(cfg["student_model"]))

    for index, (_key, gate_ids, alpha) in enumerate(groups):
        file_ids = expand_gate_ids_to_file_ids(cfg, split, gate_ids)
        group_dir = root / "groups" / f"group_{index:03d}"
        tau_path = tau_dir / f"group_{index:03d}.pt"
        if force or not tau_path.is_file():
            tau_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mix_smooth(tau_g, taus, alpha, beta), tau_path)
        if qwen:
            ctm = decode_qwen(
                corpus=corpus,
                split=split,
                out_dir=group_dir,
                device=device,
                student_model=str(cfg["student_model"]),
                tau_path=str(tau_path),
                lambda_val=lam,
                overwrite=force,
                max_new_tokens=int(cfg.get("official_max_new_tokens", 512)),
                file_ids=file_ids,
            )
        else:
            ctm = decode_whisper(
                corpus=corpus,
                split=split,
                out_dir=group_dir,
                model_name=str(cfg.get("official_whisper_model", "base")),
                device=device,
                beam_size=int(cfg.get("official_beam_size", 5)),
                overwrite=force,
                limit=0,
                model_dir="",
                student_model=str(cfg["student_model"]),
                tau_path=str(tau_path),
                lambda_val=lam,
                file_ids=file_ids,
            )
        group_ctms.append(ctm)

    merged = merge_ctms(group_ctms, root / "decode" / "asr.ctm")
    return merged, {
        "beta": beta,
        "lambda": lam,
        "n_groups": len(groups),
        "selection_metric": "segment WER on dev-asr",
    }


def evaluate(
    cfg: Dict[str, Any],
    split: str,
    corpus: Path,
    device: str,
    force: bool,
    score_only: bool = False,
) -> None:
    root = results_dir(cfg) / "official_spwer" / split
    merged = root / "decode" / "asr.ctm"
    if score_only:
        if not merged.is_file():
            raise FileNotFoundError(f"Missing {merged}; run official decode before --score_only")
        selected = load_best(cfg, "pgcv")
        n_groups = len(list((root / "groups").glob("group_*/asr.ctm")))
        ctm = merged
        selection = {
            "beta": float(selected["beta"]),
            "lambda": float(selected["best_lambda"]),
            "n_groups": n_groups,
            "selection_metric": "segment WER on dev-asr",
        }
        print(f"Score existing CTM without decode: {ctm}", flush=True)
    else:
        ctm, selection = decode_groups(cfg, split, corpus, device, force)
    overall = score_ctm(
        ctm=ctm,
        out_dir=root / "score_overall",
        corpus=corpus,
        split=split,
        force=True,
    )

    mapping = file_level_map(cfg, split)
    by_level: Dict[str, Dict[str, Any]] = {}
    ref_root = root / "references"
    for level in cfg_levels(cfg):
        ids = [file_id for file_id, label in mapping.items() if label == level]
        if not ids:
            continue
        flist, stm = subset_reference_files(corpus, split, level.replace("+", "plus"), ids, ref_root)
        by_level[level] = score_ctm(
            ctm=ctm,
            out_dir=root / "score_by_level" / level.replace("+", "plus"),
            corpus=corpus,
            split=split,
            flist=flist,
            stm=stm,
            force=True,
        )

    payload = {"split": split, "selection": selection, "overall": overall, "by_level": by_level}
    (root / "pgcv_spwer_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        f"# S&I PGCV official SpWER ({family_slug(cfg['student_model'])})",
        "",
        f"- Selected on dev segment WER: β={selection['beta']:.2f}, λ={selection['lambda']:.2f}",
        f"- Quantized attention groups: {selection['n_groups']}",
        f"- {split} overall SpWER: **{float(overall['err']):.1f}**",
        "",
        "| CEFR | SpWER | Sub | Del | Ins |",
        "|---|---:|---:|---:|---:|",
    ]
    for level in cfg_levels(cfg):
        score = by_level.get(level)
        if score:
            lines.append(
                f"| {level} | {float(score['err']):.1f} | {float(score['sub']):.1f} | "
                f"{float(score['del']):.1f} | {float(score['ins']):.1f} |"
            )
    (root / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Official S&I SpWER for selected PGCV")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="eval", choices=("dev", "eval"))
    parser.add_argument("--corpus", default=str(CORPUS_DEFAULT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--score_only",
        action="store_true",
        help="Reuse decode/asr.ctm; skip group decode. Missing STM fileids score as empty hyp.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    evaluate(
        cfg,
        args.split,
        Path(args.corpus).resolve(),
        args.device,
        bool(args.force),
        score_only=bool(args.score_only),
    )


if __name__ == "__main__":
    main()
