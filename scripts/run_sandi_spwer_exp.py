#!/usr/bin/env python3
"""Unified SANDI official-SpWER experiments.

Models: whisper_base | qwen3_0.6b
Exps:   pretrained | full_ft | lora_ft | sandi_error_gt | teemi_error_gt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from decode_sandi_qwen_ctm import decode as qwen_decode  # noqa: E402
from decode_sandi_whisper_longform import decode as whisper_decode  # noqa: E402
from run_sandi_error_gt_spwer import (  # noqa: E402
    collect_curve,
    lambda_tag,
    parse_lambdas,
    select_best,
    write_curve_csv,
)
from score_sandi_spwer import score_ctm  # noqa: E402

DEFAULT_LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
CORPUS = Path(os.environ.get("SANDI_CORPUS", "/share/corpus/sandi2025-challenge"))

SPECS: Dict[str, Dict[str, str]] = {
    "whisper_base": {
        "student": "openai/whisper-base",
        "family": "whisper",
        "openai_name": "base",
        "label": "Whisper-base",
        "full_ft_yaml": "configs/baseline/sandi_full_ft_spwer.yaml",
        "lora_ft_yaml": "configs/baseline/sandi_lora_ft_spwer.yaml",
        "data_select_yaml": "configs/sandi_data_select_tau.yaml",
        "sandi_tau": "vectors/sandi/data_select_tau/whisper_base/tau_error_gt.pt",
        "teemi_tau": "vectors/teemi/a01_only/data_select_tau/whisper_base/tau_error_gt.pt",
        "full_ft_ckpt": "checkpoints/sandi/spwer/whisper_base/full_ft",
        "lora_ft_ckpt": "checkpoints/sandi/spwer/whisper_base/lora_ft",
    },
    "qwen3_0.6b": {
        "student": "Qwen/Qwen3-ASR-0.6B-hf",
        "family": "qwen",
        "openai_name": "",
        "label": "Qwen3-ASR-0.6B",
        "full_ft_yaml": "configs/baseline/sandi_full_ft_spwer_qwen3_0.6b.yaml",
        "lora_ft_yaml": "configs/baseline/sandi_lora_ft_spwer_qwen3_0.6b.yaml",
        "data_select_yaml": "configs/sandi_data_select_tau_qwen3_asr_0.6b.yaml",
        "sandi_tau": "vectors/sandi/data_select_tau/qwen3_asr_0_6b/tau_error_gt.pt",
        "teemi_tau": "vectors/teemi/a01_only/data_select_tau/qwen3_asr_0_6b/tau_error_gt.pt",
        "full_ft_ckpt": "checkpoints/sandi/spwer/qwen3_asr_0_6b/full_ft",
        "lora_ft_ckpt": "checkpoints/sandi/spwer/qwen3_asr_0_6b/lora_ft",
    },
}

EXP_LABELS = {
    "pretrained": "pretrained (no FT, no τ)",
    "full_ft": "SANDI all-data full fine-tune",
    "lora_ft": "SANDI all-data LoRA fine-tune",
    "sandi_error_gt": "SANDI error-GT correction vector τ",
    "teemi_error_gt": "TEEMI error-GT correction vector τ",
}


def repo(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def result_root(model: str, exp: str) -> Path:
    return PROJECT_ROOT / "results" / "sandi" / "spwer" / model / exp


def resolve_ckpt(ckpt_dir: Path, override: str = "") -> Path:
    if override:
        path = Path(override)
        return path if path.is_absolute() else repo(override)
    if (ckpt_dir / "model.safetensors").is_file() or (ckpt_dir / "config.json").is_file():
        return ckpt_dir
    ckpts = sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError(f"No checkpoint under {ckpt_dir}")


def decode_to_ctm(
    *,
    spec: Dict[str, str],
    split: str,
    out_dir: Path,
    device: str,
    tau_path: str = "",
    lambda_val: float = 0.0,
    hf_ckpt: str = "",
    overwrite: bool = False,
    limit: int = 0,
) -> Path:
    if spec["family"] == "whisper":
        return whisper_decode(
            corpus=CORPUS,
            split=split,
            out_dir=out_dir,
            model_name=spec["openai_name"] or "base",
            device=device,
            beam_size=5,
            overwrite=overwrite,
            limit=limit,
            model_dir="",
            student_model=spec["student"],
            tau_path=tau_path,
            lambda_val=lambda_val,
            hf_ckpt=hf_ckpt,
        )
    return qwen_decode(
        corpus=CORPUS,
        split=split,
        out_dir=out_dir,
        device=device,
        student_model=spec["student"],
        tau_path=tau_path,
        lambda_val=lambda_val,
        hf_ckpt=hf_ckpt,
        overwrite=overwrite,
        limit=limit,
    )


def score_existing(split: str, decode_dir: Path, score_dir: Path, force: bool = True) -> Dict[str, Any]:
    ctm = decode_dir / "asr.ctm"
    if not ctm.is_file():
        raise FileNotFoundError(f"Missing {ctm}; run decode first")
    summary = score_ctm(ctm=ctm, out_dir=score_dir, corpus=CORPUS, split=split, force=force)
    score_dir.mkdir(parents=True, exist_ok=True)
    (score_dir / "spwer_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def decode_and_score(
    *,
    spec: Dict[str, str],
    split: str,
    decode_dir: Path,
    score_dir: Path,
    device: str,
    tau_path: str = "",
    lambda_val: float = 0.0,
    hf_ckpt: str = "",
    force: bool = False,
    limit: int = 0,
) -> Dict[str, Any]:
    summary_path = score_dir / "spwer_summary.json"
    if summary_path.is_file() and not force:
        print(f"Skip existing {summary_path}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    decode_to_ctm(
        spec=spec,
        split=split,
        out_dir=decode_dir,
        device=device,
        tau_path=tau_path,
        lambda_val=lambda_val,
        hf_ckpt=hf_ckpt,
        overwrite=force,
        limit=limit,
    )
    summary = score_existing(split, decode_dir, score_dir, force=True)
    summary["lambda"] = float(lambda_val)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def write_point_report(path: Path, *, model: str, exp: str, spec: Dict[str, str], summary: Dict[str, Any], extra: str = "") -> None:
    lines = [
        f"# SANDI SpWER — {spec['label']} — {EXP_LABELS[exp]}",
        "",
        f"- Model: `{spec['student']}`",
        f"- Experiment: `{exp}`",
        f"- Split: `{summary.get('split', 'eval')}`",
        f"- SpWER: **{float(summary['err']):.1f}** (Sub {float(summary['sub']):.1f} / Del {float(summary['del']):.1f} / Ins {float(summary['ins']):.1f})",
        "",
    ]
    if extra:
        lines.extend([extra, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tau_report(
    path: Path,
    *,
    spec: Dict[str, str],
    exp: str,
    tau_path: Path,
    dev_rows: List[Dict[str, Any]],
    best: Dict[str, Any],
    eval_base: Optional[Dict[str, Any]],
    eval_best: Optional[Dict[str, Any]],
) -> None:
    donor = "SANDI train-asr error-GT" if exp == "sandi_error_gt" else "TEEMI A01 error-GT"
    lines = [
        f"# SANDI SpWER — {spec['label']} — {EXP_LABELS[exp]}",
        "",
        "## Scope",
        "",
        f"- Student: `{spec['student']}`",
        f"- τ donor: {donor}",
        f"- τ file: `{tau_path}`",
        "- Geometry: `θ' = θ_now + λ τ`",
        "- Score: official GLM + sclite SpWER on full files.",
        "- λ is chosen on official **dev-asr**. eval-asr is held out.",
        "",
        "## Dev-asr λ curve",
        "",
        "| λ | SpWER | Sub | Del | Ins |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(dev_rows, key=lambda r: float(r["lambda"])):
        mark = " ← selected" if abs(float(row["lambda"]) - float(best["lambda"])) < 1e-9 else ""
        lines.append(
            f"| {row['lambda']:.1f} | {row['spwer']:.1f} | {row['sub']:.1f} | "
            f"{row['del']:.1f} | {row['ins']:.1f} |{mark}"
        )
    lines.extend(
        [
            "",
            f"Selected λ = **{float(best['lambda']):.1f}** (dev SpWER {float(best['spwer']):.1f}).",
            "",
            "## Eval-asr",
            "",
            "| System | λ | SpWER | Sub | Del | Ins |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if eval_base is not None:
        lines.append(
            f"| Pretrained {spec['label']} | 0.0 | {eval_base['err']:.1f} | "
            f"{eval_base['sub']:.1f} | {eval_base['del']:.1f} | {eval_base['ins']:.1f} |"
        )
    if eval_best is not None:
        gain = None
        if eval_base is not None:
            gain = float(eval_base["err"]) - float(eval_best["err"])
        gain_s = f"{gain:.1f}" if gain is not None else "—"
        lines.append(
            f"| {EXP_LABELS[exp]} | {float(eval_best.get('lambda', best['lambda'])):.1f} | "
            f"{eval_best['err']:.1f} | {eval_best['sub']:.1f} | {eval_best['del']:.1f} | "
            f"{eval_best['ins']:.1f} |"
        )
        lines.extend(["", f"Gain vs pretrained: **{gain_s}** SpWER (positive = better)."])
    else:
        lines.append(f"| {EXP_LABELS[exp]} | — | pending | — | — | — |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_supervised_ft(spec: Dict[str, str], exp: str, resume: str) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_sandi_full_ft.py"),
        "--config",
        str(repo(spec[f"{exp}_yaml"])),
    ]
    if resume:
        cmd.extend(["--resume_from_checkpoint", resume])
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def run_data_select(spec: Dict[str, str], device: str) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_teemi_a01_only_data_select_tau.py"),
        "--config",
        str(repo(spec["data_select_yaml"])),
        "--stage",
        "1,2,3,4",
        "--device",
        device,
    ]
    print(" ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def sweep_tau(
    *,
    spec: Dict[str, str],
    tau_path: Path,
    out_root: Path,
    device: str,
    lambdas: List[float],
    force: bool,
    limit: int,
    split: str,
) -> List[Dict[str, Any]]:
    if not tau_path.is_file():
        raise FileNotFoundError(tau_path)
    for lam in lambdas:
        print(f"==== λ={lam:.1f} split={split} ====", flush=True)
        base = out_root / f"lambda_{lambda_tag(lam)}" / split
        summary = decode_and_score(
            spec=spec,
            split=split,
            decode_dir=base / "decode",
            score_dir=base / "score",
            device=device,
            tau_path="" if abs(lam) < 1e-12 else str(tau_path),
            lambda_val=float(lam),
            force=force,
            limit=limit,
        )
        print(json.dumps({"lambda": lam, "split": split, "err": summary.get("err")}, indent=2), flush=True)
    rows = collect_curve(out_root, split, lambdas)
    write_curve_csv(out_root / f"lambda_curve_{split}.csv", rows)
    return rows


def run_pretrained(spec: Dict[str, str], args: argparse.Namespace, out_root: Path) -> None:
    split = args.split
    decode_dir = out_root / split / "decode"
    score_dir = out_root / split / "score"
    if args.stage == "decode":
        decode_to_ctm(
            spec=spec,
            split=split,
            out_dir=decode_dir,
            device=args.device,
            limit=int(args.limit),
            overwrite=args.force,
        )
        return
    if args.stage == "score":
        summary = score_existing(split, decode_dir, score_dir, force=True)
    else:
        summary = decode_and_score(
            spec=spec,
            split=split,
            decode_dir=decode_dir,
            score_dir=score_dir,
            device=args.device,
            force=args.force,
            limit=int(args.limit),
        )
    write_point_report(out_root / "experiment_report.md", model=args.model, exp="pretrained", spec=spec, summary=summary)
    print(json.dumps({"split": split, "err": summary.get("err")}, indent=2), flush=True)


def run_supervised_ft_exp(spec: Dict[str, str], args: argparse.Namespace, out_root: Path) -> None:
    ckpt_dir = repo(spec[f"{args.exp}_ckpt"])
    if args.stage in {"train", "all"}:
        run_supervised_ft(spec, args.exp, args.resume_from_checkpoint)
    if args.stage == "train":
        return
    ckpt = resolve_ckpt(ckpt_dir, args.hf_ckpt)
    print(f"hf_ckpt={ckpt}", flush=True)
    decode_dir = out_root / args.split / "decode"
    score_dir = out_root / args.split / "score"
    if args.stage == "decode":
        decode_to_ctm(
            spec=spec,
            split=args.split,
            out_dir=decode_dir,
            device=args.device,
            hf_ckpt=str(ckpt),
            overwrite=args.force,
            limit=int(args.limit),
        )
        return
    if args.stage == "score":
        summary = score_existing(args.split, decode_dir, score_dir, force=True)
    else:
        summary = decode_and_score(
            spec=spec,
            split=args.split,
            decode_dir=decode_dir,
            score_dir=score_dir,
            device=args.device,
            hf_ckpt=str(ckpt),
            force=args.force,
            limit=int(args.limit),
        )
    write_point_report(
        out_root / "experiment_report.md",
        model=args.model,
        exp=args.exp,
        spec=spec,
        summary=summary,
        extra=f"- Checkpoint: `{ckpt}`",
    )
    print(json.dumps({"split": args.split, "err": summary.get("err"), "ckpt": str(ckpt)}, indent=2), flush=True)


def run_tau_exp(spec: Dict[str, str], args: argparse.Namespace, out_root: Path) -> None:
    tau_key = "sandi_tau" if args.exp == "sandi_error_gt" else "teemi_tau"
    tau_path = repo(args.tau_path) if str(getattr(args, "tau_path", "") or "").strip() else repo(spec[tau_key])
    skip_train = bool(str(getattr(args, "tau_path", "") or "").strip())
    print(f"  tau: {tau_path}  skip_train={skip_train}", flush=True)
    if args.exp == "sandi_error_gt" and args.stage in {"train_tau", "all"} and not skip_train:
        run_data_select(spec, args.device)
    if args.exp == "sandi_error_gt" and args.stage == "train_tau":
        return
    if not tau_path.is_file():
        raise FileNotFoundError(
            f"Missing τ {tau_path}. For sandi_error_gt run stage train_tau first; "
            "for teemi_error_gt the TEEMI data-select τ must already exist."
        )
    lambdas = parse_lambdas(args.lambdas) if args.lambdas else list(DEFAULT_LAMBDAS)
    if args.stage in {"sweep", "all"}:
        sweep_tau(
            spec=spec,
            tau_path=tau_path,
            out_root=out_root,
            device=args.device,
            lambdas=lambdas,
            force=args.force,
            limit=int(args.limit),
            split="dev",
        )
        if args.stage == "sweep":
            return
    if args.stage in {"eval", "report", "all"}:
        rows = collect_curve(out_root, "dev", DEFAULT_LAMBDAS)
        best = select_best(rows)
        (out_root / "best_lambda.json").write_text(json.dumps(best, indent=2) + "\n")
        print(f"Selected λ={best['lambda']:.1f} (dev SpWER {best['spwer']:.1f})", flush=True)
        if args.stage != "report":
            eval_lams = [float(best["lambda"])]
            if abs(eval_lams[0]) >= 1e-12:
                eval_lams.append(0.0)
            for lam in eval_lams:
                base = out_root / f"lambda_{lambda_tag(lam)}" / "eval"
                decode_and_score(
                    spec=spec,
                    split="eval",
                    decode_dir=base / "decode",
                    score_dir=base / "score",
                    device=args.device,
                    tau_path="" if abs(lam) < 1e-12 else str(tau_path),
                    lambda_val=lam,
                    force=args.force,
                    limit=int(args.limit),
                )
        eval_best = json.loads(
            (out_root / f"lambda_{lambda_tag(float(best['lambda']))}" / "eval" / "score" / "spwer_summary.json").read_text()
        ) if (out_root / f"lambda_{lambda_tag(float(best['lambda']))}" / "eval" / "score" / "spwer_summary.json").is_file() else None
        eval_base_path = out_root / "lambda_0p0" / "eval" / "score" / "spwer_summary.json"
        pretrained_eval = result_root(args.model, "pretrained") / "eval" / "score" / "spwer_summary.json"
        eval_base = None
        if eval_base_path.is_file():
            eval_base = json.loads(eval_base_path.read_text())
        elif pretrained_eval.is_file():
            eval_base = json.loads(pretrained_eval.read_text())
        write_tau_report(
            out_root / "experiment_report.md",
            spec=spec,
            exp=args.exp,
            tau_path=tau_path,
            dev_rows=rows,
            best=best,
            eval_base=eval_base,
            eval_best=eval_best,
        )
        print(f"Wrote {out_root / 'experiment_report.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SANDI SpWER experiment driver")
    parser.add_argument("--model", required=True, choices=tuple(SPECS))
    parser.add_argument(
        "--exp",
        required=True,
        choices=("pretrained", "full_ft", "lora_ft", "sandi_error_gt", "teemi_error_gt"),
    )
    parser.add_argument("--stage", default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="eval", choices=("dev", "eval"))
    parser.add_argument("--lambdas", default="")
    parser.add_argument("--hf_ckpt", default="")
    parser.add_argument("--resume_from_checkpoint", default="")
    parser.add_argument("--tau_path", default="", help="Override τ .pt (skip train_tau when set)")
    parser.add_argument("--out_dir", default="", help="Override results root (seed-specific SpWER)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    spec = SPECS[args.model]
    out_root = Path(args.out_dir) if str(args.out_dir).strip() else result_root(args.model, args.exp)
    if not out_root.is_absolute():
        out_root = repo(str(out_root))
    print(f"SANDI SpWER  model={args.model}  exp={args.exp}  stage={args.stage}", flush=True)
    print(f"  out: {out_root}", flush=True)

    if args.exp == "pretrained":
        run_pretrained(spec, args, out_root)
        return
    if args.exp in {"full_ft", "lora_ft"}:
        run_supervised_ft_exp(spec, args, out_root)
        return
    run_tau_exp(spec, args, out_root)


if __name__ == "__main__":
    main()
