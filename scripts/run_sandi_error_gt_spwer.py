#!/usr/bin/env python3
"""SANDI in-domain error-GT τ, official long-form decode + SpWER.

This is NOT the TEEMI-trained τ. It loads
``vectors/sandi/data_select_tau/whisper_base/tau_error_gt.pt``
(trained on SANDI train-asr error-GT). λ is chosen on official dev-asr.

Geometry: θ' = θ_now + λ τ, θ_now = openai/whisper-base.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from decode_sandi_whisper_longform import decode  # noqa: E402
from score_sandi_spwer import score_ctm  # noqa: E402

DEFAULT_TAU = PROJECT_ROOT / "vectors/sandi/data_select_tau/whisper_base/tau_error_gt.pt"
DEFAULT_OUT = PROJECT_ROOT / "results/sandi/spwer_probe/error_gt_tau"
DEFAULT_LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
PRETRAINED_EVAL = PROJECT_ROOT / "results/sandi/spwer_probe/whisper_base/score/spwer_summary.json"


def lambda_tag(lam: float) -> str:
    return f"{float(lam):.1f}".replace(".", "p")


def parse_lambdas(text: str) -> List[float]:
    if not text.strip():
        return list(DEFAULT_LAMBDAS)
    return [float(x) for x in text.split(",") if x.strip()]


def run_dir(root: Path, lam: float, split: str) -> Path:
    return root / f"lambda_{lambda_tag(lam)}" / split


def load_summary(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def decode_and_score(
    *,
    lam: float,
    split: str,
    root: Path,
    corpus: Path,
    tau_path: Path,
    student_model: str,
    device: str,
    force: bool,
    limit: int,
) -> Dict[str, Any]:
    base = run_dir(root, lam, split)
    decode_dir = base / "decode"
    score_dir = base / "score"
    summary_path = score_dir / "spwer_summary.json"
    existing = load_summary(summary_path)
    if existing is not None and not force:
        print(f"Skip λ={lam:.1f} {split}: {summary_path}", flush=True)
        return existing

    if (
        split == "eval"
        and abs(lam) < 1e-12
        and PRETRAINED_EVAL.is_file()
        and not force
    ):
        summary = json.loads(PRETRAINED_EVAL.read_text(encoding="utf-8"))
        score_dir.mkdir(parents=True, exist_ok=True)
        summary = dict(summary)
        summary["lambda"] = 0.0
        summary["reused_pretrained_eval"] = True
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Reuse pretrained eval SpWER for λ=0.0: {PRETRAINED_EVAL}", flush=True)
        return summary

    ctm = decode(
        corpus=corpus,
        split=split,
        out_dir=decode_dir,
        model_name="base",
        device=device,
        beam_size=5,
        overwrite=force,
        limit=limit,
        model_dir="",
        student_model=student_model,
        tau_path="" if abs(lam) < 1e-12 else str(tau_path),
        lambda_val=float(lam),
    )
    summary = score_ctm(
        ctm=ctm,
        out_dir=score_dir,
        corpus=corpus,
        split=split,
        force=True,
    )
    summary["lambda"] = float(lam)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def discover_lambdas(root: Path, split: str) -> List[float]:
    found = []
    if not root.is_dir():
        return found
    for path in sorted(root.glob("lambda_*/" + split + "/score/spwer_summary.json")):
        tag = path.parents[2].name  # lambda_0p5
        if not tag.startswith("lambda_"):
            continue
        found.append(float(tag.split("_", 1)[1].replace("p", ".")))
    return found


def collect_curve(root: Path, split: str, lambdas: Sequence[float]) -> List[Dict[str, Any]]:
    wanted = list(lambdas) if lambdas else discover_lambdas(root, split)
    if not wanted:
        wanted = discover_lambdas(root, split)
    # Union requested + anything already on disk so parallel sweeps don't clobber.
    extra = discover_lambdas(root, split)
    ordered = []
    seen = set()
    for lam in list(wanted) + extra:
        key = round(float(lam), 1)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(float(lam))
    rows: List[Dict[str, Any]] = []
    for lam in sorted(ordered):
        summary = load_summary(run_dir(root, lam, split) / "score" / "spwer_summary.json")
        if summary is None:
            continue
        rows.append(
            {
                "lambda": float(summary.get("lambda", lam)),
                "split": split,
                "spwer": float(summary["err"]),
                "sub": float(summary["sub"]),
                "del": float(summary["del"]),
                "ins": float(summary["ins"]),
                "n_words": int(summary["n_words"]),
            }
        )
    return rows


def write_curve_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["lambda", "split", "spwer", "sub", "del", "ins", "n_words"]
        )
        writer.writeheader()
        writer.writerows(rows)


def select_best(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise FileNotFoundError("No dev SpWER summaries yet")
    return min(rows, key=lambda r: (float(r["spwer"]), float(r["lambda"])))


def write_report(
    path: Path,
    *,
    tau_path: Path,
    dev_rows: Sequence[Dict[str, Any]],
    best: Dict[str, Any],
    eval_base: Optional[Dict[str, Any]],
    eval_best: Optional[Dict[str, Any]],
) -> None:
    lines = [
        "# Experiment Report (SANDI in-domain error-GT τ, official SpWER)",
        "",
        "## Scope",
        "",
        "This is **SANDI-trained** error-GT τ, not TEEMI.",
        f"- τ file: `{tau_path}`",
        "- Student: `openai/whisper-base`",
        "- Geometry: `θ' = θ_now + λ τ`, `τ = θ_error-GT − θ_now` (SANDI train-asr error subset).",
        "- Decode: openai-whisper long-form (same flags as the 13.7 pretrained probe).",
        "- Score: official GLM + sclite SpWER.",
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
            "## Eval-asr vs pretrained 13.7",
            "",
            "| System | λ | SpWER | Sub | Del | Ins |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if eval_base is not None:
        lines.append(
            f"| Pretrained Whisper-base | 0.0 | {eval_base['err']:.1f} | "
            f"{eval_base['sub']:.1f} | {eval_base['del']:.1f} | {eval_base['ins']:.1f} |"
        )
    if eval_best is not None:
        gain = None
        if eval_base is not None:
            gain = float(eval_base["err"]) - float(eval_best["err"])
        gain_s = f"{gain:.1f}" if gain is not None else "—"
        lines.append(
            f"| SANDI error-GT τ | {float(eval_best.get('lambda', best['lambda'])):.1f} | "
            f"{eval_best['err']:.1f} | {eval_best['sub']:.1f} | {eval_best['del']:.1f} | "
            f"{eval_best['ins']:.1f} |"
        )
        lines.extend(["", f"Gain vs pretrained: **{gain_s}** SpWER (positive = better)."])
    else:
        lines.append("| SANDI error-GT τ | — | pending | — | — | — |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_convert(student_model: str, device: str) -> None:
    import torch
    import whisper
    from transformers import WhisperForConditionalGeneration

    from src.whisper_openai_bridge import copy_hf_state_into_openai

    oa = whisper.load_model("base", device="cpu")
    before = {k: v.detach().cpu().clone() for k, v in oa.state_dict().items() if torch.is_floating_point(v)}
    hf = WhisperForConditionalGeneration.from_pretrained(student_model)
    copied, skipped = copy_hf_state_into_openai(oa, hf.state_dict())
    diffs = []
    after = oa.state_dict()
    for key, tensor in before.items():
        if key not in after:
            continue
        other = after[key].detach().cpu().float()
        ref = tensor.float()
        if other.shape != ref.shape:
            continue
        diffs.append((key, float((other - ref).abs().max())))
    diffs.sort(key=lambda x: -x[1])
    print(f"copied={copied} skipped={skipped} max_abs={diffs[0][1] if diffs else 'n/a'} @ {diffs[0][0] if diffs else ''}")
    for key, val in diffs[:8]:
        print(f"  {val:.6g}  {key}")
    if diffs and diffs[0][1] > 1e-3:
        raise SystemExit("HF→openai conversion mismatch is too large")


def main() -> None:
    parser = argparse.ArgumentParser(description="SANDI in-domain error-GT τ + official SpWER")
    parser.add_argument("--stage", default="sweep_dev", choices=("check", "sweep_dev", "select", "eval", "report", "all"))
    parser.add_argument("--split", default="dev", choices=("dev", "eval"))
    parser.add_argument("--lambdas", default="", help="Comma-separated λ list")
    parser.add_argument("--tau", default=str(DEFAULT_TAU))
    parser.add_argument("--student_model", default="openai/whisper-base")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--corpus", default="/share/corpus/sandi2025-challenge")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    tau_path = Path(args.tau)
    out_dir = Path(args.out_dir)
    corpus = Path(args.corpus)
    lambdas = parse_lambdas(args.lambdas)
    if not tau_path.is_file():
        raise FileNotFoundError(tau_path)

    if args.stage == "check":
        check_convert(args.student_model, args.device)
        return

    if args.stage in {"sweep_dev", "all"} and args.stage != "eval":
        split = "dev" if args.stage == "sweep_dev" else args.split
        if args.stage == "all":
            split = "dev"
        for lam in lambdas:
            print(f"==== λ={lam:.1f} split={split} ====", flush=True)
            decode_and_score(
                lam=lam,
                split=split,
                root=out_dir,
                corpus=corpus,
                tau_path=tau_path,
                student_model=args.student_model,
                device=args.device,
                force=args.force,
                limit=int(args.limit),
            )
        if args.stage == "sweep_dev":
            rows = collect_curve(out_dir, "dev", lambdas)
            write_curve_csv(out_dir / "lambda_curve_dev.csv", rows)
            return

    if args.stage in {"select", "eval", "report", "all"}:
        rows = collect_curve(out_dir, "dev", DEFAULT_LAMBDAS)
        write_curve_csv(out_dir / "lambda_curve_dev.csv", rows)
        if args.stage == "select":
            best = select_best(rows)
            print(json.dumps(best, indent=2))
            (out_dir / "best_lambda.json").write_text(json.dumps(best, indent=2) + "\n")
            return

    if args.stage in {"eval", "all"}:
        rows = collect_curve(out_dir, "dev", DEFAULT_LAMBDAS)
        best = select_best(rows)
        (out_dir / "best_lambda.json").write_text(json.dumps(best, indent=2) + "\n")
        print(f"Selected λ={best['lambda']:.1f} (dev SpWER {best['spwer']:.1f})", flush=True)
        decode_and_score(
            lam=float(best["lambda"]),
            split="eval",
            root=out_dir,
            corpus=corpus,
            tau_path=tau_path,
            student_model=args.student_model,
            device=args.device,
            force=args.force,
            limit=int(args.limit),
        )
        if abs(float(best["lambda"])) >= 1e-12:
            decode_and_score(
                lam=0.0,
                split="eval",
                root=out_dir,
                corpus=corpus,
                tau_path=tau_path,
                student_model=args.student_model,
                device=args.device,
                force=False,
                limit=int(args.limit),
            )

    if args.stage in {"report", "eval", "all"}:
        rows = collect_curve(out_dir, "dev", DEFAULT_LAMBDAS)
        best_path = out_dir / "best_lambda.json"
        best = json.loads(best_path.read_text()) if best_path.is_file() else select_best(rows)
        eval_best = load_summary(run_dir(out_dir, float(best["lambda"]), "eval") / "score" / "spwer_summary.json")
        eval_base = load_summary(PRETRAINED_EVAL) or load_summary(
            run_dir(out_dir, 0.0, "eval") / "score" / "spwer_summary.json"
        )
        write_report(
            out_dir / "experiment_report.md",
            tau_path=tau_path,
            dev_rows=rows,
            best=best,
            eval_base=eval_base,
            eval_best=eval_best,
        )
        print(f"Wrote {out_dir / 'experiment_report.md'}", flush=True)


if __name__ == "__main__":
    main()
