#!/usr/bin/env python3
"""Build SANDI ASR metadata CSVs inside this repo.

Reads the official Speak & Improve 2025 challenge tree as **read-only**.
Never writes, unpacks, links, or caches under the corpus root.

Training / HF-eval rows are **official STM scored segments** (one row per
speech-unit clip), not one row per flac. Labels are cleaned toward TEEMI
"write what you heard": drop ``<...>`` / ``%`` tags, map
``(%hesitation%)`` → ``uh``, keep spoken ``uh``/``ah``, keep partials
without parentheses. ``IGNORE_TIME_SEGMENT_IN_SCORING`` is dropped.
Clips longer than 30 s are dropped (Whisper encoder window).

Outputs (default: ``data/sandi/asr/``):
  train.csv / dev.csv / eval.csv
  skipped_rows.csv
  split_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("prepare_sandi_asr_csv")

DEFAULT_CORPUS = Path("/share/corpus/sandi2025-challenge")
SPLITS = ("train", "dev", "eval")
PARTS = (1, 3, 4, 5)
FILE_ID_RE = re.compile(r"^(?P<session>.+)-P(?P<part>[1345])(?P<utt>\d+)$")
MAX_CHUNK_SEC = 30.0
MIN_CHUNK_SEC = 0.02
IGNORE_TOKEN = "IGNORE_TIME_SEGMENT_IN_SCORING"
ANGLE_TAG_RE = re.compile(r"<[^>]*>")
HESITATION_RE = re.compile(r"\(\s*%\s*hesitation\s*%\s*\)", re.IGNORECASE)
PAREN_RE = re.compile(r"\(([^)]*)\)")

CEFR_FROM_HALF = {
    1.0: "A1",
    1.5: "A1+",
    2.0: "A2",
    2.5: "A2+",
    3.0: "B1",
    3.5: "B1+",
    4.0: "B2",
    4.5: "B2+",
    5.0: "C1",
    5.5: "C1+",
    6.0: "C2",
}


def assert_output_outside_corpus(output_dir: Path, corpus_root: Path) -> None:
    out = output_dir.resolve()
    corpus = corpus_root.resolve()
    if out == corpus or corpus in out.parents:
        raise SystemExit(
            f"Refusing to write under the SANDI corpus: {out} is inside {corpus}"
        )


def load_flist(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        file_id, rel = line.split("\t", 1)
        mapping[file_id.strip()] = rel.strip()
    return mapping


def parse_file_id(file_id: str) -> Optional[Tuple[str, int, str]]:
    match = FILE_ID_RE.match(file_id)
    if not match:
        return None
    return match.group("session"), int(match.group("part")), match.group("utt")


def sla_to_cefr(score: Optional[float]) -> str:
    if score is None:
        return ""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    half = round(value * 2.0) / 2.0
    half = min(6.0, max(1.0, half))
    return CEFR_FROM_HALF.get(half, f"{half:g}")


def load_part_scores(corpus_root: Path, split: str) -> Dict[Tuple[str, int], float]:
    marks: Dict[Tuple[str, int], float] = {}
    root = corpus_root / "reference-materials" / "sla-marks"
    for part in PARTS:
        path = root / f"{split}-sla-P{part}.tsv"
        if not path.is_file():
            logger.warning("Missing part-score file %s", path)
            continue
        df = pd.read_csv(path, sep="\t", header=None, names=["session_id", "score"])
        for session_id, score in zip(df["session_id"].astype(str), df["score"]):
            try:
                marks[(session_id, part)] = float(score)
            except (TypeError, ValueError):
                continue
    return marks


def load_overall_scores(corpus_root: Path, split: str) -> Dict[str, float]:
    path = corpus_root / "reference-materials" / "sla-marks" / f"{split}-sla-overall.tsv"
    if not path.is_file():
        return {}
    df = pd.read_csv(path, sep="\t", header=None, names=["session_id", "score"])
    out: Dict[str, float] = {}
    for session_id, score in zip(df["session_id"].astype(str), df["score"]):
        try:
            out[session_id] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def _replace_paren(match: re.Match[str]) -> str:
    inner = match.group(1).strip()
    if not inner:
        return " "
    if "%" in inner:
        return " "
    return f" {inner} "


def clean_stm_text(raw: str) -> str:
    """STM text → TEEMI-like heard words. Drops annotation glyphs, keeps fillers."""
    text = str(raw or "")
    if IGNORE_TOKEN in text:
        return ""
    text = ANGLE_TAG_RE.sub(" ", text)
    text = HESITATION_RE.sub(" uh ", text)
    text = PAREN_RE.sub(_replace_paren, text)
    text = text.replace("%", " ").replace("<", " ").replace(">", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_stm_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        raw = line.strip()
        if not raw or raw.startswith(";;"):
            continue
        parts = raw.split(None, 5)
        if len(parts) < 6:
            continue
        file_id, _ch, _spk, t0_s, t1_s, rest = parts
        try:
            t0 = float(t0_s)
            t1 = float(t1_s)
        except ValueError:
            continue
        rows.append(
            {
                "file_id": file_id,
                "start_sec": t0,
                "end_sec": t1,
                "duration_sec": t1 - t0,
                "raw_text": rest,
                "stm_line": line_no,
            }
        )
    return rows


def build_split(corpus_root: Path, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    flist_path = corpus_root / "reference-materials" / "flists.flac" / f"{split}-asr.tsv"
    stm_path = corpus_root / "reference-materials" / "stms" / f"{split}-asr.stm"
    if not flist_path.is_file():
        raise FileNotFoundError(flist_path)
    if not stm_path.is_file():
        raise FileNotFoundError(stm_path)

    flist = load_flist(flist_path)
    part_scores = load_part_scores(corpus_root, split)
    overall_scores = load_overall_scores(corpus_root, split)

    kept: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for stm in parse_stm_rows(stm_path):
        file_id = stm["file_id"]
        rel = flist.get(file_id)
        skip_base = {
            "split": split,
            "file_id": file_id,
            "start_sec": stm["start_sec"],
            "end_sec": stm["end_sec"],
            "stm_line": stm["stm_line"],
            "rel_path": rel or "",
        }
        if IGNORE_TOKEN in stm["raw_text"]:
            skipped.append({**skip_base, "reason": "ignore_time_segment"})
            continue
        if rel is None:
            skipped.append({**skip_base, "reason": "missing_flist"})
            continue
        parsed = parse_file_id(file_id)
        if parsed is None:
            skipped.append({**skip_base, "reason": "bad_file_id"})
            continue
        duration = float(stm["duration_sec"])
        if duration <= MIN_CHUNK_SEC:
            skipped.append({**skip_base, "reason": "too_short"})
            continue
        if duration > MAX_CHUNK_SEC + 1e-6:
            skipped.append({**skip_base, "reason": "over_30s"})
            continue
        text = clean_stm_text(stm["raw_text"])
        if not text:
            skipped.append({**skip_base, "reason": "empty_text"})
            continue
        audio_path = (corpus_root / rel).resolve()
        if not audio_path.is_file():
            skipped.append({**skip_base, "reason": "missing_audio"})
            continue
        session_id, part, utt = parsed
        sla_score = part_scores.get((session_id, part))
        cefr = sla_to_cefr(sla_score)
        start = float(stm["start_sec"])
        end = float(stm["end_sec"])
        audio_ref = f"{audio_path}#t={start:.2f},{end:.2f}"
        kept.append(
            {
                "audio_path": audio_ref,
                "source_audio_path": str(audio_path),
                "text": text,
                "file_id": file_id,
                "clip_id": f"{file_id}_{start:.2f}_{end:.2f}",
                "start_sec": round(start, 2),
                "end_sec": round(end, 2),
                "duration_sec": round(duration, 3),
                "session_id": session_id,
                "global_speaker_id": session_id,
                "part": part,
                "utt": utt,
                "sla_score": sla_score if sla_score is not None else "",
                "overall_sla_score": overall_scores.get(session_id, ""),
                "cefr": cefr,
                "proficiency_level": cefr,
                "split": split,
                "corpus": "sandi2025",
            }
        )
    return pd.DataFrame(kept), pd.DataFrame(skipped)


def summarize(df: pd.DataFrame) -> Dict[str, Any]:
    cefr_counts = Counter(str(x) for x in df["cefr"].tolist() if str(x).strip())
    missing_cefr = int((df["cefr"].astype(str).str.strip() == "").sum()) if len(df) else 0
    return {
        "clips": int(len(df)),
        "source_files": int(df["file_id"].nunique()) if len(df) else 0,
        "sessions": int(df["session_id"].nunique()) if len(df) else 0,
        "hours": float(df["duration_sec"].sum() / 3600.0) if len(df) else 0.0,
        "mean_sec": float(df["duration_sec"].mean()) if len(df) else 0.0,
        "max_sec": float(df["duration_sec"].max()) if len(df) else 0.0,
        "missing_cefr": missing_cefr,
        "by_cefr": dict(sorted(cefr_counts.items())),
        "by_part": {str(k): int(v) for k, v in df["part"].value_counts().sort_index().items()} if len(df) else {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SANDI STM-chunk ASR CSVs (corpus read-only)")
    parser.add_argument(
        "--corpus-root",
        type=str,
        default=str(DEFAULT_CORPUS),
        help="Read-only Speak & Improve 2025 challenge root",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/sandi/asr",
        help="CSV output directory inside this repo (never under /share/corpus)",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,dev,eval",
        help="Comma-separated official splits",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    corpus_root = Path(args.corpus_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (Path(__file__).resolve().parent.parent / output_dir).resolve()
    assert_output_outside_corpus(output_dir, corpus_root)
    if not corpus_root.is_dir():
        raise SystemExit(f"Corpus root not found: {corpus_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    all_skipped: List[pd.DataFrame] = []
    summary: Dict[str, Any] = {
        "corpus_root": str(corpus_root),
        "output_dir": str(output_dir),
        "note": "Corpus is read-only. Audio paths point into the corpus with #t=start,end slices; CSVs live in the repo.",
        "labeling": "SANDI part SLA CEFR (A2/A2+/...); not remapped to TEEMI booklet levels.",
        "transcript": (
            "STM scored segments; (%hesitation%)->uh; drop <tags>/% ; keep partials; "
            "drop IGNORE and clips >30s; no GEC"
        ),
        "unit": "stm_scored_segment",
        "splits": {},
    }
    for split in splits:
        if split not in SPLITS:
            raise SystemExit(f"Unknown split {split}; choose from {SPLITS}")
        logger.info("Building %s STM chunks from %s", split, corpus_root)
        kept, skipped = build_split(corpus_root, split)
        out_csv = output_dir / f"{split}.csv"
        kept.to_csv(out_csv, index=False)
        logger.info("Wrote %s (%d clips, skipped %d)", out_csv, len(kept), len(skipped))
        if len(skipped):
            all_skipped.append(skipped)
        payload = summarize(kept)
        payload["skipped"] = int(len(skipped))
        if len(skipped):
            payload["skipped_by_reason"] = skipped["reason"].value_counts().to_dict()
        summary["splits"][split] = payload

    skipped_path = output_dir / "skipped_rows.csv"
    if all_skipped:
        pd.concat(all_skipped, ignore_index=True).to_csv(skipped_path, index=False)
    elif skipped_path.is_file():
        skipped_path.unlink()

    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
