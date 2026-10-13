#!/usr/bin/env python3
"""Prepare the confirmed-GT TEEMI 6-booklet prompt-transfer split.

Current design:
- confirmed human-corrected booklets only:
  A01, A02, B01, B02, C01, C02
- train side booklets: A01, B01, C01
- eval side booklets: A02, B02, C02
- split unit: global speaker id shared across booklets (uXXX)
- speaker split ratio: 75 / 10 / 15

Outputs under ``data/teemi``:
- all_utterances.csv
- speaker_manifest.csv
- train.csv
- dev.csv
- test.csv
- dev_a02.csv / dev_b02.csv / dev_c02.csv
- test_a02.csv / test_b02.csv / test_c02.csv
- prepare_meta.json
- dataset_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_teemi")

BOOKLET_RE = re.compile(r"^([A-Za-z]\d+)_")
SPEAKER_RE = re.compile(r"_u(\d+)_")
NOTE_SHEETS = {"計分說明"}
TRAIN_BOOKLETS: Tuple[str, ...] = ("A01", "B01", "C01")
EVAL_BOOKLETS: Tuple[str, ...] = ("A02", "B02", "C02")
CONFIRMED_GT_BOOKLETS: Tuple[str, ...] = TRAIN_BOOKLETS + EVAL_BOOKLETS
INVALID_PROFICIENCY_LEVELS = {"No answer"}


def booklet_of(filename: str) -> str:
    stem = Path(filename).stem
    m = BOOKLET_RE.match(stem)
    return m.group(1).upper() if m else stem.split("_")[0].upper()


def speaker_num_of(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    m = SPEAKER_RE.search(stem)
    return int(m.group(1)) if m else None


def local_speaker_id_of(filename: str) -> str:
    booklet = booklet_of(filename)
    speaker_num = speaker_num_of(filename)
    return f"{booklet}_u{speaker_num}" if speaker_num is not None else booklet


def global_speaker_id_of(filename: str) -> str:
    speaker_num = speaker_num_of(filename)
    return f"u{speaker_num}" if speaker_num is not None else Path(filename).stem


def load_human_transcripts(transcript_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    files = sorted(transcript_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No xlsx under {transcript_dir}")

    for xlsx in files:
        xl = pd.ExcelFile(xlsx)
        for sheet in xl.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet)
            cols = {str(c).strip().lower(): c for c in df.columns}
            if "filename" not in cols:
                continue
            fn_col = cols["filename"]
            text_col = None
            for key in ("trans_human", "trans_human_number", "transcript", "text"):
                if key in cols:
                    text_col = cols[key]
                    break
            if text_col is None:
                continue
            for _, row in df.iterrows():
                fn = row[fn_col]
                text = row[text_col]
                if pd.isna(fn) or pd.isna(text):
                    continue
                fn = str(fn).strip()
                text = str(text).strip()
                if not fn or not text:
                    continue
                booklet = booklet_of(fn)
                if booklet not in CONFIRMED_GT_BOOKLETS:
                    continue
                rows.append(
                    {
                        "filename": Path(fn).stem,
                        "text": text,
                        "transcript_xlsx": xlsx.name,
                        "transcript_sheet": sheet,
                        "booklet": booklet,
                    }
                )

    if not rows:
        raise RuntimeError(f"No transcript rows parsed from {transcript_dir}")

    best: Dict[str, dict] = {}
    for row in rows:
        key = row["filename"]
        if key not in best or len(row["text"]) > len(best[key]["text"]):
            best[key] = row
    out = pd.DataFrame(best.values())
    logger.info("Parsed %d unique confirmed-GT transcripts from %d xlsx files", len(out), len(files))
    return out


def load_score_mapping(rubrics_path: Path) -> List[Tuple[float, str]]:
    payload = json.loads(rubrics_path.read_text(encoding="utf-8"))
    mapping: List[Tuple[float, str]] = []
    for item in payload.get("score_mapping", []):
        try:
            mapping.append((float(item["original_score"]), str(item["level"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not mapping:
        raise RuntimeError(f"No score_mapping entries found in {rubrics_path}")
    return sorted(mapping, key=lambda x: x[0])


def infer_level_from_score(score: float, mapping: Sequence[Tuple[float, str]]) -> str:
    _, level = min(mapping, key=lambda item: abs(item[0] - score))
    return level


def load_proficiency_scores(analysis_dir: Path, rubrics_path: Path) -> pd.DataFrame:
    files = sorted(analysis_dir.glob("*.xlsx"))
    if not files:
        raise FileNotFoundError(f"No xlsx under {analysis_dir}")

    score_mapping = load_score_mapping(rubrics_path)
    records: List[dict] = []
    mismatch_count = 0
    for xlsx in files:
        xl = pd.ExcelFile(xlsx, engine="openpyxl")
        for sheet in xl.sheet_names:
            if sheet in NOTE_SHEETS:
                continue
            df = pd.read_excel(xlsx, sheet_name=sheet, engine="openpyxl")
            cols = {str(c).strip().lower(): c for c in df.columns}
            needed = {"student_id", "weighted_score", "wav_paths"}
            if not needed.issubset(cols):
                continue
            content_col = cols.get("content")
            pronunciation_col = cols.get("pronunciation")
            vocab_col = cols.get("vocabulary")
            for _, row in df.iterrows():
                wav_paths = row[cols["wav_paths"]]
                student_id = row[cols["student_id"]]
                weighted_score = row[cols["weighted_score"]]
                if pd.isna(wav_paths) or pd.isna(student_id) or pd.isna(weighted_score):
                    continue
                paths = [p.strip() for p in str(wav_paths).split(",") if p and str(p).strip()]
                if not paths:
                    continue
                first_name = Path(paths[0]).stem
                booklet = booklet_of(first_name)
                if booklet not in CONFIRMED_GT_BOOKLETS:
                    continue
                speaker_num = speaker_num_of(first_name)
                if speaker_num is None:
                    continue
                try:
                    student_num = int(float(student_id))
                    weighted_val = float(weighted_score)
                except (TypeError, ValueError):
                    continue
                if student_num != speaker_num:
                    mismatch_count += 1
                    continue
                local_speaker_id = local_speaker_id_of(first_name)
                global_speaker_id = global_speaker_id_of(first_name)
                content_val = pd.to_numeric(row[content_col], errors="coerce") if content_col else None
                pronunciation_val = pd.to_numeric(row[pronunciation_col], errors="coerce") if pronunciation_col else None
                vocab_val = pd.to_numeric(row[vocab_col], errors="coerce") if vocab_col else None
                records.append(
                    {
                        "local_speaker_id": local_speaker_id,
                        "global_speaker_id": global_speaker_id,
                        "speaker_num": student_num,
                        "booklet": booklet,
                        "analysis_sheet": sheet,
                        "analysis_file": xlsx.name,
                        "weighted_score": weighted_val,
                        "content_score": content_val,
                        "pronunciation_score": pronunciation_val,
                        "vocabulary_score": vocab_val,
                        "score_row_count": 1,
                    }
                )
    if not records:
        raise RuntimeError(f"No usable proficiency rows parsed from {analysis_dir}")

    score_df = pd.DataFrame(records)
    grouped = (
        score_df.groupby("local_speaker_id", as_index=False)
        .agg(
            global_speaker_id=("global_speaker_id", "first"),
            speaker_num=("speaker_num", "first"),
            booklet=("booklet", "first"),
            weighted_score=("weighted_score", "mean"),
            content_score=("content_score", "mean"),
            pronunciation_score=("pronunciation_score", "mean"),
            vocabulary_score=("vocabulary_score", "mean"),
            score_row_count=("score_row_count", "sum"),
            analysis_sheets=("analysis_sheet", lambda s: ";".join(sorted(set(map(str, s))))),
            analysis_files=("analysis_file", lambda s: ";".join(sorted(set(map(str, s))))),
        )
        .reset_index(drop=True)
    )
    grouped["proficiency_level"] = grouped["weighted_score"].map(lambda s: infer_level_from_score(s, score_mapping))
    logger.info(
        "Parsed %d score rows -> %d booklet-local profiles from %d xlsx files (mismatch=%d)",
        len(score_df),
        len(grouped),
        len(files),
        mismatch_count,
    )
    return grouped


def index_audio(audio_root: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for booklet in CONFIRMED_GT_BOOKLETS:
        booklet_dir = audio_root / booklet
        for wav in booklet_dir.glob("*.wav"):
            mapping[wav.stem] = wav
    logger.info("Indexed %d wav files across %d confirmed-GT booklets", len(mapping), len(CONFIRMED_GT_BOOKLETS))
    return mapping


def resolve_repo_path(script_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (script_path.parent.parent / path).resolve()


def audio_duration_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    return float(info.frames) / float(info.samplerate)


def compute_duration_map(audio_paths: Sequence[str], max_workers: int) -> Dict[str, float]:
    unique_paths = list(dict.fromkeys(audio_paths))
    logger.info("Computing durations for %d audio files with %d workers", len(unique_paths), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        durations = list(ex.map(audio_duration_seconds, unique_paths))
    return dict(zip(unique_paths, durations))


def side_of_booklet(booklet: str) -> str:
    if booklet in TRAIN_BOOKLETS:
        return "seen"
    if booklet in EVAL_BOOKLETS:
        return "unseen"
    return "other"


def split_global_speakers(global_speakers: Sequence[str], train_ratio: float, dev_ratio: float, seed: int) -> Dict[str, str]:
    speakers = pd.Series(sorted(set(global_speakers))).sample(frac=1.0, random_state=seed).tolist()
    n = len(speakers)
    n_train = max(1, int(math.floor(n * train_ratio)))
    n_dev = max(1, int(math.floor(n * dev_ratio)))
    n_test = n - n_train - n_dev
    if n_test < 1:
        n_test = 1
        if n_train >= n_dev and n_train > 1:
            n_train -= 1
        elif n_dev > 1:
            n_dev -= 1
    while n_train + n_dev + n_test > n and n_train > 1:
        n_train -= 1
    while n_train + n_dev + n_test > n and n_dev > 1:
        n_dev -= 1
    n_test = n - n_train - n_dev
    assignments: Dict[str, str] = {}
    for speaker in speakers[:n_train]:
        assignments[speaker] = "train"
    for speaker in speakers[n_train : n_train + n_dev]:
        assignments[speaker] = "dev"
    for speaker in speakers[n_train + n_dev :]:
        assignments[speaker] = "test"
    return assignments


def seconds_to_hms(total_seconds: float) -> str:
    total_seconds = int(round(total_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_float(value: float) -> str:
    return f"{value:.2f}"


def split_baseline_eligible_rows(all_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prof = all_df["proficiency_level"]
    prof_norm = prof.fillna("").astype(str).str.strip()
    missing_mask = prof.isna() | prof_norm.eq("")
    invalid_mask = prof_norm.isin(INVALID_PROFICIENCY_LEVELS)

    excluded_parts: List[pd.DataFrame] = []
    if missing_mask.any():
        sub = all_df.loc[missing_mask].copy()
        sub["exclusion_reason"] = "missing_proficiency"
        excluded_parts.append(sub)
    if invalid_mask.any():
        sub = all_df.loc[invalid_mask].copy()
        sub["exclusion_reason"] = "invalid_proficiency:No answer"
        excluded_parts.append(sub)

    excluded_df = (
        pd.concat(excluded_parts, ignore_index=True)
        if excluded_parts
        else pd.DataFrame(columns=list(all_df.columns) + ["exclusion_reason"])
    )
    eligible_df = all_df.loc[~(missing_mask | invalid_mask)].copy()
    return eligible_df, excluded_df


def summarize_subset(df: pd.DataFrame) -> Dict[str, object]:
    scored = df["weighted_score"].notna() if "weighted_score" in df.columns else pd.Series([False] * len(df))
    total_seconds = float(df["duration_sec"].sum()) if len(df) else 0.0
    return {
        "utterances": int(len(df)),
        "speakers": int(df["global_speaker_id"].nunique()) if "global_speaker_id" in df.columns else 0,
        "total_seconds": total_seconds,
        "total_hms": seconds_to_hms(total_seconds),
        "mean_seconds": float(df["duration_sec"].mean()) if len(df) else 0.0,
        "median_seconds": float(df["duration_sec"].median()) if len(df) else 0.0,
        "min_seconds": float(df["duration_sec"].min()) if len(df) else 0.0,
        "max_seconds": float(df["duration_sec"].max()) if len(df) else 0.0,
        "mean_words": float(df["num_words"].mean()) if len(df) else 0.0,
        "scored_utterances": int(scored.sum()),
    }


def build_speaker_manifest(all_df: pd.DataFrame) -> pd.DataFrame:
    score_cols = [
        "global_speaker_id",
        "speaker_num",
        "booklet",
        "weighted_score",
        "proficiency_level",
        "content_score",
        "pronunciation_score",
        "vocabulary_score",
    ]
    score_df = all_df[score_cols].drop_duplicates()

    def side_coverage(booklets: Sequence[str]) -> str:
        booklet_set = set(booklets)
        has_seen = any(b in TRAIN_BOOKLETS for b in booklet_set)
        has_unseen = any(b in EVAL_BOOKLETS for b in booklet_set)
        if has_seen and has_unseen:
            return "both"
        if has_seen:
            return "seen_only"
        if has_unseen:
            return "unseen_only"
        return "other"

    def score_pairs(sub: pd.DataFrame, value_col: str) -> str:
        parts = []
        for _, row in sub.sort_values("booklet").iterrows():
            value = row[value_col]
            if pd.notna(value):
                parts.append(f"{row['booklet']}={value}")
        return ";".join(parts)

    rows = []
    for speaker_id, speaker_rows in all_df.groupby("global_speaker_id"):
        score_rows = score_df[score_df["global_speaker_id"] == speaker_id]
        booklets = sorted(set(speaker_rows["booklet"].tolist()))
        weighted_values = score_rows["weighted_score"].dropna()
        rows.append(
            {
                "global_speaker_id": speaker_id,
                "speaker_num": int(score_rows["speaker_num"].dropna().iloc[0]) if not score_rows["speaker_num"].dropna().empty else None,
                "split": speaker_rows["split"].iloc[0],
                "booklets_present": ";".join(booklets),
                "coverage_type": side_coverage(booklets),
                "num_utterances_total": int(len(speaker_rows)),
                "num_utterances_seen": int((speaker_rows["side"] == "seen").sum()),
                "num_utterances_unseen": int((speaker_rows["side"] == "unseen").sum()),
                "weighted_score_mean": float(weighted_values.mean()) if not weighted_values.empty else None,
                "weighted_score_min": float(weighted_values.min()) if not weighted_values.empty else None,
                "weighted_score_max": float(weighted_values.max()) if not weighted_values.empty else None,
                "proficiency_levels_present": ";".join(sorted(set(score_rows["proficiency_level"].dropna().astype(str).tolist()))),
                "weighted_score_by_booklet": score_pairs(score_rows, "weighted_score"),
                "content_score_by_booklet": score_pairs(score_rows, "content_score"),
                "pronunciation_score_by_booklet": score_pairs(score_rows, "pronunciation_score"),
                "vocabulary_score_by_booklet": score_pairs(score_rows, "vocabulary_score"),
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "speaker_num", "global_speaker_id"]).reset_index(drop=True)


def build_report(
    all_df: pd.DataFrame,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    speaker_manifest: pd.DataFrame,
    excluded_df: pd.DataFrame,
    meta: Dict[str, object],
) -> str:
    exclusion_counts = (
        excluded_df["exclusion_reason"].value_counts().to_dict()
        if not excluded_df.empty and "exclusion_reason" in excluded_df.columns
        else {}
    )
    lines = [
        "# TEEMI Confirmed-GT Baseline Dataset Report",
        "",
        "## Scope",
        "",
        f"- Confirmed-GT booklets: `{', '.join(CONFIRMED_GT_BOOKLETS)}`",
        f"- Train-side booklets: `{', '.join(TRAIN_BOOKLETS)}`",
        f"- Eval-side booklets: `{', '.join(EVAL_BOOKLETS)}`",
        "- Split unit: cross-booklet global speaker id (`uXXX`)",
        f"- Split ratio by speakers: {meta['speaker_split_ratio']}",
        "- Goal: build a clean TEEMI baseline with speaker-disjoint evaluation and controlled proficiency labels.",
        "- Baseline materialization excludes utterances with missing proficiency and utterances labeled `No answer`.",
        "",
        "## Leakage Control",
        "",
        "- The split is assigned at the global-speaker level before materializing train/dev/test CSVs.",
        "- Therefore the same `uXXX` cannot appear across train/dev/test, even if that student recorded multiple booklets.",
        "- `train.csv` contains only seen-side booklets (`A01/B01/C01`) from train speakers.",
        "- `dev.csv` and `test.csv` contain only unseen-side booklets (`A02/B02/C02`) from dev/test speakers.",
        "",
        "## Baseline Eligibility Filters",
        "",
        f"- Excluded for missing proficiency: {exclusion_counts.get('missing_proficiency', 0)}",
        f"- Excluded for `No answer`: {exclusion_counts.get('invalid_proficiency:No answer', 0)}",
        "- Excluded utterance rows are saved to `data/teemi/excluded_utterances.csv`.",
        "",
        "## Upstream Parsing Rules",
        "",
        "- Transcript rows with missing / empty filename or text are skipped during parsing.",
        "- Only confirmed-GT booklets (`A01/A02/B01/B02/C01/C02`) are kept.",
        "- Rows without matched audio under `/share/corpus/.../audio/<BOOKLET>/` are skipped.",
        "- Score rows missing `wav_paths`, `student_id`, or `weighted_score` are skipped.",
        "- Score rows whose `student_id` does not match the speaker id encoded in `wav_paths` are skipped.",
        "",
        "## Overall Inventory",
        "",
        f"- Total utterances across 6 booklets: {len(all_df)}",
        f"- Total global speakers: {all_df['global_speaker_id'].nunique()}",
        f"- Total audio: {seconds_to_hms(float(all_df['duration_sec'].sum()))} ({format_float(float(all_df['duration_sec'].sum()))} sec)",
        "",
        "## By Booklet",
        "",
    ]

    for booklet in CONFIRMED_GT_BOOKLETS:
        sub = all_df[all_df["booklet"] == booklet]
        stats = summarize_subset(sub)
        lines.extend(
            [
                f"### {booklet}",
                "",
                f"- Utterances: {stats['utterances']}",
                f"- Global speakers: {stats['speakers']}",
                f"- Total audio: {stats['total_hms']} ({format_float(stats['total_seconds'])} sec)",
                f"- Mean / Median duration: {format_float(stats['mean_seconds'])} / {format_float(stats['median_seconds'])} sec",
                f"- Mean words: {format_float(stats['mean_words'])}",
                f"- Utterances with score: {stats['scored_utterances']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Materialized Splits",
            "",
        ]
    )
    for name, df in (("train", train_df), ("dev", dev_df), ("test", test_df)):
        stats = summarize_subset(df)
        speaker_count = speaker_manifest[speaker_manifest["split"] == name]["global_speaker_id"].nunique()
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Global speakers assigned: {speaker_count}",
                f"- Utterances materialized: {stats['utterances']}",
                f"- Total audio: {stats['total_hms']} ({format_float(stats['total_seconds'])} sec)",
                f"- Mean / Median duration: {format_float(stats['mean_seconds'])} / {format_float(stats['median_seconds'])} sec",
                f"- Mean words: {format_float(stats['mean_words'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## Eval Booklets in Dev/Test",
            "",
        ]
    )
    for split_name, df in (("dev", dev_df), ("test", test_df)):
        lines.append(f"### {split_name}")
        lines.append("")
        for booklet in EVAL_BOOKLETS:
            sub = df[df["booklet"] == booklet]
            lines.append(
                f"- {booklet}: {len(sub)} utt / {sub['global_speaker_id'].nunique()} speakers / "
                f"{seconds_to_hms(float(sub['duration_sec'].sum()))}"
            )
        lines.append("")

    coverage_counts = speaker_manifest["coverage_type"].value_counts().to_dict()
    lines.extend(
        [
            "## Speaker Coverage Types",
            "",
            f"- both seen+unseen: {coverage_counts.get('both', 0)}",
            f"- seen_only: {coverage_counts.get('seen_only', 0)}",
            f"- unseen_only: {coverage_counts.get('unseen_only', 0)}",
            "",
            "## CEFR-like / Score Fields",
            "",
            "- `weighted_score`, `proficiency_level`, `content_score`, `pronunciation_score`, and `vocabulary_score` are preserved.",
            "- `proficiency_level` is derived from TEEMI's `rubrics.json` score mapping.",
            "- `speaker_manifest.csv` stores booklet-level score traces per global speaker for interpretability.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-root", type=str, default="/share/corpus/2023_teemiv2")
    parser.add_argument("--output-dir", type=str, default="data/teemi")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-audio-workers", type=int, default=16)
    args = parser.parse_args()

    if not math.isclose(args.train_ratio + args.dev_ratio + args.test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/dev/test ratios must sum to 1.0")

    script_path = Path(__file__).resolve()
    root = resolve_repo_path(script_path, args.corpus_root)
    out_dir = resolve_repo_path(script_path, args.output_dir)

    logger.info("Loading confirmed-GT transcripts from %s", root / "transcript")
    transcripts = load_human_transcripts(root / "transcript")
    logger.info("Indexing audio from confirmed-GT booklets")
    audio_index = index_audio(root / "audio")
    logger.info("Loading proficiency scores from %s", root / "analysis")
    prof_df = load_proficiency_scores(root / "analysis", root / "rubrics.json")
    prof_map = prof_df.set_index("local_speaker_id").to_dict(orient="index")

    rows: List[dict] = []
    missing_audio = 0
    for _, row in transcripts.iterrows():
        stem = row["filename"]
        wav = audio_index.get(stem)
        if wav is None:
            missing_audio += 1
            continue
        booklet = row["booklet"]
        local_speaker_id = local_speaker_id_of(stem)
        global_speaker_id = global_speaker_id_of(stem)
        speaker_num = speaker_num_of(stem)
        score_info = prof_map.get(local_speaker_id, {})
        payload = {
            "audio_path": str(wav.resolve()),
            "text": row["text"],
            "booklet": booklet,
            "utt_id": stem,
            "side": side_of_booklet(booklet),
            "local_speaker_id": local_speaker_id,
            "global_speaker_id": global_speaker_id,
            "speaker_num": speaker_num,
            "transcript_xlsx": row.get("transcript_xlsx", ""),
            "transcript_sheet": row.get("transcript_sheet", ""),
            "weighted_score": score_info.get("weighted_score"),
            "proficiency_level": score_info.get("proficiency_level"),
            "content_score": score_info.get("content_score"),
            "pronunciation_score": score_info.get("pronunciation_score"),
            "vocabulary_score": score_info.get("vocabulary_score"),
            "score_row_count": score_info.get("score_row_count"),
            "analysis_sheets": score_info.get("analysis_sheets"),
            "analysis_files": score_info.get("analysis_files"),
        }
        rows.append(payload)

    all_df = pd.DataFrame(rows)
    if all_df.empty:
        raise RuntimeError("No matched confirmed-GT audio/transcript pairs found")
    logger.info("Matched %d utterances across %d global speakers (missing audio=%d)", len(all_df), all_df["global_speaker_id"].nunique(), missing_audio)

    duration_map = compute_duration_map(all_df["audio_path"].tolist(), max_workers=args.max_audio_workers)
    all_df["duration_sec"] = all_df["audio_path"].map(duration_map)
    all_df["num_words"] = all_df["text"].astype(str).str.split().str.len()

    eligible_df, excluded_df = split_baseline_eligible_rows(all_df)
    excluded_reasons = excluded_df.get("exclusion_reason", pd.Series(dtype=str))
    logger.info(
        "Baseline eligibility kept %d utterances and excluded %d (missing proficiency=%d, No answer=%d)",
        len(eligible_df),
        len(excluded_df),
        int((excluded_reasons == "missing_proficiency").sum()),
        int((excluded_reasons == "invalid_proficiency:No answer").sum()),
    )

    split_map = split_global_speakers(
        eligible_df["global_speaker_id"].tolist(),
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    eligible_df["split"] = eligible_df["global_speaker_id"].map(split_map)
    if not excluded_df.empty:
        excluded_df["split"] = excluded_df["global_speaker_id"].map(split_map)

    speaker_manifest = build_speaker_manifest(eligible_df)
    train_df = eligible_df[(eligible_df["split"] == "train") & (eligible_df["booklet"].isin(TRAIN_BOOKLETS))].copy()
    dev_df = eligible_df[(eligible_df["split"] == "dev") & (eligible_df["booklet"].isin(EVAL_BOOKLETS))].copy()
    test_df = eligible_df[(eligible_df["split"] == "test") & (eligible_df["booklet"].isin(EVAL_BOOKLETS))].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    all_df.sort_values(["booklet", "global_speaker_id", "utt_id"]).to_csv(out_dir / "all_utterances.csv", index=False)
    excluded_df.sort_values(["exclusion_reason", "booklet", "global_speaker_id", "utt_id"]).to_csv(
        out_dir / "excluded_utterances.csv", index=False
    )
    speaker_manifest.to_csv(out_dir / "speaker_manifest.csv", index=False)
    train_df.sort_values(["booklet", "global_speaker_id", "utt_id"]).to_csv(out_dir / "train.csv", index=False)
    dev_df.sort_values(["booklet", "global_speaker_id", "utt_id"]).to_csv(out_dir / "dev.csv", index=False)
    test_df.sort_values(["booklet", "global_speaker_id", "utt_id"]).to_csv(out_dir / "test.csv", index=False)
    for booklet in EVAL_BOOKLETS:
        dev_df[dev_df["booklet"] == booklet].sort_values(["global_speaker_id", "utt_id"]).to_csv(out_dir / f"dev_{booklet.lower()}.csv", index=False)
        test_df[test_df["booklet"] == booklet].sort_values(["global_speaker_id", "utt_id"]).to_csv(out_dir / f"test_{booklet.lower()}.csv", index=False)

    meta = {
        "corpus_root": str(root),
        "confirmed_gt_booklets": list(CONFIRMED_GT_BOOKLETS),
        "train_booklets": list(TRAIN_BOOKLETS),
        "eval_booklets": list(EVAL_BOOKLETS),
        "speaker_split_ratio": f"{args.train_ratio:.2f}/{args.dev_ratio:.2f}/{args.test_ratio:.2f}",
        "seed": args.seed,
        "n_all_utterances": int(len(all_df)),
        "n_global_speakers": int(eligible_df["global_speaker_id"].nunique()),
        "n_excluded_utterances": int(len(excluded_df)),
        "excluded_missing_proficiency": int((excluded_reasons == "missing_proficiency").sum()),
        "excluded_no_answer": int((excluded_reasons == "invalid_proficiency:No answer").sum()),
        "n_train_utterances": int(len(train_df)),
        "n_dev_utterances": int(len(dev_df)),
        "n_test_utterances": int(len(test_df)),
        "missing_audio_rows": missing_audio,
        "transcript_sources": sorted(all_df["transcript_xlsx"].dropna().astype(str).unique().tolist()),
        "analysis_sources": sorted(
            {
                value
                for values in all_df["analysis_files"].dropna().astype(str).tolist()
                for value in values.split(";")
                if value
            }
        ),
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "dataset_report.md").write_text(
        build_report(
            all_df=all_df,
            train_df=train_df,
            dev_df=dev_df,
            test_df=test_df,
            speaker_manifest=speaker_manifest,
            excluded_df=excluded_df,
            meta=meta,
        ),
        encoding="utf-8",
    )
    logger.info("Prepared TEEMI baseline dataset at %s", out_dir)


if __name__ == "__main__":
    main()
