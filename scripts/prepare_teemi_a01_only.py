#!/usr/bin/env python3
"""Prepare an A01-only TEEMI baseline with A02 external test.

Design:
- Use only A01 to split speakers into train/dev/test.
- Materialize A01 train/dev/test as internal splits.
- Materialize A02 as an external test set, but drop any A02 rows whose
  global_speaker_id appears in the A01 split universe.
- Keep the original 6-booklet experiment outputs untouched by writing to a
  dedicated output directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd

from prepare_teemi import compute_duration_map, format_float, resolve_repo_path, split_baseline_eligible_rows, split_global_speakers, summarize_subset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_teemi_a01_only")

SOURCE_BOOKLET = "A01"
EXTERNAL_TEST_BOOKLET = "A02"
SELECTED_BOOKLETS = (SOURCE_BOOKLET, EXTERNAL_TEST_BOOKLET)


def side_of_booklet(booklet: str) -> str:
    if booklet == SOURCE_BOOKLET:
        return "source_internal"
    if booklet == EXTERNAL_TEST_BOOKLET:
        return "external_test_candidate"
    return "other"


def build_speaker_manifest(
    selected_df: pd.DataFrame,
    source_df: pd.DataFrame,
    external_test_df: pd.DataFrame,
    external_overlap_df: pd.DataFrame,
) -> pd.DataFrame:
    split_map: Dict[str, str] = {}
    split_map.update(source_df.drop_duplicates("global_speaker_id").set_index("global_speaker_id")["split"].to_dict())
    split_map.update(
        external_test_df.drop_duplicates("global_speaker_id")
        .assign(split_label="external_test")
        .set_index("global_speaker_id")["split_label"]
        .to_dict()
    )
    split_map.update(
        external_overlap_df.drop_duplicates("global_speaker_id")
        .assign(split_label="external_overlap_excluded")
        .set_index("global_speaker_id")["split_label"]
        .to_dict()
    )

    rows: List[dict] = []
    for speaker_id, sub in selected_df.groupby("global_speaker_id"):
        weighted = sub["weighted_score"].dropna()
        rows.append(
            {
                "global_speaker_id": speaker_id,
                "speaker_num": int(sub["speaker_num"].dropna().iloc[0]) if not sub["speaker_num"].dropna().empty else None,
                "assigned_split": split_map.get(speaker_id, "unassigned"),
                "booklets_present": ";".join(sorted(set(sub["booklet"].astype(str).tolist()))),
                "num_utterances_total": int(len(sub)),
                "num_utterances_a01": int((sub["booklet"] == SOURCE_BOOKLET).sum()),
                "num_utterances_a02": int((sub["booklet"] == EXTERNAL_TEST_BOOKLET).sum()),
                "weighted_score_mean": float(weighted.mean()) if not weighted.empty else None,
                "proficiency_levels_present": ";".join(sorted(set(sub["proficiency_level"].dropna().astype(str).tolist()))),
            }
        )
    return pd.DataFrame(rows).sort_values(["assigned_split", "speaker_num", "global_speaker_id"]).reset_index(drop=True)


def build_report(
    selected_df: pd.DataFrame,
    source_df: pd.DataFrame,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    external_test_df: pd.DataFrame,
    external_overlap_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    meta: Dict[str, object],
) -> str:
    exclusion_counts = (
        excluded_df["exclusion_reason"].value_counts().to_dict()
        if not excluded_df.empty and "exclusion_reason" in excluded_df.columns
        else {}
    )

    lines = [
        "# TEEMI A01-only Dataset Report",
        "",
        "## Scope",
        "",
        f"- Source/internal booklet: `{SOURCE_BOOKLET}`",
        f"- External test booklet: `{EXTERNAL_TEST_BOOKLET}`",
        "- Split unit: global speaker id (`uXXX`)",
        f"- Split ratio on A01 speakers: {meta['speaker_split_ratio']}",
        "- A01 is used to materialize internal train/dev/test only.",
        "- A02 is reserved as an external test set after removing any speaker that appears in the A01 split universe.",
        "- Missing proficiency and `No answer` utterances are excluded before split materialization, consistent with the baseline policy.",
        "",
        "## Leakage Control",
        "",
        "- `train.csv`, `dev.csv`, and `test.csv` are derived only from A01 speakers.",
        "- `test_a02_external.csv` excludes all rows whose `global_speaker_id` appears anywhere in the A01 split universe.",
        "- This means the external A02 test is speaker-disjoint from every A01 internal split.",
        "",
        "## Eligibility Filters",
        "",
        f"- Excluded for missing proficiency: {exclusion_counts.get('missing_proficiency', 0)}",
        f"- Excluded for `No answer`: {exclusion_counts.get('invalid_proficiency:No answer', 0)}",
        f"- Excluded from A02 external test due to overlap with A01 speakers: {len(external_overlap_df)} utterances / {external_overlap_df['global_speaker_id'].nunique() if len(external_overlap_df) else 0} speakers",
        "",
        "## Selected Booklets",
        "",
    ]

    for booklet in SELECTED_BOOKLETS:
        sub = selected_df[selected_df["booklet"] == booklet]
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
                "",
            ]
        )

    lines.extend(["## Materialized Splits", ""])
    for name, df in (
        ("train (A01 internal)", train_df),
        ("dev (A01 internal)", dev_df),
        ("test (A01 internal)", test_df),
        ("test_a02_external", external_test_df),
    ):
        stats = summarize_subset(df)
        lines.extend(
            [
                f"### {name}",
                "",
                f"- Utterances: {stats['utterances']}",
                f"- Global speakers: {stats['speakers']}",
                f"- Total audio: {stats['total_hms']} ({format_float(stats['total_seconds'])} sec)",
                f"- Mean / Median duration: {format_float(stats['mean_seconds'])} / {format_float(stats['median_seconds'])} sec",
                f"- Mean words: {format_float(stats['mean_words'])}",
                "",
            ]
        )

    source_stats = summarize_subset(source_df)
    lines.extend(
        [
            "## A01 Split Universe",
            "",
            f"- Eligible A01 utterances before split: {source_stats['utterances']}",
            f"- Eligible A01 speakers before split: {source_stats['speakers']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-all-utterances-csv", type=str, default="data/teemi/all_utterances.csv")
    parser.add_argument("--output-dir", type=str, default="data/teemi/a01_only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-audio-workers", type=int, default=16)
    args = parser.parse_args()

    if not math.isclose(args.train_ratio + args.dev_ratio + args.test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/dev/test ratios must sum to 1.0")

    script_path = Path(__file__).resolve()
    base_all_path = resolve_repo_path(script_path, args.base_all_utterances_csv)
    out_dir = resolve_repo_path(script_path, args.output_dir)

    selected_df = pd.read_csv(base_all_path)
    selected_df = selected_df[selected_df["booklet"].isin(SELECTED_BOOKLETS)].copy()
    if selected_df.empty:
        raise RuntimeError(f"No A01/A02 rows found in {base_all_path}")
    selected_df["side"] = selected_df["booklet"].map(side_of_booklet)
    if "num_words" not in selected_df.columns:
        selected_df["num_words"] = selected_df["text"].astype(str).str.split().str.len()
    if "duration_sec" not in selected_df.columns:
        logger.info("duration_sec missing in %s; recomputing from audio_path", base_all_path)
        selected_df["duration_sec"] = selected_df["audio_path"].map(
            compute_duration_map(selected_df["audio_path"].tolist(), max_workers=args.max_audio_workers)
        )

    eligible_df, excluded_df = split_baseline_eligible_rows(selected_df)
    source_df = eligible_df[eligible_df["booklet"] == SOURCE_BOOKLET].copy()
    external_df = eligible_df[eligible_df["booklet"] == EXTERNAL_TEST_BOOKLET].copy()

    split_map = split_global_speakers(
        source_df["global_speaker_id"].tolist(),
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    source_df["split"] = source_df["global_speaker_id"].map(split_map)
    source_speakers = set(split_map)

    external_overlap_mask = external_df["global_speaker_id"].isin(source_speakers)
    external_overlap_df = external_df.loc[external_overlap_mask].copy()
    if not external_overlap_df.empty:
        external_overlap_df["exclusion_reason"] = "speaker_overlap_with_a01_split_universe"
    external_test_df = external_df.loc[~external_overlap_mask].copy()
    external_test_df["split"] = "external_test"

    train_df = source_df[source_df["split"] == "train"].copy()
    dev_df = source_df[source_df["split"] == "dev"].copy()
    test_df = source_df[source_df["split"] == "test"].copy()

    speaker_manifest = build_speaker_manifest(
        selected_df=eligible_df,
        source_df=source_df,
        external_test_df=external_test_df,
        external_overlap_df=external_overlap_df,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    selected_df.sort_values(["booklet", "global_speaker_id", "utt_id"]).to_csv(out_dir / "all_utterances.csv", index=False)
    excluded_df.sort_values(["exclusion_reason", "booklet", "global_speaker_id", "utt_id"]).to_csv(
        out_dir / "excluded_utterances.csv", index=False
    )
    external_overlap_df.sort_values(["booklet", "global_speaker_id", "utt_id"]).to_csv(
        out_dir / "excluded_a02_overlap.csv", index=False
    )
    speaker_manifest.to_csv(out_dir / "speaker_manifest.csv", index=False)
    train_df.sort_values(["global_speaker_id", "utt_id"]).to_csv(out_dir / "train.csv", index=False)
    dev_df.sort_values(["global_speaker_id", "utt_id"]).to_csv(out_dir / "dev.csv", index=False)
    test_df.sort_values(["global_speaker_id", "utt_id"]).to_csv(out_dir / "test.csv", index=False)
    external_test_df.sort_values(["global_speaker_id", "utt_id"]).to_csv(out_dir / "test_a02_external.csv", index=False)

    excluded_reasons = excluded_df.get("exclusion_reason", pd.Series(dtype=str))
    meta = {
        "base_all_utterances_csv": str(base_all_path),
        "selected_booklets": list(SELECTED_BOOKLETS),
        "source_booklet": SOURCE_BOOKLET,
        "external_test_booklet": EXTERNAL_TEST_BOOKLET,
        "speaker_split_ratio": f"{args.train_ratio:.2f}/{args.dev_ratio:.2f}/{args.test_ratio:.2f}",
        "seed": args.seed,
        "n_selected_utterances": int(len(selected_df)),
        "n_source_speakers": int(source_df["global_speaker_id"].nunique()),
        "n_train_utterances": int(len(train_df)),
        "n_dev_utterances": int(len(dev_df)),
        "n_test_utterances": int(len(test_df)),
        "n_external_test_utterances": int(len(external_test_df)),
        "n_external_overlap_utterances": int(len(external_overlap_df)),
        "n_external_overlap_speakers": int(external_overlap_df["global_speaker_id"].nunique()) if len(external_overlap_df) else 0,
        "excluded_missing_proficiency": int((excluded_reasons == "missing_proficiency").sum()),
        "excluded_no_answer": int((excluded_reasons == "invalid_proficiency:No answer").sum()),
    }
    (out_dir / "prepare_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "dataset_report.md").write_text(
        build_report(
            selected_df=selected_df,
            source_df=source_df,
            train_df=train_df,
            dev_df=dev_df,
            test_df=test_df,
            external_test_df=external_test_df,
            external_overlap_df=external_overlap_df,
            excluded_df=excluded_df,
            meta=meta,
        ),
        encoding="utf-8",
    )
    logger.info("Prepared TEEMI A01-only dataset at %s", out_dir)


if __name__ == "__main__":
    main()
