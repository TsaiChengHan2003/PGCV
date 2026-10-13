#!/usr/bin/env python3
"""Build SANDI metadata whose CEFR label comes from the overall SLA score."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "dev", "eval")
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


def resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def overall_score_to_cefr(score: Any) -> str:
    """Round to the nearest half point, with .25 ties rounded upward."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        return ""
    if math.isnan(value):
        return ""
    half = math.floor(value * 2.0 + 0.5) / 2.0
    half = min(6.0, max(1.0, half))
    return CEFR_FROM_HALF[half]


def convert_split(src: Path, dest: Path) -> dict[str, Any]:
    frame = pd.read_csv(src)
    required = {"session_id", "sla_score", "overall_sla_score", "cefr", "proficiency_level"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{src} is missing required columns: {missing}")

    input_rows = len(frame)
    input_sessions = frame["session_id"].nunique()
    frame["part_cefr"] = frame["cefr"]
    frame["part_proficiency_level"] = frame["proficiency_level"]
    frame["cefr"] = frame["overall_sla_score"].map(overall_score_to_cefr)
    frame["proficiency_level"] = frame["cefr"]

    missing_mask = frame["cefr"].astype(str).str.strip().eq("")
    dropped_rows = int(missing_mask.sum())
    dropped_sessions = int(frame.loc[missing_mask, "session_id"].nunique())
    frame = frame.loc[~missing_mask].copy()

    # An overall label is a session-level property and must be unique per learner.
    labels_per_session = frame.groupby("session_id")["proficiency_level"].nunique()
    if len(labels_per_session) and int(labels_per_session.max()) != 1:
        raise ValueError(f"{src} produced inconsistent overall labels within a session")

    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return {
        "input_rows": input_rows,
        "output_rows": len(frame),
        "dropped_missing_overall_rows": dropped_rows,
        "input_sessions": int(input_sessions),
        "output_sessions": int(frame["session_id"].nunique()),
        "dropped_missing_overall_sessions": dropped_sessions,
        "utterances_by_overall_cefr": dict(
            sorted(Counter(frame["proficiency_level"].astype(str)).items())
        ),
        "sessions_by_overall_cefr": dict(
            sorted(
                Counter(
                    frame[["session_id", "proficiency_level"]]
                    .drop_duplicates()["proficiency_level"]
                    .astype(str)
                ).items()
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/sandi/asr")
    parser.add_argument("--output-dir", default="data/sandi/asr_overall")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    summary = {
        "label_source": "overall_sla_score",
        "rounding": "nearest 0.5; exact .25 ties round upward",
        "splits": {},
    }
    for split in SPLITS:
        summary["splits"][split] = convert_split(
            input_dir / f"{split}.csv",
            output_dir / f"{split}.csv",
        )
    summary_path = output_dir / "distribution.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
