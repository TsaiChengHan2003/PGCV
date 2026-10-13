#!/usr/bin/env python3
"""Count manifest rows and report how many audio files are readable locally."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULTS = [
    Path("data/teemi/a01_only/train.csv"),
    Path("data/teemi/a01_only/dev.csv"),
    Path("data/teemi/a01_only/test.csv"),
    Path("data/teemi/a01_only/test_a02_external.csv"),
    Path("data/sandi/asr/train.csv"),
    Path("data/sandi/asr/dev.csv"),
    Path("data/sandi/asr/eval.csv"),
    Path("data/sandi/asr_overall/train.csv"),
    Path("data/sandi/asr_overall/dev.csv"),
    Path("data/sandi/asr_overall/eval.csv"),
]


def audio_file(value: str) -> Path:
    return Path(value.split("#", 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*", type=Path, default=DEFAULTS)
    args = parser.parse_args()
    missing_total = 0
    for path in args.csv:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        missing = sum(1 for row in rows if not audio_file(row["audio_path"]).is_file())
        missing_total += missing
        print(f"{path}: rows={len(rows)} missing_audio={missing}")
    if missing_total:
        raise SystemExit(
            f"{missing_total} audio paths are not readable. "
            "Obtain the corpora and run scripts/remap_audio_paths.py."
        )


if __name__ == "__main__":
    main()
