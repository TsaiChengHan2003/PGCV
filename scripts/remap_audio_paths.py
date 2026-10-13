#!/usr/bin/env python3
"""Rewrite corpus path prefixes in the released manifests.

The CSVs store the original audio locations. They do not contain waveforms.
After obtaining TEEMI and Speak & Improve, point those prefixes at the local
copies. Clip suffixes such as ``#t=0.00,18.78`` are preserved.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


COLUMNS = ("audio_path", "source_audio_path")


def rewrite(path: Path, old: str, new: str) -> int:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)
    changed = 0
    for row in rows:
        for column in COLUMNS:
            value = row.get(column)
            if value and value.startswith(old):
                row[column] = new + value[len(old) :]
                changed += 1
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="Prefix currently stored in the CSV")
    parser.add_argument("--new", required=True, help="Local prefix that replaces --old")
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    old = args.old.rstrip("/")
    new = args.new.rstrip("/")
    for path in args.csv:
        changed = rewrite(path, old, new)
        print(f"{path}: rewrote {changed} path fields")


if __name__ == "__main__":
    main()
