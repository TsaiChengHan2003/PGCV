#!/usr/bin/env bash
# Fail fast unless PYTHON_BIN has transformers>=5.13 (Qwen3-ASR native checkpoints).
set -euo pipefail

PYTHON_BIN="${1:-${PYTHON_BIN:-python}}"

"$PYTHON_BIN" - <<'PY'
import sys

try:
    import transformers
except ImportError:
    print("transformers is not installed in this Python.", file=sys.stderr)
    raise SystemExit(1)

parts = []
for chunk in transformers.__version__.split("."):
    digits = ""
    for char in chunk:
        if char.isdigit():
            digits += char
        else:
            break
    parts.append(int(digits) if digits else 0)
    if len(parts) >= 3:
        break
while len(parts) < 3:
    parts.append(0)

print(f"  python: {sys.executable}")
print(f"  transformers: {transformers.__version__} ({transformers.__file__})")
try:
    import torch

    print(f"  torch: {torch.__version__} ({torch.__file__})")
except ImportError:
    print("  torch: not installed", file=sys.stderr)
    raise SystemExit(1)
if tuple(parts[:3]) < (5, 13, 0):
    print(
        "Qwen3-ASR needs transformers>=5.13.0. "
        "Do not use eztalking_v2 / the Whisper env.\n"
        "  conda activate l2-pseudo2real\n"
        "  # or: PYTHON_BIN=/path/to/l2-pseudo2real/bin/python",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
