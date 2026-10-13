#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash run_single_gpu.sh <config_path> [stage] [gpu_id]" >&2
  exit 1
fi

CONFIG_PATH="$1"
STAGE="${2:-all}"
GPU_ID="${3:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$GPU_ID" == "cpu" ]]; then
  DEVICE="cpu"
  unset CUDA_VISIBLE_DEVICES || true
else
  DEVICE="cuda"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

echo "Running single-device pipeline"
echo "  config: $CONFIG_PATH"
echo "  stage:  $STAGE"
echo "  device: $DEVICE"
if [[ "$DEVICE" == "cuda" ]]; then
  echo "  gpu:    $GPU_ID"
fi

if ! "$PYTHON_BIN" - <<'PY'
import sys

missing = []
for name in ["torch", "transformers", "soundfile", "librosa"]:
    try:
        __import__(name)
    except Exception:
        missing.append(name)

if missing:
    print("Missing Python packages: " + ", ".join(missing), file=sys.stderr)
    raise SystemExit(1)
PY
then
  echo "Please activate the correct environment or install dependencies from requirements.txt." >&2
  exit 1
fi

exec "$PYTHON_BIN" run_pipeline.py --config "$CONFIG_PATH" --stage "$STAGE" --device "$DEVICE"
