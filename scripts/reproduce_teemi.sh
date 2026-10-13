#!/usr/bin/env bash
# Reproduce the TEEMI A01 / A02 experiments.
# Usage: bash scripts/reproduce_teemi.sh whisper 0
#        bash scripts/reproduce_teemi.sh qwen 0
set -euo pipefail

MODEL="${1:-whisper}"
GPU_ID="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "$GPU_ID" == "cpu" ]]; then
  DEVICE="cpu"
  unset CUDA_VISIBLE_DEVICES || true
else
  DEVICE="cuda"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

case "$MODEL" in
  whisper)
    BASELINE="configs/baseline/teemi_a01_only.yaml"
    DATA_SELECT="configs/teemi_a01_only_data_select_tau.yaml"
    LEVEL_ERROR="configs/teemi_a01_only_level_error_tau.yaml"
    LEVEL_FT="configs/teemi_a01_only_level_full_ft_whisper_base.yaml"
    EMBED="configs/teemi_a01_only_speaker_cluster_tau.yaml"
    PGCV="configs/teemi_a01_only_pgcv_whisper_base.yaml"
    ;;
  qwen)
    bash "$ROOT/scripts/require_qwen3_asr_env.sh" "$PYTHON_BIN"
    export HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD:-1}"
    BASELINE="configs/baseline/teemi_a01_only_qwen3_asr_0.6b.yaml"
    DATA_SELECT="configs/teemi_a01_only_data_select_tau_qwen3_asr_0.6b.yaml"
    LEVEL_ERROR="configs/teemi_a01_only_level_error_tau_qwen3_asr_0.6b.yaml"
    LEVEL_FT="configs/teemi_a01_only_level_full_ft_qwen3_asr_0.6b.yaml"
    EMBED="configs/teemi_a01_only_speaker_embeddings_qwen3_asr_0.6b.yaml"
    PGCV="configs/teemi_a01_only_pgcv_qwen3_asr_0.6b.yaml"
    ;;
  *)
    echo "MODEL must be whisper or qwen" >&2
    exit 2
    ;;
esac

echo "TEEMI reproduce: model=$MODEL device=$DEVICE"
"$PYTHON_BIN" scripts/check_audio_manifests.py \
  data/teemi/a01_only/train.csv data/teemi/a01_only/dev.csv \
  data/teemi/a01_only/test.csv data/teemi/a01_only/test_a02_external.csv

# Full fine-tuning baseline: 4000 steps, warmup 500, effective batch 12.
bash "$ROOT/run_single_gpu.sh" "$BASELINE" all "$GPU_ID"

# Global error-targeted correction: score the training set, train the
# 2000-step error subset, and write tau_g.
"$PYTHON_BIN" scripts/run_teemi_a01_only_data_select_tau.py \
  --config "$DATA_SELECT" --stage all --device "$DEVICE"

# Proficiency-specific error corrections. Stages 1-4 train the bank.
# Stages 5-7 are diagnostics and are not required by the PGCV table.
"$PYTHON_BIN" scripts/run_teemi_a01_only_level_error_tau.py \
  --config "$LEVEL_ERROR" --stage 1,2,3,4 --device "$DEVICE"

# Oracle proficiency-specific full fine-tuning, one model per gold level.
"$PYTHON_BIN" scripts/run_teemi_a01_only_level_full_ft.py \
  --config "$LEVEL_FT" --stage all --device "$DEVICE"

# Frozen-encoder speaker embeddings used by the gate.
"$PYTHON_BIN" scripts/run_teemi_a01_only_speaker_cluster_tau.py \
  --config "$EMBED" --stage 2 --device "$DEVICE"

# Train the gate and select beta / lambda on A01 dev, then evaluate A01 and A02.
"$PYTHON_BIN" scripts/run_teemi_a01_only_smooth_tau.py \
  --config "$PGCV" --stage all --device "$DEVICE"
