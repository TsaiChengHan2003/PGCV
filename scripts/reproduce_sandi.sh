#!/usr/bin/env bash
# Reproduce the Speak & Improve experiments.
# Usage: bash scripts/reproduce_sandi.sh whisper 0
#        bash scripts/reproduce_sandi.sh qwen 0
# Official SpWER reads scoring tools from $SANDI_CORPUS
# (default: /share/corpus/sandi2025-challenge).
set -euo pipefail

MODEL="${1:-whisper}"
GPU_ID="${2:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
export SANDI_CORPUS="${SANDI_CORPUS:-/share/corpus/sandi2025-challenge}"

if [[ "$GPU_ID" == "cpu" ]]; then
  DEVICE="cpu"
  unset CUDA_VISIBLE_DEVICES || true
else
  DEVICE="cuda"
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

case "$MODEL" in
  whisper)
    SPWER_MODEL="whisper_base"
    DATA_SELECT="configs/sandi_data_select_tau.yaml"
    LEVEL_ERROR="configs/sandi_overall_cefr_level_error_tau_whisper_base.yaml"
    EMBED="configs/sandi_overall_cefr_pgcv_embeddings_whisper_base.yaml"
    PGCV="configs/sandi_overall_cefr_pgcv_whisper_base.yaml"
    FULL_FT="configs/baseline/sandi_full_ft_spwer.yaml"
    ;;
  qwen)
    bash "$ROOT/scripts/require_qwen3_asr_env.sh" "$PYTHON_BIN"
    export HF_DEACTIVATE_ASYNC_LOAD="${HF_DEACTIVATE_ASYNC_LOAD:-1}"
    SPWER_MODEL="qwen3_0.6b"
    DATA_SELECT="configs/sandi_data_select_tau_qwen3_asr_0.6b.yaml"
    LEVEL_ERROR="configs/sandi_overall_cefr_level_error_tau_qwen3_asr_0.6b.yaml"
    EMBED="configs/sandi_overall_cefr_pgcv_embeddings_qwen3_asr_0.6b.yaml"
    PGCV="configs/sandi_overall_cefr_pgcv_qwen3_asr_0.6b.yaml"
    FULL_FT="configs/baseline/sandi_full_ft_spwer_qwen3_0.6b.yaml"
    ;;
  *)
    echo "MODEL must be whisper or qwen" >&2
    exit 2
    ;;
esac

echo "Speak & Improve reproduce: model=$MODEL device=$DEVICE corpus=$SANDI_CORPUS"
"$PYTHON_BIN" scripts/check_audio_manifests.py \
  data/sandi/asr/train.csv data/sandi/asr/dev.csv data/sandi/asr/eval.csv \
  data/sandi/asr_overall/train.csv data/sandi/asr_overall/dev.csv data/sandi/asr_overall/eval.csv

# Full fine-tuning baseline on the official train-asr split.
"$PYTHON_BIN" scripts/run_sandi_full_ft.py --config "$FULL_FT"

# Pretrained and full-FT official SpWER. The error-GT tau is scored after
# data selection writes vectors/sandi/data_select_tau/.../tau_error_gt.pt.
"$PYTHON_BIN" scripts/run_sandi_spwer_exp.py \
  --model "$SPWER_MODEL" --exp pretrained --split eval --device "$DEVICE"
"$PYTHON_BIN" scripts/run_sandi_spwer_exp.py \
  --model "$SPWER_MODEL" --exp full_ft --split eval --device "$DEVICE"

# Global error-targeted tau on train-asr, then per-level taus on the
# overall-CEFR subset.
"$PYTHON_BIN" scripts/run_teemi_a01_only_data_select_tau.py \
  --config "$DATA_SELECT" --stage all --device "$DEVICE"
"$PYTHON_BIN" scripts/run_teemi_a01_only_level_error_tau.py \
  --config "$LEVEL_ERROR" --stage 1,2,3,4 --device "$DEVICE"
"$PYTHON_BIN" scripts/run_teemi_a01_only_speaker_cluster_tau.py \
  --config "$EMBED" --stage 2 --device "$DEVICE"
"$PYTHON_BIN" scripts/run_teemi_a01_only_smooth_tau.py \
  --config "$PGCV" --stage all --device "$DEVICE"

# Official long-form SpWER for the selected PGCV system.
"$PYTHON_BIN" scripts/run_sandi_pgcv_spwer.py \
  --config "$PGCV" --split eval --device "$DEVICE" --corpus "$SANDI_CORPUS"
