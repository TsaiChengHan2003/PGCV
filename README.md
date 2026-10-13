# Proficiency-Guided Correction Vectors for L2 English ASR

Reviewer reproduction package for the anonymous submission. The code trains
error-targeted correction vectors, a speech-only proficiency gate, and the
reported full-fine-tuning baselines for Whisper-base and Qwen3-ASR-0.6B on
TEEMI and Speak & Improve.

TEEMI and Speak & Improve audio are private and are not in this repository.
The released CSVs record transcripts, labels, and the original audio paths.
Those paths do not grant access to the waveforms. See `data/README.md`.

1. Install the Python dependencies and obtain both corpora under their own
   licenses.
2. Point `audio_path` at the local audio, or mount the corpora at the
   prefixes already stored in the CSVs.
3. Run one backbone at a time on a single 24 GB GPU. The paper used one
   NVIDIA GeForce RTX 3090.

```bash
python -m pip install -r requirements.txt
python scripts/check_audio_manifests.py
bash scripts/reproduce_teemi.sh whisper 0
bash scripts/reproduce_sandi.sh whisper 0
```

Replace `whisper` with `qwen` for Qwen3-ASR-0.6B. Qwen requires
`transformers>=5.13.0`. The Qwen scripts set `HF_DEACTIVATE_ASYNC_LOAD=1`
to avoid a Python 3.10 `re.compile` race while weights load.

Speak & Improve official SpWER also needs the challenge scoring tools.
If they are not at `/share/corpus/sandi2025-challenge`, set:

```bash
export SANDI_CORPUS=/path/to/sandi2025-challenge
```

## Method, as implemented

For a pretrained model with parameters \(\theta_{\mathrm{pretrain}}\):

1. Decode the training utterances and keep those whose hypothesis differs
   from the reference. That is the error subset.
2. Fine-tune one global model on the whole error subset, and one model on
   each proficiency slice. Every error-targeted run starts from the same
   pretrained checkpoint and uses 2000 AdamW steps, learning rate
   \(1\times10^{-5}\), weight decay 0.01, 200 warmup steps, and an
   effective batch size of 12.
3. A correction vector is the parameter difference from the pretrained
   checkpoint.
4. The gate mean-pools frozen encoder states by speaker (TEEMI) or by
   session (Speak & Improve). It is a scaled dot-product affinity with
   \(d=64\) and dropout 0.3, trained with class-weighted cross-entropy.
   The backbone and the vectors stay frozen. Optimization is AdamW at
   \(1\times10^{-3}\), weight decay 0.05, batch size 64, for at most 200
   epochs. A stratified 20% of training speakers is held out, and training
   stops after 20 epochs without an improvement in validation
   cross-entropy.
5. On the development set, search
   \(\beta\in\{0,0.25,0.5,0.75,1\}\) and
   \(\lambda\in\{0.25,0.5,0.7,0.75,1\}\). The selected pair is then frozen
   for test. Proficiency labels are not an inference input.

Full fine-tuning and the proficiency-specific full-fine-tuning oracle use
the same optimizer, learning rate, weight decay, and effective batch, for
4000 steps and 500 warmup steps. The oracle selects the checkpoint with
the gold proficiency label and is not an inference system.

Effective batch 12 is 12×1 for Whisper. Qwen uses 4×3 on TEEMI and 1×12
on Speak & Improve, because the longer S&I audio does not fit a larger
per-device batch on a 24 GB card. Qwen training is bf16; Whisper training
is fp16. The seed is 42.

Reported development choices, which the search should recover:

| Setting | \(\beta\) | \(\lambda\) |
| --- | ---: | ---: |
| TEEMI, Whisper-base | 0.25 | 1.0 |
| TEEMI, Qwen3-ASR-0.6B | 0.25 | 0.7 |
| Speak & Improve, Whisper-base | 0.5 | 1.0 |
| Speak & Improve, Qwen3-ASR-0.6B | 0.25 | 1.0 |

## Command map

`scripts/reproduce_teemi.sh` and `scripts/reproduce_sandi.sh` call these
stages in order. Run them from the repository root.

| Paper piece | Command |
| --- | --- |
| TEEMI full fine-tuning | `bash run_single_gpu.sh configs/baseline/teemi_a01_only.yaml all 0` |
| TEEMI global error vector | `python scripts/run_teemi_a01_only_data_select_tau.py --config configs/teemi_a01_only_data_select_tau.yaml --stage all` |
| TEEMI per-level error vectors | `python scripts/run_teemi_a01_only_level_error_tau.py --config configs/teemi_a01_only_level_error_tau.yaml --stage 1,2,3,4` |
| TEEMI oracle per-level full fine-tuning | `python scripts/run_teemi_a01_only_level_full_ft.py --config configs/teemi_a01_only_level_full_ft_whisper_base.yaml --stage all` |
| TEEMI gate and PGCV | embedding stage 2, then `python scripts/run_teemi_a01_only_smooth_tau.py --config configs/teemi_a01_only_pgcv_whisper_base.yaml --stage all` |
| S&I full fine-tuning | `python scripts/run_sandi_full_ft.py --config configs/baseline/sandi_full_ft_spwer.yaml` |
| S&I pretrained / full-FT SpWER | `python scripts/run_sandi_spwer_exp.py --model whisper_base --exp pretrained` and `--exp full_ft` |
| S&I PGCV SpWER | `python scripts/run_sandi_pgcv_spwer.py --config configs/sandi_overall_cefr_pgcv_whisper_base.yaml --split eval` |

Qwen file names replace `whisper_base` with `qwen3_asr_0.6b` or
`qwen3_0.6b`, as listed in `scripts/reproduce_sandi.sh`.

TEEMI writes `results/teemi/a01_only/pgcv/<backbone>/eval_summary.json`.
Speak & Improve official PGCV scoring writes
`results/sandi/overall_cefr/pgcv/<backbone>/official_spwer/eval/`.

After those results exist, the paper figures are:

```bash
python scripts/plot_pgcv_parameter_wer.py
python scripts/plot_teemi_pgcv_error_composition.py
python scripts/plot_affinity_heatmaps.py
python scripts/analyze_pgcv_composition.py
```

Figures are written under `analysis/`.

## Data policy

Do not redistribute the audio. The manifests may be shared with reviewers
because a stored path is not a copy of the recording. Rebuilding the
splits from the raw releases is optional and requires the private
workbooks or challenge references:

```bash
python scripts/prepare_teemi_a01_only.py
python scripts/prepare_sandi_asr_csv.py
python scripts/prepare_sandi_overall_cefr.py
```

The checked-in manifests are the ones used for the reported experiments.
`data/sandi/asr/train.csv` keeps sessions that have no overall score,
because full fine-tuning uses the official train-asr split. PGCV's
per-level models and gate use `data/sandi/asr_overall/`, which drops
those sessions. The global S&I correction is still trained from the
official train-asr error subset.

## Layout

```text
src/            model, training, gate, and task-vector code
scripts/        paper stages, scoring, plots, and path checks
configs/        hyperparameters for the reported runs
data/           utterance manifests only
run_pipeline.py TEEMI supervised fine-tuning entry point
```

Checkpoints, vectors, and result directories are created locally and are
git-ignored.
