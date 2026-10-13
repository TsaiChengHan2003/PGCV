# Manifests

These CSVs are the utterance manifests used by the paper. They contain
transcripts, proficiency labels, split ids, and the original audio locations.
They do not contain audio.

| File | Rows | Role |
| --- | ---: | --- |
| `teemi/a01_only/train.csv` | 4626 | TEEMI A01 train |
| `teemi/a01_only/dev.csv` | 609 | TEEMI A01 development |
| `teemi/a01_only/test.csv` | 928 | TEEMI A01 test |
| `teemi/a01_only/test_a02_external.csv` | 1368 | TEEMI A02, speaker-disjoint from A01 |
| `sandi/asr/{train,dev,eval}.csv` | 10986 / 8636 / 8586 | Speak & Improve official ASR splits, including sessions without an overall score |
| `sandi/asr_overall/{train,dev,eval}.csv` | 6909 / 8636 / 8586 | Same ASR splits after dropping sessions with no overall SLA score |

TEEMI levels are `pre-A`, `A1`, `A1A2`, `A2`, `A2B1`, `B1`, `B1B2`, `B2`
(`A1A2` is A1+, and so on). Speak & Improve overall grades in
`asr_overall` are `A2`, `A2+`, `B1`, `B1+`, `B2`, `B2+`, `C1`, `C1+`.
The mapping is recorded in `sandi/asr_overall/distribution.json`: each
session overall score is the mean of the four part scores, rounded to the
nearest 0.5, with exact quarter-point ties rounded upward.

## Audio access

Both corpora require a separate application:

- TEEMI audio is not included. Paths currently point at
  `/share/corpus/2023_teemiv2/audio/...`.
- Speak & Improve audio is not included. Paths currently point at
  `/share/corpus/sandi2025-challenge/data/flac/...`. A path may end in
  `#t=start,end`; that suffix is a clip window into the FLAC file, not a
  separate download.

A path string does not grant access to the waveform. After the corpora are
approved, either mount them at those prefixes or rewrite the prefixes:

```bash
python scripts/remap_audio_paths.py \
  --old /share/corpus/2023_teemiv2 \
  --new /data/teemi \
  data/teemi/a01_only/*.csv

python scripts/remap_audio_paths.py \
  --old /share/corpus/sandi2025-challenge \
  --new /data/sandi2025-challenge \
  data/sandi/asr/*.csv data/sandi/asr_overall/*.csv

python scripts/check_audio_manifests.py
```

Official Speak & Improve SpWER also needs the challenge scoring tools
(`sclite`, STM, GLM) from the corpus release. Pass that directory as
`--corpus` or set `SANDI_CORPUS`.
