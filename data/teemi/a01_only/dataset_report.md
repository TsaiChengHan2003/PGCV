# TEEMI A01-only Dataset Report

## Scope

- Source/internal booklet: `A01`
- External test booklet: `A02`
- Split unit: global speaker id (`uXXX`)
- Split ratio on A01 speakers: 0.75/0.10/0.15
- A01 is used to materialize internal train/dev/test only.
- A02 is reserved as an external test set after removing any speaker that appears in the A01 split universe.
- Missing proficiency and `No answer` utterances are excluded before split materialization, consistent with the baseline policy.

## Leakage Control

- `train.csv`, `dev.csv`, and `test.csv` are derived only from A01 speakers.
- `test_a02_external.csv` excludes all rows whose `global_speaker_id` appears anywhere in the A01 split universe.
- This means the external A02 test is speaker-disjoint from every A01 internal split.

## Eligibility Filters

- Excluded for missing proficiency: 77
- Excluded for `No answer`: 15
- Excluded from A02 external test due to overlap with A01 speakers: 1710 utterances / 340 speakers

## Selected Booklets

### A01

- Utterances: 6206
- Global speakers: 1252
- Total audio: 27:46:28 (99988.06 sec)
- Mean / Median duration: 16.11 / 16.04 sec
- Mean words: 14.33

### A02

- Utterances: 3127
- Global speakers: 629
- Total audio: 13:58:13 (50292.90 sec)
- Mean / Median duration: 16.08 / 16.04 sec
- Mean words: 14.32

## Materialized Splits

### train (A01 internal)

- Utterances: 4626
- Global speakers: 929
- Total audio: 20:41:41 (74501.43 sec)
- Mean / Median duration: 16.10 / 16.04 sec
- Mean words: 14.54

### dev (A01 internal)

- Utterances: 609
- Global speakers: 123
- Total audio: 02:44:12 (9852.23 sec)
- Mean / Median duration: 16.18 / 16.04 sec
- Mean words: 14.10

### test (A01 internal)

- Utterances: 928
- Global speakers: 187
- Total audio: 04:09:21 (14960.94 sec)
- Mean / Median duration: 16.12 / 16.04 sec
- Mean words: 13.80

### test_a02_external

- Utterances: 1368
- Global speakers: 274
- Total audio: 06:07:27 (22047.02 sec)
- Mean / Median duration: 16.12 / 16.04 sec
- Mean words: 14.07

## A01 Split Universe

- Eligible A01 utterances before split: 6163
- Eligible A01 speakers before split: 1239
