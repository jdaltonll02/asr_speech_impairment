# Architecture — ASR Sprint Rwanda 2026

## Overview

The codebase is organised around three layers:

1. **`src/eval/`** — language-agnostic evaluation primitives reused across all RQs
2. **`src/rq1/`** — RQ1-specific pipeline stages (one file per stage)
3. **`src/utils/`** — infrastructure (config, data loading, audio, slurm)

Each RQ gets its own subpackage (`src/rq2/`, `src/rq3/`, …) that imports from `src/eval/` and `src/utils/` rather than duplicating code.

---

## Directory layout

```
asr-sprint/
├── config/
│   └── base.yaml               # Single source of truth for all paths, model IDs,
│                               # thresholds, and Slurm settings
├── docs/
│   ├── architecture.md         # This file
│   └── (future: rq2_design.md, …)
├── jobs/
│   ├── rq1_baseline.slurm      # GPU: baseline inference (KY + EN, dev + test)
│   ├── rq1_interventions.slurm # GPU: 6 intervention re-decoding strategies
│   └── rq1_analysis.slurm      # CPU: failure + regression + report (no GPU needed)
├── results -> /data/user_data/jgibson2/asr_sprint/results/
│                               # Symlink — all outputs live on fast storage
├── scripts/
│   ├── smoke_test.py           # Pre-submission validation (CPU, ~1 sec)
│   └── run_rq1_analysis.sh     # Shell wrapper for CPU analysis steps
├── src/
│   ├── eval/                   # Shared evaluation primitives
│   ├── rq1/                    # RQ1 pipeline
│   └── utils/                  # Infrastructure
└── tests/                      # (stubs; extended as pipeline grows)
```

---

## Data flow

```
HuggingFace (gated)
      │
      │  python -m src.rq1.baseline --download
      ▼
/data/.../data/
  ky_dev.tsv / ky_test.tsv       ← metadata + audio paths
  en_dev.tsv / en_test.tsv
  audio/ky/{dev,test}/*.wav      ← raw 16 kHz audio
  audio/en/{dev,test}/*.wav
  ky_speaker_metadata.csv        ← severity, etiology, gender, age
  en_speaker_metadata.csv
      │
      │  src/utils/data.load_split()  ← joins speaker metadata on speaker_id
      ▼
  DataFrame (per utterance)
      │
      │  sbatch jobs/rq1_baseline.slurm
      │  src/eval/harness.transcribe_dataset()
      ▼
results/rq1/
  baseline_{lang}_{split}.parquet    ← WER, CER, edit ops, log-probs, silence, SNR
  baseline_summary.csv
      │
      │  python -m src.rq1.failure
      │  src/rq1/failure.add_failure_diagnostics()
      │  src/rq1/failure.flag_catastrophic()
      ▼
results/rq1/failure/
  failures_{lang}_{split}.parquet    ← + length_ratio, repeated_4gram_rate,
  failure_summary.csv                  is_catastrophic
  catastrophic_utterances.csv
  wer_distribution.png
      │
      ├─ python -m src.rq1.regression
      │  src/rq1/regression.build_feature_matrix()
      │  src/rq1/regression.fit_regression()
      │  src/rq1/regression.compute_bootstrap_cis()
      ▼
  regression_results.json
  regression_coefs.png
      │
      ├─ sbatch jobs/rq1_interventions.slurm
      │  src/rq1/interventions.run_all_interventions()
      ▼
  interventions/
    interventions_{lang}_{split}.parquet
  intervention_summary.csv
      │
      │  python -m src.rq1.report
      ▼
  table_baseline_wer.csv
  table_catastrophic_share.csv
  fig_wer_distribution.png
  fig_logit_coef.png
  fig_interventions.png           ← after interventions job completes
  RQ1_results.md                  ← narrative results report
```

---

## Module contracts

### `src/eval/normalise.py`

| Function | Signature | Notes |
|----------|-----------|-------|
| `normalize(text, language)` | `str → str` | Apply KY elision **before** BasicTextNormalizer. Always call on both ref and hyp before any metric. |

### `src/eval/metrics.py`

| Function | Returns | Notes |
|----------|---------|-------|
| `word_errors(ref, hyp)` | `dict` | keys: substitutions, deletions, insertions, total_ref_words, total_hyp_words |
| `corpus_wer(refs, hyps)` | `float` | Pool all words across utterances |
| `per_utterance_wer(ref, hyp, cap=1.0)` | `float` | Capped at 1.0 per utterance |
| `uncapped_wer(ref, hyp)` | `float` | Can exceed 1.0; used for catastrophic flagging |
| `cer(ref, hyp)` | `float` | Character error rate |
| `repeated_ngram_rate(text, n=4)` | `float` | Fraction of n-grams that repeat |
| `build_row(...)` | `dict` | Convenience: assemble a full result row |
| `compute_edit_ops(ref, hyp)` | `list` | Raw Levenshtein operations |

### `src/eval/harness.py`

| Function | Notes |
|----------|-------|
| `load_model(model_id, cache_dir)` | fp16 on CUDA, fp32 on CPU |
| `decode_single(waveform, sr, model, processor, generate_kwargs)` | Returns `{hypothesis, compression_ratio, avg_log_prob}` |
| `transcribe_dataset(model, processor, dataset_df, language, decode_kwargs, batch_size, results_root, checkpoint_path, checkpoint_every, prior_rows)` | Checkpoints every N utterances; `prior_rows` for preemption resume |

**Preemption resume pattern:**
```python
# In baseline.py — before calling transcribe_dataset
if ckpt.exists():
    ckpt_df = pd.read_parquet(ckpt)
    prior_rows = ckpt_df.to_dict("records")
    already_done = set(ckpt_df["utt_id"])
    split_df = split_df[~split_df["utt_id"].isin(already_done)]

new_df = transcribe_dataset(..., prior_rows=prior_rows)
# new_df contains prior_rows prepended + new rows
```

### `src/utils/data.py`

| Function | Notes |
|----------|-------|
| `load_split(split, language, data_root)` | Joins `{language}_speaker_metadata.csv`; renames `severity_speech_impairment → severity` |
| `validate_data_root(data_root)` | Raises `DataNotReadyError` if splits missing |
| `download_dataset(language, data_root, cfg)` | Uses `Audio(decode=False)` to skip torchcodec; writes `.wav` files |

**Important**: `speaker_id` column in source data has a trailing space for RW004 (`"RW004 "`). `load_split` strips it. Always strip before joining.

### `src/rq1/failure.py`

| Function | Notes |
|----------|-------|
| `add_failure_diagnostics(df, config)` | Adds length_ratio, compression_ratio (filled), repeated_4gram_rate |
| `set_repeated_4gram_threshold_on_dev(dev_df)` | Maximises Youden's J on dev; returns float threshold |
| `flag_catastrophic(df, threshold_4gram, config)` | WER>1.0 OR length_ratio>2.0 OR ngram_rate>threshold |
| `compute_summary(df)` | Returns dict with share_utterances, share_word_errors |
| `compute_failure_features(df)` | Alias for use in tests/notebooks (no config arg) |
| `apply_catastrophic_flag(df, ...)` | Alias with explicit threshold args |

### `src/rq1/regression.py`

Features used: `audio_length_sec`, `leading_silence_s`, `trailing_silence_s`, `snr_db`, `prompt_type` (dummy), `severity` (dummy, joined at runtime from speaker_metadata), `speaker_id` (dummy fixed effects).

Note: `recording_environment` had zero variance in this dataset (all indoor low-noise) — produces no dummy columns after `drop_first=True`.

Train/test alignment: combined DataFrame is built then split by `_fold` column to ensure identical dummy columns across dev (train) and test.

### `src/rq1/interventions.py`

Six strategies, each a standalone function `(model, processor, row_dict, language, config) → str`:

| ID | Name | Mechanism |
|----|------|-----------|
| A | `repetition_penalty` | `no_repeat_ngram_size=4` |
| B | `max_length_from_audio` | `max_new_tokens = audio_len_sec × multiplier` |
| C | `temperature_fallback` | `compression_ratio_threshold` in generate() |
| D | `beam_search` | `num_beams=5` |
| E | `vad_trim` | silero-VAD silence removal before decode |
| F | `ctc_fallback` | Placeholder — returns Whisper output unchanged until a KY CTC model is configured |

Run on **full split** (`--all-utterances`) to measure corpus and per-utterance WER change, not just on catastrophic utterances.

---

## Config (`config/base.yaml`) key sections

```yaml
data_root:     /data/user_data/jgibson2/asr_sprint/data
models_root:   /data/user_data/jgibson2/asr_sprint/models
results_root:  /data/user_data/jgibson2/asr_sprint/results
cache_dir:     /data/user_data/jgibson2/asr_sprint/hf_cache

models:
  ky:
    model_id: cdli/whisper-large-v3_finetuned_rwandan_kinyarwanda_nonstandard_speech_v1.0
    language_token: sw      # KY uses Swahili token — do not change
    task: transcribe
    ctc_fallback_id: TODO   # update when KY CTC model available
  en:
    model_id: cdli/whisper-large-v3_finetuned_rwandan_english_nonstandard_speech_v1.0
    language_token: en
    task: transcribe

datasets:
  ky:
    hf_id: cdli/rwandan_kinyarwanda_nonstandard_speech_v1.0
    audio_column: audio
    reference_column: transcription
  en:
    hf_id: cdli/rwandan_english_nonstandard_speech_v1.0
    ...

catastrophic:
  uncapped_wer_threshold: 1.0
  length_ratio_threshold: 2.0
  repeated_4gram_threshold: 0.1   # overridden at runtime by Youden's J on dev

decode:
  beam_size: 1               # greedy baseline
  checkpoint_every: 50

slurm:
  partition: preempt
  qos: preempt_qos
```

---

## Adding a new RQ

1. Create `src/rqN/` with `__init__.py` and one file per pipeline stage
2. Import from `src/eval/` and `src/utils/` — do not copy-paste
3. Add a `jobs/rqN_*.slurm` for any GPU stage, a `scripts/run_rqN_*.sh` for CPU stages
4. Add RQ-specific config keys under a `rqN:` block in `base.yaml`
5. Add smoke test cases to `scripts/smoke_test.py`
6. Add a results section to `results/rqN/RQN_results.md` after running

---

## Known issues / TODOs

| Item | File | Notes |
|------|------|-------|
| CTC fallback placeholder | `interventions.py`, `base.yaml` | Update `ctc_fallback_id` when KY CTC model available (RQ2 candidate) |
| `speaker_id` dummies zero out on test | `regression.py` | Expected — dev and test speakers differ; speaker effects are only interpretable on dev |
| `avg_log_prob` from `sequences_scores` | `harness.py` | Only available with `return_dict_in_generate=True`; `None` if model doesn't return scores |
| silero-vad version | `audio.py` | API changed between v3 and v4; pinned to `>=4.0.0` in requirements.txt |
