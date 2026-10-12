# RQ1 Implementation Plan — CDLI Rwandan NSS ASR Failure Diagnosis

## Model IDs (resolved from HuggingFace API)

| Alias | HuggingFace slug | Language token |
|-------|-----------------|----------------|
| KY model | `cdli/whisper-large-v3_finetuned_rwandan_kinyarwanda_nonstandard_speech_v1.0` | `sw` (published setting — Swahili token used as proxy for Kinyarwanda) |
| EN model | `cdli/whisper-large-v3_finetuned_rwandan_english_nonstandard_speech_v1.0` | `en` |
| KY dataset | `cdli/rwandan_kinyarwanda_nonstandard_speech_v1.0` | splits: train/validation/test |
| EN dataset | `cdli/rwandan_english_nonstandard_speech_v1.0` | splits: train/validation/test |

Both datasets are **gated** (manual access on HuggingFace). A valid HF token with accepted terms must be configured in `~/.cache/huggingface/token` or via `HF_TOKEN` env var before running any job.

---

## Design decisions

### 1. Central config — `config/base.yaml` + `src/utils/config.py`

Use a single YAML file as the source of truth for all paths, model IDs, hyperparameters, and thresholds. `config.py` loads it once with `pydantic-settings` (or plain `pydantic.BaseModel` + `PyYAML`) and exposes a frozen `Cfg` dataclass so every module imports `from src.utils.config import load_config` and calls `cfg = load_config()`. Rationale: one place to change a path or threshold; typed fields catch mistakes early.

### 2. Data directory root

`/data/user_data/jgibson2/asr_sprint/` is the **data root** for audio, model caches, checkpoints, parquet outputs, and result artefacts (plots + metric JSON/CSV). This path is stored in `base.yaml` as `data_root` and never hard-coded in Python.

### 3. Per-utterance DataFrame schema

All intermediate results are stored as Parquet files under `{data_root}/results/rq1/`. The canonical per-utterance schema (enforced via a `pandas` dtype dict in `metrics.py`) is:

```
utt_id          str       # unique utterance identifier
split           str       # "dev" or "test"
model           str       # "KY" or "EN"
audio_path      str       # absolute path
duration_s      float64   # audio duration in seconds
reference       str       # normalised reference text
hypothesis      str       # normalised hypothesis text
n_ref_words     int32     # reference word count
n_hyp_words     int32     # hypothesis word count
substitutions   int32
deletions       int32
insertions      int32
wer_uncapped    float64   # (S+D+I) / max(n_ref_words, 1)
length_ratio    float64   # n_hyp_words / max(n_ref_words, 1)
compression_ratio float64 # Whisper compression ratio (from generate metadata)
avg_log_prob    float64   # mean token log-prob (from generate metadata)
repeated_4gram_rate float64 # fraction of 4-grams that are repeats
is_catastrophic bool      # derived flag (set by failure.py)
prompt_type     str       # e.g. "no_prompt", "text_prompt"
recording_env   str       # metadata field from dataset (e.g. "studio", "phone")
speaker_id      str
leading_silence_s  float64
trailing_silence_s float64
snr_db          float64   # estimated signal-to-noise ratio
```

### 4. Whisper generate() → metadata extraction

Use `transformers >= 4.36`. Call `model.generate()` with:
```python
outputs = model.generate(
    input_features,
    return_dict_in_generate=True,
    output_scores=True,
    language=cfg.model.language_token,
    task="transcribe",
    do_sample=False,          # greedy
)
```
The returned `GenerateBeamDecoderOnlyOutput` / `GenerateDecoderOnlyOutput` exposes:
- `outputs.sequences_scores` — overall sequence log-prob (scalar per beam); divide by token count for `avg_log_prob`.
- Compression ratio is obtained from `WhisperForConditionalGeneration`'s processor **before** generate via `feature_extractor.compute_compression_ratio(audio_array)` — or extracted post-hoc from `outputs.encoder_outputs.last_hidden_state` shape vs. input length. Simpler: use the `WhisperProcessor`'s `feature_extractor` to compute `compression_ratio = len(audio_array) / (feature_extractor.chunk_length * feature_extractor.sampling_rate)` as a proxy, or use `model.generation_config.compression_ratio_threshold` logic.

**Authoritative approach chosen:** extract `compression_ratio` from the raw audio before the model call (ratio of uncompressed audio length to the Whisper mel-frame budget = `n_samples / (3000 * hop_length)`). Extract `avg_log_prob` from `outputs.sequences_scores[0] / (len(outputs.sequences[0]) - 1)` (subtract the initial BOS token). Both are stored in the baseline Parquet by `harness.py`.

### 5. Interface: harness.py → failure.py

`harness.py` is the only place that calls the model. It writes a Parquet with all columns in §3 (including `compression_ratio`, `avg_log_prob`). `failure.py` reads that Parquet — it never loads the model. This strict read-only contract means `failure.py` can run on the login node.

### 6. Interface: harness.py → interventions.py

`interventions.py` imports `harness.decode_single` (or `harness.decode_batch`) and passes an `override: dict` that merges into the `generate()` call kwargs. Each intervention is a named function that returns an override dict. Rationale: avoids duplicating the data-loading and model-loading boilerplate; interventions differ only in `generate()` parameters or pre/post-processing.

### 7. Text normalisation

`normalise.py` provides a single `normalise(text: str, lang: str) -> str` function. For English: lowercase, expand contractions, strip punctuation (except apostrophes in words), remove extra whitespace. For Kinyarwanda: lowercase, strip punctuation, remove diacritics only if not linguistically contrastive (no known contrastive diacritics in standard orthography — strip all). Normalisation is applied identically to reference and hypothesis before any metric computation.

### 8. Catastrophic failure threshold

The repeated 4-gram threshold is set **on the dev split** by choosing the 95th percentile of `repeated_4gram_rate` among non-catastrophic utterances (those where WER ≤ 1.0 AND length_ratio ≤ 2.0). This threshold is written to `{data_root}/results/rq1/thresholds.json` by `failure.py` and reused for the test split. Definition: `is_catastrophic = wer_uncapped > 1.0 OR length_ratio > 2.0 OR repeated_4gram_rate > threshold`.

### 9. Slurm partition

All `.slurm` files use `#SBATCH --partition=preempt`. Jobs must handle preemption by checkpointing the Parquet incrementally (write after every batch, not only at the end).

### 10. Results location

All plots (PNG) and metric tables (CSV/JSON) go to `{data_root}/results/rq1/`. Sub-directories: `baseline/`, `failure/`, `interventions/`, `report/`.

---

## Ambiguities resolved

| Ambiguity | Resolution |
|-----------|------------|
| `language=sw` for KY | Published decoding setting — Swahili token used as acoustic language hint; stored in `base.yaml` under `models.ky.language_token: "sw"` |
| `compression_ratio` source | Computed from raw audio length vs. Whisper mel frame budget before `generate()`, stored in baseline Parquet by `harness.py` |
| `avg_log_prob` source | `outputs.sequences_scores[0] / (output_len - 1)` from `return_dict_in_generate=True`; stored by `harness.py` |
| 4-gram threshold | Data-driven: 95th percentile of `repeated_4gram_rate` on dev non-catastrophic set; written to `thresholds.json` |
| Repeated n-gram metric | Count distinct 4-grams that appear >1 time / total 4-grams in hypothesis |
| interventions.py ↔ harness.py | `interventions.py` calls `harness.decode_single(audio, model, processor, **override)` — one shared decode function, different override kwargs per intervention |
| CTC fallback | Uses `facebook/wav2vec2-large-xlsr-53` (for EN) or a Kinyarwanda CTC model as fallback; model IDs stored in `base.yaml` as `models.ky.ctc_fallback_id` and `models.en.ctc_fallback_id` |
| VAD trimming | Uses `silero-vad` (torch hub) for silence detection; `audio.py` exposes `trim_silence(waveform, sr) -> waveform` |
| SNR estimation | Computed with `wada_snr` algorithm from `audiomentations` library; stored in baseline Parquet |
| Dataset access | Both CDLI Rwanda datasets are gated; HF token needed; documented in README |
| Data split naming | HuggingFace datasets use `validation` for dev; `base.yaml` maps `dev -> validation`, `test -> test` |

---

## Implementation plan

- [ ] 1. **Bootstrap repository structure and config skeleton**

  Create every file and directory listed in the spec. Files are empty stubs (just a module docstring and any essential imports) except `config/base.yaml` and `requirements.txt` which must be functional from the start.

  `config/base.yaml` must contain:
  ```yaml
  data_root: /data/user_data/jgibson2/asr_sprint
  results_root: /data/user_data/jgibson2/asr_sprint/results/rq1
  cache_dir: /data/user_data/jgibson2/asr_sprint/model_cache

  models:
    ky:
      model_id: cdli/whisper-large-v3_finetuned_rwandan_kinyarwanda_nonstandard_speech_v1.0
      language_token: sw
      task: transcribe
      ctc_fallback_id: facebook/wav2vec2-large-xlsr-53  # placeholder; replace with KY CTC model if available
    en:
      model_id: cdli/whisper-large-v3_finetuned_rwandan_english_nonstandard_speech_v1.0
      language_token: en
      task: transcribe
      ctc_fallback_id: facebook/wav2vec2-large-960h

  datasets:
    ky:
      hf_id: cdli/rwandan_kinyarwanda_nonstandard_speech_v1.0
      audio_column: audio
      reference_column: sentence
      split_map: {dev: validation, test: test}
    en:
      hf_id: cdli/rwandan_english_nonstandard_speech_v1.0
      audio_column: audio
      reference_column: sentence
      split_map: {dev: validation, test: test}

  decode:
    batch_size: 8
    greedy: true   # do_sample=False; num_beams=1
    return_dict_in_generate: true
    output_scores: true

  failure:
    wer_threshold: 1.0
    length_ratio_threshold: 2.0
    repeated_4gram_percentile: 95   # threshold set on dev

  interventions:
    repetition_penalty: 1.3
    max_length_multiplier: 2.0      # max_new_tokens = audio_duration_s * tokens_per_second * multiplier
    tokens_per_second: 5            # approximate for Whisper
    temperature_fallback:
      temperature: 0.2
      compression_ratio_threshold: 2.4   # Whisper default
    beam_size: 5
    vad_aggressiveness: 2           # silero-vad threshold
    ctc_flag_wer_threshold: 1.0

  slurm:
    partition: preempt
    nodes: 1
    gpus_per_node: 1
    cpus_per_task: 4
    mem_gb: 32
    time_baseline: "04:00:00"
    time_interventions: "06:00:00"
    time_analysis: "02:00:00"
  ```

  `requirements.txt` must pin:
  ```
  torch>=2.1.0
  transformers>=4.36.0
  datasets>=2.16.0
  accelerate>=0.25.0
  soundfile>=0.12.1
  librosa>=0.10.1
  pydantic>=2.5.0
  pyyaml>=6.0.1
  pandas>=2.1.0
  pyarrow>=14.0.0
  scikit-learn>=1.3.0
  statsmodels>=0.14.0
  matplotlib>=3.8.0
  seaborn>=0.13.0
  jiwer>=3.0.3
  audiomentations>=0.32.0
  silero-vad>=4.0.0
  ```

  Files to create:
  - `config/base.yaml`
  - `requirements.txt`
  - `src/__init__.py` (empty)
  - `src/eval/__init__.py`, `src/eval/normalise.py`, `src/eval/metrics.py`, `src/eval/harness.py`
  - `src/rq1/__init__.py`, `src/rq1/baseline.py`, `src/rq1/failure.py`, `src/rq1/regression.py`, `src/rq1/interventions.py`, `src/rq1/report.py`
  - `src/utils/__init__.py`, `src/utils/audio.py`, `src/utils/data.py`, `src/utils/slurm.py`, `src/utils/config.py`
  - `jobs/rq1_baseline.slurm`, `jobs/rq1_interventions.slurm`, `jobs/rq1_analysis.slurm`

  Verify: `python -c "import src; from src.utils.config import load_config; cfg = load_config(); print(cfg.data_root)"` prints the data root path without errors.

---

- [ ] 2. **`src/utils/config.py` — typed config loader**

  Implement `load_config(path: str | None = None) -> Cfg` where `Cfg` is a `pydantic.BaseModel` tree that mirrors `base.yaml`. Use `pydantic` v2 with `model_config = ConfigDict(frozen=True)`. If `path` is `None`, look for `CONFIG_PATH` env var, then fall back to `<repo_root>/config/base.yaml` (found via `pathlib.Path(__file__).parents[2] / "config" / "base.yaml"`). Cache the loaded config in a module-level `_cfg` variable so repeated calls are free.

  Key sub-models: `ModelCfg`, `DatasetCfg`, `DecodeCfg`, `FailureCfg`, `InterventionsCfg`, `SlurmCfg`.

  Files: `src/utils/config.py`

  Verify: `python -m pytest src/utils/test_config.py -v` (write a minimal test inline — 3 assertions: `cfg.data_root` is a string, `cfg.models.ky.language_token == "sw"`, `cfg.failure.wer_threshold == 1.0`). Place test at `tests/utils/test_config.py`.

---

- [ ] 3. **`src/eval/normalise.py` — text normalisation**

  Implement `normalise(text: str, lang: str) -> str`:
  - `lang="en"`: lowercase → expand contractions (use `contractions` library or a hand-rolled dict for the 20 most common) → strip all punctuation except internal apostrophes → collapse whitespace.
  - `lang="rw"` (Kinyarwanda): lowercase → strip all punctuation → collapse whitespace. No diacritic stripping needed for standard Kinyarwanda.
  - Both: strip leading/trailing whitespace; return empty string for None/empty input.

  Files: `src/eval/normalise.py`, `tests/eval/test_normalise.py`

  Verify: `python -m pytest tests/eval/test_normalise.py -v` — cover empty input, contractions for EN, punctuation-only string, multi-space collapse.

---

- [ ] 4. **`src/eval/metrics.py` — per-utterance metric computation**

  Implement:
  - `compute_edit_ops(ref: str, hyp: str) -> dict` — returns `{substitutions, deletions, insertions, n_ref_words, n_hyp_words}` using `jiwer.process_words`.
  - `wer_uncapped(ref: str, hyp: str) -> float` — `(S+D+I) / max(n_ref, 1)`.
  - `length_ratio(ref: str, hyp: str) -> float` — `n_hyp / max(n_ref, 1)`.
  - `repeated_ngram_rate(text: str, n: int = 4) -> float` — tokenise on whitespace; count 4-grams that appear more than once; return `(count of repeated 4-gram tokens) / max(total 4-grams, 1)`.
  - `SCHEMA: dict[str, str]` — the pandas dtype mapping for the canonical per-utterance DataFrame (matches the schema in §3 of this plan).
  - `build_row(utt_id, split, model_name, audio_path, duration_s, ref, hyp, compression_ratio, avg_log_prob, prompt_type, recording_env, speaker_id, leading_silence_s, trailing_silence_s, snr_db) -> dict` — calls all the above, returns a flat dict matching SCHEMA.

  Files: `src/eval/metrics.py`, `tests/eval/test_metrics.py`

  Verify: `python -m pytest tests/eval/test_metrics.py -v` — cover zero-length reference, perfect match, all-insertions, repeated 4-gram detection.

---

- [ ] 5. **`src/utils/audio.py` — audio utilities**

  Implement:
  - `load_audio(path: str, target_sr: int = 16000) -> tuple[np.ndarray, int]` — load via `soundfile` or `librosa`, resample to `target_sr`, return mono float32.
  - `measure_silence(waveform: np.ndarray, sr: int, frame_ms: int = 25) -> tuple[float, float]` — use energy-based VAD (threshold at -40 dBFS) to find leading/trailing silence duration in seconds.
  - `estimate_snr(waveform: np.ndarray, sr: int) -> float` — use WADA-SNR via `wada_snr` from `audiomentations` utilities, or implement a simple power-ratio estimator if unavailable.
  - `trim_silence(waveform: np.ndarray, sr: int, aggressiveness: float = 0.5) -> np.ndarray` — for intervention (e), use silero-vad to trim leading/trailing silence.
  - `compute_compression_ratio(waveform: np.ndarray, sr: int, hop_length: int = 160, n_frames: int = 3000) -> float` — returns `len(waveform) / (n_frames * hop_length)`.

  Files: `src/utils/audio.py`, `tests/utils/test_audio.py`

  Verify: `python -m pytest tests/utils/test_audio.py -v` — use a synthetic sine-wave fixture; check load+resample round-trips and silence measurements are within 50 ms of ground truth.

---

- [ ] 6. **`src/utils/data.py` — HuggingFace dataset loader**

  Implement:
  - `load_split(hf_id: str, split: str, cache_dir: str, audio_column: str = "audio", text_column: str = "sentence") -> datasets.Dataset` — calls `datasets.load_dataset(hf_id, split=split, cache_dir=cache_dir)`. Raises `RuntimeError` with clear message if the dataset is gated and no HF token is found.
  - `iter_batches(dataset: datasets.Dataset, batch_size: int) -> Iterator[list[dict]]` — yields lists of row dicts; each dict has keys `utt_id`, `audio_array`, `sampling_rate`, `reference`, `speaker_id`, `recording_env`, and any available metadata.
  - `utt_id_from_row(row: dict, model_name: str) -> str` — constructs a stable unique ID from model + dataset index or native ID field.

  Files: `src/utils/data.py`, `tests/utils/test_data.py`

  Verify: `python -m pytest tests/utils/test_data.py -v` — mock the `datasets.load_dataset` call; test batch iteration and ID construction without network access.

---

- [ ] 7. **`src/eval/harness.py` — decode-and-measure core**

  This is the central module. Implement:

  `decode_single(waveform: np.ndarray, sr: int, model, processor, generate_kwargs: dict) -> dict`:
  - Resample to 16 kHz if needed.
  - Run `processor.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt")`.
  - Compute `compression_ratio` via `audio.compute_compression_ratio(waveform, sr)` **before** `generate()`.
  - Call `model.generate(input_features, return_dict_in_generate=True, output_scores=True, **generate_kwargs)`.
  - Extract `avg_log_prob = outputs.sequences_scores[0].item() / max(outputs.sequences[0].shape[-1] - 1, 1)`.
  - Decode tokens: `processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]`.
  - Return `{"hypothesis": str, "compression_ratio": float, "avg_log_prob": float}`.

  `run_baseline(model_alias: str, split: str, cfg: Cfg, output_parquet: str, checkpoint_every: int = 100) -> pd.DataFrame`:
  - Loads model + processor from `cfg.models[model_alias].model_id` to GPU if available.
  - Iterates dataset batches; for each utterance calls `decode_single`, then `audio.measure_silence`, `audio.estimate_snr`, `metrics.build_row`.
  - Appends rows incrementally to a Parquet (uses `fastparquet` append mode or writes a new Parquet every `checkpoint_every` rows and merges on completion) to survive preemption.
  - Returns final DataFrame.

  Files: `src/eval/harness.py`

  Verify: write `tests/eval/test_harness.py` that mocks `model.generate()` to return a fixed `GenerateDecoderOnlyOutput`, asserts the returned dict has the right keys and value types, and checks the checkpoint-resume logic with a temp directory.
  Run: `python -m pytest tests/eval/test_harness.py -v`.

---

- [ ] 8. **`src/rq1/baseline.py` — baseline orchestrator**

  Implement `main(cfg: Cfg)`:
  - For each `(model_alias, split)` in `[("ky","dev"), ("ky","test"), ("en","dev"), ("en","test")]`:
    - Determine output path `{cfg.results_root}/baseline/{model_alias}_{split}.parquet`.
    - Skip if already exists and `--resume` flag is set.
    - Call `harness.run_baseline(model_alias, split, cfg, output_path)`.
    - Log summary WER to stdout.
  - When called as `__main__`, parse args: `--config`, `--model` (default: both), `--split` (default: both), `--resume`.

  Files: `src/rq1/baseline.py`

  Verify: `python -m pytest tests/rq1/test_baseline.py -v` (mock `harness.run_baseline`; test argument parsing and skip-if-exists logic).

---

- [ ] 9. **`src/rq1/failure.py` — catastrophic failure analysis**

  Implement:

  `set_threshold(dev_parquet: str, cfg: Cfg) -> float`:
  - Load dev Parquet.
  - Filter to non-catastrophic by WER+length criteria.
  - Take `np.percentile(df["repeated_4gram_rate"], cfg.failure.repeated_4gram_percentile)`.
  - Write `{cfg.results_root}/failure/thresholds.json` with all three threshold values.
  - Return the 4-gram threshold.

  `flag_catastrophic(df: pd.DataFrame, threshold_4gram: float, cfg: Cfg) -> pd.DataFrame`:
  - Adds/overwrites `is_catastrophic` column per §8 of design decisions.

  `compute_summary(df: pd.DataFrame) -> dict`:
  - Returns `{share_utterances, share_word_errors}` for catastrophic subset.

  `main(cfg: Cfg)`:
  - Load all four baseline Parquets.
  - Call `set_threshold` on dev data.
  - Flag all splits.
  - Compute summaries.
  - Write flagged Parquets to `{cfg.results_root}/failure/`.
  - Print summary table.

  Files: `src/rq1/failure.py`, `tests/rq1/test_failure.py`

  Verify: `python -m pytest tests/rq1/test_failure.py -v` — synthetic DataFrame with known catastrophic rows; assert threshold is computed correctly and flagging logic matches §3.

---

- [ ] 10. **`src/rq1/regression.py` — logistic regression on catastrophic flag**

  Implement `run_regression(df: pd.DataFrame, feature_cols: list[str], cfg: Cfg) -> dict`:
  - Feature columns: `["prompt_type_enc", "duration_s", "repeated_4gram_rate", "speaker_id_enc", "recording_env_enc", "leading_silence_s", "trailing_silence_s", "snr_db", "compression_ratio", "avg_log_prob"]`.
  - Encode categoricals with `sklearn.preprocessing.LabelEncoder`.
  - Standardise numerics with `StandardScaler`.
  - Fit `sklearn.linear_model.LogisticRegression(max_iter=500, class_weight="balanced")`.
  - Return dict with `coef`, `feature_names`, `auc_roc`, `report` (from `classification_report`).
  - Save coefficients plot to `{cfg.results_root}/failure/logit_coef.png`.

  `stratified_sample(df: pd.DataFrame, n: int = 50, random_seed: int = 42) -> pd.DataFrame`:
  - Stratify on `(model, recording_env, prompt_type)`.
  - Return sample for manual listening (R4).

  Files: `src/rq1/regression.py`, `tests/rq1/test_regression.py`

  Verify: `python -m pytest tests/rq1/test_regression.py -v` — fit on a 200-row synthetic DataFrame; assert AUC > 0.5 and coefficients dict has expected keys.

---

- [ ] 11. **`src/rq1/interventions.py` — re-decode with all six interventions**

  Each intervention is a function `intervention_X(cfg: Cfg) -> dict` returning a `generate_kwargs` override dict plus optional pre/post-processing hooks. The six interventions:

  | ID | Name | Override |
  |----|------|----------|
  | A | repetition_penalty | `{"repetition_penalty": cfg.interventions.repetition_penalty}` |
  | B | max_length | `{"max_new_tokens": int(duration_s * cfg.interventions.tokens_per_second * cfg.interventions.max_length_multiplier)}` (per-utterance) |
  | C | temperature_fallback | decode greedy first; if `compression_ratio > cfg.interventions.temperature_fallback.compression_ratio_threshold`, re-decode with `{"do_sample": True, "temperature": cfg.interventions.temperature_fallback.temperature}` |
  | D | beam_search | `{"num_beams": cfg.interventions.beam_size, "do_sample": False}` |
  | E | vad_trim | pre-process: `audio.trim_silence(waveform, sr)` before `decode_single` |
  | F | ctc_fallback | post-process: if `is_catastrophic`, replace hypothesis with output of `ctc_model.transcribe(waveform)` using `cfg.models[alias].ctc_fallback_id` |

  `run_all_interventions(model_alias: str, split: str, flagged_parquet: str, cfg: Cfg) -> pd.DataFrame`:
  - Loads only the rows where `is_catastrophic == True` for efficiency.
  - Runs each intervention independently (not stacked).
  - Returns a DataFrame with columns: `utt_id`, `intervention`, `wer_before`, `wer_after`, `delta_wer`.
  - Writes to `{cfg.results_root}/interventions/{model_alias}_{split}.parquet`.

  Files: `src/rq1/interventions.py`, `tests/rq1/test_interventions.py`

  Verify: `python -m pytest tests/rq1/test_interventions.py -v` — mock `harness.decode_single`; check each override dict is constructed correctly; assert delta_wer column is present.

---

- [ ] 12. **`src/rq1/report.py` — result aggregation and plots**

  Implement `generate_report(cfg: Cfg)`:
  - Load all Parquets from `baseline/`, `failure/`, `interventions/`.
  - Produce (and save to `{cfg.results_root}/report/`):
    1. `table_baseline_wer.csv` — WER by model × split.
    2. `fig_wer_distribution.png` — histogram of `wer_uncapped` with catastrophic threshold marked.
    3. `fig_catastrophic_share.png` — bar chart: % utterances, % word errors, by model.
    4. `fig_logit_coef.png` — already written by `regression.py`; copy/symlink here.
    5. `table_interventions.csv` — mean ∆WER per intervention per model.
    6. `fig_interventions.png` — grouped bar chart of ∆WER per intervention.
  - Print a plain-text summary to stdout.

  Files: `src/rq1/report.py`

  Verify: `python -m pytest tests/rq1/test_report.py -v` — mock all Parquet loads; assert output file paths are created in a temp directory.

---

- [ ] 13. **`src/utils/slurm.py` — Slurm job helpers**

  Implement:
  - `render_slurm_header(cfg: Cfg, job_name: str, time: str) -> str` — renders the `#SBATCH` block from the slurm section of `cfg`.
  - `write_job_script(path: str, header: str, body: str)` — writes a `.slurm` file.

  Files: `src/utils/slurm.py` (no test required; purely a template helper).

---

- [ ] 14. **Slurm job scripts**

  Write three complete, runnable `.slurm` files. Each must:
  - Use `#SBATCH --partition=preempt`.
  - Activate the correct conda/venv environment (use placeholder `source /path/to/venv/activate` — user must customise).
  - Export `HF_TOKEN`, `HF_HOME={data_root}/hf_cache`, `TRANSFORMERS_CACHE={data_root}/model_cache`.
  - Call the appropriate Python module entry point.
  - Handle preemption with `--signal=B:USR1@60` and a Python `signal.SIGUSR1` handler in the called script that flushes the in-progress Parquet checkpoint.

  **`jobs/rq1_baseline.slurm`**: runs `python -m src.rq1.baseline --resume` for both models, both splits.
  **`jobs/rq1_interventions.slurm`**: runs `python -m src.rq1.interventions --resume`.
  **`jobs/rq1_analysis.slurm`**: runs `python -m src.rq1.failure`, then `python -m src.rq1.regression`, then `python -m src.rq1.report` in sequence (CPU-only; lower memory/time).

  Files: `jobs/rq1_baseline.slurm`, `jobs/rq1_interventions.slurm`, `jobs/rq1_analysis.slurm`

  Verify: `bash -n jobs/rq1_baseline.slurm && bash -n jobs/rq1_interventions.slurm && bash -n jobs/rq1_analysis.slurm` — syntax check passes.

---

- [ ] 15. **Wire up `__main__` entry points and smoke-test**

  Add `if __name__ == "__main__": main(load_config())` to `baseline.py`, `failure.py`, `regression.py`, `interventions.py`, `report.py`. Run the full test suite:

  `python -m pytest tests/ -v --tb=short`

  All tests pass (mocked, no network or GPU required). Fix any import errors that surfaced during integration.

  Files: minor edits to all `src/rq1/*.py` files.
  Verify: `python -m pytest tests/ -v --tb=short` exits 0.

---

## Dependency order

```
Step 1 (structure) → Step 2 (config) → Step 3 (normalise) → Step 4 (metrics)
                                      → Step 5 (audio)
                                      → Step 6 (data)
Steps 3+4+5+6 → Step 7 (harness) → Step 8 (baseline) → Step 9 (failure)
                                                       → Step 11 (interventions)
Step 9 → Step 10 (regression) → Step 12 (report)
Step 11 → Step 12 (report)
Step 2 → Step 13 (slurm utils) → Step 14 (slurm scripts)
All → Step 15 (wire-up + smoke-test)
```

## Notes for the implementer

1. **HF gated datasets**: Both `cdli/rwandan_*` datasets require accepting CDLI's terms at `https://huggingface.co/datasets/cdli/rwandan_kinyarwanda_nonstandard_speech_v1.0` and `https://huggingface.co/datasets/cdli/rwandan_english_nonstandard_speech_v1.0`. Set `HF_TOKEN` before running.

2. **`language=sw` for Kinyarwanda**: This is the published decoding setting from CDLI. Whisper does not natively support Kinyarwanda (`rw`), so `sw` (Swahili) is used as the closest supported language token. Do not change this to `rw` — it will cause the model to fall back to English.

3. **GPU memory**: `whisper-large-v3` requires ~10 GB VRAM. With `batch_size=8` and 16-bit inference expect ~14 GB. Use `model.half()` (float16) and consider `batch_size=4` if OOM.

4. **Incremental Parquet**: Use `pyarrow.parquet.write_to_dataset` with `append=True` or maintain a running list and write a new file every N rows, keeping track of the last written index in a sidecar `.json` checkpoint file. On resume, load the last row's `utt_id` and skip the dataset iterator forward.

5. **CTC fallback model for KY**: No CDLI Kinyarwanda CTC model is currently published. The placeholder `facebook/wav2vec2-large-xlsr-53` will decode to whatever language it was fine-tuned for (multilingual). If a better KY CTC model becomes available, update `models.ky.ctc_fallback_id` in `base.yaml`.
