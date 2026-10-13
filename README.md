# ASR Sprint Rwanda 2026 — Research & Modelling Tracks

Diagnosing and closing the reliability gap in ASR for Kinyarwanda and Rwandan English non-standard speech.

**Team:** 5 members · **Track:** Research + ASR Modelling (integrated)  
**Sprint window:** 5 Oct – 28 Nov 2026 · **Demo Day:** 28 Nov 2026  
**Data:** CDLI Rwandan non-standard speech datasets v1.0 (gated, HuggingFace)

---

## Quick start

```bash
# 1. Activate the venv
source /data/user_data/jgibson2/asr_sprint/venv/bin/activate

# 2. Smoke test before any job submission
python scripts/smoke_test.py

# 3. Submit a GPU job
sbatch jobs/rq1_baseline.slurm
```

---

## Research questions

| RQ | Question | Priority | Status |
|----|----------|----------|--------|
| RQ1 | What explains the corpus vs per-utterance WER gap in KY, and how much do catastrophic outputs contribute? | Core | 🔄 In progress |
| RQ2 | Do native-KY base models (Omnilingual CTC, w2v-BERT) beat Whisper for KY NSS? | Core | ⏳ Pending |
| RQ3 | Which adaptation strategies (LoRA, full FT, augmentation) give best gains under small data? | Core | ⏳ Pending |
| RQ4 | How well does WER reflect meaning preservation for Kinyarwanda? | Core | ⏳ Pending |
| RQ5 | How many utterances are needed for personalised adaptation? | Stretch | ⏳ Pending |
| RQ6 | What do SLPs and users consider acceptable accuracy and failure behaviour? | Core (qualitative) | ⏳ Pending |

---

## Project structure

```
asr-sprint/
├── config/base.yaml              # All paths, model IDs, thresholds, slurm settings
├── docs/
│   └── architecture.md           # Module contracts, data flow, design decisions
├── jobs/
│   ├── rq1_baseline.slurm        # GPU: CDLI baseline inference
│   ├── rq1_interventions.slurm   # GPU: 6 decoding intervention strategies
│   └── rq1_analysis.slurm        # CPU: failure + regression + report
├── results -> /data/.../results/ # Symlink to fast storage
├── scripts/
│   ├── smoke_test.py             # Pre-submission validation (~1 sec, no GPU)
│   └── run_rq1_analysis.sh       # CPU pipeline wrapper
├── src/
│   ├── eval/                     # Shared: normalise, metrics, harness
│   ├── rq1/                      # RQ1: baseline, failure, regression, interventions, report
│   └── utils/                    # config, data, audio, slurm
└── tests/
```

**Fast storage paths:**

| Path | Contents |
|------|----------|
| `/data/user_data/jgibson2/asr_sprint/data/` | TSV splits, audio files, speaker metadata |
| `/data/user_data/jgibson2/asr_sprint/models/` | Downloaded model weights |
| `/data/user_data/jgibson2/asr_sprint/results/` | All outputs (symlinked as `results/`) |
| `/data/user_data/jgibson2/asr_sprint/venv/` | Python 3.14 virtual environment |
| `/data/user_data/jgibson2/asr_sprint/hf_cache/` | HuggingFace model/dataset cache |

---

## Setup

```bash
# Create venv (once)
python -m venv /data/user_data/jgibson2/asr_sprint/venv
source /data/user_data/jgibson2/asr_sprint/venv/bin/activate
pip install -r requirements.txt

# Authenticate with HuggingFace (needed for gated CDLI datasets and models)
hf auth login
```

---

## RQ1 — Failure diagnosis pipeline

### Step 1: Download data (done ✓)
```bash
python -m src.rq1.baseline --download --model ky
python -m src.rq1.baseline --download --model en
```

### Step 2: Baseline inference (GPU)
```bash
python scripts/smoke_test.py          # verify before submitting
sbatch jobs/rq1_baseline.slurm
# monitor: tail -f results/rq1/logs/baseline_<jobid>.log
```
Reproduces CDLI published WER within ±0.02. Results in `results/rq1/baseline_*.parquet`.

### Step 3: Failure analysis + regression (CPU, login node)
```bash
python -m src.rq1.failure             # catastrophic flagging, Table 1
python -m src.rq1.regression          # logistic regression on covariates
python -m src.rq1.report              # consolidated plots and tables
```

### Step 4: Interventions (GPU)
```bash
python scripts/smoke_test.py
sbatch jobs/rq1_interventions.slurm
# then re-run report to add intervention plots:
python -m src.rq1.report
```

### Step 5: Manual review
Open `results/rq1/catastrophic_sample_50.csv` — stratified 50-utterance sample for R4 qualitative listening.

---

## Configuration

All hyperparameters, paths and model IDs live in `config/base.yaml`. Override for a run:
```bash
python -m src.rq1.baseline --config /path/to/custom.yaml
```

Key settings:

| Key | Default | Notes |
|-----|---------|-------|
| `decode.beam_size` | 1 | Greedy baseline; interventions override |
| `catastrophic.uncapped_wer_threshold` | 1.0 | Fixed |
| `catastrophic.length_ratio_threshold` | 2.0 | Fixed |
| `catastrophic.repeated_4gram_threshold` | 0.1 | Overridden at runtime by Youden's J on dev |
| `decode.checkpoint_every` | 50 | Utterances between checkpoint flushes |
| `slurm.partition` | preempt | Only partition available |
| `slurm.qos` | preempt_qos | Required — submission fails without it |

---

## Slurm notes

- **Partition:** `preempt` only — jobs can be preempted at any time
- **QOS:** must specify `--qos=preempt_qos` in every job file
- **Requeue:** all GPU jobs use `--requeue` to automatically resubmit after preemption
- **Resume:** all inference jobs use `--resume` flag — final parquets skip completed splits; checkpoint parquets resume mid-split
- **Signal:** `--signal=B:USR1@120` gives 120s to flush checkpoint before time limit

---

## Important notes

- **KY language token:** Kinyarwanda uses `language=sw` (Swahili token) — the published CDLI setting. Do not change to `rw`.
- **Normalisation order:** KY elision (`n'abantu → nabantu`) must run *before* Whisper BasicTextNormalizer, which strips apostrophes.
- **Speaker metadata:** `RW004` has a trailing space in the source CSV (`"RW004 "`). `load_split` strips it automatically.
- **Recording environment:** All utterances are `indoor with low background noise` — this feature produces no dummy columns in regression.
- **CTC fallback (Intervention F):** Placeholder only. Update `models.ky.ctc_fallback_id` in `base.yaml` when a KY CTC model is selected (RQ2).
- **Test discipline:** Test splits are evaluated once per frozen system at Gate G4. Do not iterate on test.

---

## Results summary (RQ1 — in progress)

See [`results/rq1/RQ1_results.md`](results/rq1/RQ1_results.md) for the full results report.

| Metric | KY test | EN test | Published KY | Published EN |
|--------|---------|---------|-------------|-------------|
| Corpus WER | 0.7242 | 0.2354 | 0.72 | 0.24 |
| Mean per-utt WER | 0.5260 | 0.1668 | 0.38 | 0.25 |
| Catastrophic utterances | 2.6% | 0.5% | — | — |
| % word errors from catastrophic utts | 47.2% | 1.4% | — | — |

---

## Gates and timeline

| Gate | Date | Criterion | Status |
|------|------|-----------|--------|
| G0 Charter | 9 Oct | Signed charter, roles, data access | ✅ |
| G1 Workplan | 10 Oct | RQs drafted, environment working | ✅ |
| G2 Diagnosis | 24 Oct | Baselines reproduced, error-analysis report, shortlist | 🔄 |
| G3 Testing-ready | 7 Nov | Demo v1, protocol, ≥1 model beats baseline | ⏳ |
| G4 Results freeze | 14 Nov | Test evaluated once, CIs, model cards | ⏳ |
| G5 Demo-ready | 25 Nov | Report final, pitch rehearsed, demo deployed | ⏳ |
| G6 Demo Day | 28 Nov | Presentation delivered | ⏳ |
