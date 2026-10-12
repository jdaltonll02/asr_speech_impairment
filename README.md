# ASR Sprint — CDLI Rwandan NSS Failure Diagnosis (RQ1)

## Project structure

```
asr-sprint/
├── config/base.yaml          # Central config (paths, model IDs, thresholds)
├── requirements.txt          # Python package pins
├── src/
│   ├── eval/
│   │   ├── normalise.py      # Text normalisation (Whisper BasicTextNormalizer + KY rules)
│   │   ├── metrics.py        # WER / CER / repeated n-gram metrics
│   │   └── harness.py        # Whisper model loading + per-utterance decode loop
│   ├── rq1/
│   │   ├── baseline.py       # Reproduce CDLI published WER numbers
│   │   ├── failure.py        # Catastrophic failure diagnostics
│   │   ├── regression.py     # Logistic regression on catastrophic flag
│   │   ├── interventions.py  # Six re-decoding interventions
│   │   └── report.py         # Plots and summary tables
│   └── utils/
│       ├── config.py         # YAML config loader (dot-access)
│       ├── data.py           # Local data loading + HF download helper
│       ├── audio.py          # Silence detection, SNR, VAD trim
│       └── slurm.py          # Slurm script writer / submitter
├── jobs/
│   ├── rq1_baseline.slurm    # GPU job: baseline inference (submit to preempt)
│   └── rq1_interventions.slurm  # GPU job: intervention re-decoding
└── scripts/
    └── run_rq1_analysis.sh   # CPU analysis pipeline (run on login node)
```

## Data paths

| Location | Purpose |
|----------|---------|
| `/data/user_data/jgibson2/asr_sprint/data/` | Local TSV splits + speaker_metadata.csv |
| `/data/user_data/jgibson2/asr_sprint/models/` | Downloaded model weights |
| `/data/user_data/jgibson2/asr_sprint/results/rq1/` | Parquet outputs, plots, CSVs |
| `/data/user_data/jgibson2/asr_sprint/venv/` | Python virtual environment |

## Setup

```bash
# Create virtual environment
python -m venv /data/user_data/jgibson2/asr_sprint/venv
source /data/user_data/jgibson2/asr_sprint/venv/bin/activate
pip install -r requirements.txt

# Set HuggingFace token (needed for gated CDLI datasets and models)
export HF_TOKEN=<your_token>
# or: huggingface-cli login
```

## Download data (once HF access is granted)

```bash
# Download KY dataset to local data root
python -m src.rq1.baseline --model ky --download

# Download EN dataset
python -m src.rq1.baseline --model en --download
```

The `--download` flag fetches splits from HuggingFace and writes TSV files to
`/data/user_data/jgibson2/asr_sprint/data/`. It exits after downloading without
running any inference.

## Run baseline inference (Slurm)

```bash
sbatch jobs/rq1_baseline.slurm
# Runs python -m src.rq1.baseline --resume for both KY and EN, dev and test.
```

Or run interactively on a GPU node:
```bash
python -m src.rq1.baseline --model ky --split dev
python -m src.rq1.baseline --model ky --split test
python -m src.rq1.baseline --model en --split dev --resume
```

## Run failure analysis + regression + report (CPU, login node)

```bash
bash scripts/run_rq1_analysis.sh               # both languages
bash scripts/run_rq1_analysis.sh --lang ky     # KY only
bash scripts/run_rq1_analysis.sh --lang ky --split test
```

Or run each step individually:
```bash
python -m src.rq1.failure --lang ky --split dev
python -m src.rq1.failure --lang both --split both
python -m src.rq1.regression --lang both
python -m src.rq1.report --lang both
```

## Run interventions (Slurm)

```bash
sbatch jobs/rq1_interventions.slurm
```

Or interactively:
```bash
python -m src.rq1.interventions --model ky --split test
python -m src.rq1.interventions --intervention beam_search --model en
```

## Configuration

All paths and hyperparameters are in `config/base.yaml`.  Override with:
```bash
python -m src.rq1.baseline --config /path/to/custom.yaml
# or:
CONFIG_PATH=/path/to/custom.yaml python -m src.rq1.baseline
```

## Notes

- **Language token**: Kinyarwanda uses `language=sw` (Swahili) — the published
  CDLI decoding setting. Do not change to `rw`.
- **Gated datasets**: Both CDLI datasets require accepting terms at
  https://huggingface.co/datasets/cdli/rwandan_kinyarwanda_nonstandard_speech_v1.0
  and the equivalent EN URL.
- **CTC fallback**: Intervention F uses a placeholder model ID. Update
  `models.ky.ctc_fallback_id` in `base.yaml` when a KY CTC model becomes
  available.
- **GPU memory**: `whisper-large-v3` needs ~10 GB VRAM. Use `batch_size: 4` in
  `base.yaml` if you hit OOM.
