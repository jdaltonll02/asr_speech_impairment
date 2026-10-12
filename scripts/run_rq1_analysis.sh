#!/bin/bash
# Run the full RQ1 analysis pipeline on the login node (CPU-only, no GPU needed).
#
# Prerequisites:
#   - Baseline parquets written by rq1_baseline.slurm must exist in
#     /data/user_data/jgibson2/asr_sprint/results/rq1/
#   - Virtual environment must be active, or call with the full path below.
#
# Usage:
#   bash scripts/run_rq1_analysis.sh               # both languages, both splits
#   bash scripts/run_rq1_analysis.sh --lang ky     # Kinyarwanda only
#   bash scripts/run_rq1_analysis.sh --lang en --split test
#
# Pass any extra flags and they are forwarded to the Python scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate venv if not already active
VENV_PYTHON="${VENV_PYTHON:-/data/user_data/jgibson2/asr_sprint/venv/bin/python}"
if ! command -v python &>/dev/null || [[ "$(python -c 'import sys; print(sys.prefix)')" != */venv* ]]; then
    if [[ -f "$VENV_PYTHON" ]]; then
        PYTHON="$VENV_PYTHON"
    else
        PYTHON="python"
    fi
else
    PYTHON="python"
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export HF_HOME=/data/user_data/jgibson2/asr_sprint/hf_cache
export TRANSFORMERS_CACHE=/data/user_data/jgibson2/asr_sprint/model_cache

echo "=== RQ1 Analysis Pipeline ==="
echo "Project root: $PROJECT_ROOT"
echo "Python:       $PYTHON"
echo "Args forwarded: $*"
echo ""

echo "[1/3] Running failure diagnostics..."
"$PYTHON" -m src.rq1.failure "$@"

echo ""
echo "[2/3] Running logistic regression..."
# regression.py only accepts --lang and --config (not --split)
LANG_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --lang*|--config*) LANG_ARGS="$LANG_ARGS $arg" ;;
    esac
done
"$PYTHON" -m src.rq1.regression $LANG_ARGS

echo ""
echo "[3/3] Generating report..."
"$PYTHON" -m src.rq1.report $LANG_ARGS

echo ""
echo "=== Analysis complete. ==="
echo "Outputs in: /data/user_data/jgibson2/asr_sprint/results/rq1/"
