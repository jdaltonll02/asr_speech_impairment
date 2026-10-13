#!/bin/bash
#SBATCH --job-name=rq1_interventions
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/data/user_data/jgibson2/asr_sprint/results/rq1/logs/interventions_%j.log
#SBATCH --signal=B:USR1@120

source /data/user_data/jgibson2/asr_sprint/venv/bin/activate
export PYTHONPATH=/home/jgibson2/projects/asr-sprint:$PYTHONPATH
export HF_HOME=/data/user_data/jgibson2/asr_sprint/hf_cache
export TRANSFORMERS_CACHE=/data/user_data/jgibson2/asr_sprint/model_cache

echo "Job started: $(date)  node: $(hostname)  jobid: $SLURM_JOB_ID"

# Run interventions on KY only (EN has ~6 catastrophic utterances — not informative).
# --all-utterances: run each strategy on the full split, not just catastrophic ones,
# so we can report corpus and per-utterance WER change as the plan requires.
python -m src.rq1.interventions --model ky --split dev --all-utterances --resume
python -m src.rq1.interventions --model ky --split test --all-utterances --resume

echo "Job finished: $(date)"
