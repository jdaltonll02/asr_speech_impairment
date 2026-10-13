#!/bin/bash
#SBATCH --job-name=rq1_baseline
#SBATCH --partition=preempt
#SBATCH --requeue                        # automatically requeue on preemption
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --qos=preempt_qos
#SBATCH --output=/data/user_data/jgibson2/asr_sprint/results/rq1/logs/baseline_%j.log
#SBATCH --signal=B:USR1@120             # send SIGUSR1 120s before time limit

source /data/user_data/jgibson2/asr_sprint/venv/bin/activate
export PYTHONPATH=/home/jgibson2/projects/asr-sprint:$PYTHONPATH
export HF_HOME=/data/user_data/jgibson2/asr_sprint/hf_cache
export TRANSFORMERS_CACHE=/data/user_data/jgibson2/asr_sprint/model_cache

echo "Job started: $(date)  node: $(hostname)  jobid: $SLURM_JOB_ID"

# --resume: skip any split whose final parquet already exists.
# Checkpoint files handle mid-split recovery automatically.
python -m src.rq1.baseline --resume

echo "Job finished: $(date)"
