#!/bin/bash
#SBATCH --job-name=rq1_baseline
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/user_data/jgibson2/asr_sprint/results/rq1/logs/baseline_%j.log
#SBATCH --signal=B:USR1@60

source /data/user_data/jgibson2/asr_sprint/venv/bin/activate
export PYTHONPATH=/home/jgibson2/projects/asr-sprint:$PYTHONPATH
export HF_HOME=/data/user_data/jgibson2/asr_sprint/hf_cache
export TRANSFORMERS_CACHE=/data/user_data/jgibson2/asr_sprint/model_cache
# HF_TOKEN must be set in your environment or ~/.cache/huggingface/token
# export HF_TOKEN=<your_token>

python -m src.rq1.baseline --resume
