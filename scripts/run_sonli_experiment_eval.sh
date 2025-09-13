#!/bin/bash
#SBATCH --job-name=sonli_exp_eval_oss
#SBATCH --output=logs/sonli_exp_eval_oss_%j.out
#SBATCH --error=logs/sonli_exp_eval_oss_%j.err
#SBATCH --time=24:00
#SBATCH --partition=ba100
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:/home/adeo1/SoNLI"

# Navigate to project directory
cd /home/adeo1/SoNLI

# Activate research conda environment
source activate research

# Install vLLM if not already installed (uncomment if needed)
# pip install vllm

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"

# Run the experiment with all open source models
# Using entire eval split (no --eval-samples limit)
# High worker count for parallelization
# HuggingFace inference with 4 GPUs
# All open source models (excluding OpenAI gpt-4o and gpt-4o-mini)

python src/experiments/sonli_experiment_one.py \
    --eval-split \
    -mw 24 \
    -im huggingface \
    -gpu 4 \
    -m llama-3.1-8b-instruct deepseek-v3-chat llama-3.3-70b-instruct qwen3-32b \
    -ci 100

echo "Job completed at: $(date)"

# Optional: Send notification when job completes
# echo "SoNLI experiment completed on $(hostname) at $(date)" | mail -s "Job $SLURM_JOB_ID Complete" your.email@domain.com
