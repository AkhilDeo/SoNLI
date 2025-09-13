#!/bin/bash
#SBATCH --job-name=sonli_exp_eval_conservative
#SBATCH --output=logs/sonli_exp_eval_conservative_%j.out
#SBATCH --error=logs/sonli_exp_eval_conservative_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${PYTHONPATH}:/home/adeo1/SoNLI"

# Navigate to project directory
cd /home/adeo1/SoNLI

# Activate research conda environment
source activate research

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"

# Run the experiment with smaller models first, then larger ones
# Process models one at a time to avoid memory issues

echo "Processing llama-3.1-8b-instruct..."
python src/experiments/sonli_experiment_one.py \
    --eval-split \
    -mw 16 \
    -im huggingface \
    -gpu 2 \
    -m llama-3.1-8b-instruct \
    -ci 50

echo "Processing qwen3-32b..."
python src/experiments/sonli_experiment_one.py \
    --eval-split \
    -mw 16 \
    -im huggingface \
    -gpu 2 \
    -m qwen3-32b \
    -ci 50

echo "Processing llama-3.3-70b-instruct..."
python src/experiments/sonli_experiment_one.py \
    --eval-split \
    -mw 8 \
    -im huggingface \
    -gpu 2 \
    -m llama-3.3-70b-instruct \
    -ci 50

echo "Processing deepseek-v3-chat..."
python src/experiments/sonli_experiment_one.py \
    --eval-split \
    -mw 8 \
    -im huggingface \
    -gpu 2 \
    -m deepseek-v3-chat \
    -ci 50

echo "Job completed at: $(date)"
