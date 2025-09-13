#!/bin/bash
#SBATCH --job-name=sonli_single_model
#SBATCH --output=logs/sonli_single_model_%j.out
#SBATCH --error=logs/sonli_single_model_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Usage: sbatch --export=MODEL_NAME=llama-3.1-8b-instruct run_single_model.sbatch

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${PYTHONPATH}:/home/adeo1/SoNLI"

# Navigate to project directory
cd /home/adeo1/SoNLI

# Activate research conda environment
source activate research

# Check if MODEL_NAME is set
if [ -z "$MODEL_NAME" ]; then
    echo "ERROR: MODEL_NAME environment variable not set"
    echo "Usage: sbatch --export=MODEL_NAME=llama-3.1-8b-instruct run_single_model.sbatch"
    exit 1
fi

# Validate model name
case $MODEL_NAME in
    "llama-3.1-8b-instruct"|"deepseek-v3-chat"|"llama-3.3-70b-instruct"|"qwen3-32b")
        echo "Processing model: $MODEL_NAME"
        ;;
    *)
        echo "ERROR: Invalid model name: $MODEL_NAME"
        echo "Valid options: llama-3.1-8b-instruct, deepseek-v3-chat, llama-3.3-70b-instruct, qwen3-32b"
        exit 1
        ;;
esac

# Adjust workers based on model size
if [[ "$MODEL_NAME" == "llama-3.3-70b-instruct" || "$MODEL_NAME" == "deepseek-v3-chat" ]]; then
    WORKERS=8  # Fewer workers for large models
else
    WORKERS=16  # More workers for smaller models
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Model: $MODEL_NAME"
echo "Workers: $WORKERS"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# Run the experiment for the specified model
python src/experiments/sonli_experiment_one.py \
    --eval-split \
    -mw $WORKERS \
    -im huggingface \
    -gpu 2 \
    -m $MODEL_NAME \
    -ci 50

echo "Job completed at: $(date)"
