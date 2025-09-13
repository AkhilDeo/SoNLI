#!/bin/bash
#SBATCH --job-name=sonli_exp_eval_openrouter
#SBATCH --output=logs/sonli_exp_eval_openrouter_%j.out
#SBATCH --error=logs/sonli_exp_eval_openrouter_%j.err
#SBATCH --time=24:00
#SBATCH --partition=ba100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G

# Create logs directory if it doesn't exist
mkdir -p logs

# No GPUs requested for this job

# # Set environment variables
# export PYTHONPATH="${PYTHONPATH}:/home/adeo1/SoNLI"

# # If you keep your API keys in a .env file at project root, the script
# # will load them automatically (load_dotenv in Python). Otherwise, uncomment:
# # export OPENROUTER_API_KEY=your_key_here

# # Navigate to project directory
# cd /home/adeo1/SoNLI

# Activate research conda environment
source activate research || true

# Print system info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"

# Run the experiment over the full eval split using OpenRouter (CPU only)
python src/experiments/sonli_experiment_one.py \
    --eval-split \
    --eval-samples 2 \
    -mw 24 \
    -im openrouter \
    -m llama-3.1-8b-instruct deepseek-v3-chat llama-3.3-70b-instruct qwen3-32b \
    -ci 100

echo "Job completed at: $(date)"


