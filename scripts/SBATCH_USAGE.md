# SoNLI Experiment SBATCH Scripts Usage

## Overview
Three SBATCH scripts have been created for running the SoNLI experiment on the full evaluation dataset:

1. **`run_sonli_experiment_eval.sbatch`** - High-performance version (all models)
2. **`run_sonli_experiment_eval_conservative.sbatch`** - Conservative version (all models)
3. **`run_single_model.sbatch`** - Single model processing

## Script Comparison

### High-Performance Version (`run_sonli_experiment_eval.sbatch`)
- **Resources**: 4 GPUs, 32 CPUs, 256GB RAM, 48 hours
- **Approach**: Runs all models simultaneously with high parallelization
- **Workers**: 32 concurrent workers
- **Best for**: Clusters with abundant resources

### Conservative Version (`run_sonli_experiment_eval_conservative.sbatch`)
- **Resources**: 2 GPUs, 16 CPUs, 128GB RAM, 24 hours  
- **Approach**: Processes models sequentially to avoid memory issues
- **Workers**: 8-16 workers depending on model size
- **Best for**: Clusters with limited resources or when testing

### Single Model Version (`run_single_model.sbatch`)
- **Resources**: 2 GPUs, 16 CPUs, 128GB RAM, 12 hours
- **Approach**: Processes one specified model at a time
- **Workers**: 8-16 workers (auto-adjusted based on model size)
- **Best for**: Testing individual models or distributed processing

## Key Features

Both scripts:
- ✅ Use **entire eval dataset** (no sample limit)
- ✅ Use **HuggingFace inference** with vLLM for fast local inference
- ✅ **Automatically activate research conda environment**
- ✅ Include **all open source models** (excludes OpenAI models):
  - `llama-3.1-8b-instruct`
  - `deepseek-v3-chat` 
  - `llama-3.3-70b-instruct`
  - `qwen3-32b`
- ✅ **High worker counts** for maximum parallelization
- ✅ **Large checkpoint intervals** (50-100) for efficiency

## Usage

### 1. Prepare Environment
```bash
# Create logs directory
mkdir -p logs

# Activate research conda environment (done automatically in scripts)
conda activate research

# All dependencies (Python, CUDA, GCC, vLLM) are already in the research environment
# No module loads needed!

# Set up environment variables if needed  
export OPENAI_API_KEY="your_key_here"  # Still needed for caching/fallback
```

### 2. Submit Job
```bash
# For high-performance clusters (all models)
sbatch run_sonli_experiment_eval.sbatch

# For conservative resource usage (all models)
sbatch run_sonli_experiment_eval_conservative.sbatch

# For single model processing
sbatch --export=MODEL_NAME=llama-3.1-8b-instruct run_single_model.sbatch
sbatch --export=MODEL_NAME=deepseek-v3-chat run_single_model.sbatch
sbatch --export=MODEL_NAME=llama-3.3-70b-instruct run_single_model.sbatch
sbatch --export=MODEL_NAME=qwen3-32b run_single_model.sbatch
```

### 3. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor output logs
tail -f logs/sonli_exp_eval_full_<JOBID>.out

# Monitor error logs  
tail -f logs/sonli_exp_eval_full_<JOBID>.err
```

## Expected Outputs

Results will be saved to timestamped directories:
```
/home/adeo1/SoNLI/outputs/exp_one_eval_<samples>samples_<timestamp>/
├── llama-3.1-8b-instruct/
│   ├── results/
│   ├── checkpoints/
│   └── plots/
├── deepseek-v3-chat/
├── llama-3.3-70b-instruct/
└── qwen3-32b/
```

## Customization

### Modify Resources
Edit the `#SBATCH` directives:
```bash
#SBATCH --gres=gpu:4        # Number of GPUs
#SBATCH --cpus-per-task=32  # Number of CPUs
#SBATCH --mem=256G          # Memory allocation
#SBATCH --time=48:00:00     # Time limit
```

### Adjust Model Parameters
Modify the Python command:
```bash
-mw 32          # Max workers (adjust based on CPU count)
-gpu 4          # Number of GPUs for vLLM
-ci 100         # Checkpoint interval
```

### Run Subset of Models
Modify the `-m` parameter:
```bash
-m llama-3.1-8b-instruct qwen3-32b  # Only run these models
```

## Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce `-mw` workers or `-gpu` count
2. **Model Loading Errors**: Ensure vLLM is properly installed
3. **Permission Errors**: Check file permissions and CUDA access
4. **Time Limit**: Increase `--time` for larger datasets

### Performance Tips
1. **GPU Memory**: Larger models (70B) may need more GPUs or tensor parallelism
2. **CPU Workers**: Match worker count to CPU cores for optimal throughput  
3. **Checkpointing**: Lower checkpoint intervals for safer recovery but slightly slower performance

## Estimated Runtime

With full eval dataset (~1000-2000 samples):
- **High-performance version**: ~12-24 hours
- **Conservative version**: ~18-36 hours

Times vary based on:
- Model sizes (8B vs 70B parameters)
- Hardware specifications
- Dataset size
- Worker configuration
