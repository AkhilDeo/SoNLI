# SoNLI Experiment One - LLM-as-Judge Guide

This guide explains how to run the first social reasoning experiment using LLM-as-judge methodology with various models and inference methods.

## Quick Test

To run a quick test with 2 samples from the eval split:

```bash
./run_test_experiment.py
```

This will:
- Use 2 samples from the eval split (`socialnli_human_eval_split.json`)
- Test with `gpt-4o-mini` and `llama-3.3-8b-instruct` models
- Use OpenRouter for inference
- Use LLM-as-judge scoring (DeepSeek-R1) only
- Save results to `outputs/sonli_experiment_one_eval_2samples/`

## Manual Experiment Runs

### Basic Usage

```bash
python src/experiments/sonli_experiment_one.py [OPTIONS]
```

### Key Options

- `--eval-split`: Use eval split instead of main split
- `--eval-samples N`: Number of samples to take from eval split (default: 2)
- `--inference-method {openrouter,huggingface}`: Choose inference method
- `--models MODEL1 MODEL2 ...`: Specify which models to test
- `--num-gpus N`: Number of GPUs for vLLM (huggingface method only)
- `--limit N`: Limit number of items from main split
- `--max-workers N`: Number of parallel workers

### Available Models

- `gpt-4o` (OpenAI)
- `gpt-4o-mini` (OpenAI)  
- `llama-3.3-8b-instruct` (OpenRouter/HuggingFace)
- `deepseek-v3-chat` (OpenRouter/HuggingFace)
- `llama-3.3-70b-instruct` (OpenRouter/HuggingFace)
- `qwen3-32b` (OpenRouter/HuggingFace)

## Example Commands

### Small Test with Eval Split
```bash
# Test with 5 samples and 2 models
python src/experiments/sonli_experiment_one.py --eval-split --eval-samples 5 --models gpt-4o-mini llama-3.3-8b-instruct

# Test with all models, 10 samples
python src/experiments/sonli_experiment_one.py --eval-split --eval-samples 10
```

### Full Experiment with Main Split
```bash
# Run all models with OpenRouter
python src/experiments/sonli_experiment_one.py --models gpt-4o gpt-4o-mini llama-3.3-8b-instruct deepseek-v3-chat llama-3.3-70b-instruct qwen3-32b

# Run with limited samples from main split
python src/experiments/sonli_experiment_one.py --limit 100 --models gpt-4o-mini llama-3.3-8b-instruct
```

### Using HuggingFace/vLLM (Local Inference)
```bash
# Single GPU
python src/experiments/sonli_experiment_one.py --inference-method huggingface --models llama-3.3-8b-instruct --eval-split --eval-samples 5

# Multi-GPU (4 GPUs)
python src/experiments/sonli_experiment_one.py --inference-method huggingface --num-gpus 4 --models llama-3.3-70b-instruct --eval-split --eval-samples 10
```

## Output Structure

Results are saved to model-specific directories:
- **Eval experiments**: `outputs/sonli_experiment_one_eval_Nsamples/`
- **Main split experiments**: `outputs/sonli_experiment_one/`

Each model gets its own organized directory:
```
outputs/sonli_experiment_one/
├── gpt-4o/
│   ├── results/
│   │   └── experiment_results_20250113_143022.json
│   ├── plots/
│   │   └── judge_bayes_distributions_20250113_143022.png
│   └── checkpoints/
│       ├── checkpoint_20250113_143022_stage1_complete_count_1000.json
│       └── checkpoint_20250113_143022_stage2_complete_count_1000.json
├── gpt-4o-mini/
│   ├── results/
│   ├── plots/
│   └── checkpoints/
├── llama-3.3-8b-instruct/
│   ├── results/
│   ├── plots/
│   └── checkpoints/
└── ... (other models)
```

## Workflow

For each model, the experiment:
1. **Stage 1**: Generate supporting and opposing explanations
2. **Stage 2**: Score explanations with judge model (DeepSeek-R1)
3. **Stage 3**: Skipped (UNLI scoring removed - LLM-as-judge only)
4. **Stage 4**: Restructure data
5. **Stage 5**: Calculate Bayes scores using judge scores
6. **Stage 6**: Save results and generate plots

## Environment Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```bash
OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key
HTTP_REFERER=https://www.clsp.jhu.edu/
X_TITLE=Social Reasoning NLI
```

3. For vLLM (optional):
```bash
pip install vllm
```

## Notes

- **Temperature**: 0.7 (consistent across all models)
- **Max tokens**: 5000 (consistent across all models)
- **Judge model**: DeepSeek-R1 via OpenRouter (LLM-as-judge methodology)
- **UNLI model**: Removed for this experiment (focusing on LLM-as-judge only)
- **Checkpointing**: Automatic saves at configurable intervals
- **Caching**: LangChain SQLite cache for efficiency
- **Bayes calculation**: Based only on judge scores (supporting and opposing)
