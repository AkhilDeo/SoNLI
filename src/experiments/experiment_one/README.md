# Experiment 1 – Counterfactual explanation pipeline

`experiment_one.py` reproduces the counterfactual reasoning study from the SocialNLI paper. Given a set of (dialogue, question, inference) triples it:
1. Generates supporting explanations.
2. Generates opposing explanations.
3. Scores each explanation with an LLM judge (0–10 scale).
4. Computes Bayes-style plausibility scores.
5. Saves checkpoints, final JSON outputs, and score distribution plots.

## Note

The script uses a LangChain SQLite cache stored at `.langchain_sonli_exp1_cache.db`; it is cleared automatically before each run.

## Input data
By default the script looks for:
- `datasets/socialnli/auto.json`
- `datasets/socialnli/eval.json`

## Example commands
Run GPT-4o-mini and DeepSeek-V3 via OpenRouter on the first 20 items:
```bash
python src/experiments/experiment_one/experiment_one.py \
  --models gpt-4o-mini deepseek-v3-chat \
  --limit 20 \
  --max-workers 4 \
  --checkpoint-interval 10
```

Evaluate the human split on 100 random samples with Qwen3-32B served locally through vLLM:
```bash
python src/experiments/experiment_one/experiment_one.py \
  --models qwen3-32b \
  --eval-split --eval-samples 100 \
  --inference-method huggingface \
  --num-gpus 2 \
  --max-workers 2
```

## Output layout
Each run creates a timestamped directory under `outputs/`.

Sample artifacts from the paper (September 2025 runs) are preserved in `src/experiments/experiment_one/artifacts/` for reference.
