# Experiment 1 – Counterfactual explanation pipeline

`experiment_one.py` reproduces the counterfactual reasoning study from the SocialNLI paper. Given a set of (dialogue, question, inference) triples it:
1. Generates supporting explanations.
2. Generates opposing explanations.
3. Scores each explanation with an LLM judge (0–10 scale).
4. Computes Bayes-style plausibility scores.
5. Saves checkpoints, final JSON outputs, and score distribution plots.

## Prerequisites
- Python 3.11+ with the packages listed in `requirements.txt` (`langchain`, `tqdm`, `matplotlib`, `seaborn`, `pandas`, `torch`, etc.).
- API access for the models you intend to call:
  - `OPENAI_API_KEY` for GPT-4o / GPT-4o-mini.
  - `OPENROUTER_API_KEY` (plus optional `OPENROUTER_API_BASE`, `HTTP_REFERER`, `X_TITLE`, `OPENROUTER_CALLS_PER_PAUSE`, `OPENROUTER_PAUSE_SECONDS`) for OpenRouter-hosted models such as DeepSeek, Llama 3.1, Qwen3.
- Optional: install `vllm>=0.4.0` if you want to run open-weight models locally via `--inference-method huggingface`.

The script uses a LangChain SQLite cache stored at `.langchain_sonli_exp1_cache.db`; it is cleared automatically before each run.

## Input data
By default the script looks for:
- `datasets/socialnli/socialnli_main_split.json`
- `datasets/socialnli/socialnli_human_eval_split.json`

In this repository the released files are named `auto.json` and `eval.json`. Create lightweight symlinks before running:
```bash
ln -sf auto.json datasets/socialnli/socialnli_main_split.json
ln -sf eval.json datasets/socialnli/socialnli_human_eval_split.json
```

## CLI arguments
```
python experiment_one.py [options]

General
  -l, --limit INT              Process only the first N items from the input split.
  -t, --test                   Shortcut that processes the first 3 items.
  --eval-split                 Switch to the human evaluation split.
  --eval-samples INT           Random sample (with seed 42) from the eval split.

Generation / scoring
  -m, --models M1 [M2 ...]     Models to run (see EXPLANATION_MODELS_CONFIG).
  -mw, --max-workers INT       Thread pool size (default 3).
  -ci, --checkpoint-interval   Save partial checkpoints every N items (default 50).

Inference backend
  -im, --inference-method      `openrouter` (default) or `huggingface` (vLLM).
  -gpu, --num-gpus             Number of GPUs for vLLM when using `--inference-method huggingface`.
```

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
```
outputs/
  exp_one_YYYYMMDD_HHMMSS/
    <model_name>/
      checkpoints/   # periodic partial saves of intermediate results
      results/       # final JSON (one file per run)
      plots/         # KDE/Histogram PNGs of judge and Bayes scores
```
The JSON in `results/` mirrors the input records and adds an `explanations_from_models` array with the generated explanations, judge scores, raw judge messages, and `final_bayes_score`.

## Tips & troubleshooting
- **Rate limiting:** adjust `OPENROUTER_CALLS_PER_PAUSE` (default 160) and `OPENROUTER_PAUSE_SECONDS` (default 5) if you observe HTTP 429 responses from OpenRouter.
- **Long contexts:** some Friends transcripts exceed 2k tokens; ensure the target models support long contexts or reduce `--limit` while debugging.
- **Missing files:** the script fails fast if the expected JSON input is absent. Verify the symlinks or update the source paths inside `experiment_one.py` if you keep custom filenames.
- **GPU memory:** when using vLLM with 32B+ models, allocate enough GPU memory and consider reducing `--max-workers` to 1.

Sample artifacts from the paper (September 2025 runs) are preserved in `src/experiments/experiment_one/artifacts/` for reference.
