# SocialNLI inference generation

`generate_inferences.py` expands curated FriendsQA question-answer pairs into SocialNLI-style inference sets by running two LLM passes per item (chain-of-thought and direct decoding). Each run collects model reasoning traces, normalised JSON payloads, and latency/error metadata for reproducibility.

## Input data
By default the script processes `datasets/socialnli_sources/curated/friendsqa_dialogues_qa_sarcasm_irony_only.json`, which contains Friends dialogue snippets with nested `qas` entries (dialogue text plus question/answer metadata). Use `--input` to point at another file with the same structure. Apply `--limit` during smoke tests to avoid consuming the whole corpus.

## Running the script
OpenAI-only run that writes to `outputs/socialnli_inference_regen_<timestamp>/`:
```bash
python src/dataset/generate_inferences.py \
  --cot-model gpt-4o \
  --no-cot-model gpt-3.5-turbo \
  --limit 10
```

Mixed providers with a slower cadence to respect rate limits:
```bash
python src/dataset/generate_inferences.py \
  --cot-provider openrouter --cot-model deepseek/deepseek-r1 \
  --no-cot-provider openai --no-cot-model gpt-4o-mini \
  --sleep 1.5
```

Enable the optional LangChain cache while saving raw responses for a tiny smoke test:
```bash
SONLI_ENABLE_LANGCHAIN_CACHE=1 python src/dataset/generate_inferences.py \
  --run-name smoke_test \
  --limit 3 \
  --include-raw
```

Pass `--include-raw` to persist the verbatim model responses alongside the parsed fields.

## Outputs
Each run creates a timestamped subdirectory under the chosen `--output-dir` (default `outputs/`). The directory contains `socialnli_inferences.json` with:
- `metadata`: run timestamp, resolved input path, and the models/providers used (`cot_model` and `no_cot_model`).
- `data`: one entry per question, including dialogue context, the original answer set, and two `inference_sets`. Each set stores up to five hypotheses, the parsed JSON payload, extracted `<think>` reasoning (if present), raw latency, and any collection errors.

Warnings are emitted whenever fewer than five hypotheses are recovered from a model, helping flag prompt or API issues early.
