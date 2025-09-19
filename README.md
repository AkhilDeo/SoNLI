# SocialNLI: A Dialogue-Centric Social Inference Dataset

SocialNLI (SoNLI) is a dialogue-centric natural language inference benchmark that probes whether language and reasoning models can recover subtle social intent such as sarcasm, irony, and unstated motives. The corpus pairs multi-party television transcripts with free-form hypotheses, scalar plausibility judgments, and supporting/contradicting explanations collected from both humans and models. This repository accompanies the SocialNLI paper and provides the full data release, prompt templates, and experiment code used in the manuscript.

## Repository map
- `datasets/socialnli/` – primary dataset release (auto-labeled training split with 3.9k examples and 1.4k human-annotated examples).
- `datasets/socialnli_sources/` – original FriendsQA sources plus intermediate filtered/augmented JSON used during curation.
- `src/experiments/experiment_one/` – pipeline for generating counterfactual explanations and UNLI-style plausibility scores with different LLMs/LRMs.
- `src/experiments/experiment_two/` – human evaluation summaries contrasting language and reasoning models on a 27-example subset.
- `src/prompts/` – prompt programs used for filtering, inference generation, explanation drafting, and judging.
- `src/utils/` – helper modules (OpenRouter client with rate limiting, UNLI scorer wrapper).
- `outputs/` – sample outputs produced by the authors when running the experiments in September 2025.

## SocialNLI dataset release
SocialNLI combines two complementary splits:
- `auto.json` (3,920 items) – automatically generated inferences with counterfactual explanations and UNLI proxy scores.
- `eval.json` (1,400 items) – the human annotation split with plausibility ratings and free-form justifications collected from curated Mechanical Turk workers.

Each example contains the dialogue snippet, targeting question, inference hypothesis, metadata indicating whether it arose from chain-of-thought prompting, model-produced supporting and opposing explanations, judge model scores, and Bayes-style posteriors. The human split additionally records the raw slider score (`human_annotated_score`) and explanation text supplied by annotators.

Detailed field documentation and loading tips for the JSON files live in `datasets/socialnli/README.md`.

## Source data and intermediate assets
The FriendsQA dataset (Apache 2.0) provides the base transcripts. We filtered, augmented, and re-questioned these dialogues to foreground sarcasm/irony phenomena before soliciting inferences and annotations. Intermediate artifacts—including filtered transcript lists and augmented question sets—are documented under `datasets/socialnli_sources/`.

## Experiments
### Experiment 1 – Counterfactual explanation generation
`src/experiments/experiment_one/experiment_one.py` orchestrates six stages per model:
1. Generate supporting explanations.
2. Generate opposing explanations.
3. Judge each explanation on a 0–10 UNLI-inspired scale (normalized to 0-1).
4. Restructure outputs per inference.
5. Compute Bayes posterior plausibility estimates.
6. Persist artifacts (checkpoint JSON, plots, final results).

The script can call OpenAI models directly or OpenRouter-hosted / vLLM-served models. See the experiment-specific README for CLI arguments, environment variables, and output structure. Sample artifacts matching the paper appear in `src/experiments/experiment_one/artifacts/`.

### Experiment 2 – Human-judged reasoning vs. language models
The second experiment contrasts three reasoning-focused models (o1, DeepSeek-R1, QwQ-32B) with three instruction-tuned LLMs (GPT-4o, DeepSeek-V3, Qwen2.5-32B-Instruct) on a 27 example subset. Authors marked each model’s supporting/opposing explanations as correct or incorrect. Aggregated scores and per-example judgments are stored in `src/experiments/experiment_two/artifacts/experiment_two.json`; methodology notes and analysis guidance sit in the accompanying README.

## Setup and usage
1. Create a Python 3.11 environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure credentials for external models:
   - `OPENAI_API_KEY` (for GPT-4o family).
   - `OPENROUTER_API_KEY`, optionally `OPENROUTER_API_BASE`, `HTTP_REFERER`, `X_TITLE`, `OPENROUTER_CALLS_PER_PAUSE`, `OPENROUTER_PAUSE_SECONDS`.
4. Optional: install `vllm` if you intend to run open-weight models locally via the `--inference-method huggingface` flag.

Example run (OpenRouter hosted models, limited to 10 items):
```bash
python src/experiments/experiment_one/experiment_one.py \
  --models gpt-4o-mini deepseek-v3-chat \
  --limit 10 \
  --max-workers 4 \
  --checkpoint-interval 20
```
Outputs land under `outputs/exp_one_<timestamp>/<model>/` with checkpoints, plots, and JSON results.

## Citation
If you use the dataset or accompanying code, please cite the SocialNLI paper (citation forthcoming). A BibTeX entry will be added once the manuscript is public.

## License and acknowledgements
- Original SocialNLI materials are licensed under the Apache License 2.0 (see `LICENSE`). `NOTICE` records attribution details and key modifications, including removal of the `human_eval_uuid_source` field from `datasets/socialnli/eval.json`.
- Derived transcripts originate from FriendsQA (Apache 2.0). The original FriendsQA license and notice appear in `third_party_licenses/FriendsQA-APACHE-2.0.txt`, and their attribution requirements are preserved in this release.
- Television dialogue excerpts remain the property of their respective rightsholders ("Friends" © Warner Bros. Entertainment Inc.).
- OpenAI, DeepSeek, Meta, and Qwen models are accessed through their respective APIs; abide by their terms of service when reproducing the experiments.

For questions or clarifications, please open an issue or contact the SocialNLI authors.
