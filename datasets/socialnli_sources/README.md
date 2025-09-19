# SocialNLI source materials

This directory houses the upstream transcripts and intermediate assets used while constructing SocialNLI. They are provided for transparency and reproducibility of the curation pipeline.

## Contents
- `friendsqa_original/` – original FriendsQA release files (`friendsqa_trn.json`, `friendsqa_dev.json`, `friendsqa_tst.json`) under Apache 2.0. Each item contains the raw dialogue snippet, question, and answer span metadata.
- `curated/` – filtered and augmented subsets that isolate sarcasm/irony-centric scenes, along with newly phrased questions that explicitly target social nuance. Detailed descriptions for each JSON live in `curated/README.md`.

## Curation process (high-level)
1. **Dialogue selection** – we started from FriendsQA and retained transcripts exhibiting sarcasm, irony, or related pragmatic phenomena. The filtering step leveraged the prompt in `src/prompts/classify_sarcasm_necessity_prompt.py` with DeepSeek-R1 as a judge.
2. **Question augmentation** – for transcripts where the original FriendsQA question did not emphasise sarcasm/irony, we generated new questions using GPT-4o with prompts in `src/prompts/generate_sarcasm_irony_question_prompt.py` and `generate_sarcasm_irony_answer_prompt.py`.
3. **Inference generation** – the curated transcripts and questions formed the input to the inference generation prompts (`inference_cot.py`, `inference_no_cot.py`), producing 10 hypotheses per (dialogue, question) pair.
4. **Counterfactual reasoning** – supporting/opposing explanations and UNLI-style scores were produced using the pipeline in `src/experiments/experiment_one` before the human annotation pass.

These intermediate JSON files are not meant for training directly; they document the decisions that led to the final SocialNLI release and can be used to regenerate subsets if model prompts evolve.

## Licensing
- FriendsQA remains Apache 2.0. Retain the original attribution if you redistribute the transcripts.
- Generated questions and filtered lists in `curated/` are released for research use and inherit the overall SocialNLI repository terms.
