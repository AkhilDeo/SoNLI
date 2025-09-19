# SocialNLI dataset release

This directory contains the JSON releases for the SocialNLI benchmark described in the accompanying manuscript. SocialNLI pairs Friends dialogue snippets with free-form inference hypotheses, counterfactual explanations, and scalar plausibility judgments.

## Files
- `auto.json` – 3,920 inferences, with automatically generated supporting/opposing explanations and scores from the judge model.
- `eval.json` – 1,400 examples with human slider scores (0–1) and written justifications. These correspond to the evaluation slice used for the paper’s alignment analysis.

Together the splits yield 5,320 inference triples over 243 core dialogues. The dialogues average 25 turns (4.8 speakers), making SocialNLI substantially longer and more multi-party than prior dialogue NLI corpora.

## Record structure
Every JSON entry is a dictionary with the following fields:

- `uuid` – globally unique identifier for the (dialogue, question, inference) triple.
- `dialogue` – transcript snippet (string with turn-by-turn lines).
- `question` – targeting question used when prompting models to produce the inference.
- `inference` – hypothesis statement whose plausibility is being scored.
- `classification` – coarse label for the hypothesis type (`concerning reality`, `belief`, or `emotion`).
- `inference_type` – whether the hypothesis was produced with chain-of-thought prompting (`cot`) or direct decoding (`nocot`).
- `model` – generator responsible for the hypothesis (e.g., `gpt-4o`, `gpt-3.5-turbo`).
- `supporting_explanation_reasoning` / `opposing_explanation_reasoning` – raw “thinking” traces returned by the explanation model (if available).
- `supporting_explanation` / `opposing_explanation` – concise explanation strings establishing why the hypothesis could be true/false.
- `supporting_explanation_score` / `opposing_explanation_score` – UNLI-style scalar scores supplied by the automatic judge (0–1 after normalisation).
- `supporting_judge_raw_output` / `opposing_judge_raw_output` – verbatim judge responses (include rubric reasoning and raw 0–10 score before normalisation).
- `counterfactual_score` – Bayes posterior computed from supporting/opposing judge scores using the equation in §3.1 of the paper.
- `human_annotated_score` / `human_annotated_explanation` – **only in `eval.json`**. Final slider score (0–1) and short justification from human annotators.

A minimal example (truncated for brevity):

```json
{
  "uuid": "2ef7f230-aa7e-443c-9c6d-d34d54459a3d",
  "dialogue": "#NOTE#: [ Scene: Phoebe's grandmother's place. ]\nPhoebe Buffay: ...",
  "question": "Why does Phoebe figuratively smell smoke?",
  "inference": "Phoebe learns that her father is a pharmacist, not the adventurous person she was told about.",
  "classification": "concerning reality",
  "inference_type": "cot",
  "model": "gpt-4o",
  "supporting_explanation": "Phoebe uses the phrase ... which leads to her grandmother admitting the truth.",
  "opposing_explanation": "There is no evidence that contradicts the inference.",
  "supporting_explanation_score": 0.9,
  "opposing_explanation_score": 0.2,
  "human_annotated_score": 1.0,
  "human_annotated_explanation": "Her grandmother reveals ..."
}
```

## Loading the data
```python
import json
from pathlib import Path

with Path("datasets/socialnli/eval.json").open() as fp:
    eval_split = json.load(fp)

print(len(eval_split))  # 1400
print(eval_split[0]["inference"])
```

The JSON files contain UTF-8 text and can be parsed with standard Python tooling. Because some rationale strings include newline characters, do not expect TSV/CSV compatibility without additional escaping.

## Recommended usage
- Use `auto.json` for model training or prompt-based fine-tuning. It covers the full label spectrum and contains the counterfactual explanations needed to reproduce the Bayes scoring procedure.
- Evaluate on `eval.json` by comparing predicted plausibility scores against `human_annotated_score` (MAE, Pearson, or thresholded accuracy). Human explanations may also be used for qualitative studies.
- The raw judge messages allow you to swap in alternative scoring functions if desired. If you recompute scores, update `counterfactual_score` accordingly.

## Licensing
This directory incorporates content derived from FriendsQA, which remains under Apache 2.0 (original license and notice reproduced in `third_party_licenses/FriendsQA-APACHE-2.0.txt`). Dialogue excerpts remain subject to the underlying show rightsholders.
