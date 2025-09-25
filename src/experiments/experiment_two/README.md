# Experiment 2 – Human judgment study

Experiment 2 examines whether reasoning-focused models produce higher quality counterfactual explanations than instruction-tuned LLMs. Instead of relying on an automated judge, the authors collected human correctness labels for supporting/opposing explanations produced by six models across 27 inference prompts.

## Artifact
- `artifacts/experiment_two.json` – list of 96 records, one per `(dialogue, question, inference, model)` combination. Each record contains:
  - `id` – dialogue identifier.
  - `question` – sarcasm/irony-focused question shown to annotators.
  - `dialogue` – list of transcript turns.
  - `answers` – ground-truth short answers (when available from FriendsQA).
  - `results` – list of explanation evaluations for the given model. Fields per item:
    - `inference` – hypothesis under consideration.
    - `supporting_explanation` / `opposing_explanation` – model-generated rationales.
    - `unli_support_score` / `unli_oppose_score` – automatic scores from the UNLI classifier.
    - `llm_support_score` / `llm_oppose_score` – scores assigned by the model-based judge during Experiment 1.
    - `human_support_score` / `human_oppose_score` – scalar ratings from human annotators (0–1 scale).
    - `human_support_correct` / `human_oppose_correct` – boolean flags indicating whether annotators marked the explanation as correct.
    - `relevant_inference` – whether annotators agreed the inference pertains to the dialogue.
  - `model` – name of the generator (`o1`, `deepseek-r1`, `qwq-32b`, `gpt-4o`, `deepseek-v3`, `qwen-2.5-32b-instruct`, `llama-3.1-8b`, `qwen-2.5-7b-instruct`).

## Re-running the study
We do not ship the annotation interface; however, you can reproduce the model outputs by running Experiment 1 with the same prompts and then collecting human judgments using your preferred tool (e.g., Qualtrics, MTurk). The curated prompt set for this experiment matches the entries with `model == 'o1'` in the JSON file.

For transparency and reproducibility, the raw human decisions are left untouched in `artifacts/experiment_two.json`. Please anonymise or aggregate further if you plan to redistribute the file.
