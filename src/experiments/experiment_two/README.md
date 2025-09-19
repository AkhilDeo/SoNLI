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
    - `llm_support_score` / `llm_oppose_score` – scores assigned by the model-based judge during Experiment 1.
    - `human_support_score` / `human_oppose_score` – scalar ratings from human annotators (0–1 scale).
    - `human_support_correct` / `human_oppose_correct` – boolean flags indicating whether annotators marked the explanation as correct.
    - `relevant_inference` – whether annotators agreed the inference pertains to the dialogue.
  - `model` – name of the generator (`o1`, `deepseek-r1`, `qwq-32b`, `gpt-4o`, `deepseek-v3`, `qwen-2.5-32b-instruct`, `llama-3.1-8b`, `qwen-2.5-7b-instruct`).

## Quick analysis example
```python
import json
from pathlib import Path
from collections import defaultdict

data = json.loads(Path('src/experiments/experiment_two/artifacts/experiment_two.json').read_text())

summary = defaultdict(lambda: {'n': 0, 'support_correct': 0, 'oppose_correct': 0})
for entry in data:
    model = entry['model']
    for res in entry['results']:
        summary[model]['n'] += 1
        summary[model]['support_correct'] += int(bool(res.get('human_support_correct')))
        summary[model]['oppose_correct'] += int(bool(res.get('human_oppose_correct')))

for model, stats in summary.items():
    support_acc = stats['support_correct'] / stats['n']
    oppose_acc = stats['oppose_correct'] / stats['n']
    print(f"{model:>20}: support={support_acc:.2%}, oppose={oppose_acc:.2%}, n={stats['n']}")
```

The official paper aggregates these into overall factuality accuracies (see Figure 4). You can reproduce the bar chart with your own plotting code using the above summary statistic scaffold.

## Methodology recap
- 27 dialogue-inference pairs were sampled from SocialNLI after filtering to ensure they centre sarcasm/irony.
- Each of the eight models generated both a supporting and an opposing explanation.
- Authors double-checked explanations for topical relevance, discarding degenerate outputs before annotation.
- Human raters judged whether each explanation was factually correct with respect to the dialogue (binary label) and provided optional scalar scores.

## Re-running the study
We do not ship the annotation interface; however, you can reproduce the model outputs by running Experiment 1 with the same prompts and then collecting human judgments using your preferred tool (e.g., Qualtrics, MTurk). The curated prompt set for this experiment matches the entries with `model == 'o1'` in the JSON file.

For transparency and reproducibility, the raw human decisions are left untouched in `experiment_two.json`. Please anonymise or aggregate further if you plan to redistribute the file.
