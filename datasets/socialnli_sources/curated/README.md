# Curated sarcasm/irony subsets

The JSON files in this directory document intermediate subsets derived from FriendsQA after filtering for sarcasm and irony. They capture the scenes and questions that ultimately seed SocialNLI.

## Files
- `friendsqa_dialogues_sarcasm_irony_only.json` – 243 raw transcripts annotated with boolean flags `contains_sarcasm` / `contains_irony`. The `qas` list retains every original FriendsQA question for the scene; use this file if you need the full context for downstream filtering.
- `friendsqa_dialogues_qa_sarcasm_irony_only.json` – the same 243 transcripts, but each `qas` list is trimmed to only those questions the DeepSeek-R1 filter marked as likely to require sarcasm/irony understanding. These are the prompts we conditioned on when generating inferences.
- `friendsqa_sarcasm_irony_dialogues_qa_augmented.json` – extends the trimmed question lists with automatically generated, sarcasm-targeted questions. Augmented entries appear inside `qas` alongside the original items and are tagged with `question_source` (e.g., `deepseek_r1_sarcasm_augmentation`) and a synthetic `question_id`.

All three files share the same schema:

- `title` – Friends episode / scene identifier.
- `dialogue` – list of string turns (includes stage directions prefixed with `#NOTE#`).
- `contains_irony` / `contains_sarcasm` – boolean indicators from the filtering pass.
- `qas` – list of dicts. Each dict contains at minimum `question` and `answers`; original FriendsQA entries retain their `id`, while augmented questions may include `question_id`, `question_source`, or other provenance hints.

Use these resources to trace how a SocialNLI example was created, regenerate prompts with updated models, or conduct ablations on the filtering heuristics.
