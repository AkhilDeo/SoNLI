def classify_sarcasm_necessity_prompt(dialogue: str, question: str) -> str:

    return f"""<role> You are an AI assistant specializing in analyzing the nuances of language in questions and dialogues. </role>

<task> Your task is to analyze the provided question, and the accompanying dialogue, to determine the likelihood that understanding sarcasm or irony in the dialogue is necessary to answer the question accurately. If the question doesn't require understanding sarcasm or irony, choose "very unlikely" or "unlikely", and justify your choice. If the question requires understanding sarcasm or irony, choose "very likely" or "likely", and justify your choice. If you are unsure or it is possible, choose "possibly" and justify your choice.</task>

<response_format> You MUST choose your answer exclusively from the following five categories:
- very unlikely
- unlikely
- possibly
- likely
- very likely

Wrap your thoughts in <think> </think> tags and your final answer in <answer> </answer> tags.
</response_format>

<dialogue>
{dialogue}
</dialogue>

<question> {question} </question>
"""
