def generate_sarcasm_irony_answer_prompt(dialogue: str, question: str) -> str:
    return f"""<role> You are an AI assistant specialized in interpreting dialogue, including sarcasm and irony, to provide accurate answers to questions. </role>

<task> Your task is to carefully read the provided dialogue and the question. Based on your understanding of the dialogue, including any sarcastic or ironic intent, generate a concise and accurate answer to the question. The answer should directly address the question and reflect the true meaning or implication derived from the dialogue, especially if sarcasm/irony is involved.

Ensure the answer is:
- Grammatically correct and clearly phrased.
- Directly responsive to the question asked.
- Accurately reflects the information (explicit or implicit due to sarcasm/irony) in the dialogue.
- As brief as possible while still being complete.
</task>

<dialogue>
{dialogue}
</dialogue>

<question>
{question}
</question>

<response_format>
Provide only the generated answer. Do not include any preamble, explanation, or XML tags in your response. If the question asks "What does Speaker A *really* think...", your answer should state what Speaker A really thinks.
For example, if the question is "What does Speaker A *really* think about the news?" and Speaker A sarcastically said "Oh, I'm just thrilled," a good answer might be: "Speaker A is actually very unhappy about the news."
</response_format>

<answer_output_placeholder>
[Your generated answer here]
</answer_output_placeholder>
"""
