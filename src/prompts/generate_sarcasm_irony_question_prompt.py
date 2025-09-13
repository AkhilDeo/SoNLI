def generate_sarcasm_irony_question_prompt(dialogue: str) -> str:
    return f"""<role> You are an AI assistant specialized in understanding and identifying sarcasm and irony in conversations. Your goal is to create insightful questions that probe the sarcastic or ironic elements within a dialogue. </role>

<task> Your task is to carefully read the provided dialogue and generate one high-quality question that specifically targets or requires understanding of the sarcasm or irony present in the dialogue. The question should make sense in the context of the dialogue and should not be answerable without recognizing the sarcastic or ironic intent.

Ensure the question is:
- Grammatically correct and clearly phrased.
- Directly related to the sarcastic or ironic content.
- Open-ended or requires more than a simple yes/no answer if possible, encouraging deeper reasoning about the sarcastic/irony.
- Not a repeat of existing questions (if any were implicitly provided or are obvious).
</task>

<dialogue>
{dialogue}
</dialogue>

<response_format>
Provide only the generated question. Do not include any preamble, explanation, or XML tags in your response.
For example, if the dialogue implies someone is "thrilled" about bad news sarcastically, a good question might be: "What does Speaker A *really* think about the news?" or "How does Speaker B's tone when saying 'fantastic' suggest their true feelings?"
</response_format>

<question_output_placeholder>
[Your generated question here]
</question_output_placeholder>
"""
