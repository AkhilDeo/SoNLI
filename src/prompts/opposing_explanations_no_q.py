def opposing_explanations_prompt_with_no_question(scene_dialogue, inference):
    return f"""<role> You are an AI assistant specializing in creating concise, evidence-based explanations to prove inferences false. </role>

<task> Create an explanation that proves that the inference is false. Find and cite specific evidence that directly contradicts the inference. Focus on relevant details that prove the inference is false. If there is no evidence that contradicts the inference, simply state that there is no evidence that contradicts the inference. Do not repeat the inference or dialogue in the explanation. </task>

<tone> The explanation should be simple, concise and declarative. It should be a single sentence that opposes the inference. </tone>

<format> Write your thoughts in <think> </think> tags. Do not include any additional text, explanations, or formatting in the answer. </format>

<dialogue>
{scene_dialogue}
</dialogue>

<inference> {inference} </inference>

<think>
"""