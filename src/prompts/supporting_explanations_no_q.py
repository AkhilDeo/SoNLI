def supporting_explanations_prompt_with_no_question(scene_dialogue, inference):
    return f"""<role> You are an AI assistant specializing in creating concise, evidence-based explanations to support inferences. </role>

<task> Create an explanation that directly supports the inference. Find and cite specific evidence that directly supports the inference. Focus on relevant details that prove the inference is true. If there is no evidence that supports the inference, simply state that there is no evidence that supports the inference. Do not repeat the inference or dialogue in the explanation. </task>

<tone> The explanation should be simple, concise and declarative. It should be a single sentence that directly supports the inference. </tone>

<format> Write your thoughts in <think> </think> tags. Do not include any additional text, explanations, or formatting in the answer. </format>

<dialogue>
{scene_dialogue}
</dialogue>

<inference> {inference} </inference>

<think>
"""