def classify_inference_type(question, scene_dialogue, inference):
    return f"""**Instructions**: For the following question and scene dialogue, classify the provided inference as either concerning reality, a belief, or an emotion. Reason through the inference, the question and the scene dialogue step by step, and determine the most appropriate classification. Provide a brief explanation for your reasoning.

**Output Format:**
Wrap the reasoning process in <think> </think> tags and the classification in <answer> </answer> tags. The answer should be one of the three values: "concerning reality," "belief," or "emotion." Do not include any additional text, explanations, or formatting in the answer.

**Example Output Format:**
<think>
<reasoning step 1>
<reasoning step 2>
...
</think>
<answer>
concerning reality
</answer>

**Question**: {question}

**Scene**: {scene_dialogue}

**Inference**: {inference}

**Inference Classification and Reasoning**:"""