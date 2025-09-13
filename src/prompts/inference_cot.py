def generate_social_nli_cot(question, scene_dialogue):
    return f"""**Instructions**: For the following question and scene dialogue, write five distinct and non-overlapping inferences entailed by the scene. Generate the inferences by reasoning through the scene dialogue and the question in a step by step manner. Ensure the inferences are logical and relevant to the scene. The inferences should resemble short, factual statements about the scene and should help answer the question using component reasoning steps.

**Output Format:**
Wrap the reasoning process in <think> </think> tags and the inferences with ```json ``` markdown tags. The inferences should be in JSON format with keys "1" through "5". Do not include any additional text, explanations, or formatting in the JSON.

**Example Output Format:**
<think>
<reasoning step 1>
<reasoning step 2>
...
</think>
```json
{{
  "1": "Inference one.",
  "2": "Inference two.",
  "3": "Inference three.",
  "4": "Inference four.",
  "5": "Inference five."
}}```

**Question**: {question}

**Scene**: {scene_dialogue}

**Inferences (5 total)**:"""