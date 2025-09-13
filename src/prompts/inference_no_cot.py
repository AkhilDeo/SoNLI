def generate_social_nli_no_cot(question, scene_dialogue):
    return f"""**Instructions**: For the following question and scene dialogue, write five distinct and non-overlapping inferences entailed by the scene. The inferences should resemble short, factual statements about the scene and should help answer the question using component reasoning steps.

**Output Format:**
Return the inferences in JSON format with keys "1" through "5". Do not include any additional text, explanations, or formatting.

**Example Format:**
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