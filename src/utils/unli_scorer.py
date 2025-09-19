import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class UNLIScorer:
    def __init__(self):
        self.model_name = "Zhengping/roberta-large-unli"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def score(self, premise: str, hypothesis: str) -> float:
        """Returns a probability score for the hypothesis given the premise."""
        inputs = self.tokenizer(
            premise, 
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use sigmoid instead of softmax for probability
            probs = torch.sigmoid(outputs.logits)
            entail_prob = probs[0][0].item()
            
        return entail_prob