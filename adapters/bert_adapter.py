# adapters/bert_adapter.py
from transformers import pipeline
from .model_adapter import ModelAdapter

class BertAdapter(ModelAdapter):
    """
    Adapter for the BERT model using Hugging Face's transformers library.
    """

    def __init__(self):
        self.model = pipeline("sentiment-analysis")

    def predict(self, text: str) -> dict:
        result = self.model(text)
        return {
            "label": result[0]['label'],
            "score": result[0]['score']
        }
