# adapters/distilbert_adapter.py
from transformers import pipeline
from .model_adapter import ModelAdapter

class DistilBertAdapter(ModelAdapter):
    """
    Adapter for the DistilBERT model using Hugging Face's transformers library.
    DistilBERT is a smaller, faster, and lighter version of BERT.
    """

    def __init__(self):
        # Initialize the DistilBERT pipeline for sentiment analysis
        self.model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def predict(self, text: str) -> dict:
        """
        Perform sentiment analysis on the input text using DistilBERT and return the result.
        """
        result = self.model(text)
        return {
            "label": result[0]['label'],  # 'POSITIVE', 'NEGATIVE', etc.
            "score": result[0]['score']   # Confidence score for the label
        }
