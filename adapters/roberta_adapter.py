# adapters/roberta_adapter.py
from transformers import pipeline
from .model_adapter import ModelAdapter

class RoBertaAdapter(ModelAdapter):
    """
    Adapter for the RoBERTa model using Hugging Face's transformers library.
    """

    def __init__(self):
        # Initialize the RoBERTa pipeline for sentiment analysis
        self.model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

    def predict(self, text: str) -> dict:
        """
        Perform sentiment analysis on the input text and return the result.
        """
        result = self.model(text)
        # Extract the label and score from the result
        label = result[0]['label']
        score = result[0]['score']

        # Map the label to a human-readable format, if necessary
        if label == 'LABEL_2':  # Positive sentiment
            label = 'POSITIVE'
        elif label == 'LABEL_0':  # Negative sentiment
            label = 'NEGATIVE'
        elif label == 'LABEL_1':  # Neutral sentiment
            label = 'NEUTRAL'

        # Return the result in the desired format
        return {
            "result": {
                "label": label,
                "score": score
            }
        }