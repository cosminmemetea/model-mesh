# inference_engine.py

from models.model_manager import ModelManager


class InferenceEngine:
    """
    Handles running inference for different models using lazy-loaded ModelManager.
    """

    def __init__(self):
        self.model_manager = ModelManager()

    def predict(self, model_name: str, texts: list):
        """
        Run inference using the specified model.
        """
        # Retrieve the model lazily from the model manager
        model = self.model_manager.get_model(model_name)
        
        # Get predictions for all texts
        predictions = [model.predict(text) for text in texts]
        return predictions
