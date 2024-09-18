# adapters/model_adapter.py
from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    """
    Abstract adapter to define the common structure for all ML models.
    """

    @abstractmethod
    def predict(self, text: str) -> dict:
        """
        Method to predict the sentiment from the input text.
        """
        pass
