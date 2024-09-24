# models/model_manager.py
from adapters.bert_adapter import BertAdapter
from adapters.distilbert_adapter import DistilBertAdapter
from adapters.gpt_adapter import GPTAdapter
from adapters.roberta_adapter import RoBertaAdapter

class ModelManager:
    """
    Manages the loading and switching of models.
    """
    
    def __init__(self):
        self.models = {
            "bert": BertAdapter(),
            "roberta": RoBertaAdapter(),
            "distilbert": DistilBertAdapter()
            # You can add more models like LLaMA, etc.
        }

    def get_model(self, model_name: str):
        """
        Retrieve the model adapter based on the model name.
        """
        model_name = model_name.lower()  # Ensure case-insensitive matching
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model '{model_name}' is not available.")

