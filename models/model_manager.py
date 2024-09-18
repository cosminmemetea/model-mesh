# models/model_manager.py
from adapters.bert_adapter import BertAdapter

class ModelManager:
    """
    Manages the loading and switching of models.
    """
    
    def __init__(self):
        self.models = {
            "bert": BertAdapter(),
            # You can add more models like GPT, LLaMA, etc.
        }

    def get_model(self, model_name: str):
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not available.")
