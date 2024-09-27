# models/model_manager.py

class ModelManager:
    """
    Manages the loading and switching of models.
    Implements lazy loading to load models only when needed.
    """
    
    def __init__(self):
        self.models = {}  # Keep an empty dict initially

    def get_model(self, model_name: str):
        """
        Retrieve the model adapter based on the model name.
        If the model is not loaded, it loads the model lazily.
        """
        model_name = model_name.lower()  # Ensure case-insensitive matching
        
        if model_name not in self.models:
            # Lazy loading of models
            if model_name == "bert":
                from adapters.bert_adapter import BertAdapter
                self.models[model_name] = BertAdapter()
            elif model_name == "roberta":
                from adapters.roberta_adapter import RoBertaAdapter
                self.models[model_name] = RoBertaAdapter()
            elif model_name == "distilbert":
                from adapters.distilbert_adapter import DistilBertAdapter
                self.models[model_name] = DistilBertAdapter()
            else:
                raise ValueError(f"Model '{model_name}' is not available.")
        
        return self.models[model_name]
