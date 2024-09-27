# pipeline.py
from benchmarking.inference_engine import InferenceEngine
from benchmarking.evaluation import evaluate_metrics

def benchmark_model(model_name, data_loader):
    """
    Load data, run inference, and evaluate the specified model.
    """
    # Load data using the provided data loader
    texts, true_labels = data_loader.load_data()
    
    # Initialize the InferenceEngine to use lazy loading of models
    inference_engine = InferenceEngine()
    
    # Get predictions
    predicted_labels = inference_engine.predict(model_name, texts)
    
    # Evaluate metrics
    metrics = evaluate_metrics(true_labels, predicted_labels)
    
    # Display results
    print(f"Benchmarking results for {model_name}:")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1_score']}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    return metrics

if __name__ == "__main__":
    from benchmarking.dataloaders.data_loader import CSVDataLoader
    
    # Create the data loader
    data_loader = CSVDataLoader('data/processed/dataset.csv')
    
    # Benchmark models
    benchmark_model("roberta", data_loader)
    benchmark_model("distilbert", data_loader)
