import pandas as pd
from benchmarking.dataloaders.data_loader import DataLoader

class CSVDataLoader(DataLoader):
    """
    A concrete data loader for loading data from CSV files.
    """
    
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Load data from a CSV file.
        """
        data = pd.read_csv(self.file_path)
        texts = data['text'].tolist()
        labels = data['label'].tolist()
        return texts, labels
