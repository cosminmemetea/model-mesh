import requests

from benchmarking.dataloaders.data_loader import DataLoader

class APIDataLoader(DataLoader):
    """
    A concrete data loader for fetching data from an API.
    """
    
    def __init__(self, api_url):
        self.api_url = api_url

    def load_data(self):
        """
        Fetch data from the API.
        """
        response = requests.get(self.api_url)
        response.raise_for_status()  # Check for errors
        data = response.json()
        texts = [item['text'] for item in data]
        labels = [item['label'] for item in data]
        return texts, labels
