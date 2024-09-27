from abc import ABC, abstractmethod

class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    Defines the interface for loading data from various sources.
    """
    
    @abstractmethod
    def load_data(self):
        """
        Load the data and return it in a format usable by models.
        Should return a tuple of texts and labels.
        """
        pass
