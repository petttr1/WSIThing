from abc import ABC, abstractmethod


class Expert(ABC):
    def __init__(self, name):
        super().__init__()
        self.name = name

    @abstractmethod
    def predict(self):
        """Create a predictiong for a sample
        """
        pass
