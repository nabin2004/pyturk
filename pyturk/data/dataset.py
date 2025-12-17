# pyturk/data/dataset.py
import torch

class Dataset:
    """Abstract base class for all datasets."""
    def __init__(self):
        self.X = None
        self.y = None

    def generate(self):
        """Generate dataset: must return (X, y) as torch tensors."""
        raise NotImplementedError

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def plot(self):
        """Optional: visualize dataset in 2D."""
        import matplotlib.pyplot as plt
        if self.X.shape[1] == 1:
            plt.scatter(range(len(self.X)), self.y, c=self.y, cmap='jet')
        else:
            plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='jet')
        plt.show()
