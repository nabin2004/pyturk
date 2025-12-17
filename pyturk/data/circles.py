# pyturk/data/circles.py
from .dataset import Dataset
from sklearn.datasets import make_circles
import torch

class CirclesDataset(Dataset):
    def __init__(self, n_samples=100, noise=0.05):
        super().__init__()
        self.n_samples = n_samples
        self.noise = noise
        self.X, self.y = self.generate()

    def generate(self):
        X, y = make_circles(n_samples=self.n_samples, noise=self.noise)
        y = y * 2 - 1
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
