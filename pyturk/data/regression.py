# pyturk/data/regression.py
from pyturk.data.dataset import Dataset
from sklearn.datasets import make_regression
import torch

class RegressionDataset(Dataset):
    def __init__(self, n_samples=100, n_features=1, noise=15.0):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.X, self.y = self.generate()

    def generate(self):
        X, y = make_regression(n_samples=self.n_samples, n_features=self.n_features, noise=self.noise)
        y = y.reshape(-1,1)
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
