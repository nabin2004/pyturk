"""Regression dataset for pyturk."""

from __future__ import annotations
from pyturk.data.dataset import Dataset


class RegressionDataset(Dataset):
    def __init__(self, n_samples: int = 100, n_features: int = 1, noise: float = 15.0):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.X, self.y = self.generate()

    def generate(self):
        from sklearn.datasets import make_regression
        X, y = make_regression(
            n_samples=self.n_samples,
            n_features=self.n_features,
            noise=self.noise,
        )
        return X.tolist(), y.reshape(-1).tolist()
