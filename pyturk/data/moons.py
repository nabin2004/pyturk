"""Moons dataset for pyturk."""

from __future__ import annotations
from pyturk.data.dataset import Dataset


class MoonsDataset(Dataset):
    def __init__(self, n_samples: int = 100, noise: float = 0.1):
        super().__init__()
        self.n_samples = n_samples
        self.noise = noise
        self.X, self.y = self.generate()

    def generate(self):
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise)
        y = y * 2 - 1  # map {0,1} -> {-1,1}
        return X.tolist(), y.tolist()
