"""Blobs dataset for pyturk."""

from __future__ import annotations
from pyturk.data.dataset import Dataset


class BlobsDataset(Dataset):
    def __init__(self, n_samples: int = 100, centers: int = 3, cluster_std: float = 1.0):
        super().__init__()
        self.n_samples = n_samples
        self.centers = centers
        self.cluster_std = cluster_std
        self.X, self.y = self.generate()

    def generate(self):
        from sklearn.datasets import make_blobs
        X, y = make_blobs(
            n_samples=self.n_samples,
            centers=self.centers,
            cluster_std=self.cluster_std,
        )
        return X.tolist(), y.tolist()
