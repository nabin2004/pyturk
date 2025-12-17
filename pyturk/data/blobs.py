# pyturk/data/blobs.py
from .dataset import Dataset
from sklearn.datasets import make_blobs
import torch

class BlobsDataset(Dataset):
    def __init__(self, n_samples=100, centers=3, cluster_std=1.0):
        super().__init__()
        self.n_samples = n_samples
        self.centers = centers
        self.cluster_std = cluster_std
        self.X, self.y = self.generate()

    def generate(self):
        X, y = make_blobs(n_samples=self.n_samples, centers=self.centers, cluster_std=self.cluster_std)
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
