"""XOR dataset for pyturk."""

from __future__ import annotations
from pyturk.data.dataset import Dataset


class XORDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = self.generate()

    def generate(self):
        X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        y = [-1, 1, 1, -1]
        return X, y
