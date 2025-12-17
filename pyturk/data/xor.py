# pyturk/data/xor.py
from pyturk.data.dataset import Dataset
import torch
import numpy as np

class XORDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = self.generate()

    def generate(self):
        X = np.array([[0,0],[0,1],[1,0],[1,1]])
        y = np.array([-1, 1, 1, -1])
        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float)
