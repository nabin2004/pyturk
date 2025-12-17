from pyturk.data.dataset import Dataset

class MoonsDataset(Dataset):
    def __init__(self, n_samples=100, noise=0.1):
        self.n_samples = n_samples
        self.noise = noise
        self.X, self.y = self.generate()
    
    def generate(self):
        from sklearn.datasets import make_moons
        import torch
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise)
        y = y*2 - 1
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return X, y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
