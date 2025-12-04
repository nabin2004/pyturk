
class Dataset:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.n_samples = len(X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return self.n_samples
