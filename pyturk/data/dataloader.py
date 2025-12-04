import random

class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __iter__(self):
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(self.indices)
        self.ptr = 0
        return self
    
    def __next__(self):
        if self.ptr >= len(self.indices):
            raise StopIteration
        batch_indices = self.indices[self.ptr:self.ptr+self.batch_size]
        batch = [self.dataset[i] for i in batch_indices]
        self.ptr += self.batch_size
        X_batch, y_batch = zip(*batch)
        return list(X_batch), list(y_batch)
