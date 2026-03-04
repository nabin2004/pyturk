"""DataLoader for pyturk — batches data from a Dataset."""

from __future__ import annotations
import random
from typing import List, Tuple


class DataLoader:
    """
    Loads data from a Dataset in batches.

    Args:
        dataset:    a Dataset instance
        batch_size: samples per batch (default: 1)
        shuffle:    whether to shuffle each epoch (default: False)
    """

    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch = [self.dataset[i] for i in batch_indices]
            X_batch, y_batch = zip(*batch)
            yield list(X_batch), list(y_batch)

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
