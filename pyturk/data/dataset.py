"""
Base Dataset class for pyturk.

Data is stored as Python lists (no torch dependency) — the autograd
engine works with scalar Values, so data stays as plain numbers until
fed through the model.
"""

from __future__ import annotations
from typing import Tuple, Any


class Dataset:
    """
    Abstract base class for all datasets.

    Subclasses should call ``self.generate()`` in ``__init__`` and
    populate ``self.X`` (list of samples) and ``self.y`` (list of labels).
    """

    def __init__(self):
        self.X = None
        self.y = None

    def generate(self):
        """Generate dataset — must set self.X and self.y."""
        raise NotImplementedError

    def __len__(self) -> int:
        if self.X is None:
            return 0
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        return self.X[idx], self.y[idx]

    def plot(self):
        """Visualize the dataset (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            X = np.array(self.X)
            y = np.array(self.y)
            if X.ndim == 1 or (X.ndim == 2 and X.shape[1] == 1):
                plt.scatter(range(len(X)), y, c=y, cmap='jet', s=20)
            else:
                plt.scatter(X[:, 0], X[:, 1], c=y, cmap='jet', s=20)
            plt.title(type(self).__name__)
            plt.show()
        except ImportError:
            print("matplotlib is required for plotting")
