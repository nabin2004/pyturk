"""
pyturk — A mini deep learning framework for learning and experimentation.

Built on a scalar autograd engine (inspired by micrograd), with a
PyTorch-like API for neural networks, optimizers, and data loading.

Quick start::

    from pyturk import Value
    import pyturk.nn as nn
    import pyturk.optim as optim

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Forward
    out = model([Value(1.0), Value(2.0)])
    loss = (out - Value(1.0)) ** 2

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""

from pyturk.autograd import Value
from pyturk.nn.parameter import Parameter

__version__ = "0.1.0"
__all__ = ["Value", "Parameter", "nn", "optim", "data", "losses", "utils"]
