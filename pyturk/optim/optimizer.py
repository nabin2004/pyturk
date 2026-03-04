"""
Base Optimizer class for pyturk.

All optimizers inherit from this and implement ``step()``.
"""

from __future__ import annotations
from typing import List


class Optimizer:
    """
    Base class for all optimizers.

    Args:
        parameters: list of Parameters to optimize
        lr:         learning rate
    """

    def __init__(self, parameters: List, lr: float = 0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        """Perform a single optimization step (must be overridden)."""
        raise NotImplementedError

    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for p in self.parameters:
            p.grad = 0.0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(lr={self.lr}, params={len(self.parameters)})"
