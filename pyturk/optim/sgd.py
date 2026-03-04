"""SGD optimizer for pyturk (with optional momentum)."""

from __future__ import annotations
from pyturk.optim.optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent.

    Args:
        parameters: iterable of parameters to optimize
        lr:         learning rate (default: 0.01)
        momentum:   momentum factor (default: 0)
    """

    def __init__(self, parameters, lr: float = 0.01, momentum: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self._velocity = {id(p): 0.0 for p in self.parameters}

    def step(self):
        for p in self.parameters:
            grad = p.grad.data if hasattr(p.grad, 'data') else p.grad
            pid = id(p)
            if self.momentum > 0:
                self._velocity[pid] = self.momentum * self._velocity[pid] + grad
                p.data -= self.lr * self._velocity[pid]
            else:
                p.data -= self.lr * grad
