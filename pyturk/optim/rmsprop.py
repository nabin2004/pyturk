"""RMSProp optimizer for pyturk."""

from __future__ import annotations
import math
from pyturk.optim.optimizer import Optimizer


class RMSProp(Optimizer):
    """
    RMSProp optimizer (Hinton, 2012).

    Args:
        parameters: iterable of parameters to optimize
        lr:         learning rate (default: 0.001)
        alpha:      smoothing constant / decay rate (default: 0.9)
        eps:        numerical stability term (default: 1e-8)
    """

    def __init__(self, parameters, lr=0.001, alpha=0.9, eps=1e-8):
        super().__init__(parameters, lr)
        self.alpha = alpha
        self.eps = eps
        self.v = {id(p): 0.0 for p in self.parameters}

    def step(self):
        for p in self.parameters:
            g = p.grad.data if hasattr(p.grad, 'data') else p.grad
            pid = id(p)
            self.v[pid] = self.alpha * self.v[pid] + (1 - self.alpha) * (g ** 2)
            p.data -= self.lr * g / (math.sqrt(self.v[pid]) + self.eps)
