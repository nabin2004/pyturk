"""Adam optimizer for pyturk."""

from __future__ import annotations
import math
from pyturk.optim.optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2014).

    Args:
        parameters: iterable of parameters to optimize
        lr:         learning rate (default: 0.001)
        betas:      coefficients for running averages (default: (0.9, 0.999))
        eps:        numerical stability term (default: 1e-8)
    """

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = {id(p): 0.0 for p in self.parameters}  # first moment
        self.v = {id(p): 0.0 for p in self.parameters}  # second moment

    def step(self):
        self.t += 1
        for p in self.parameters:
            g = p.grad.data if hasattr(p.grad, 'data') else p.grad
            pid = id(p)
            self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * g
            self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (g ** 2)

            m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
            v_hat = self.v[pid] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
