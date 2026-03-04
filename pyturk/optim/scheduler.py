"""Learning rate schedulers for pyturk."""

from __future__ import annotations
import math


class LRScheduler:
    """Base class for learning rate schedulers."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
        self._step_count = 0

    def step(self):
        raise NotImplementedError

    def get_lr(self) -> float:
        return self.optimizer.lr


class StepLR(LRScheduler):
    """Reduce LR by ``gamma`` every ``step_size`` epochs."""

    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self):
        self._step_count += 1
        if self._step_count % self.step_size == 0:
            self.optimizer.lr *= self.gamma


class ExponentialLR(LRScheduler):
    """Multiply LR by ``gamma`` every epoch."""

    def __init__(self, optimizer, gamma: float = 0.95):
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self):
        self._step_count += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** self._step_count)


class CosineAnnealingLR(LRScheduler):
    """Cosine annealing schedule over ``T_max`` steps."""

    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = T_max
        self.eta_min = eta_min

    def step(self):
        self._step_count += 1
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * \
            (1 + math.cos(math.pi * self._step_count / self.T_max)) / 2
