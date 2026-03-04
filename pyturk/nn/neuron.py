"""
Single Neuron module (micrograd-compatible).

Kept for backward compatibility and educational purposes.
For production-style layers, use ``Linear`` instead.
"""

from __future__ import annotations
from pyturk.autograd import Value
from pyturk.nn.module import Module
from pyturk.nn.parameter import Parameter


class Neuron(Module):
    """A single neuron with configurable activation."""

    def __init__(self, nin: int, activation: str = 'tanh'):
        self.w = [Parameter.uniform(-1, 1, label=f'w{i}') for i in range(nin)]
        self.b = Parameter.uniform(-1, 1, label='b')
        self.activation = activation

    def forward(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        return act  # linear / no activation

    def __repr__(self) -> str:
        return f"Neuron(nin={len(self.w)}, act={self.activation})"
