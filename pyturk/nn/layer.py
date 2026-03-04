"""
A layer of Neurons (micrograd-compatible).

Kept for backward compatibility and educational purposes.
For production-style layers, use ``Linear`` instead.
"""

from __future__ import annotations
from pyturk.nn.module import Module
from pyturk.nn.neuron import Neuron


class Layer(Module):
    """A layer of parallel neurons."""

    def __init__(self, nin: int, nout: int, activation: str = 'tanh'):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def forward(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def __repr__(self) -> str:
        nin = len(self.neurons[0].w) if self.neurons else 0
        return f"Layer(nin={nin}, nout={len(self.neurons)})"
