"""
pyturk.nn — Neural network modules.

Provides Module base class, layers, activations, and containers
following PyTorch-like API conventions.

Quick reference::

    import pyturk.nn as nn

    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
"""

from pyturk.nn.module import Module
from pyturk.nn.parameter import Parameter
from pyturk.nn.linear import Linear
from pyturk.nn.sequential import Sequential
from pyturk.nn.activations import ReLU, Tanh, Sigmoid
from pyturk.nn.mlp import MLP
from pyturk.nn.neuron import Neuron
from pyturk.nn.layer import Layer

__all__ = [
    'Module', 'Parameter', 'Linear', 'Sequential',
    'ReLU', 'Tanh', 'Sigmoid', 'MLP',
    'Neuron', 'Layer',
]
