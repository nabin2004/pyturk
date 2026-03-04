"""
Multi-Layer Perceptron for pyturk.
"""

from __future__ import annotations
from typing import List, Union

from pyturk.autograd import Value
from pyturk.nn.module import Module
from pyturk.nn.linear import Linear
from pyturk.nn.activations import ReLU, Tanh, Sigmoid
from pyturk.nn.sequential import Sequential


_ACTIVATIONS = {
    'relu': ReLU,
    'tanh': Tanh,
    'sigmoid': Sigmoid,
    'none': None,
}


class MLP(Module):
    """
    Multi-Layer Perceptron built from Linear layers and activations.

    Args:
        nin:        number of input features
        nouts:      list of layer output sizes (e.g. [16, 16, 1])
        activation: activation between hidden layers ('relu', 'tanh', 'sigmoid', 'none')

    Example::

        >>> model = MLP(2, [16, 16, 1])
        >>> out = model([Value(1.0), Value(2.0)])
    """

    def __init__(self, nin: int, nouts: List[int], activation: str = 'tanh'):
        sizes = [nin] + nouts
        act_cls = _ACTIVATIONS.get(activation.lower(), Tanh)

        layers = []
        for i in range(len(nouts)):
            layers.append(Linear(sizes[i], sizes[i + 1]))
            # Activation after every layer except the last
            if act_cls is not None and i < len(nouts) - 1:
                layers.append(act_cls())

        self.net = Sequential(*layers)

    def forward(self, x: Union[List, Value]) -> Union[Value, List[Value]]:
        return self.net(x)

    def __repr__(self) -> str:
        return f"MLP(\n  {self.net}\n)"
