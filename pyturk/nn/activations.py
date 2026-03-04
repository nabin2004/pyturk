"""
Activation function modules for pyturk.

Each activation is a Module that can be used standalone or composed
inside a Sequential container — just like ``torch.nn.ReLU()``.
"""

from __future__ import annotations
from typing import List, Union

from pyturk.autograd import Value
from pyturk.nn.module import Module


class ReLU(Module):
    """Applies ReLU element-wise: max(0, x)"""

    def forward(self, x: Union[Value, List[Value]]) -> Union[Value, List[Value]]:
        if isinstance(x, (list, tuple)):
            return [v.relu() for v in x]
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    """Applies Tanh element-wise."""

    def forward(self, x: Union[Value, List[Value]]) -> Union[Value, List[Value]]:
        if isinstance(x, (list, tuple)):
            return [v.tanh() for v in x]
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class Sigmoid(Module):
    """Applies Sigmoid element-wise: 1 / (1 + exp(-x))"""

    def forward(self, x: Union[Value, List[Value]]) -> Union[Value, List[Value]]:
        if isinstance(x, (list, tuple)):
            return [v.sigmoid() for v in x]
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"
