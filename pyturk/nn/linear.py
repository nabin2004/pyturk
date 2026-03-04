"""
Linear (fully connected) layer for pyturk.
"""

from __future__ import annotations
import random
from typing import List, Union

from pyturk.autograd import Value
from pyturk.nn.module import Module
from pyturk.nn.parameter import Parameter


class Linear(Module):
    """
    Applies a linear transformation: y = xW^T + b

    Args:
        in_features:  size of each input sample
        out_features: size of each output sample
        bias:         if True, adds a learnable bias (default: True)

    Example:
        >>> layer = Linear(3, 2)
        >>> x = [Value(1.0), Value(2.0), Value(3.0)]
        >>> out = layer(x)  # returns list of 2 Values
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        # Kaiming-uniform-like initialization
        bound = (1.0 / in_features) ** 0.5
        self.weight = [
            [Parameter.uniform(-bound, bound, label=f'w{i}_{j}')
             for j in range(in_features)]
            for i in range(out_features)
        ]
        self.bias = [
            Parameter.uniform(-bound, bound, label=f'b{i}')
            for i in range(out_features)
        ] if bias else None

    def forward(self, x: List[Union[Value, float]]) -> Union[Value, List[Value]]:
        # Wrap raw numbers as Values
        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]

        outs = []
        for i in range(self.out_features):
            act = sum(
                (w * xi for w, xi in zip(self.weight[i], x)),
                Value(0.0),
            )
            if self.bias is not None:
                act = act + self.bias[i]
            outs.append(act)
        return outs[0] if len(outs) == 1 else outs

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )
