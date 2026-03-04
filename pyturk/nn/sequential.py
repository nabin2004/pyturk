"""
Sequential container for pyturk.

Passes input through a chain of modules in order,
just like ``torch.nn.Sequential``.
"""

from __future__ import annotations
from pyturk.nn.module import Module


class Sequential(Module):
    """
    A sequential container — modules are applied in the order they are passed.

    Example::

        model = Sequential(
            Linear(2, 4),
            ReLU(),
            Linear(4, 1),
            Tanh(),
        )
        out = model([Value(1.0), Value(2.0)])
    """

    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def add(self, layer: Module) -> Sequential:
        """Append a layer to the sequence."""
        self.layers.append(layer)
        return self

    def __getitem__(self, idx):
        return self.layers[idx]

    def __len__(self):
        return len(self.layers)

    def __repr__(self) -> str:
        lines = ["Sequential("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)
