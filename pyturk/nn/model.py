"""
Model wrapper for pyturk.

Provides a configurable ModelBlock that wraps an MLP with named
architecture settings — useful for registries and experimentation.
"""

from __future__ import annotations
from pyturk.nn.mlp import MLP


class ModelBlock:
    """
    A block that wraps a neural network with configurable parameters.
    """

    def __init__(
        self,
        name: str,
        input_size: int = 2,
        hidden_layers: list = None,
        output_size: int = 1,
        activation: str = 'tanh',
    ):
        self.name = name
        self.input_size = input_size
        self.hidden_layers = hidden_layers or [5]
        self.output_size = output_size
        self.activation = activation
        self.model = self._build()

    def _build(self) -> MLP:
        """Create a fresh MLP with current parameters."""
        return MLP(
            nin=self.input_size,
            nouts=self.hidden_layers + [self.output_size],
            activation=self.activation,
        )

    def reset(self, **kwargs) -> None:
        """Rebuild the model with updated configuration."""
        for key, val in kwargs.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)
        self.model = self._build()

    def __repr__(self) -> str:
        return (
            f"ModelBlock(name='{self.name}', "
            f"arch=[{self.input_size}, {self.hidden_layers}, {self.output_size}])"
        )
