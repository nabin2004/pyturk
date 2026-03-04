"""
Parameter class for pyturk.

A Parameter is a Value that is recognized as a learnable parameter
by Module.parameters(). This mirrors torch.nn.Parameter.
"""

from __future__ import annotations
import random
from pyturk.autograd import Value


class Parameter(Value):
    """
    A Value that represents a learnable parameter.

    Modules use Parameter to distinguish learnable weights/biases
    from intermediate computation Values.

    Example:
        >>> p = Parameter(0.5, label='weight')
        >>> p
        Parameter(data=0.5000, grad=0.0000)
    """

    def __init__(self, data: float = 0.0, label: str = ''):
        super().__init__(data, label=label)

    def __repr__(self) -> str:
        return f"Parameter(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- Convenient factory methods ---

    @staticmethod
    def uniform(low: float = -1.0, high: float = 1.0, label: str = '') -> Parameter:
        """Create a parameter initialized from U(low, high)."""
        return Parameter(random.uniform(low, high), label=label)

    @staticmethod
    def zeros(label: str = '') -> Parameter:
        """Create a zero-initialized parameter."""
        return Parameter(0.0, label=label)
