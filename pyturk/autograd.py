"""
Autograd engine for pyturk.

Implements reverse-mode automatic differentiation on scalar values.
This is the foundation of the framework — all neural network operations
flow through Value's computational graph.

Example:
    >>> a = Value(2.0, label='a')
    >>> b = Value(3.0, label='b')
    >>> c = a * b + a
    >>> c.backward()
    >>> a.grad  # dc/da = b + 1 = 4.0
    4.0
"""

from __future__ import annotations
import math
from typing import Union, Tuple, Set, List

Numeric = Union[int, float]


class Value:
    """
    A scalar value that tracks its computational graph for automatic differentiation.

    Supports arithmetic operations (+, -, *, /, **) and common math functions
    (exp, log, tanh, sigmoid, relu). Calling .backward() computes gradients
    for all ancestor Values via reverse-mode autodiff.
    """

    __slots__ = ('data', 'grad', '_backward', '_prev', '_op', 'label')

    def __init__(
        self,
        data: Numeric,
        _children: Tuple['Value', ...] = (),
        _op: str = '',
        label: str = '',
    ):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev: Set[Value] = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------------ #
    #  Arithmetic operators                                                #
    # ------------------------------------------------------------------ #

    def __add__(self, other: Union[Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Numeric) -> Value:
        return self + other

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Union[Value, Numeric]) -> Value:
        return self + (-other)

    def __rsub__(self, other: Numeric) -> Value:
        return Value(other) + (-self)

    def __mul__(self, other: Union[Value, Numeric]) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Numeric) -> Value:
        return self * other

    def __truediv__(self, other: Union[Value, Numeric]) -> Value:
        return self * other ** -1

    def __rtruediv__(self, other: Numeric) -> Value:
        return Value(other) * self ** -1

    def __pow__(self, other: Union[int, float]) -> Value:
        assert isinstance(other, (int, float)), "only supporting int/float powers"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Comparison operators (non-differentiable, for control flow)         #
    # ------------------------------------------------------------------ #

    def __gt__(self, other: Union[Value, Numeric]) -> bool:
        return self.data > (other.data if isinstance(other, Value) else other)

    def __lt__(self, other: Union[Value, Numeric]) -> bool:
        return self.data < (other.data if isinstance(other, Value) else other)

    def __ge__(self, other: Union[Value, Numeric]) -> bool:
        return self.data >= (other.data if isinstance(other, Value) else other)

    def __le__(self, other: Union[Value, Numeric]) -> bool:
        return self.data <= (other.data if isinstance(other, Value) else other)

    # NOTE: __eq__ is intentionally NOT overridden so that Value objects
    # remain hashable via identity (id-based), which is required for use
    # in sets and as dict keys throughout the autograd graph.

    # ------------------------------------------------------------------ #
    #  Activation / math functions                                         #
    # ------------------------------------------------------------------ #

    def relu(self) -> Value:
        """Rectified Linear Unit: max(0, x)"""
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        """Hyperbolic tangent."""
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        """Logistic sigmoid: 1 / (1 + exp(-x))"""
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        """Exponential function."""
        out = Value(math.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        """Natural logarithm (requires data > 0)."""
        assert self.data > 0, f"log undefined for data={self.data}"
        out = Value(math.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def abs(self) -> Value:
        """Absolute value."""
        out = Value(abs(self.data), (self,), 'abs')

        def _backward():
            self.grad += (1.0 if self.data >= 0 else -1.0) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    #  Autograd                                                            #
    # ------------------------------------------------------------------ #

    def backward(self) -> None:
        """Compute gradients via reverse-mode autodiff (backpropagation).

        Must be called on a scalar output (the loss). Populates .grad for
        every Value that is an ancestor in the computational graph.
        """
        # Topological sort
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Backward pass
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
