"""Statistical utilities for pyturk."""

from __future__ import annotations
from typing import List, Union
from pyturk.autograd import Value


def mean(values: List[Union[Value, float]]) -> Value:
    """Compute the mean of a list of Values."""
    n = len(values)
    total = sum(v if isinstance(v, Value) else Value(v) for v in values)
    return total * (1.0 / n)


def variance(values: List[Union[Value, float]]) -> Value:
    """Compute the variance of a list of Values."""
    m = mean(values)
    n = len(values)
    return sum(
        (v - m) ** 2 if isinstance(v, Value) else (Value(v) - m) ** 2
        for v in values
    ) * (1.0 / n)
