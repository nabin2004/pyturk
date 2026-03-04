"""Mean Squared Error loss for pyturk."""

from __future__ import annotations
from typing import List, Union

from pyturk.autograd import Value
from pyturk.nn.module import Module


class MSELoss(Module):
    """
    Mean Squared Error Loss.

    ``loss = (1/n) * sum((pred_i - target_i)^2)``
    """

    def forward(
        self,
        predictions: List[Value],
        targets: List[Union[Value, float]],
    ) -> Value:
        n = len(predictions)
        assert n == len(targets), "predictions and targets must have same length"
        loss = sum(
            (pred - (t if isinstance(t, Value) else Value(t))) ** 2
            for pred, t in zip(predictions, targets)
        )
        return loss * (1.0 / n)
