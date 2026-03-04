"""
Functional API for pyturk.nn

Stateless functions for activations, loss computation, and common
operations — mirrors ``torch.nn.functional``.
"""

from __future__ import annotations
from typing import List, Union

from pyturk.autograd import Value


# ------------------------------------------------------------------ #
#  Activations                                                         #
# ------------------------------------------------------------------ #

def relu(x: Union[Value, List[Value]]) -> Union[Value, List[Value]]:
    """Apply ReLU element-wise."""
    if isinstance(x, (list, tuple)):
        return [v.relu() for v in x]
    return x.relu()


def tanh(x: Union[Value, List[Value]]) -> Union[Value, List[Value]]:
    """Apply tanh element-wise."""
    if isinstance(x, (list, tuple)):
        return [v.tanh() for v in x]
    return x.tanh()


def sigmoid(x: Union[Value, List[Value]]) -> Union[Value, List[Value]]:
    """Apply sigmoid element-wise."""
    if isinstance(x, (list, tuple)):
        return [v.sigmoid() for v in x]
    return x.sigmoid()


def softmax(logits: List[Value]) -> List[Value]:
    """Numerically stable softmax over a list of Values."""
    max_val = max(v.data for v in logits)
    exps = [(v - Value(max_val)).exp() for v in logits]
    total = sum(exps, Value(0.0))
    return [e / total for e in exps]


# ------------------------------------------------------------------ #
#  Loss functions                                                      #
# ------------------------------------------------------------------ #

def mse_loss(
    predictions: List[Value],
    targets: List[Union[Value, float]],
) -> Value:
    """Mean Squared Error loss."""
    n = len(predictions)
    assert n == len(targets), "predictions and targets must have same length"
    loss = sum(
        (pred - (t if isinstance(t, Value) else Value(t))) ** 2
        for pred, t in zip(predictions, targets)
    )
    return loss * (1.0 / n)


def cross_entropy_loss(logits: List[Value], target: int) -> Value:
    """Cross-entropy loss for single-label classification."""
    probs = softmax(logits)
    return -probs[target].log()


def hinge_loss(
    predictions: List[Value],
    targets: List[Union[Value, float]],
    margin: float = 1.0,
) -> Value:
    """SVM max-margin hinge loss."""
    n = len(predictions)
    losses = []
    for pred, t in zip(predictions, targets):
        t_val = t.data if isinstance(t, Value) else t
        loss_i = (Value(margin) - pred * Value(t_val)).relu()
        losses.append(loss_i)
    return sum(losses) * (1.0 / n)
