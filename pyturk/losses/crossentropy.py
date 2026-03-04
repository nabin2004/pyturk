"""Cross-Entropy loss for pyturk."""

from __future__ import annotations
from typing import List

from pyturk.autograd import Value
from pyturk.nn.module import Module
from pyturk.nn.functional import softmax


class CrossEntropyLoss(Module):
    """
    Cross-Entropy loss for multi-class classification.

    Expects:
        logits: list of Value objects (one per class)
        target: integer index of the correct class
    """

    def forward(self, logits: List[Value], target: int) -> Value:
        probs = softmax(logits)
        return -probs[target].log()
