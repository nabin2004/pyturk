"""
Generic trainer for pyturk models.
"""

from __future__ import annotations
from typing import Callable, List, Tuple


class Trainer:
    """
    Generic trainer for pyturk models.

    Args:
        model_block: a ModelBlock (or any object with a .model Module), or a Module directly
        dataset:     a Dataset with .X and .y attributes
        loss_fn:     callable(model, X, y) -> (loss_value, accuracy)
        optimizer:   an Optimizer instance (defaults to SGD)
        lr:          learning rate (used if optimizer is None)
        steps:       number of training steps
    """

    def __init__(
        self,
        model_block,
        dataset,
        loss_fn: Callable,
        optimizer=None,
        lr: float = 0.01,
        steps: int = 100,
    ):
        self.model_block = model_block
        self.model = model_block.model if hasattr(model_block, 'model') else model_block
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.lr = lr
        self.steps = steps

        if optimizer is None:
            from pyturk.optim.sgd import SGD
            self.optimizer = SGD(self.model.parameters(), lr=self.lr)
        else:
            self.optimizer = optimizer

    def train(self, verbose: bool = True) -> List[Tuple[float, float]]:
        """Run the training loop.  Returns list of (loss, accuracy) per step."""
        history: List[Tuple[float, float]] = []

        for step in range(self.steps):
            # Forward pass
            total_loss, acc = self.loss_fn(self.model, self.dataset.X, self.dataset.y)

            # Zero gradients, then backward pass
            self.optimizer.zero_grad()
            total_loss.backward()

            # Update parameters
            self.optimizer.step()

            # Record
            loss_val = total_loss.data if hasattr(total_loss, 'data') else total_loss
            if verbose:
                print(f"step {step:>4d} | loss {loss_val:.6f} | accuracy {acc * 100:.1f}%")
            history.append((loss_val, acc))

        return history
