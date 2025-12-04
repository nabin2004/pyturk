# pyturk/losses/mse.py
from pyturk.core.value import Value

class MSE:
    """
    Mean Squared Error Loss
    Expects:
    - predictions: list of Value objects
    - targets: list of numbers (float/int)
    """

    def __call__(self, predictions, targets):
        assert len(predictions) == len(targets), "predictions and targets must be same length"
        loss = sum((pred - Value(t) )**2 for pred, t in zip(predictions, targets))
        return loss / len(predictions)
