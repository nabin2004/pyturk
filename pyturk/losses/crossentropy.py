import math
from pyturk.value import Value

class CrossEntropy:
    """
    CrossEntropy loss for multi-class classification.
    Expects:
    - logits: list of Value objects, one per class
    - target: integer index of the correct class
    """

    def __call__(self, logits, target):
        # Compute softmax
        max_logit = max(logits, key=lambda x: x.data)  # for numerical stability
        exps = [ (l - max_logit).exp() for l in logits ]  # e^(logit - max)
        sum_exps = sum(exps)
        softmax = [ e / sum_exps for e in exps ]

        # Cross-entropy loss: -log(p_target)
        loss = -(softmax[target].log())
        return loss
