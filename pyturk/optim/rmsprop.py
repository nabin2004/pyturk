import math

class RMSProp:
    """
    RMSProp optimizer for PyTurk Value parameters.
    """

    def __init__(self, parameters, lr=0.001, gamma=0.9, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.gamma = gamma  # decay rate for moving average of squared gradients
        self.eps = eps

        # Initialize squared gradient moving averages
        self.s = {p: 0 for p in self.parameters}

    def step(self):
        for p in self.parameters:
            g = p.grad
            # Update moving average of squared gradients
            self.s[p] = self.gamma * self.s[p] + (1 - self.gamma) * (g * g)

            # Update parameters
            p.data -= self.lr * g / (math.sqrt(self.s[p]) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
