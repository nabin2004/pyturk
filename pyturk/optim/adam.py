# pyturk/optim/adam.py
import math

class Adam:
    """
    Adam optimizer for PyTurk Value parameters.
    """

    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # Initialize moving averages
        self.m = {p: 0 for p in self.parameters}  # first moment
        self.v = {p: 0 for p in self.parameters}  # second moment

    def step(self):
        self.t += 1
        for p in self.parameters:
            g = p.grad
            # Update biased first and second moment estimates
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * g
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (g * g)

            # Bias-corrected moments
            m_hat = self.m[p] / (1 - self.beta1**self.t)
            v_hat = self.v[p] / (1 - self.beta2**self.t)

            # Parameter update
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0
