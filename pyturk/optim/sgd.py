# pyturk/optim/sgd.py
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            grad = p.grad.data if hasattr(p.grad, 'data') else p.grad
            p.data -= self.lr * grad
