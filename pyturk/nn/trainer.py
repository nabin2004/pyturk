from pyturk.nn.model_registry import MODEL_REGISTRY

class Trainer:
    """
    Generic trainer for pyturk models.
    Works with Value-based autograd.
    """

    def __init__(self, model_block, dataset, loss_fn, optimizer=None, lr=1.0, steps=100):
        self.model_block = model_block
        self.model = model_block.model
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.steps = steps

        # if no optimizer provided, default to simple SGD
        if self.optimizer is None:
            from pyturk.optim.sgd import SGD
            self.optimizer = SGD(self.model.parameters(), lr=self.lr)

    def zero_grad(self):
        for p in self.model.parameters():
            p.grad = type(p)(0.0)  # reset Value gradients

    def train(self, verbose=True):
        history = []

        for step in range(self.steps):
            # forward
            total_loss, acc = self.loss_fn(self.model, self.dataset.X, self.dataset.y)

            # backward
            self.zero_grad()
            total_loss.backward()

            # update parameters
            self.optimizer.step()

            # logging
            if verbose:
                loss_val = total_loss.data if hasattr(total_loss, 'data') else total_loss
                print(f"step {step} loss {loss_val}, accuracy {acc*100:.1f}%")

            history.append((loss_val, acc))

        return history
