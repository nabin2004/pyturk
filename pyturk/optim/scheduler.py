class LRScheduler:
    """
    Base class for learning rate schedulers.
    Expects an optimizer with a 'lr' attribute or dictionary of parameters.
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step(self):
        raise NotImplementedError("Scheduler must implement step()")


class StepLR(LRScheduler):
    """
    Step learning rate scheduler.
    Reduces learning rate by 'gamma' every 'step_size' calls to step().
    """
    def __init__(self, optimizer, step_size, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.last_step = 0

    def step(self):
        self.last_step += 1
        if self.last_step % self.step_size == 0:
            self.optimizer.lr *= self.gamma
