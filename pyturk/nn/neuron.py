import random
from pyturk.value import Value

class Neuron:

  def __init__(self, nin):
    self.w = [Value(random.uniform(-1,1), label='W') for _ in range(nin)]
    self.b = Value(random.uniform(-1,1), label='b')

  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.tanh()
    return out

  def parameters(self):
    return self.w + [self.b]
