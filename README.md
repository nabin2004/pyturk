# pyturk

`pyturk` is a minimal educational deep-learning framework inspired by micrograd and PyTorch. It is designed for clarity and teaching, not for production use.

Features

- Simple scalar autograd (`Value`)
- PyTorch-like `nn` modules: `Module`, `Linear`, `Sequential`, `MLP`
- Optimizers: `SGD`, `Adam`, `RMSProp` and LR schedulers
- Lightweight datasets and `DataLoader` for experiments
- Utilities: graph visualization, logging, and simple stats

Quick start

```python
from pyturk import Value
import pyturk.nn as nn
import pyturk.optim as optim

model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

optimizer = optim.SGD(model.parameters(), lr=1e-2)

# Single sample forward
x = [Value(1.0), Value(2.0)]
out = model(x)
loss = (out - Value(1.0)) ** 2

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

See `docs/usage.md` and `docs/api.md` for more examples and reference.
