# Usage

This document shows simple examples to get started with `pyturk`.

## Training XOR (small example)

See `examples/train_xor.py` for a runnable script.

### Quick training loop (in-memory)

```python
from pyturk import Value
import pyturk.nn as nn
import pyturk.optim as optim
from pyturk.data import XORDataset

# Dataset
 ds = XORDataset()

# Model
 model = nn.MLP(2, [8, 8, 1], activation='tanh')
 optimizer = optim.Adam(model.parameters(), lr=0.05)

# Simple epoch loop
for epoch in range(50):
    total_loss = Value(0.0)
    for x, y in zip(ds.X, ds.y):
        pred = model(x)
        loss = (pred - Value(y)) ** 2
        total_loss = total_loss + loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

print('Training complete')
```

## Building models

Use `nn.Sequential` to compose layers or create a custom `Module` subclass and implement `forward()`.

## Visualizing computation graphs

Use `pyturk.utils.draw_dot(value)` to get a Graphviz `Digraph` representing the graph. Render with `dot.render()` or view in notebooks.
