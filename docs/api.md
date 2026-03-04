# API Reference (brief)

## `pyturk.autograd.Value`

Scalar value with autograd. Key methods:
- `data`, `grad`
- arithmetic: `+ - * / **`
- activations: `relu(), tanh(), sigmoid(), exp(), log()`
- `backward()` — compute gradients


## `pyturk.nn`

- `Module` — base class for models and layers
- `Parameter` — a learnable parameter (subclass of `Value`)
- `Linear(in_features, out_features)` — fully connected layer
- `Sequential(*layers)` — compose modules
- `MLP(nin, nouts, activation)` — convenience MLP
- `ReLU()`, `Tanh()`, `Sigmoid()` — activations


## `pyturk.optim`

- `Optimizer` — base class
- `SGD(parameters, lr, momentum=0.0)`
- `Adam(parameters, lr, betas=(0.9,0.999))`
- `RMSProp(parameters, lr)`
- schedulers: `StepLR`, `ExponentialLR`, `CosineAnnealingLR`


## `pyturk.data`

- `Dataset` — base dataset
- `DataLoader(dataset, batch_size, shuffle)` — yields batches
- small datasets: `XORDataset`, `MoonsDataset`, `CirclesDataset`, `BlobsDataset`, `RegressionDataset`


## `pyturk.losses`

- `MSELoss()`, `CrossEntropyLoss()`


## Utilities

- `pyturk.utils.draw_dot(value)` — graphviz visualization
- `pyturk.utils.Logger` — basic metric logger
