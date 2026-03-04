# Changelog

## 0.1.0 - (unreleased)

- Refactor: new modular API with `Module`, `Parameter`, `Linear`, `Sequential`.
- Autograd: `Value` is standalone and supports additional ops (`log`, `sigmoid`, `abs`, r-ops).
- Optimizers: refactored `SGD`, added `Optimizer` base, `Adam`, `RMSProp`, schedulers.
- Data: removed `torch` dependency; datasets return plain Python lists.
- Documentation: added `README`, `docs/usage.md`, `docs/api.md` and example `examples/train_xor.py`.
