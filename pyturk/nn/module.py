"""
Module base class for pyturk.

All neural network layers and models should subclass Module.
Provides automatic parameter discovery, gradient zeroing,
train/eval mode switching, and a clean forward/__call__ interface.
"""

from __future__ import annotations
from typing import List, Iterator, Tuple


class Module:
    """
    Base class for all neural network modules (mirrors torch.nn.Module).

    Subclasses should implement ``forward()``.  Parameters are automatically
    discovered by scanning instance attributes for Parameter and Module objects
    (including inside lists/tuples/dicts).

    Example::

        class MyModel(Module):
            def __init__(self):
                self.layer1 = Linear(2, 4)
                self.layer2 = Linear(4, 1)

            def forward(self, x):
                x = self.layer1(x)
                x = [v.relu() for v in x]
                return self.layer2(x)
    """

    training: bool = True

    # ------------------------------------------------------------------ #
    #  Forward / call                                                      #
    # ------------------------------------------------------------------ #

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------ #
    #  Parameter & module discovery                                        #
    # ------------------------------------------------------------------ #

    def parameters(self) -> List:
        """Recursively gather all Parameters from this module and its children."""
        from pyturk.nn.parameter import Parameter

        params: list = []
        visited: set = set()

        def _collect(obj):
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)

            if isinstance(obj, Parameter):
                params.append(obj)
            elif isinstance(obj, Module):
                for name, val in vars(obj).items():
                    if name.startswith('_'):
                        continue
                    _collect(val)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    _collect(item)

        _collect(self)
        return params

    def named_parameters(self, prefix: str = '') -> Iterator[Tuple[str, object]]:
        """Yield ``(name, Parameter)`` pairs for all parameters."""
        from pyturk.nn.parameter import Parameter

        def _yield_from(obj, name):
            if isinstance(obj, Parameter):
                yield name, obj
            elif isinstance(obj, Module):
                yield from obj.named_parameters(name)
            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    yield from _yield_from(item, f"{name}.{i}")

        for name, val in vars(self).items():
            if name.startswith('_'):
                continue
            full_name = f"{prefix}.{name}" if prefix else name
            yield from _yield_from(val, full_name)

    def children(self) -> Iterator[Module]:
        """Yield immediate child Modules."""
        for _name, val in vars(self).items():
            if isinstance(val, Module):
                yield val
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        yield item

    # ------------------------------------------------------------------ #
    #  Gradient utilities                                                  #
    # ------------------------------------------------------------------ #

    def zero_grad(self) -> None:
        """Set gradients of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0.0

    # ------------------------------------------------------------------ #
    #  Train / eval mode                                                   #
    # ------------------------------------------------------------------ #

    def train(self, mode: bool = True) -> Module:
        """Set training mode (affects dropout, batchnorm, etc.)."""
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> Module:
        """Set evaluation mode."""
        return self.train(False)

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        """Return total number of learnable parameters."""
        return len(self.parameters())

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}("]
        for name, val in vars(self).items():
            if name.startswith('_'):
                continue
            if isinstance(val, Module):
                lines.append(f"  ({name}): {val}")
            elif isinstance(val, (list, tuple)) and val and isinstance(val[0], Module):
                for i, item in enumerate(val):
                    lines.append(f"  ({name}.{i}): {item}")
        lines.append(")")
        return "\n".join(lines) if len(lines) > 2 else f"{type(self).__name__}()"
