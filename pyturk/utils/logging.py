"""Simple metric logger for pyturk experiments."""

from __future__ import annotations
from typing import Dict, List, Any


class Logger:
    """Accumulate named scalar metrics during training."""

    def __init__(self):
        self.logs: Dict[str, List[float]] = {}

    def log(self, name: str, value: float) -> None:
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append(value)

    def summary(self, name: str) -> Dict[str, float]:
        vals = self.logs.get(name, [])
        if vals:
            return {
                'mean': sum(vals) / len(vals),
                'max': max(vals),
                'min': min(vals),
                'last': vals[-1],
                'count': len(vals),
            }
        return {}

    def reset(self, name: str = None) -> None:
        """Reset one metric or all metrics."""
        if name is not None:
            self.logs.pop(name, None)
        else:
            self.logs = {}

    def __repr__(self) -> str:
        return f"Logger(metrics={list(self.logs.keys())})"
