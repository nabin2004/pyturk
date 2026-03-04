"""Data transforms (tokenization, padding, etc.)."""

from __future__ import annotations
from typing import List, Dict


class TextToIds:
    """Convert text to a list of token IDs using a vocabulary dict."""

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab

    def __call__(self, text: str) -> List[int]:
        return [self.vocab.get(tok, 0) for tok in text.split()]


class PadSequence:
    """Pad or truncate a sequence to a fixed length."""

    def __init__(self, max_length: int, pad_id: int = 0):
        self.max_length = max_length
        self.pad_id = pad_id

    def __call__(self, seq: List[int]) -> List[int]:
        if len(seq) >= self.max_length:
            return seq[: self.max_length]
        return seq + [self.pad_id] * (self.max_length - len(seq))
