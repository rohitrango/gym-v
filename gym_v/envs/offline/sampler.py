"""Index sampling strategies for offline environments."""

from __future__ import annotations

from collections import deque

from gym_v.utils import np_random


class IndexSampler:
    """Sample indices with optional shuffling.

    Args:
        size: Total number of indices (0 to size-1)
        shuffle: If True, shuffle indices each epoch; otherwise sequential
    """

    def __init__(self, size: int, shuffle: bool = True):
        self._size = size
        self._shuffle = shuffle
        self._rng = None
        self._indices: deque[int] = deque()

    def reset(self, seed: int | None = None) -> None:
        """Reset sampler with optional new seed."""
        if seed is not None or self._rng is None:
            self._rng, _ = np_random(seed)
            self._indices.clear()

    def __call__(self) -> int:
        if not self._indices:
            self._refill()
        return self._indices.popleft()

    def _refill(self) -> None:
        """Refill the index queue for a new epoch."""
        indices = list(range(self._size))
        if self._shuffle:
            self._rng.shuffle(indices)
        self._indices = deque(indices)
