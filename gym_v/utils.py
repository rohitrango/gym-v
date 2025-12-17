from __future__ import annotations

import numpy as np


def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Returns a NumPy random number generator (RNG) along with seed value from the inputted seed."""
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        if isinstance(seed, int) is False:
            raise ValueError(
                f"Seed must be a python integer, actual type: {type(seed)}"
            )
        else:
            raise ValueError(
                f"Seed must be greater or equal to zero, actual value: {seed}"
            )

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed
