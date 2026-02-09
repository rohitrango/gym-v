from __future__ import annotations

from PIL import Image


def select_indices(indices, options, rng):
    if options:
        if "indices" in options:
            indices = list(options["indices"])
        elif "max_samples" in options:
            max_samples = int(options["max_samples"])
            indices = indices[:max_samples]
        if options.get("shuffle"):
            indices = list(indices)
            rng.shuffle(indices)
    return indices