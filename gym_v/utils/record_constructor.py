"""Allows attributes passed to `RecordConstructorArgs` to be saved. This is used by the `Wrapper.spec` to know the constructor arguments of implemented wrappers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class RecordConstructorArgs:
    """Records all arguments passed to constructor to `_saved_kwargs`."""

    def __init__(self, *, _disable_deepcopy: bool = False, **kwargs: Any):
        """Records all arguments passed to constructor to `_saved_kwargs`."""
        # See class docstring for explanation
        if not hasattr(self, "_saved_kwargs"):
            if _disable_deepcopy is False:
                kwargs = deepcopy(kwargs)
            self._saved_kwargs: dict[str, Any] = kwargs
