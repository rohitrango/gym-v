"""
Shared image utilities for gym_v (Client-side).
Should NOT depend on torch/cuda.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class _WritableImage(Image.Image):
    """PIL Image subclass whose ``__array_interface__`` points to a writable ndarray.

    Modern Pillow (>=10) returns ``data`` as immutable bytes in the standard
    ``__array_interface__``, so ``np.asarray(pil_image)`` is always read-only.
    That, in turn, makes ``torch.from_numpy`` in downstream VLM processors emit
    a non-writable-tensor ``UserWarning``. This subclass overrides the
    interface to expose a lazily-materialized writable buffer while preserving
    every PIL operation (mode/size/convert/resize/etc.). Laziness matters for
    Ray/pickle: after deserialization the buffer is re-materialized on first
    array access instead of being dropped.
    """

    _writable_arr: np.ndarray | None = None

    @property
    def __array_interface__(self) -> dict:  # type: ignore[override]
        if self._writable_arr is None:
            # Materialize once via the parent's (read-only) interface;
            # np.array with copy=True yields a fresh writable buffer.
            base = np.array(super().__array_interface__["data"], copy=False)
            shape = super().__array_interface__["shape"]
            typestr = super().__array_interface__["typestr"]
            arr = np.frombuffer(bytes(base), dtype=np.dtype(typestr)).reshape(shape).copy()
            self._writable_arr = arr
        return self._writable_arr.__array_interface__


def _wrap_writable_pil(img: Image.Image) -> Image.Image:
    """Return a PIL image whose ``np.asarray`` view is writable."""
    wrapped = _WritableImage()
    wrapped.__dict__.update(img.__dict__)
    # Eagerly materialize; subsequent pickling drops it and lazy path re-fills.
    wrapped._writable_arr = np.array(img)
    return wrapped


def ensure_writable_image(img: Any) -> Any:
    """Return an image whose backing buffer is writable.

    Downstream VLM processors call ``torch.from_numpy(np.asarray(img))``; when
    ``img`` is a PIL Image lazily loaded from a PNG buffer, or a NumPy array
    that came out of ``matplotlib``'s canvas / Ray's object store / any
    ``np.frombuffer`` path, the buffer is read-only and PyTorch emits a
    ``UserWarning: The given NumPy array is not writable``.

    Idempotent: already-writable inputs are returned unchanged.
    """
    if img is None:
        return None
    if isinstance(img, list | tuple):
        return type(img)(ensure_writable_image(x) for x in img)
    if isinstance(img, np.ndarray):
        if img.flags.writeable and img.flags.owndata:
            return img
        return np.array(img)
    if isinstance(img, _WritableImage):
        return img
    if isinstance(img, Image.Image):
        return _wrap_writable_pil(img)
    return img


def _to_uint8(array):
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError("Images must be 2D or 3D arrays.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(arr.max()) if arr.size else 1.0
            min_val = float(arr.min()) if arr.size else 0.0
            if max_val <= 1.0 and min_val >= 0.0:
                arr = (arr * 255.0).round()
            else:
                arr = np.clip(arr, 0.0, 255.0).round()
        else:
            arr = np.clip(arr, 0, 255)
        arr = arr.astype(np.uint8)
    return arr


def to_pil_list(images) -> list[Image.Image]:
    """Convert inputs to a list of PIL Images."""
    if isinstance(images, Image.Image):
        return [images if images.mode == "RGB" else images.convert("RGB")]
    if isinstance(images, str | Path):
        with Image.open(images) as img:
            return [img.convert("RGB")]
    if isinstance(images, bytes | bytearray | memoryview):
        with Image.open(io.BytesIO(images)) as img:
            return [img.convert("RGB")]
    if isinstance(images, list | tuple):
        out = []
        for img in images:
            if isinstance(img, Image.Image):
                out.append(img if img.mode == "RGB" else img.convert("RGB"))
            elif isinstance(img, str | Path):
                with Image.open(img) as pil:
                    out.append(pil.convert("RGB"))
            elif isinstance(img, bytes | bytearray | memoryview):
                with Image.open(io.BytesIO(img)) as pil:
                    out.append(pil.convert("RGB"))
            elif isinstance(img, np.ndarray):
                out.append(Image.fromarray(_to_uint8(img)).convert("RGB"))
            # Removed torch support
            else:
                try:
                    # Fallback check for torch tensors without importing torch
                    type_str = str(type(img))
                    if "torch" in type_str and "Tensor" in type_str:
                        # We can't handle it here if we want to be torch-free
                        raise TypeError(
                            "gym_v client does not support torch tensors. Please convert to numpy/PIL."
                        )
                except Exception:
                    pass
                raise TypeError(f"Unsupported element type in list: {type(img)}")
        return out
    if isinstance(images, np.ndarray):
        arr = images
        if arr.ndim == 3:
            arr = arr[None, ...]
        return [Image.fromarray(_to_uint8(frame)).convert("RGB") for frame in arr]

    # Removed torch check
    type_str = str(type(images))
    if "torch" in type_str and "Tensor" in type_str:
        raise TypeError(
            "gym_v client does not support torch tensors. Please convert to numpy/PIL."
        )

    raise TypeError(f"Unsupported image type: {type(images)}")
