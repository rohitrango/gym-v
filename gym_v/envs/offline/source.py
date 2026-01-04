"""Offline data sources."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

import jsonlines
from PIL import Image
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .utils import decode_base64_image


class OfflineSample(BaseModel):
    """A single-turn offline sample.

    Expected semantics:
    - text: textual prompt/context (optional but typical)
    - image: PIL Image object (optional for text-only tasks)
    - answer: oracle / ground-truth answer (optional for some tasks)
    - metadata: arbitrary extra information
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    text: str | None = None
    image: Image.Image | None = None
    answer: str | None = None
    metadata: dict[str, Any] | None = None

    @field_validator("image", mode="before")
    @classmethod
    def _parse_image(cls, v: Any) -> Image.Image | None:
        """Parse image from base64 string or pass through PIL Image."""
        if v is None:
            return None
        if isinstance(v, Image.Image):
            return v
        if isinstance(v, str):
            return decode_base64_image(v)
        raise TypeError(f"expected str or PIL.Image, got {type(v).__name__}")

    @model_validator(mode="after")
    def _require_content(self) -> OfflineSample:
        """Validate that sample has at least text or image."""
        if self.text is None and self.image is None:
            raise ValueError("sample must contain at least 'text' or 'image'")
        return self


class DataSource(Protocol):
    """A random-access collection of offline samples."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> OfflineSample: ...

    def __iter__(self) -> Iterator[OfflineSample]: ...


class JsonlDataSource:
    """Loads newline-delimited JSON examples into memory.

    JSONL schema (per line):
    - text: str (optional)
    - image: base64 str (optional)
    - answer: str (optional)
    - metadata: dict (optional)
    """

    def __init__(self, data_path: str | Path):
        self._data_path = Path(data_path)
        self._samples: list[OfflineSample] = []

        with jsonlines.open(self._data_path, "r") as reader:
            for line in reader:
                sample = OfflineSample.model_validate(line)
                self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> OfflineSample:
        return self._samples[index]

    def __iter__(self) -> Iterator[OfflineSample]:
        return iter(self._samples)


DATASOURCE_REGISTRY: dict[str, type[DataSource]] = {
    "jsonl": JsonlDataSource,
}
