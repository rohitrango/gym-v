"""Offline data sources."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import os
from pathlib import Path
from typing import Any, Protocol

from PIL import Image
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from gym_v.envs.offline.utils import decode_base64_image, fast_json_loads


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
    """Lazy-loading JSONL data source with O(1) random access.

    Builds byte offset index in memory, enabling direct seek to any line
    without loading entire file contents.

    Args:
        data_path: Path to JSONL file
        num_workers: Number of parallel workers for index building
    """

    def __init__(
        self,
        data_path: str | Path,
        num_workers: int | None = None,
    ):
        self._data_path = Path(data_path)
        self._num_workers = num_workers or os.cpu_count() or 1

        self._offsets = self._build_index()
        self._file = open(self._data_path)

    def _scan_chunk(self, start_pos: int, end_pos: int) -> list[int]:
        """Scan a file chunk for line start positions."""
        offsets = []

        with open(self._data_path, "rb") as f:
            f.seek(start_pos)
            if start_pos != 0:
                f.readline()
                start_pos = f.tell()

            current_pos = start_pos
            if current_pos < end_pos:
                offsets.append(current_pos)

            while current_pos < end_pos:
                line = f.readline()
                if not line:
                    break
                current_pos = f.tell()
                if current_pos < end_pos:
                    offsets.append(current_pos)

            return offsets

    def _build_index(self) -> list[int]:
        """Build byte offset index using parallel workers."""
        file_size = self._data_path.stat().st_size
        chunk_size = math.ceil(file_size / self._num_workers)

        chunks = []
        for i in range(self._num_workers):
            start_pos = i * chunk_size
            end_pos = min((i + 1) * chunk_size, file_size)
            if start_pos >= file_size:
                break
            chunks.append((start_pos, end_pos))

        with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
            future_to_idx = {
                executor.submit(self._scan_chunk, start_pos, end_pos): idx
                for idx, (start_pos, end_pos) in enumerate(chunks)
            }

            ordered_results = [None] * len(chunks)

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                ordered_results[idx] = future.result()

        offsets = []
        for chunked_offsets in ordered_results:
            offsets.extend(chunked_offsets)

        return offsets

    def __len__(self) -> int:
        return len(self._offsets)

    def __getitem__(self, index: int) -> OfflineSample:
        offset = self._offsets[index]
        self._file.seek(offset)
        line = self._file.readline()
        data = fast_json_loads(line)
        return OfflineSample.model_validate(data)

    def __iter__(self) -> Iterator[OfflineSample]:
        with open(self._data_path) as f:
            for line in f:
                if line.strip():
                    data = fast_json_loads(line)
                    yield OfflineSample.model_validate(data)

    def __del__(self):
        if self._file:
            self._file.close()
            self._file = None


DATASOURCE_REGISTRY: dict[str, type[DataSource]] = {
    "jsonl": JsonlDataSource,
}
