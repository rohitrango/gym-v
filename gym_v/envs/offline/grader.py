"""Built-in graders for offline environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .utils import normalize_em


def exact_match(action: str | None, answer: str | None) -> tuple[float, dict[str, Any]]:
    """Exact match reward: 1.0 if normalized strings match, else 0.0."""
    if action is None:
        return 0.0, {"correct": False, "reason": "missing action"}
    if answer is None:
        return 0.0, {"correct": False, "reason": "missing answer"}
    pred = normalize_em(action)
    gt = normalize_em(answer)
    correct = pred == gt
    return (1.0 if correct else 0.0), {"correct": correct, "pred": action, "gt": answer}


GRADER_REGISTRY: dict[str, Callable] = {
    "exact_match": exact_match,
}
