"""Safety adapters around the unchanged official evaluator."""

from __future__ import annotations

import math
from typing import Iterable, Sequence, TypeVar

from evaluate import evaluate as official_evaluate

from .contracts import ContractError, ValidationMetrics

T = TypeVar("T")


def _materialize(values: Iterable[T] | Sequence[T]) -> Sequence[T]:
    return values if hasattr(values, "__len__") else list(values)


def evaluate_checked(user_ids, labels, predictions, k: int = 5) -> dict:
    """Require aligned arrays and finite predictions before official scoring."""
    if k != 5:
        raise ContractError("Agent A protocol fixes nDCG at k=5")
    users = _materialize(user_ids)
    truth = _materialize(labels)
    scores = _materialize(predictions)
    lengths = (len(users), len(truth), len(scores))
    if len(set(lengths)) != 1:
        raise ContractError(
            "user_ids, labels, and predictions must have identical lengths; "
            f"got {lengths}"
        )
    if any(not math.isfinite(float(score)) for score in scores):
        raise ContractError("predictions must be finite")
    metrics = official_evaluate(users, truth, scores, k=k)
    if int(metrics.get("rows", -1)) != len(truth):
        raise ContractError("official evaluator returned an inconsistent row count")
    ValidationMetrics.from_mapping(metrics)
    return metrics
