"""Strict contracts shared by the runner, ledger, and selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


class ContractError(ValueError):
    """Raised before invalid metrics can enter persistent research state."""


def reject_test_data(value: Any, path: str = "result") -> None:
    """Reject any test namespace; trial selection is validation-only."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            token = str(key).lower().replace("-", "_")
            if token == "test" or token.startswith("test_") or token.endswith("_test"):
                raise ContractError(f"test data is forbidden in trial results: {path}.{key}")
            reject_test_data(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            reject_test_data(child, f"{path}[{index}]")


@dataclass(frozen=True)
class ValidationMetrics:
    GAUC: float
    nDCG_at_5: float
    primary: float
    users: int | None = None
    rows: int | None = None

    @classmethod
    def from_mapping(cls, metrics: Mapping[str, Any]) -> "ValidationMetrics":
        reject_test_data(metrics, "validation")
        allowed = {"GAUC", "nDCG@5", "primary", "users", "rows"}
        extra = set(metrics) - allowed
        if extra:
            raise ContractError(f"unexpected validation metrics: {sorted(extra)}")
        try:
            gauc = float(metrics["GAUC"])
            ndcg = float(metrics["nDCG@5"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractError("validation requires numeric GAUC and nDCG@5") from exc
        primary = (gauc + ndcg) / 2.0
        supplied = metrics.get("primary", primary)
        try:
            supplied = float(supplied)
        except (TypeError, ValueError) as exc:
            raise ContractError("validation primary must be numeric") from exc
        if not all(math.isfinite(v) for v in (gauc, ndcg, primary, supplied)):
            raise ContractError("validation metrics must be finite")
        # The official evaluator may combine NumPy float32 labels with Python
        # floats, producing a few ulps of casting noise while using this formula.
        if not math.isclose(supplied, primary, rel_tol=0.0, abs_tol=1e-7):
            raise ContractError("primary must equal mean(GAUC, nDCG@5)")
        users = None if metrics.get("users") is None else int(metrics["users"])
        rows = None if metrics.get("rows") is None else int(metrics["rows"])
        return cls(gauc, ndcg, primary, users, rows)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["nDCG@5"] = result.pop("nDCG_at_5")
        return {key: value for key, value in result.items() if value is not None}


@dataclass(frozen=True)
class TrialOutcome:
    validation: ValidationMetrics
    best_step: int
    stop_reason: str
    history: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "validation": self.validation.to_dict(),
            "best_step": int(self.best_step),
            "stop_reason": self.stop_reason,
            "history": list(self.history),
            "artifacts": list(self.artifacts),
        }
        reject_test_data(payload)
        return payload
