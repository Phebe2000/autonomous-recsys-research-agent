"""Unified capability dispatcher and validation-only trial facade."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import time
from typing import Any, Callable, Mapping

from .candidate import CandidateSpec
from .contracts import ContractError, ValidationMetrics, reject_test_data
from .fingerprint import canonical_json
from .runner import TrialRunner, TrialSpec
from .store import ResearchStore


ROUTE_BINDINGS = {
    "baseline": "tracks.agent_a.pipeline:train_baseline_candidate",
    "listwise": "tracks.agent_a.pipeline:train_listnet_candidate",
    "history": "tracks.agent_a.history_pipeline:_train_history",
    "bpr": "tracks.agent_a.bpr_pipeline:_train_bpr",
    "multitask": "tracks.agent_a.multitask_pipeline:_train_multitask",
    "ranker": "tracks.agent_a.ranker:train_ranker_candidate",
}


class TrainingNotAuthorized(RuntimeError):
    pass


def dispatch_route(spec: CandidateSpec) -> str:
    config = spec.config
    if config.ranker.enabled:
        return "ranker"
    if not config.listwise.enabled:
        if config.history.enabled or config.bpr.enabled or config.auxiliary.enabled:
            raise ValueError("baseline route cannot enable ranking modules")
        return "baseline"
    if config.history.enabled:
        return "history"
    if config.bpr.enabled:
        return "bpr"
    if config.auxiliary.enabled:
        return "multitask"
    return "listwise"


def resolve_implementation(spec: CandidateSpec) -> Callable:
    module_name, function_name = ROUTE_BINDINGS[dispatch_route(spec)].split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


@dataclass(frozen=True)
class UnifiedResult:
    trial_id: str
    status: str
    config_identity: str
    dataset_fingerprint: str
    validation: dict[str, Any] | None
    best_step: int | None
    checkpoint: dict[str, Any] | None
    runtime_seconds: float | None
    test_metrics_used: bool
    provenance: dict[str, Any]
    selection_key: str = "validation.primary"

    def __post_init__(self) -> None:
        reject_test_data({
            "validation": self.validation,
            "checkpoint": self.checkpoint,
            "provenance": self.provenance,
        })
        if self.status == "completed":
            if self.validation is None or self.best_step is None:
                raise ValueError("completed trials require validation and best_step")
            ValidationMetrics.from_mapping(self.validation)
        elif self.validation is not None:
            raise ValueError("non-completed trials cannot carry selection metrics")
        if self.test_metrics_used:
            raise ContractError("test metrics are forbidden")
        if self.selection_key != "validation.primary":
            raise ContractError("selection key must be official validation primary")
        if self.best_step is not None and self.best_step < 0:
            raise ValueError("invalid best step")
        if self.runtime_seconds is not None and self.runtime_seconds < 0:
            raise ValueError("invalid best step or runtime")

    @classmethod
    def from_trial(
        cls,
        trial: Mapping[str, Any],
        spec: CandidateSpec,
        runtime_seconds: float | None = None,
    ) -> "UnifiedResult":
        result = trial.get("result") or {}
        artifacts = result.get("artifacts", [])
        checkpoint = artifacts[0] if artifacts else None
        return cls(
            trial_id=str(trial["trial_id"]),
            status=str(trial["status"]),
            config_identity=spec.identity,
            dataset_fingerprint=spec.dataset_fingerprint,
            validation=None if trial.get("validation") is None else dict(trial["validation"]),
            best_step=None if result.get("best_step") is None else int(result["best_step"]),
            checkpoint=checkpoint,
            runtime_seconds=runtime_seconds,
            test_metrics_used=False,
            provenance={
                "method": trial["method"],
                "seed": trial["seed"],
                "legacy_config_hash": trial["config_hash"],
                "code_version": spec.code_version,
                "schema_version": spec.schema_version,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "status": self.status,
            "config_identity": self.config_identity,
            "dataset_fingerprint": self.dataset_fingerprint,
            "validation": self.validation,
            "best_step": self.best_step,
            "checkpoint": self.checkpoint,
            "runtime_seconds": self.runtime_seconds,
            "test_metrics_used": self.test_metrics_used,
            "provenance": self.provenance,
            "selection_key": self.selection_key,
        }


class UnifiedTrialRunner:
    """Facade over existing trainers; exact duplicates are reused before reservation."""

    def __init__(self, store: ResearchStore):
        self.store = store

    def inspect(self, spec: CandidateSpec, method: str | None = None) -> dict[str, Any]:
        if spec.dataset_fingerprint != self.store.dataset_fingerprint:
            raise ValueError("candidate/store fingerprint mismatch")
        exact = self.find_exact(spec, method)
        return {
            "route": dispatch_route(spec),
            "implementation": ROUTE_BINDINGS[dispatch_route(spec)],
            "config_identity": spec.identity,
            "exact_trial_id": None if exact is None else exact["trial_id"],
            "exact_trial_status": None if exact is None else exact["status"],
            "budget": {"used": self.store.consumed, "remaining": self.store.remaining},
        }

    def find_exact(self, spec: CandidateSpec, method: str | None = None) -> dict | None:
        expected_hash = hashlib.sha256(canonical_json(spec.ledger_config()).encode()).hexdigest()
        for trial in self.store.trials():
            same_identity = trial["config"].get("config_identity") == spec.identity
            if (trial["config_hash"] == expected_hash or same_identity) and (
                method is None or trial["method"] == method
            ):
                return trial
        return None

    def execute(
        self,
        spec: CandidateSpec,
        method: str,
        hypothesis: str,
        candidate: Callable[[dict], Any],
        allow_training: bool = False,
    ) -> tuple[UnifiedResult, bool]:
        if spec.dataset_fingerprint != self.store.dataset_fingerprint:
            raise ValueError("candidate/store fingerprint mismatch")
        dispatch_route(spec)
        existing = self.find_exact(spec, method)
        if existing is not None and existing["status"] in {
            "completed", "failed", "pruned"
        }:
            return UnifiedResult.from_trial(existing, spec), True
        if not allow_training:
            raise TrainingNotAuthorized("unified runner is inspect-only unless training is explicit")
        runner = TrialRunner(self.store)
        trial_spec = TrialSpec(method, hypothesis, spec.ledger_config(), spec.config.seed)
        resume_id = None if existing is None else existing["trial_id"]
        started = time.monotonic()
        trial = runner.execute(trial_spec, candidate, resume_trial_id=resume_id)
        return UnifiedResult.from_trial(trial, spec, time.monotonic() - started), False
