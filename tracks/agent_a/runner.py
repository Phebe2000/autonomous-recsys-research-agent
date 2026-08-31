"""Reserve-before-run contract for resumable, budget-accounted trials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .contracts import TrialOutcome
from .store import ResearchStore


class TrialPruned(RuntimeError):
    pass


@dataclass(frozen=True)
class TrialSpec:
    method: str
    hypothesis: str
    config: dict
    seed: int = 0


class TrialRunner:
    def __init__(self, store: ResearchStore):
        self.store = store

    def execute(
        self,
        spec: TrialSpec,
        candidate: Callable[[dict], TrialOutcome],
        resume_trial_id: str | None = None,
    ) -> dict:
        trial = (
            self.store.reserve(spec.method, spec.hypothesis, spec.config, spec.seed)
            if resume_trial_id is None
            else self.store.resume(resume_trial_id)
        )
        self.store.mark_running(trial["trial_id"])
        try:
            outcome = candidate(trial)
            if not isinstance(outcome, TrialOutcome):
                raise TypeError("candidate must return TrialOutcome")
            return self.store.complete(trial["trial_id"], outcome)
        except TrialPruned as exc:
            return self.store.fail(trial["trial_id"], str(exc), pruned=True)
        except Exception as exc:
            self.store.fail(trial["trial_id"], f"{type(exc).__name__}: {exc}")
            raise
