"""Immutable judged-run policy, persistent wall clock, and reviewable logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
from typing import Any, Callable

from .fingerprint import canonical_json


class WallClockExceeded(RuntimeError):
    pass


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class JudgedRunPolicy:
    dataset_fingerprint: str
    benchmark: str = "KuaiRand-Pure"
    epsilon: float = 0.002
    convergence_n: int = 3
    minimum_scored_iterations: int = 9
    max_iterations: int = 50
    wall_clock_seconds: int = 6 * 60 * 60
    hidden_test_evaluations: int = 1
    selection_metric: str = "validation.primary"
    selection_tie_break: str = "lowest_trial_ordinal"
    code_diff_required: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.dataset_fingerprint.startswith("sha256:"):
            raise ValueError("judged policy requires a content fingerprint")
        if self.benchmark != "KuaiRand-Pure":
            raise ValueError("this judged-run implementation is isolated to KuaiRand-Pure")
        if self.epsilon < 0 or self.convergence_n <= 0:
            raise ValueError("invalid convergence policy")
        if self.minimum_scored_iterations < self.convergence_n + 1:
            raise ValueError("minimum scored floor must leave a pre-window reference")
        if not 1 <= self.max_iterations <= 50:
            raise ValueError("judged run cannot exceed 50 iterations")
        if not 1 <= self.wall_clock_seconds <= 6 * 60 * 60:
            raise ValueError("judged run cannot exceed the 6 hour wall-clock cap")
        if self.hidden_test_evaluations != 1:
            raise ValueError("hidden test must be scored exactly once")

    @property
    def identity(self) -> str:
        return hashlib.sha256(canonical_json(asdict(self)).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "policy_identity": self.identity}


class JudgedRunCompliance:
    def __init__(
        self,
        state_dir: Path,
        policy: JudgedRunPolicy,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.policy = policy
        self.now = now
        self.policy_path = self.state_dir / "run_policy.json"
        self.timing_path = self.state_dir / "run_timing.json"
        self.usage_path = self.state_dir / "resource_usage.json"
        self.audit_path = self.state_dir / "iteration_audit_log.json"
        self.decision_journal_path = self.state_dir / "agent_decision_journal.json"
        self._initialize()

    def _initialize(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        expected = self.policy.to_dict()
        if self.policy_path.exists():
            if json.loads(self.policy_path.read_text()) != expected:
                raise ValueError("judged run policy is immutable after initialization")
        else:
            _atomic_json(self.policy_path, expected)
        if not self.timing_path.exists():
            _atomic_json(
                self.timing_path,
                {
                    "schema_version": 1,
                    "policy_identity": self.policy.identity,
                    "started_at": None,
                    "deadline_at": None,
                },
            )
        timing = json.loads(self.timing_path.read_text())
        if timing["policy_identity"] != self.policy.identity:
            raise ValueError("timing state belongs to a different run policy")
        if not self.usage_path.exists():
            _atomic_json(
                self.usage_path,
                {
                    "schema_version": 1,
                    "llm_input_tokens": 0,
                    "llm_output_tokens": 0,
                    "gpu_seconds": 0.0,
                    "manual_interventions": 0,
                    "updated_at": None,
                },
            )

    def start_if_needed(self) -> None:
        timing = json.loads(self.timing_path.read_text())
        if timing["started_at"] is not None:
            return
        started = self.now()
        deadline = started.timestamp() + self.policy.wall_clock_seconds
        timing["started_at"] = started.isoformat()
        timing["deadline_at"] = datetime.fromtimestamp(deadline, timezone.utc).isoformat()
        _atomic_json(self.timing_path, timing)

    def elapsed_seconds(self) -> float:
        timing = json.loads(self.timing_path.read_text())
        if timing["started_at"] is None:
            return 0.0
        started = datetime.fromisoformat(timing["started_at"])
        return max(0.0, (self.now() - started).total_seconds())

    def wall_clock_exhausted(self) -> bool:
        timing = json.loads(self.timing_path.read_text())
        if timing["deadline_at"] is None:
            return False
        return self.now() >= datetime.fromisoformat(timing["deadline_at"])

    def execute_with_deadline(self, function):
        """Interrupt a training iteration at the persistent six-hour deadline."""
        timing = json.loads(self.timing_path.read_text())
        if timing["deadline_at"] is None:
            raise RuntimeError("judged clock must start before training")
        remaining = (
            datetime.fromisoformat(timing["deadline_at"]) - self.now()
        ).total_seconds()
        if remaining <= 0:
            raise WallClockExceeded("judged-run wall-clock deadline reached")
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("deadline-enforced training must run on the main thread")

        def expired(_signum, _frame):
            raise WallClockExceeded("training crossed the judged-run wall-clock deadline")

        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, expired)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            return function()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    def record_usage(
        self,
        *,
        llm_input_tokens: int = 0,
        llm_output_tokens: int = 0,
        gpu_seconds: float = 0.0,
        manual_interventions: int = 0,
    ) -> dict[str, Any]:
        increments = (llm_input_tokens, llm_output_tokens, gpu_seconds, manual_interventions)
        if any(value < 0 for value in increments):
            raise ValueError("resource-usage increments must be non-negative")
        usage = json.loads(self.usage_path.read_text())
        usage["llm_input_tokens"] += int(llm_input_tokens)
        usage["llm_output_tokens"] += int(llm_output_tokens)
        usage["gpu_seconds"] += float(gpu_seconds)
        usage["manual_interventions"] += int(manual_interventions)
        usage["updated_at"] = self.now().isoformat()
        _atomic_json(self.usage_path, usage)
        return usage

    def write_audit(self, store) -> dict[str, Any]:
        events = store.events()
        decisions = store.decisions()
        by_trial: dict[str, list[dict[str, Any]]] = {}
        decisions_by_trial: dict[str, list[dict[str, Any]]] = {}
        for event in events:
            if event["trial_id"] is not None:
                by_trial.setdefault(event["trial_id"], []).append(event)
                if event["kind"] == "agent_decision":
                    decisions_by_trial.setdefault(event["trial_id"], []).append(event)
        iterations = []
        for trial in store.trials():
            action = trial["config"].get("research_action", {})
            if self.policy.code_diff_required and "code_diff" not in action:
                raise ValueError(f"{trial['trial_id']} is missing its recorded code diff")
            if "code_diff" in action and action.get("code_diff_sha256") != hashlib.sha256(
                action["code_diff"].encode()
            ).hexdigest():
                raise ValueError(f"{trial['trial_id']} code diff digest mismatch")
            iterations.append(
                {
                    "ordinal": trial["ordinal"],
                    "trial_id": trial["trial_id"],
                    "status": trial["status"],
                    "hypothesis": trial["hypothesis"],
                    "code_diff": action.get("code_diff"),
                    "code_diff_sha256": action.get("code_diff_sha256"),
                    "change_kind": action.get("change_kind"),
                    "research_action": action,
                    "config_identity": trial["config"].get("config_identity"),
                    "validation": trial["validation"],
                    "error": trial["error"],
                    "created_at": trial["created_at"],
                    "updated_at": trial["updated_at"],
                    "agent_decisions": decisions_by_trial.get(trial["trial_id"], []),
                    "events": by_trial.get(trial["trial_id"], []),
                }
            )
        payload = {
            "schema_version": 1,
            "dataset_fingerprint": self.policy.dataset_fingerprint,
            "policy": self.policy.to_dict(),
            "timing": {
                **json.loads(self.timing_path.read_text()),
                "elapsed_seconds": self.elapsed_seconds(),
                "wall_clock_exhausted": self.wall_clock_exhausted(),
            },
            "resource_usage": json.loads(self.usage_path.read_text()),
            "iterations": iterations,
            "budget": {
                "used": store.consumed,
                "remaining": store.remaining,
                "maximum": self.policy.max_iterations,
            },
            "test_metrics_used": False,
            "hidden_labels_loaded": False,
        }
        journal = {
            "schema_version": 1,
            "dataset_fingerprint": self.policy.dataset_fingerprint,
            "append_only_source": "research.sqlite3.memory_events",
            "decision_count": len(decisions),
            "final_decision_sha256": (
                None if not decisions else decisions[-1]["payload"]["decision_sha256"]
            ),
            "decisions": decisions,
            "data_scope": "train_and_validation_only",
        }
        _atomic_json(self.decision_journal_path, journal)
        payload["agent_decision_journal"] = {
            "path": str(self.decision_journal_path),
            "decision_count": journal["decision_count"],
            "final_decision_sha256": journal["final_decision_sha256"],
        }
        _atomic_json(self.audit_path, payload)
        return payload
