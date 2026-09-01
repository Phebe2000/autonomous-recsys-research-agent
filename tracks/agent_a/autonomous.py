"""Optuna TPE research loop with ResearchStore-authoritative budgeting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import optuna
from optuna.trial import TrialState
from sqlalchemy.pool import NullPool

from .candidate import (
    AuxiliaryConfig,
    BackboneConfig,
    BPRConfig,
    CandidateConfig,
    CandidateSpec,
    HistoryConfig,
    ListwiseConfig,
    OptimizerConfig,
    RankerConfig,
)
from .compliance import JudgedRunCompliance, WallClockExceeded
from .contracts import ContractError, TrialOutcome, ValidationMetrics, reject_test_data
from .runner import TrialPruned, TrialRunner, TrialSpec
from .selection import write_top1
from .store import BudgetExhausted, ResearchStore


MAX_TRIALS = 50
EPSILON = 0.002
CONVERGENCE_N = 3
SEARCH_SCHEMA = Path(__file__).with_name("configs") / "autonomous_search_space.json"


def phase_for_ordinal(ordinal: int) -> str:
    if not 1 <= ordinal <= 50:
        raise ValueError("trial ordinal must be in [1, 50]")
    if ordinal <= 6:
        return "controlled_anchors"
    if ordinal <= 14:
        return "single_module_screening"
    if ordinal <= 34:
        return "tpe_conditional_search"
    if ordinal <= 42:
        return "local_refinement"
    if ordinal <= 47:
        return "automatic_ablation"
    if ordinal <= 49:
        return "finalist_verification"
    return "reserve"


def _atomic_json(path: Path, payload: dict) -> None:
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


def _spec_mapping(spec: CandidateSpec) -> dict:
    return {
        "dataset_fingerprint": spec.dataset_fingerprint,
        "config": spec.config.to_dict(),
        "code_version": spec.code_version,
        "schema_version": spec.schema_version,
    }


@dataclass(frozen=True)
class ResearchState:
    dataset_fingerprint: str
    evidence: dict[str, Any]
    history: tuple[dict[str, Any], ...]
    learning_curves: tuple[dict[str, Any], ...]
    phase: str
    budget: dict[str, int]
    best_primary: float | None
    recent_trend: tuple[float, ...]
    test_metrics_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        reject_test_data({key: value for key, value in payload.items() if key != "test_metrics_used"})
        return payload


class ValidationReporter:
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def report(self, validation: dict[str, Any], step: int) -> None:
        metrics = ValidationMetrics.from_mapping(validation)
        self.trial.report(metrics.primary, int(step))
        if self.trial.should_prune():
            raise TrialPruned(f"Optuna pruned at validation step {step}")


class SyntheticExecutor:
    """Deterministic validation-only simulator; never a production artifact."""

    def __call__(
        self, trial: dict, spec: CandidateSpec, module: str, reporter: ValidationReporter
    ) -> TrialOutcome:
        lr = spec.config.optimizer.learning_rate
        gain = {
            "baseline": -0.004,
            "listwise": 0.0,
            "history": 0.004,
            "bpr": -0.0005,
            "click": 0.003,
            "play": 0.0025,
            "ranker": 0.006,
            "ensemble": 0.007,
            "ranker_wide": 0.0055,
        }[module]
        smooth = max(0.0, 0.001 - 0.0008 * abs(math.log10(lr) - math.log10(1e-6)))
        primary = 0.60 + gain + smooth
        intermediate = {
            "GAUC": primary + 0.049,
            "nDCG@5": primary - 0.051,
            "primary": primary - 0.001,
            "rows": 100,
            "users": 20,
        }
        reporter.report(intermediate, 1)
        validation = {
            "GAUC": primary + 0.05,
            "nDCG@5": primary - 0.05,
            "primary": primary,
            "rows": 100,
            "users": 20,
        }
        return TrialOutcome(
            ValidationMetrics.from_mapping(validation),
            best_step=2,
            stop_reason="synthetic_completed",
            history=({"step": 1, "validation": intermediate}, {"step": 2, "validation": validation}),
            artifacts=(),
        )


class AutonomousResearchLoop:
    def __init__(
        self,
        state_dir: Path,
        dataset_fingerprint: str,
        *,
        simulation: bool,
        max_trials: int = 50,
        convergence_enabled: bool = True,
        compliance: JudgedRunCompliance | None = None,
        research_code_diff: str = "",
    ) -> None:
        if not 1 <= max_trials <= MAX_TRIALS:
            raise ValueError("max_trials must be in [1, 50]")
        synthetic = dataset_fingerprint.startswith("sha256:synthetic-")
        if simulation != synthetic:
            raise ValueError("simulation requires an explicit synthetic fingerprint and vice versa")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = dataset_fingerprint
        self.simulation = simulation
        self.max_trials = max_trials
        self.convergence_enabled = convergence_enabled
        self.compliance = compliance
        self.research_code_diff = research_code_diff
        if compliance is not None:
            if compliance.policy.dataset_fingerprint != dataset_fingerprint:
                raise ValueError("compliance policy fingerprint mismatch")
            if compliance.policy.max_iterations != max_trials:
                raise ValueError("compliance policy budget mismatch")
        self.store = ResearchStore(
            self.state_dir / "research.sqlite3", dataset_fingerprint, max_trials=max_trials
        )
        storage_url = f"sqlite:///{(self.state_dir / 'optuna.sqlite3').resolve()}"
        storage = optuna.storages.RDBStorage(
            storage_url,
            engine_kwargs={"poolclass": NullPool},
        )
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.study = optuna.create_study(
            study_name="agent-a-" + hashlib.sha256(dataset_fingerprint.encode()).hexdigest()[:16],
            storage=storage,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=0),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=1),
        )
        self.search_space = json.loads(SEARCH_SCHEMA.read_text())

    def _base(self, lr: float = 1e-6, warmup: int = 50) -> CandidateConfig:
        return CandidateConfig(optimizer=OptimizerConfig(learning_rate=lr, warmup_steps=warmup))

    def _module_config(
        self,
        module: str,
        lr: float = 1e-6,
        warmup: int = 50,
        trial: optuna.Trial | None = None,
    ) -> CandidateConfig:
        base = self._base(lr, warmup)
        if module == "baseline":
            return replace(base, listwise=ListwiseConfig(enabled=False))
        if module == "listwise":
            return base
        if module == "history":
            last_n = 20 if trial is None else trial.suggest_categorical("history_last_n", [20, 50, 100])
            gate = -0.05 if trial is None else trial.suggest_float("history_gate", -0.1, -0.01)
            return replace(
                base,
                history=HistoryConfig(
                    True, last_n, gate, False, False,
                    "causal_positive_mean_pool", 16, None, None, 0,
                ),
            )
        if module == "bpr":
            weight = 0.01 if trial is None else trial.suggest_float("bpr_weight", 0.005, 0.1, log=True)
            return replace(base, bpr=BPRConfig(True, weight))
        if module in {"click", "play"}:
            click = 0.01 if module == "click" else 0.0
            play = 0.01 if module == "play" else 0.0
            if trial is not None:
                if module == "click":
                    click = trial.suggest_float("click_weight", 0.005, 0.05, log=True)
                else:
                    play = trial.suggest_float("play_weight", 0.005, 0.05, log=True)
            return replace(
                base,
                auxiliary=AuxiliaryConfig(
                    True, click, play,
                    "train_log1p_zscore" if play > 0 else None,
                    0.001, 1.0,
                ),
            )
        if module in {"ranker", "ensemble", "ranker_wide"}:
            ensemble = module == "ensemble"
            leaves = 63 if module == "ranker_wide" else 31
            ranker_lr = 0.04
            estimators = 160
            if trial is not None:
                leaves = trial.suggest_categorical("ranker_num_leaves", [15, 31, 63])
                ranker_lr = trial.suggest_float("ranker_learning_rate", 0.02, 0.08, log=True)
                estimators = trial.suggest_categorical("ranker_n_estimators", [100, 160, 240])
            return replace(
                base,
                backbone=BackboneConfig(
                    "fm_lambdarank_ensemble" if ensemble else "lambdarank", 16
                ),
                ranker=RankerConfig(
                    True, "causal_behavioral_v1", 20.0, estimators,
                    ranker_lr, leaves, 50, 20,
                ),
                listwise=ListwiseConfig(enabled=False),
            )
        raise ValueError(f"unsupported module: {module}")

    def _anchor(self, ordinal: int) -> tuple[str, str, CandidateConfig]:
        definitions = {
            1: ("A-01", "baseline"),
            2: ("A-02", "listwise"),
            3: ("A-03", "history"),
            4: ("A-04", "ranker"),
            5: ("A-05", "ensemble"),
            6: ("A-06", "ranker_wide"),
        }
        anchor_id, module = definitions[ordinal]
        return anchor_id, module, self._module_config(module)

    def _evidence(self) -> dict[str, Any]:
        completed = [trial for trial in self.store.trials() if trial["status"] == "completed"]
        anchors = {
            trial["config"].get("autonomous", {}).get("anchor_id"): trial
            for trial in completed
            if trial["config"].get("autonomous", {}).get("anchor_id")
        }
        reference = anchors.get("A-02")
        result = {"listwise": {"eligible": True, "gain": 0.0}}
        for anchor_id, module in (
            ("A-03", "history"), ("A-04", "ranker"),
            ("A-05", "ensemble"), ("A-06", "ranker_wide"),
        ):
            observed = anchors.get(anchor_id)
            gain = None if reference is None or observed is None else (
                observed["validation"]["primary"] - reference["validation"]["primary"]
            )
            result[module] = {"eligible": gain is not None and gain > 0, "gain": gain}
        return result

    def eligible_modules(self) -> list[str]:
        return [name for name, item in self._evidence().items() if item["eligible"]]

    def _best_config(self) -> CandidateConfig:
        best = self.store.best_trial()
        if best is None:
            return self._base()
        payload = best["config"].get("unified_candidate", {}).get("candidate")
        return self._base() if payload is None else CandidateConfig.from_mapping(payload)

    def suggest(self, trial: optuna.Trial, ordinal: int) -> tuple[str | None, str, CandidateSpec]:
        phase = phase_for_ordinal(ordinal)
        if phase == "controlled_anchors":
            anchor_id, module, config = self._anchor(ordinal)
        elif phase == "single_module_screening":
            # A-05 was the only controlled ranker anchor with positive evidence.
            # Trials 7-14 are a predeclared local screen around its 0.5 blend,
            # 160 trees and 31 leaves. Each row changes one axis where possible.
            screens = {
                7: (0.35, 160, 31),
                8: (0.45, 160, 31),
                9: (0.55, 160, 31),
                10: (0.65, 160, 31),
                11: (0.50, 100, 31),
                12: (0.50, 240, 31),
                13: (0.50, 160, 15),
                14: (0.50, 160, 63),
            }
            blend, estimators, leaves = screens[ordinal]
            module = "ensemble"
            base = self._module_config(module)
            config = replace(
                base,
                ranker=replace(
                    base.ranker,
                    blend_weight=blend,
                    n_estimators=estimators,
                    num_leaves=leaves,
                ),
            )
            anchor_id = None
        elif phase in {"tpe_conditional_search", "local_refinement"}:
            eligible = self.eligible_modules() or ["listwise"]
            module = trial.suggest_categorical("module", eligible)
            if phase == "tpe_conditional_search":
                lr = trial.suggest_float("learning_rate", 3e-7, 3e-6, log=True)
            else:
                center = self._best_config().optimizer.learning_rate
                lr = trial.suggest_float("refined_learning_rate", max(3e-7, center / 1.8), min(3e-6, center * 1.8), log=True)
            warmup = trial.suggest_categorical("warmup_steps", [20, 50, 100])
            config = self._module_config(module, lr, warmup, trial)
            anchor_id = None
        else:
            best = self._best_config()
            module = self._module_name(best)
            factor = 1.0 + (ordinal - 45) * 0.003 + trial.number * 1e-6
            lr = min(3e-6, max(3e-7, best.optimizer.learning_rate * factor))
            if phase == "automatic_ablation":
                module = ["listwise"] + self.eligible_modules()
                module = module[(ordinal - 43) % len(module)]
            config = self._module_config(module, lr, best.optimizer.warmup_steps)
            anchor_id = None
        spec = CandidateSpec(self.fingerprint, config)
        return anchor_id, module, spec

    @staticmethod
    def _module_name(config: CandidateConfig) -> str:
        if config.ranker.enabled:
            return (
                "ensemble"
                if config.backbone.kind == "fm_lambdarank_ensemble"
                else "ranker"
            )
        if not config.listwise.enabled:
            return "baseline"
        if config.history.enabled:
            return "history"
        if config.bpr.enabled:
            return "bpr"
        if config.auxiliary.enabled:
            if config.auxiliary.play_time_weight and config.auxiliary.play_time_weight > 0:
                return "play"
            return "click"
        return "listwise"

    def state(self) -> ResearchState:
        trials = self.store.trials()
        completed = [trial for trial in trials if trial["status"] == "completed"]
        curves = tuple(
            {"trial_id": trial["trial_id"], "history": (trial["result"] or {}).get("history", [])}
            for trial in completed
        )
        values = [trial["validation"]["primary"] for trial in completed]
        trend = tuple(values[index] - values[index - 1] for index in range(max(1, len(values) - 3), len(values)))
        ordinal = min(self.store.consumed + 1, 50)
        state = ResearchState(
            self.fingerprint,
            self._evidence(),
            tuple({"trial_id": trial["trial_id"], "status": trial["status"], "config": trial["config"], "validation": trial["validation"]} for trial in trials),
            curves,
            phase_for_ordinal(ordinal),
            {"used": self.store.consumed, "remaining": self.store.remaining, "maximum": self.max_trials},
            None if not values else max(values),
            trend,
        )
        state.to_dict()
        return state

    def stop_reason(self) -> str | None:
        if (self.state_dir / "locked_manifest.json").exists():
            return "user_lock"
        fatal = self.state_dir / "fatal_stop.json"
        if fatal.exists():
            return json.loads(fatal.read_text())["reason"]
        if self.compliance is not None and self.compliance.wall_clock_exhausted():
            return "wall_clock_exhausted"
        if self.store.consumed >= self.max_trials:
            return "budget_exhausted"
        epsilon = EPSILON if self.compliance is None else self.compliance.policy.epsilon
        convergence_n = (
            CONVERGENCE_N if self.compliance is None else self.compliance.policy.convergence_n
        )
        minimum_scored = (
            6 + convergence_n
            if self.compliance is None
            else self.compliance.policy.minimum_scored_iterations
        )
        if not self.convergence_enabled:
            return None
        completed = [trial for trial in self.store.trials() if trial["status"] == "completed"]
        if len(completed) < minimum_scored:
            return None
        values = [trial["validation"]["primary"] for trial in completed]
        prior_best = max(values[:-convergence_n])
        if max(values[-convergence_n:]) <= prior_best + epsilon:
            return f"converged_no_{epsilon}_gain_for_{convergence_n}_scored_iterations"
        return None

    def _fatal_stop(self, reason: str) -> None:
        path = self.state_dir / "fatal_stop.json"
        if not path.exists():
            _atomic_json(path, {"reason": f"fatal_guard_failure: {reason}"})

    def _running_optuna(self) -> optuna.Trial | None:
        running = self.study.get_trials(deepcopy=False, states=(TrialState.RUNNING,))
        staged = [item for item in running if "candidate_spec" in item.user_attrs]
        if not staged:
            return None
        frozen = min(staged, key=lambda item: item.number)
        return optuna.Trial(self.study, frozen._trial_id)

    def _stage(self) -> tuple[optuna.Trial, str | None, str, CandidateSpec, int]:
        active = self._running_optuna()
        if active is not None:
            attrs = active.user_attrs
            return (
                active,
                attrs.get("anchor_id"),
                attrs["module"],
                CandidateSpec.from_mapping(attrs["candidate_spec"]),
                int(attrs["ledger_ordinal"]),
            )
        ordinal = self.store.consumed + 1
        if ordinal > self.max_trials:
            raise BudgetExhausted("dataset trial budget exhausted")
        optuna_trial = self.study.ask()
        try:
            anchor_id, module, spec = self.suggest(optuna_trial, ordinal)
        except Exception as exc:
            self.study.tell(optuna_trial, state=TrialState.FAIL)
            if isinstance(exc, (ContractError, ValueError)):
                self._fatal_stop(str(exc))
            raise
        optuna_trial.set_user_attr("candidate_spec", _spec_mapping(spec))
        optuna_trial.set_user_attr("config_identity", spec.identity)
        optuna_trial.set_user_attr("module", module)
        optuna_trial.set_user_attr("anchor_id", anchor_id)
        optuna_trial.set_user_attr("ledger_ordinal", ordinal)
        optuna_trial.set_user_attr("simulation", self.simulation)
        return optuna_trial, anchor_id, module, spec, ordinal

    def _find_binding(self, optuna_number: int) -> dict | None:
        for trial in self.store.trials():
            if trial["config"].get("optuna", {}).get("trial_number") == optuna_number:
                return trial
        return None

    def _find_identity(self, identity: str) -> dict | None:
        for trial in self.store.trials():
            if trial["config"].get("config_identity") == identity:
                return trial
        return None

    @staticmethod
    def _method_for_module(module: str) -> str:
        return {
            "baseline": "official_fm_baseline_reproduction",
            "listwise": "fm_user_soft_target_listnet_finetune",
            "history": "fm_user_soft_target_listnet_causal_history_fixed_gate",
            "bpr": "fm_user_soft_target_listnet_same_user_bpr_regularized",
            "click": "fm_user_soft_target_listnet_multitask_auxiliary",
            "play": "fm_user_soft_target_listnet_multitask_auxiliary",
            "ranker": "train_causal_behavioral_lambdarank",
            "ensemble": "train_causal_behavioral_lambdarank_fm_ensemble",
            "ranker_wide": "train_causal_behavioral_lambdarank_wide",
        }[module]

    def _step_impl(
        self,
        executor: Callable[[dict, CandidateSpec, str, ValidationReporter], TrialOutcome],
        *,
        interrupt_after_ask: bool = False,
        interrupt_after_reserve: bool = False,
    ) -> dict[str, Any]:
        reason = self.stop_reason()
        if reason:
            return {"status": "stopped", "stop_reason": reason}
        optuna_trial, anchor_id, module, spec, ordinal = self._stage()
        if interrupt_after_ask:
            self.store.record_agent_decision(
                stage="recovery",
                decision="pause_after_optuna_ask",
                rationale="The interruption fixture stops after suggestion so resume behavior is auditable.",
                evidence={"optuna_trial_number": optuna_trial.number, "ordinal": ordinal},
                alternatives=("continue_to_budget_reservation",),
                selected_action="persist_optuna_mapping_and_resume_later",
                actor="agent-a-autonomous-loop",
                decision_key=f"optuna-{optuna_trial.number}:pause-after-ask",
            )
            return {"status": "interrupted_after_ask", "optuna_trial_number": optuna_trial.number}
        bound = self._find_binding(optuna_trial.number)
        exact = self._find_identity(spec.identity)
        if exact is not None and bound is None and exact["status"] == "completed":
            reward = exact["validation"]["primary"]
            optuna_trial.set_user_attr("reused_trial_id", exact["trial_id"])
            self.study.tell(optuna_trial, reward)
            if not self.simulation:
                write_top1(self.store, self.state_dir / "top1.json")
            self.store.record_agent_decision(
                stage="candidate_reuse",
                decision="reuse_exact_completed_candidate",
                rationale="The canonical identity already has a completed validation result, so retraining would waste budget.",
                evidence={"config_identity": spec.identity, "validation": exact["validation"]},
                alternatives=("reserve_duplicate_training_trial",),
                selected_action=f"reuse_{exact['trial_id']}",
                actor="agent-a-autonomous-loop",
                trial_id=exact["trial_id"],
                decision_key=f"optuna-{optuna_trial.number}:exact-reuse",
            )
            return {"status": "reused", "trial_id": exact["trial_id"], "reward": reward}
        if bound is None:
            hypothesis = (
                f"Evaluate {module} during {phase_for_ordinal(ordinal)} using only "
                "training data and validation.primary feedback."
            )
            code_diff = "" if anchor_id else self.research_code_diff
            ledger_config = {
                **spec.ledger_config(),
                "optuna": {"study_name": self.study.study_name, "trial_number": optuna_trial.number},
                "autonomous": {"phase": phase_for_ordinal(ordinal), "anchor_id": anchor_id, "module": module},
                "research_action": {
                    "hypothesis": hypothesis,
                    "code_diff": code_diff,
                    "code_diff_sha256": hashlib.sha256(code_diff.encode()).hexdigest(),
                    "change_kind": "preimplemented_anchor" if anchor_id else "configuration_only",
                    "generated_by": "agent-a-autonomous-planner",
                    "manual_intervention": False,
                },
                "simulation": self.simulation,
            }
            bound = self.store.reserve(
                self._method_for_module(module),
                hypothesis,
                ledger_config,
                spec.config.seed,
            )
            optuna_trial.set_user_attr("ledger_trial_id", bound["trial_id"])
            self.store.record_agent_decision(
                stage="candidate_selection",
                decision="execute_candidate",
                rationale=hypothesis,
                evidence={
                    "phase": phase_for_ordinal(ordinal),
                    "module": module,
                    "anchor_id": anchor_id,
                    "config_identity": spec.identity,
                    "budget_before_execution": {
                        "used": self.store.consumed,
                        "remaining": self.store.remaining,
                    },
                },
                alternatives=("skip_candidate", "reuse_exact_candidate_if_available"),
                selected_action=f"train_and_validate_{bound['trial_id']}",
                actor="agent-a-autonomous-planner",
                trial_id=bound["trial_id"],
                decision_key=f"{bound['trial_id']}:execute",
            )
        if interrupt_after_reserve:
            self.store.record_agent_decision(
                stage="recovery",
                decision="pause_after_budget_reservation",
                rationale="The interruption fixture preserves the consumed reservation for resume verification.",
                evidence={"optuna_trial_number": optuna_trial.number},
                alternatives=("start_training_now",),
                selected_action="resume_same_reserved_trial_later",
                actor="agent-a-autonomous-loop",
                trial_id=bound["trial_id"],
                decision_key=f"{bound['trial_id']}:pause-after-reserve",
            )
            return {
                "status": "interrupted_after_reserve",
                "optuna_trial_number": optuna_trial.number,
                "trial_id": bound["trial_id"],
            }
        if bound["status"] == "completed":
            reward = bound["validation"]["primary"]
            self.study.tell(optuna_trial, reward)
            if not self.simulation:
                write_top1(self.store, self.state_dir / "top1.json")
            self.store.record_agent_decision(
                stage="recovery",
                decision="restore_completed_trial_to_sampler",
                rationale="Resume found a completed ledger result whose Optuna state still required tell().",
                evidence={"validation": bound["validation"], "optuna_trial_number": optuna_trial.number},
                alternatives=("retrain_completed_candidate", "discard_completed_result"),
                selected_action="tell_existing_validation_reward",
                actor="agent-a-autonomous-loop",
                trial_id=bound["trial_id"],
                decision_key=f"{bound['trial_id']}:recover-completed",
            )
            return {"status": "recovered_completed", "trial_id": bound["trial_id"], "reward": reward}
        if bound["status"] in {"failed", "pruned"}:
            state = TrialState.PRUNED if bound["status"] == "pruned" else TrialState.FAIL
            self.study.tell(optuna_trial, state=state)
            self.store.record_agent_decision(
                stage="recovery",
                decision=f"restore_{bound['status']}_trial_to_sampler",
                rationale="Resume preserves the terminal ledger status without retrying or refunding budget.",
                evidence={"status": bound["status"], "error": bound["error"]},
                alternatives=("retry_and_double_count", "erase_terminal_trial"),
                selected_action=f"tell_optuna_{bound['status']}",
                actor="agent-a-autonomous-loop",
                trial_id=bound["trial_id"],
                decision_key=f"{bound['trial_id']}:recover-{bound['status']}",
            )
            return {"status": f"recovered_{bound['status']}", "trial_id": bound["trial_id"]}

        reporter = ValidationReporter(optuna_trial)
        runner = TrialRunner(self.store)
        runner_spec = TrialSpec(bound["method"], bound["hypothesis"], bound["config"], bound["seed"])

        def candidate(current_trial):
            return executor(current_trial, spec, module, reporter)

        try:
            completed = runner.execute(runner_spec, candidate, resume_trial_id=bound["trial_id"])
        except Exception as exc:
            self.study.tell(optuna_trial, state=TrialState.FAIL)
            self.store.record_agent_decision(
                stage="candidate_disposition",
                decision="reject_failed_candidate",
                rationale="Candidate execution did not produce an admissible completed validation result.",
                evidence={"error_type": type(exc).__name__, "error": str(exc)},
                alternatives=("promote_without_validation", "refund_budget"),
                selected_action="retain_failure_and_exclude_from_top1",
                actor="agent-a-autonomous-loop",
                trial_id=bound["trial_id"],
                decision_key=f"{bound['trial_id']}:failed-disposition",
            )
            if isinstance(exc, (ContractError, ValueError)):
                self._fatal_stop(str(exc))
            if isinstance(exc, WallClockExceeded):
                return {
                    "status": "failed",
                    "trial_id": bound["trial_id"],
                    "stop_reason": "wall_clock_exhausted",
                }
            raise
        if completed["status"] == "pruned":
            self.study.tell(optuna_trial, state=TrialState.PRUNED)
            self.store.record_agent_decision(
                stage="candidate_disposition",
                decision="retain_pruned_candidate_as_evidence",
                rationale="Validation-only pruning stopped the candidate and the consumed budget remains permanent.",
                evidence={"status": "pruned", "error": completed["error"]},
                alternatives=("promote_pruned_candidate", "refund_budget"),
                selected_action="exclude_from_top1",
                actor="agent-a-autonomous-loop",
                trial_id=completed["trial_id"],
                decision_key=f"{completed['trial_id']}:pruned-disposition",
            )
            return {"status": "pruned", "trial_id": completed["trial_id"]}
        reward = completed["validation"]["primary"]
        self.study.tell(optuna_trial, reward)
        if not self.simulation:
            write_top1(self.store, self.state_dir / "top1.json")
        best = self.store.best_trial()
        promoted = best is not None and best["trial_id"] == completed["trial_id"]
        self.store.record_agent_decision(
            stage="candidate_disposition",
            decision="promote_validation_top1" if promoted else "retain_as_ablation_evidence",
            rationale=(
                "Candidate is the stable validation.primary leader."
                if promoted else
                "Candidate completed but did not exceed the stable validation.primary leader."
            ),
            evidence={
                "candidate_validation": completed["validation"],
                "top1_trial_id": None if best is None else best["trial_id"],
                "top1_validation": None if best is None else best["validation"],
                "selection_rule": "validation.primary_then_lowest_ordinal",
            },
            alternatives=("select_by_auxiliary_metric", "select_by_hidden_result"),
            selected_action="update_top1_manifest" if promoted else "keep_current_top1",
            actor="agent-a-autonomous-loop",
            trial_id=completed["trial_id"],
            decision_key=f"{completed['trial_id']}:completed-disposition",
        )
        return {"status": "completed", "trial_id": completed["trial_id"], "reward": reward}

    def step(
        self,
        executor: Callable[[dict, CandidateSpec, str, ValidationReporter], TrialOutcome],
        *,
        interrupt_after_ask: bool = False,
        interrupt_after_reserve: bool = False,
    ) -> dict[str, Any]:
        if self.compliance is not None and self.stop_reason() is None:
            self.compliance.start_if_needed()
        try:
            return self._step_impl(
                executor,
                interrupt_after_ask=interrupt_after_ask,
                interrupt_after_reserve=interrupt_after_reserve,
            )
        finally:
            if self.compliance is not None:
                self.compliance.write_audit(self.store)

    def run(
        self,
        executor: Callable[[dict, CandidateSpec, str, ValidationReporter], TrialOutcome],
        target_trials: int | None = None,
    ) -> dict[str, Any]:
        target = self.max_trials if target_trials is None else min(target_trials, self.max_trials)
        events = []
        attempts = 0
        while self.store.consumed < target and self.stop_reason() is None:
            events.append(self.step(executor))
            attempts += 1
            if attempts > target * 5:
                raise RuntimeError("too many duplicate suggestions without budget progress")
        self.record_stop_decision()
        report = self.report(events)
        _atomic_json(self.state_dir / "autonomous_report.json", report)
        return report

    def record_stop_decision(self) -> None:
        """Persist the predeclared terminal decision exactly once."""
        reason = self.stop_reason()
        if reason is not None:
            self.store.record_agent_decision(
                stage="stopping",
                decision="stop_research_loop",
                rationale=f"The predeclared stopping policy returned: {reason}.",
                evidence={
                    "stop_reason": reason,
                    "budget": {"used": self.store.consumed, "remaining": self.store.remaining},
                    "best_validation": None if self.store.best_trial() is None else self.store.best_trial()["validation"],
                },
                alternatives=("continue_beyond_policy",),
                selected_action="preserve_validation_best_and_stop",
                actor="agent-a-autonomous-loop",
                decision_key=f"stop:{reason}",
            )

    def report(self, events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if self.compliance is not None:
            self.compliance.freeze_at_recorded_stop(self.store)
        best = self.store.best_trial()
        mapping = [
            {
                "ledger_trial_id": trial["trial_id"],
                "optuna_trial_number": trial["config"].get("optuna", {}).get("trial_number"),
                "phase": trial["config"].get("autonomous", {}).get("phase"),
                "module": trial["config"].get("autonomous", {}).get("module"),
                "status": trial["status"],
            }
            for trial in self.store.trials()
        ]
        return {
            "schema_version": 1,
            "simulation": self.simulation,
            "production_top1_eligible": not self.simulation,
            "dataset_fingerprint": self.fingerprint,
            "sampler": {"name": "Optuna.TPESampler", "seed": 0, "version": optuna.__version__},
            "budget": {"used": self.store.consumed, "remaining": self.store.remaining, "maximum": self.max_trials},
            "phase": phase_for_ordinal(min(self.store.consumed + 1, 50)),
            "trial_mapping": mapping,
            "best_validation": None if best is None else best["validation"],
            "best_trial_id": None if best is None else best["trial_id"],
            "stop_reason": self.stop_reason(),
            "epsilon": EPSILON if self.compliance is None else self.compliance.policy.epsilon,
            "convergence_n": CONVERGENCE_N if self.compliance is None else self.compliance.policy.convergence_n,
            "minimum_scored_iterations": None if self.compliance is None else self.compliance.policy.minimum_scored_iterations,
            "wall_clock_seconds": None if self.compliance is None else self.compliance.policy.wall_clock_seconds,
            "elapsed_wall_clock_seconds": None if self.compliance is None else self.compliance.elapsed_seconds(),
            "events": [] if events is None else events,
            "test_metrics_used": False,
        }

    def lock(self) -> dict[str, Any]:
        path = self.state_dir / "locked_manifest.json"
        if path.exists():
            return json.loads(path.read_text())
        best = self.store.best_trial()
        if best is None:
            raise RuntimeError("cannot lock without a completed validation trial")
        manifest = {
            "schema_version": 1,
            "immutable": True,
            "simulation": self.simulation,
            "production_top1_eligible": not self.simulation,
            "dataset_fingerprint": self.fingerprint,
            "trial_id": best["trial_id"],
            "validation": best["validation"],
            "selection": "validation.primary with lowest ordinal stable tie-break",
            "test_metrics_used": False,
            "hidden_labels_loaded": False,
            "run_policy_identity": None if self.compliance is None else self.compliance.policy.identity,
        }
        self.store.record_agent_decision(
            stage="final_selection",
            decision="lock_validation_best_checkpoint",
            rationale="The final checkpoint is selected only by validation.primary with the predeclared stable tie-break.",
            evidence={
                "trial_id": best["trial_id"],
                "validation": best["validation"],
                "selection_rule": "validation.primary_then_lowest_ordinal",
            },
            alternatives=("select_by_hidden_result", "select_by_auxiliary_metric"),
            selected_action=f"lock_{best['trial_id']}",
            actor="agent-a-autonomous-loop",
            trial_id=best["trial_id"],
            decision_key="final-lock",
        )
        _atomic_json(path, manifest)
        if self.compliance is not None:
            self.compliance.write_audit(self.store)
        return manifest
