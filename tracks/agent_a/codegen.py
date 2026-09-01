"""Auditable LLM code-generation loop for post-anchor judged iterations."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Protocol

import numpy as np

from .compliance import JudgedRunCompliance
from .contracts import TrialOutcome, ValidationMetrics, reject_test_data
from .guards import evaluate_checked
from .selection import write_top1
from .safe_data import load_safe_side_features


GENERATED_METHOD = "llm_generated_numpy_candidate"
CODEGEN_SCHEMA_VERSION = 1
MAX_SOURCE_BYTES = 100_000
MAX_STATE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class ResearchProposal:
    hypothesis: str
    rationale: str
    expected_metric_effect: str
    source_code: str
    reflection_plan: str
    resource_estimate: str

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ResearchProposal":
        expected = {
            "hypothesis", "rationale", "expected_metric_effect", "source_code",
            "reflection_plan", "resource_estimate",
        }
        if set(value) != expected or not all(isinstance(value[key], str) for key in expected):
            raise ValueError("coding provider returned an invalid proposal schema")
        proposal = cls(**value)
        if not proposal.hypothesis.strip() or not proposal.source_code.strip():
            raise ValueError("proposal hypothesis and source code must be non-empty")
        reject_test_data(value, "research_proposal")
        return proposal


@dataclass(frozen=True)
class ProviderResult:
    proposal: ResearchProposal
    provider: str
    model: str | None
    input_tokens: int
    output_tokens: int
    raw_log_path: str | None = None


class CodingProvider(Protocol):
    def propose(self, context: dict[str, Any], artifact_dir: Path) -> ProviderResult: ...

    def reflect(
        self, context: dict[str, Any], artifact_dir: Path
    ) -> tuple[dict[str, Any], int, int]: ...


class ProviderInfrastructureError(RuntimeError):
    pass


PROPOSAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hypothesis", "rationale", "expected_metric_effect", "source_code",
        "reflection_plan", "resource_estimate",
    ],
    "properties": {
        key: {"type": "string"}
        for key in (
            "hypothesis", "rationale", "expected_metric_effect", "source_code",
            "reflection_plan", "resource_estimate",
        )
    },
}

REFLECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hypothesis_supported", "diagnosis", "next_action"],
    "properties": {
        "hypothesis_supported": {"type": "boolean"},
        "diagnosis": {"type": "string"},
        "next_action": {"type": "string"},
    },
}


class CodexCodingProvider:
    """Use the installed Codex coding agent in read-only proposal mode."""

    def __init__(self, repo_root: Path, executable: str | None = None, model: str | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.executable = executable or shutil.which("codex")
        self.model = model or "gpt-5.6-sol"
        if not self.executable:
            raise FileNotFoundError("codex executable is required for the judged coding provider")

    @staticmethod
    def _tokens(jsonl: str) -> tuple[int, int]:
        input_tokens = output_tokens = 0

        def visit(value):
            nonlocal input_tokens, output_tokens
            if isinstance(value, dict):
                for key, child in value.items():
                    if key in {"input_tokens", "input_token_count"} and isinstance(child, int):
                        input_tokens = max(input_tokens, child)
                    elif key in {"output_tokens", "output_token_count"} and isinstance(child, int):
                        output_tokens = max(output_tokens, child)
                    else:
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        for line in jsonl.splitlines():
            try:
                visit(json.loads(line))
            except json.JSONDecodeError:
                continue
        return input_tokens, output_tokens

    def _run(self, prompt: str, schema: dict, artifact_dir: Path, stem: str):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        schema_path = artifact_dir / f"{stem}_schema.json"
        output_path = artifact_dir / f"{stem}.json"
        log_path = artifact_dir / f"{stem}_events.jsonl"
        schema_path.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
        command = [
            self.executable, "exec", "--ephemeral", "--json", "--sandbox", "read-only",
            "--output-schema", str(schema_path), "--output-last-message", str(output_path),
            "--cd", str(self.repo_root),
        ]
        if self.model:
            command.extend(("--model", self.model))
        command.append(prompt)
        process = subprocess.Popen(
            command,
            cwd=self.repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            stdout, _ = process.communicate()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise
        log_path.write_text(stdout)
        if process.returncode:
            raise ProviderInfrastructureError(
                f"Codex provider exited {process.returncode}; see {log_path}"
            )
        try:
            payload = json.loads(output_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ProviderInfrastructureError(
                f"Codex provider did not return schema-valid JSON; see {log_path}"
            ) from exc
        return payload, self._tokens(stdout), str(log_path)

    def propose(self, context: dict[str, Any], artifact_dir: Path) -> ProviderResult:
        prompt = """You are the coding researcher for a judged KuaiRand-Pure run.
Return one JSON proposal matching the supplied schema. Write a self-contained NumPy-only
candidate module with exactly two functions:
  train(X_train, y_train, users_train, train_side, X_valid, y_valid, users_valid,
        valid_side, dimension, score_validation)
  predict(X, side, state)
train returns {"state": dict[str, ndarray], "best_step": int, "history": list}.
predict returns one finite score per row. You may use public validation feedback through
score_validation(users_valid, y_valid, scores). Do not access files, environment, network,
test data, external data, subprocesses, reflection/introspection, or randomness without a
fixed seed. Imports are limited to `import numpy as np`. Keep runtime and memory modest.
train_side contains date/hourmin/time_ms/log_duration_ms plus training-only
is_click/log_play_time_ms. valid_side and final inference contain only the four common
inference fields; generated predict must not require feedback-only keys.
Study the evidence below, form a concrete hypothesis, and produce genuinely revised model
or feature/loss code rather than merely renaming a configuration.

EVIDENCE:
""" + json.dumps(context, indent=2, sort_keys=True)
        payload, tokens, log_path = self._run(prompt, PROPOSAL_SCHEMA, artifact_dir, "proposal")
        return ProviderResult(
            ResearchProposal.from_mapping(payload), "codex-cli", self.model,
            tokens[0], tokens[1], log_path,
        )

    def reflect(self, context: dict[str, Any], artifact_dir: Path):
        prompt = """Reflect on this completed or failed validation-only ML iteration.
Return JSON only. Decide whether the hypothesis was supported, diagnose metric/error
behavior, and recommend a specific next research action. Never infer or request test data.

ITERATION:
""" + json.dumps(context, indent=2, sort_keys=True)
        payload, tokens, _ = self._run(prompt, REFLECTION_SCHEMA, artifact_dir, "reflection")
        reject_test_data(payload, "reflection")
        return payload, tokens[0], tokens[1]


_BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "help", "breakpoint", "memoryview",
}
_BANNED_ATTRIBUTES = {
    "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "memmap",
    "loadtxt", "savetxt", "genfromtxt", "fromregex", "DataSource", "open_memmap",
    "dump", "dumps", "setfield", "ctypeslib", "get_include",
}


def validate_generated_source(source: str) -> ast.Module:
    if len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("generated source exceeds the 100 KB limit")
    tree = ast.parse(source, filename="generated_candidate.py")
    if any(not isinstance(node, (ast.Import, ast.FunctionDef)) for node in tree.body):
        raise ValueError("generated top level may contain only numpy import and function definitions")
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if set(functions) != {"train", "predict"}:
        raise ValueError("generated module must define exactly train and predict")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name != "numpy" or alias.asname != "np" for alias in node.names):
                raise ValueError("generated imports are limited to `import numpy as np`")
        elif isinstance(node, ast.ImportFrom):
            raise ValueError("from-import is forbidden in generated candidates")
        elif isinstance(node, ast.Name) and (
            node.id in _BANNED_NAMES or node.id.startswith("__")
        ):
            raise ValueError(f"forbidden generated name: {node.id}")
        elif isinstance(node, ast.Attribute) and (
            node.attr in _BANNED_ATTRIBUTES or node.attr.startswith("__")
        ):
            raise ValueError(f"forbidden generated attribute: {node.attr}")
        elif isinstance(node, (ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal)):
            raise ValueError(f"forbidden generated syntax: {type(node).__name__}")
    if len(functions["train"].args.args) != 10 or len(functions["predict"].args.args) != 3:
        raise ValueError("generated train/predict signatures do not match the contract")
    return tree


def load_generated_module(source: str) -> dict[str, Any]:
    tree = validate_generated_source(source)

    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "numpy" and not fromlist and level == 0:
            return np
        raise ImportError("generated candidates may import only numpy")

    builtins = {
        "__import__": safe_import,
        "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "float": float, "int": int, "len": len,
        "list": list, "max": max, "min": min, "range": range, "round": round,
        "sum": sum, "tuple": tuple, "zip": zip, "ValueError": ValueError,
        "RuntimeError": RuntimeError,
    }
    namespace = {"__builtins__": builtins, "np": np}
    exec(compile(tree, "generated_candidate.py", "exec"), namespace, namespace)
    return {"train": namespace["train"], "predict": namespace["predict"]}


def source_diff(trial_id: str, source: str) -> str:
    path = f"generated_candidates/{trial_id}.py"
    return "".join(difflib.unified_diff([], source.splitlines(keepends=True), "/dev/null", path))


class GeneratedCandidateExecutor:
    def __init__(
        self, enc: dict, dimension: int, state_dir: Path, fingerprint: str, data_dir: Path
    ):
        self.enc = enc
        self.dimension = int(dimension)
        self.state_dir = Path(state_dir)
        self.fingerprint = fingerprint
        self.train_side = load_safe_side_features(data_dir, "train")
        self.valid_side = load_safe_side_features(data_dir, "valid")

    @staticmethod
    def _readonly(array: np.ndarray) -> np.ndarray:
        result = np.asarray(array).view()
        result.flags.writeable = False
        return result

    def execute(self, trial: dict, source: str) -> TrialOutcome:
        module = load_generated_module(source)
        Xtr, ytr, utr = self.enc["train"]
        Xva, yva, uva = self.enc["valid"]
        Xtr_arg, ytr_arg = self._readonly(Xtr), self._readonly(ytr)
        Xva_arg, yva_arg = self._readonly(Xva), self._readonly(yva)
        train_side_arg = {key: self._readonly(value) for key, value in self.train_side.items()}
        valid_side_arg = {key: self._readonly(value) for key, value in self.valid_side.items()}

        def score_validation(users, labels, scores):
            if users is not uva or labels is not yva_arg:
                raise ValueError("generated validation callback accepts official validation arrays only")
            return evaluate_checked(users, labels, scores)

        payload = module["train"](
            Xtr_arg, ytr_arg, tuple(utr), train_side_arg,
            Xva_arg, yva_arg, uva, valid_side_arg,
            self.dimension, score_validation,
        )
        if not isinstance(payload, dict) or set(payload) != {"state", "best_step", "history"}:
            raise ValueError("generated train result must contain state, best_step, and history")
        state = payload["state"]
        if not isinstance(state, dict) or not state:
            raise ValueError("generated state must be a non-empty array mapping")
        arrays = {}
        total_bytes = 0
        for key, value in state.items():
            if not isinstance(key, str) or not key.replace("_", "").isalnum():
                raise ValueError("generated state keys must be simple identifiers")
            array = np.asarray(value)
            if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
                raise ValueError("generated state arrays must be finite numeric values")
            arrays[key] = array
            total_bytes += array.nbytes
        if total_bytes > MAX_STATE_BYTES:
            raise ValueError("generated checkpoint exceeds the 512 MB limit")
        scores = np.asarray(
            module["predict"](self._readonly(Xva), valid_side_arg, arrays),
            dtype=np.float64,
        )
        if scores.shape != (len(Xva),) or not np.all(np.isfinite(scores)):
            raise ValueError("generated predict must return one finite validation score per row")
        validation = ValidationMetrics.from_mapping(evaluate_checked(uva, yva, scores))
        best_step = int(payload["best_step"])
        if best_step < 0 or not isinstance(payload["history"], list):
            raise ValueError("generated best_step/history are invalid")
        reject_test_data(payload["history"], "generated_history")
        artifact_dir = self.state_dir / "artifacts" / f"{trial['trial_id']}_generated"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        source_path = artifact_dir / "candidate.py"
        checkpoint_path = artifact_dir / "checkpoint.npz"
        source_path.write_text(source)
        np.savez_compressed(
            checkpoint_path,
            **{f"state_{key}": value for key, value in arrays.items()},
            dataset_fingerprint=np.asarray(self.fingerprint),
            source_sha256=np.asarray(hashlib.sha256(source.encode()).hexdigest()),
            best_step=np.asarray(best_step),
        )

        def artifact(path: Path, kind: str) -> dict[str, str]:
            return {"path": str(path), "kind": kind, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        return TrialOutcome(
            validation,
            best_step,
            "generated_candidate_completed",
            tuple(payload["history"]),
            (
                artifact(checkpoint_path, "generated_candidate_checkpoint"),
                artifact(source_path, "generated_candidate_source"),
            ),
        )


class CodeGeneratingResearchLoop:
    def __init__(
        self,
        autonomous_loop,
        compliance: JudgedRunCompliance,
        real_executor,
        provider: CodingProvider,
    ) -> None:
        self.loop = autonomous_loop
        self.store = autonomous_loop.store
        self.compliance = compliance
        self.provider = provider
        self.generated = GeneratedCandidateExecutor(
            real_executor.enc,
            real_executor.dim,
            autonomous_loop.state_dir,
            autonomous_loop.fingerprint,
            real_executor.data_dir,
        )

    def _context(self) -> dict[str, Any]:
        return {
            "benchmark": "KuaiRand-Pure",
            "task": "within-user logged-exposure ranking",
            "label": "long_view",
            "metrics": ["GAUC", "nDCG@5", "validation.primary"],
            "budget": {"used": self.store.consumed, "remaining": self.store.remaining},
            "best": None if self.store.best_trial() is None else {
                "trial_id": self.store.best_trial()["trial_id"],
                "validation": self.store.best_trial()["validation"],
            },
            "trials": [
                {
                    "trial_id": trial["trial_id"], "status": trial["status"],
                    "hypothesis": trial["hypothesis"], "validation": trial["validation"],
                    "error": trial["error"],
                    "learning_curve": (trial["result"] or {}).get("history", [])[-20:],
                }
                for trial in self.store.trials()[-10:]
            ],
            "recent_reflection_and_recovery_events": [
                event for event in self.store.events()
                if event["kind"] in {
                    "agent_reflection", "reflection_error", "codegen_recovery_required",
                    "generated_code_validation", "agent_decision",
                }
            ][-20:],
            "constraints": {
                "train_labels_only": True,
                "validation_selection_only": True,
                "test_labels_available": False,
                "external_training_data": False,
            },
        }

    def _active(self) -> dict | None:
        active = [
            trial for trial in self.store.trials()
            if trial["method"] == GENERATED_METHOD and trial["status"] in {"reserved", "running"}
        ]
        return None if not active else active[0]

    def step(self) -> dict[str, Any]:
        reason = self.loop.stop_reason()
        if reason:
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
                actor="agent-a-codegen-loop",
                decision_key=f"stop:{reason}",
            )
            return {"status": "stopped", "stop_reason": reason}
        self.compliance.start_if_needed()
        trial = self._active()
        if trial is None:
            pending = {
                "schema_version": CODEGEN_SCHEMA_VERSION,
                "research_action": {
                    "hypothesis": "LLM proposal pending",
                    "code_diff": "",
                    "code_diff_sha256": hashlib.sha256(b"").hexdigest(),
                    "change_kind": "proposal_pending",
                    "generated_by": "codex-coding-provider",
                    "manual_intervention": False,
                },
            }
            trial = self.store.reserve(
                GENERATED_METHOD,
                "Generate and validate the next evidence-driven candidate patch.",
                pending,
                seed=0,
            )
        if trial["status"] == "reserved":
            trial = self.store.mark_running(trial["trial_id"])
        artifact_dir = self.loop.state_dir / "codegen" / trial["trial_id"]
        try:
            action = trial["config"].get("research_action", {})
            source_path = artifact_dir / "candidate.py"
            if action.get("change_kind") == "proposal_pending":
                provider_result = self.compliance.execute_with_deadline(
                    lambda: self.provider.propose(self._context(), artifact_dir)
                )
                proposal = provider_result.proposal
                validate_generated_source(proposal.source_code)
                artifact_dir.mkdir(parents=True, exist_ok=True)
                source_path.write_text(proposal.source_code)
                diff = source_diff(trial["trial_id"], proposal.source_code)
                config = {
                    "schema_version": CODEGEN_SCHEMA_VERSION,
                    "research_action": {
                        "hypothesis": proposal.hypothesis,
                        "rationale": proposal.rationale,
                        "expected_metric_effect": proposal.expected_metric_effect,
                        "reflection_plan": proposal.reflection_plan,
                        "resource_estimate": proposal.resource_estimate,
                        "code_diff": diff,
                        "code_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
                        "source_sha256": hashlib.sha256(proposal.source_code.encode()).hexdigest(),
                        "before_source_sha256": None,
                        "after_source_sha256": hashlib.sha256(proposal.source_code.encode()).hexdigest(),
                        "change_kind": "generated_candidate_module",
                        "generated_by": provider_result.provider,
                        "provider_model": provider_result.model,
                        "provider_raw_log": provider_result.raw_log_path,
                        "manual_intervention": False,
                    },
                }
                trial = self.store.update_research_action(
                    trial["trial_id"], proposal.hypothesis, config
                )
                self.compliance.record_usage(
                    llm_input_tokens=provider_result.input_tokens,
                    llm_output_tokens=provider_result.output_tokens,
                )
                self.store.add_event(
                    "generated_code_validation",
                    {
                        "ast_policy": "passed",
                        "contract_compile": "passed",
                        "actual_source_sha256": hashlib.sha256(
                            source_path.read_bytes()
                        ).hexdigest(),
                    },
                    trial["trial_id"],
                )
                self.store.record_agent_decision(
                    stage="candidate_selection",
                    decision="execute_generated_candidate",
                    rationale=proposal.rationale,
                    evidence={
                        "hypothesis": proposal.hypothesis,
                        "expected_metric_effect": proposal.expected_metric_effect,
                        "source_sha256": hashlib.sha256(proposal.source_code.encode()).hexdigest(),
                        "ast_policy": "passed",
                        "contract_compile": "passed",
                        "budget": {"used": self.store.consumed, "remaining": self.store.remaining},
                    },
                    alternatives=("reject_generated_source", "request_manual_patch"),
                    selected_action=f"train_and_validate_{trial['trial_id']}",
                    actor=f"{provider_result.provider}:{provider_result.model}",
                    trial_id=trial["trial_id"],
                    decision_key=f"{trial['trial_id']}:execute-generated",
                )
            else:
                proposal = ResearchProposal(
                    trial["hypothesis"], action.get("rationale", ""),
                    action.get("expected_metric_effect", ""), source_path.read_text(),
                    action.get("reflection_plan", ""), action.get("resource_estimate", ""),
                )
            outcome = self.compliance.execute_with_deadline(
                lambda: self.generated.execute(trial, proposal.source_code)
            )
            completed = self.store.complete(trial["trial_id"], outcome)
            write_top1(self.store, self.loop.state_dir / "top1.json")
            is_top1 = self.store.best_trial()["trial_id"] == completed["trial_id"]
            best = self.store.best_trial()
            self.store.record_agent_decision(
                stage="candidate_disposition",
                decision="promote_validation_top1" if is_top1 else "retain_as_ablation_evidence",
                rationale=(
                    "Generated candidate is the stable validation.primary leader."
                    if is_top1 else
                    "Generated candidate completed but did not exceed the stable validation.primary leader."
                ),
                evidence={
                    "candidate_validation": completed["validation"],
                    "top1_trial_id": best["trial_id"],
                    "top1_validation": best["validation"],
                    "selection_rule": "validation.primary_then_lowest_ordinal",
                },
                alternatives=("select_by_auxiliary_metric", "select_by_hidden_result"),
                selected_action="update_top1_manifest" if is_top1 else "keep_current_top1",
                actor="agent-a-codegen-loop",
                trial_id=completed["trial_id"],
                decision_key=f"{completed['trial_id']}:completed-disposition",
            )
            reflection_context = {
                "trial_id": completed["trial_id"],
                "hypothesis": completed["hypothesis"],
                "validation": completed["validation"],
                "best_trial_id": self.store.best_trial()["trial_id"],
            }
            try:
                reflection, input_tokens, output_tokens = self.compliance.execute_with_deadline(
                    lambda: self.provider.reflect(reflection_context, artifact_dir)
                )
                self.store.add_event("agent_reflection", reflection, completed["trial_id"])
                self.compliance.record_usage(
                    llm_input_tokens=input_tokens, llm_output_tokens=output_tokens
                )
            except Exception as reflection_error:
                self.store.add_event(
                    "reflection_error",
                    {"error": f"{type(reflection_error).__name__}: {reflection_error}"},
                    completed["trial_id"],
                )
            return {
                "status": "completed",
                "trial_id": completed["trial_id"],
                "validation": completed["validation"],
            }
        except Exception as exc:
            current = self.store.get(trial["trial_id"])
            if current["status"] in {"reserved", "running"}:
                self.store.fail(trial["trial_id"], f"{type(exc).__name__}: {exc}")
            self.store.add_event(
                "codegen_recovery_required",
                {"error": f"{type(exc).__name__}: {exc}"},
                trial["trial_id"],
            )
            self.store.record_agent_decision(
                stage="candidate_disposition",
                decision="reject_failed_generated_candidate",
                rationale="The generated proposal or execution failed an infrastructure, safety, contract, or training requirement.",
                evidence={"error_type": type(exc).__name__, "error": str(exc)},
                alternatives=("promote_without_validation", "refund_consumed_budget"),
                selected_action=(
                    "fatal_stop_provider_outage"
                    if isinstance(exc, ProviderInfrastructureError)
                    else "retain_failure_as_recovery_evidence"
                ),
                actor="agent-a-codegen-loop",
                trial_id=trial["trial_id"],
                decision_key=f"{trial['trial_id']}:failed-disposition",
            )
            if isinstance(exc, ProviderInfrastructureError):
                self.loop._fatal_stop(f"coding_provider_infrastructure: {exc}")
            return {"status": "failed", "trial_id": trial["trial_id"], "error": str(exc)}
        finally:
            self.compliance.write_audit(self.store)

    def run(self, target_trials: int | None = None) -> dict[str, Any]:
        target = self.store.max_trials if target_trials is None else min(target_trials, self.store.max_trials)
        events = []
        while self.store.consumed < target and self.loop.stop_reason() is None:
            events.append(self.step())
        reason = self.loop.stop_reason()
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
                actor="agent-a-codegen-loop",
                decision_key=f"stop:{reason}",
            )
        report = self.loop.report(events)
        report["research_mode"] = "llm_code_generation"
        return report
