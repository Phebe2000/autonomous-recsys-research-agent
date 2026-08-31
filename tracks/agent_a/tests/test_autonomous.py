import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import optuna

from tracks.agent_a.autonomous import (
    AutonomousResearchLoop,
    SyntheticExecutor,
    phase_for_ordinal,
)
from tracks.agent_a.autonomous_cli import inspect as cli_inspect
from tracks.agent_a.autonomous_cli import plan as cli_plan
from tracks.agent_a.autonomous_cli import run_real
from tracks.agent_a.autonomous_cli import simulate as cli_simulate
from tracks.agent_a.evidence import AUDITED_KUAIRAND_FINGERPRINT
from tracks.agent_a.candidate import CandidateSpec
from tracks.agent_a.contracts import ContractError, TrialOutcome, ValidationMetrics
from tracks.agent_a.runner import TrialPruned
from tracks.agent_a.store import BudgetExhausted, ResearchStore


FINGERPRINT = "sha256:synthetic-agent-a-tpe-v1"


def make_loop(root: Path, max_trials=50, convergence=False):
    return AutonomousResearchLoop(
        root / "synthetic-agent-a-tpe-v1",
        FINGERPRINT,
        simulation=True,
        max_trials=max_trials,
        convergence_enabled=convergence,
    )


class PhaseAndBudgetTest(unittest.TestCase):
    def test_phase_boundaries(self):
        expected = {
            1: "controlled_anchors", 6: "controlled_anchors",
            7: "single_module_screening", 14: "single_module_screening",
            15: "tpe_conditional_search", 34: "tpe_conditional_search",
            35: "local_refinement", 42: "local_refinement",
            43: "automatic_ablation", 47: "automatic_ablation",
            48: "finalist_verification", 49: "finalist_verification", 50: "reserve",
        }
        for ordinal, phase in expected.items():
            self.assertEqual(phase_for_ordinal(ordinal), phase)

    def test_full_fifty_covers_every_phase_and_fifty_first_is_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = make_loop(Path(directory))
            report = loop.run(SyntheticExecutor(), 50)
            self.assertEqual(report["budget"], {"used": 50, "remaining": 0, "maximum": 50})
            counts = {}
            for mapping in report["trial_mapping"]:
                counts[mapping["phase"]] = counts.get(mapping["phase"], 0) + 1
            self.assertEqual(counts, {
                "controlled_anchors": 6,
                "single_module_screening": 8,
                "tpe_conditional_search": 20,
                "local_refinement": 8,
                "automatic_ablation": 5,
                "finalist_verification": 2,
                "reserve": 1,
            })
            self.assertEqual(loop.step(SyntheticExecutor())["stop_reason"], "budget_exhausted")
            with self.assertRaises(BudgetExhausted):
                loop.store.reserve("extra", "51st", {})
            self.assertTrue(any(item["status"] == "pruned" for item in report["trial_mapping"]))

    def test_conditional_eligibility_uses_anchor_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = make_loop(Path(directory), max_trials=6)
            loop.run(SyntheticExecutor(), 6)
            self.assertEqual(loop.eligible_modules(), ["listwise", "history", "click", "play"])
            self.assertNotIn("bpr", loop.eligible_modules())

    def test_convergence_never_fires_before_all_anchors(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = make_loop(Path(directory), max_trials=20, convergence=True)
            for _ in range(5):
                loop.step(SyntheticExecutor())
                self.assertIsNone(loop.stop_reason())
            loop.step(SyntheticExecutor())
            self.assertIsNone(loop.stop_reason())
            for index in range(3):
                trial = loop.store.reserve("fixture", "flat", {"flat": index})
                loop.store.complete(
                    trial["trial_id"],
                    TrialOutcome(ValidationMetrics(0.6, 0.6, 0.6), 1, "fixture"),
                )
            self.assertIn("converged_no_0.002_gain_for_3", loop.stop_reason())


class OptunaPersistenceTest(unittest.TestCase):
    def test_ask_tell_mapping_persistent_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_loop(root, max_trials=20)
            first.run(SyntheticExecutor(), 10)
            first_mapping = first.report()["trial_mapping"]
            resumed = make_loop(root, max_trials=20)
            resumed.run(SyntheticExecutor(), 15)
            mapping = resumed.report()["trial_mapping"]
            self.assertEqual(mapping[:10], first_mapping)
            self.assertEqual(len(mapping), 15)
            self.assertEqual(len(resumed.study.trials), 15)
            for index, item in enumerate(mapping):
                self.assertEqual(item["optuna_trial_number"], index)

    def test_tpe_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = make_loop(Path(first_dir), max_trials=20)
            second = make_loop(Path(second_dir), max_trials=20)
            first.run(SyntheticExecutor(), 20)
            second.run(SyntheticExecutor(), 20)
            first_trials = [(trial.params, trial.value, trial.state.name) for trial in first.study.trials]
            second_trials = [(trial.params, trial.value, trial.state.name) for trial in second.study.trials]
            self.assertEqual(first_trials, second_trials)
            self.assertIsInstance(first.study.sampler, optuna.samplers.TPESampler)

    def test_interrupted_after_ask_and_after_reserve_recover_without_double_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            loop = make_loop(root, max_trials=5)
            asked = loop.step(SyntheticExecutor(), interrupt_after_ask=True)
            self.assertEqual(asked["optuna_trial_number"], 0)
            self.assertEqual(loop.store.consumed, 0)
            resumed = make_loop(root, max_trials=5)
            completed = resumed.step(SyntheticExecutor())
            self.assertEqual(completed["trial_id"], "trial-01")
            self.assertEqual(resumed.store.consumed, 1)
            reserved = resumed.step(SyntheticExecutor(), interrupt_after_reserve=True)
            self.assertEqual(resumed.store.consumed, 2)
            again = make_loop(root, max_trials=5)
            recovered = again.step(SyntheticExecutor())
            self.assertEqual(recovered["trial_id"], reserved["trial_id"])
            self.assertEqual(again.store.consumed, 2)

    def test_exact_duplicate_tells_optuna_without_ledger_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = make_loop(Path(directory), max_trials=5)
            loop.step(SyntheticExecutor())
            original = loop.study.trials[0].user_attrs["candidate_spec"]
            spec = CandidateSpec.from_mapping(original)
            before = loop.store.consumed
            with patch.object(loop, "suggest", return_value=(None, "baseline", spec)):
                reused = loop.step(SyntheticExecutor())
            self.assertEqual(reused["status"], "reused")
            self.assertEqual(loop.store.consumed, before)
            self.assertEqual(len(loop.study.trials), 2)
            self.assertEqual(loop.study.trials[1].state, optuna.trial.TrialState.COMPLETE)


class GuardAndIsolationTest(unittest.TestCase):
    def test_invalid_suggestion_and_validation_guard_have_explicit_budget_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = make_loop(Path(directory), max_trials=5)
            with patch.object(loop, "suggest", side_effect=ValueError("invalid candidate")):
                with self.assertRaises(ValueError):
                    loop.step(SyntheticExecutor())
            self.assertEqual(loop.store.consumed, 0)
            self.assertEqual(loop.study.trials[0].state, optuna.trial.TrialState.FAIL)
            self.assertIn("fatal_guard_failure", loop.stop_reason())

        with tempfile.TemporaryDirectory() as directory:
            loop = make_loop(Path(directory), max_trials=5)
            def invalid_metrics(_trial, _spec, _module, reporter):
                reporter.report(
                    {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55, "test_auc": 0.9}, 1
                )

            with self.assertRaises(ContractError):
                loop.step(invalid_metrics)
            self.assertEqual(loop.store.consumed, 1)
            self.assertEqual(loop.store.get("trial-01")["status"], "failed")
            self.assertIn("fatal_guard_failure", loop.stop_reason())

    def test_pruning_counts_and_only_accepts_validation_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = make_loop(Path(directory), max_trials=2)

            def prune(_trial, _spec, _module, reporter):
                reporter.report(
                    {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55, "rows": 2, "users": 1}, 1
                )
                raise TrialPruned("fixture")

            result = loop.step(prune)
            self.assertEqual(result["status"], "pruned")
            self.assertEqual(loop.store.consumed, 1)
            self.assertEqual(loop.store.get("trial-01")["status"], "pruned")

    def test_simulation_and_production_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = ResearchStore(root / "production.sqlite3", "sha256:production")
            before = hashlib.sha256(production.path.read_bytes()).hexdigest()
            simulation = make_loop(root / "simulation", max_trials=3)
            report = simulation.run(SyntheticExecutor(), 3)
            self.assertTrue(report["simulation"])
            self.assertFalse(report["production_top1_eligible"])
            self.assertEqual(production.consumed, 0)
            self.assertEqual(hashlib.sha256(production.path.read_bytes()).hexdigest(), before)
            with self.assertRaises(ValueError):
                AutonomousResearchLoop(root / "bad", FINGERPRINT, simulation=False)
            production_runtime = Path(__file__).parents[1] / "runtime"
            with self.assertRaisesRegex(ValueError, "production runtime"):
                cli_simulate(production_runtime, 2)

    def test_lock_is_immutable_and_blocks_future_loop_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            loop = make_loop(Path(directory), max_trials=10)
            loop.run(SyntheticExecutor(), 6)
            first = loop.lock()
            second = loop.lock()
            self.assertEqual(first, second)
            self.assertTrue(first["immutable"])
            self.assertFalse(first["production_top1_eligible"])
            before = loop.store.consumed
            stopped = loop.step(SyntheticExecutor())
            self.assertEqual(stopped["stop_reason"], "user_lock")
            self.assertEqual(loop.store.consumed, before)

    def test_plan_and_inspect_do_not_create_or_modify_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "tracks.agent_a.autonomous_cli.fingerprint_dataset",
                return_value={"dataset_fingerprint": "sha256:new"},
            ):
                planned = cli_plan(root / "data", root / "state")
                inspected = cli_inspect(root / "data", root / "state")
            self.assertTrue(planned["read_only"] and inspected["read_only"])
            self.assertEqual(planned["budget"]["used"], 0)
            self.assertFalse((root / "state" / "new" / "research.sqlite3").exists())

    def test_real_run_rejects_audited_current_fingerprint_before_state_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch(
                "tracks.agent_a.autonomous_cli.fingerprint_dataset",
                return_value={"dataset_fingerprint": AUDITED_KUAIRAND_FINGERPRINT},
            ):
                with self.assertRaisesRegex(RuntimeError, "KuaiRand"):
                    run_real(root / "data", root / "state", resume=False)
            self.assertFalse((root / "state").exists())


if __name__ == "__main__":
    unittest.main()
