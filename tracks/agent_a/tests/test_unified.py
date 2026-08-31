from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tracks.agent_a.candidate import (
    AuxiliaryConfig,
    BPRConfig,
    CandidateConfig,
    CandidateSpec,
    HistoryConfig,
    ListwiseConfig,
)
from tracks.agent_a.contracts import ContractError, TrialOutcome, ValidationMetrics
from tracks.agent_a.evidence import build_evidence_registry
from tracks.agent_a.store import ResearchStore
from tracks.agent_a.unified_inspect import inspect
from tracks.agent_a.unified_runner import (
    TrainingNotAuthorized,
    UnifiedResult,
    UnifiedTrialRunner,
    dispatch_route,
    resolve_implementation,
)


FINGERPRINT = "sha256:fixture"
TOP_PRIMARY = 0.6015123724937439
REFERENCE_PRIMARY = 0.6014719009399414


def outcome(primary, step=1):
    return TrialOutcome(
        ValidationMetrics(primary, primary, primary), step, "validation_patience"
    )


def populate_evidence_store(path: Path) -> ResearchStore:
    store = ResearchStore(path, FINGERPRINT)
    definitions = {
        3: ("official_fm_baseline_reproduction", {}, 0.6014695465564728),
        4: ("fm_user_soft_target_listnet_finetune", {}, REFERENCE_PRIMARY),
        9: (
            "fm_user_soft_target_listnet_same_user_bpr_regularized",
            {"run_id": "B-01", "bpr_weight": 0.01},
            REFERENCE_PRIMARY,
        ),
        14: (
            "fm_user_soft_target_listnet_multitask_auxiliary",
            {"run_id": "M-01"},
            REFERENCE_PRIMARY,
        ),
        15: (
            "fm_user_soft_target_listnet_causal_history_fixed_gate",
            {"run_id": "F-02"},
            TOP_PRIMARY,
        ),
    }
    for ordinal in range(1, 16):
        if ordinal in definitions:
            method, config, primary = definitions[ordinal]
            trial = store.reserve(method, "fixture", config)
            store.complete(trial["trial_id"], outcome(primary, ordinal))
        else:
            trial = store.reserve("fixture", "filler", {"ordinal": ordinal})
            store.fail(trial["trial_id"], "fixture")
    return store


class CandidateIdentityTest(unittest.TestCase):
    def test_canonical_hash_ignores_key_order_and_detects_meaningful_changes(self):
        first = CandidateSpec.from_mapping({
            "dataset_fingerprint": FINGERPRINT,
            "config": {"seed": 0, "optimizer": {"learning_rate": 1e-6, "name": "adamw"}},
        })
        second = CandidateSpec.from_mapping({
            "config": {"optimizer": {"name": "adamw", "learning_rate": 1e-6}, "seed": 0},
            "dataset_fingerprint": FINGERPRINT,
        })
        self.assertEqual(first.identity, second.identity)
        changed = CandidateSpec(FINGERPRINT, CandidateConfig(seed=1))
        self.assertNotEqual(first.identity, changed.identity)

    def test_fingerprint_code_and_schema_versions_isolate_identity(self):
        config = CandidateConfig()
        base = CandidateSpec(FINGERPRINT, config)
        variants = [
            CandidateSpec("sha256:other", config),
            CandidateSpec(FINGERPRINT, config, code_version="v2"),
            CandidateSpec(FINGERPRINT, config, schema_version=2),
        ]
        self.assertTrue(all(item.identity != base.identity for item in variants))

    def test_invalid_and_contradictory_configs_are_rejected(self):
        with self.assertRaises(ValueError):
            CandidateConfig(bpr=BPRConfig(enabled=False, weight=0.0))
        explicit_zero = CandidateConfig(bpr=BPRConfig(enabled=True, weight=0.0))
        self.assertNotEqual(explicit_zero.to_dict(), CandidateConfig().to_dict())
        with self.assertRaises(ValueError):
            CandidateConfig(
                history=HistoryConfig(
                    True, 20, -0.05, False, False,
                    "causal_positive_mean_pool", 16, None, None, 0,
                ),
                bpr=BPRConfig(True, 0.01),
            )
        with self.assertRaises(ValueError):
            CandidateConfig.from_mapping({"unknown": 1})
        with self.assertRaises(ValueError):
            CandidateConfig(auxiliary=AuxiliaryConfig(True, 0.0, 0.01, None, 0.001, 1.0))


class UnifiedRunnerTest(unittest.TestCase):
    def test_dispatch_routes_and_resolves_existing_implementations(self):
        configs = {
            "baseline": CandidateConfig(listwise=ListwiseConfig(enabled=False)),
            "listwise": CandidateConfig(),
            "history": CandidateConfig(
                history=HistoryConfig(
                    True, 20, -0.05, False, False,
                    "causal_positive_mean_pool", 16, None, None, 0,
                )
            ),
            "bpr": CandidateConfig(bpr=BPRConfig(True, 0.01)),
            "multitask": CandidateConfig(
                auxiliary=AuxiliaryConfig(True, 0.01, 0.0, None, 0.001, 1.0)
            ),
        }
        for expected, config in configs.items():
            spec = CandidateSpec(FINGERPRINT, config)
            self.assertEqual(dispatch_route(spec), expected)
            self.assertTrue(callable(resolve_implementation(spec)))

    def test_exact_duplicate_reuse_does_not_consume_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ResearchStore(Path(directory) / "ledger.sqlite3", FINGERPRINT)
            spec = CandidateSpec(FINGERPRINT, CandidateConfig())
            trial = store.reserve("unified", "fixture", spec.ledger_config())
            store.complete(trial["trial_id"], outcome(0.55))
            before = store.consumed
            called = False

            def candidate(_trial):
                nonlocal called
                called = True
                return outcome(0.6)

            reused, was_reused = UnifiedTrialRunner(store).execute(
                spec, "unified", "fixture", candidate, allow_training=False
            )
            self.assertTrue(was_reused)
            self.assertFalse(called)
            self.assertEqual(reused.trial_id, "trial-01")
            self.assertEqual(store.consumed, before)
            other = CandidateSpec(FINGERPRINT, CandidateConfig(seed=1))
            with self.assertRaises(TrainingNotAuthorized):
                UnifiedTrialRunner(store).execute(
                    other, "unified", "fixture", candidate, allow_training=False
                )
            self.assertEqual(store.consumed, before)

    def test_unified_result_rejects_bad_primary_and_selection_keys(self):
        valid = {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55}
        with self.assertRaises(ContractError):
            UnifiedResult("t", "completed", "i", FINGERPRINT, valid, 1, None, None, False, {}, "auxiliary.click")
        with self.assertRaises(ContractError):
            UnifiedResult(
                "t", "completed", "i", FINGERPRINT,
                {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.7},
                1, None, None, False, {},
            )
        with self.assertRaises(ContractError):
            UnifiedResult("t", "completed", "i", FINGERPRINT, valid, 1, None, None, True, {})


class EvidenceTest(unittest.TestCase):
    def test_read_only_reconstruction_and_current_new_fingerprint_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            store = populate_evidence_store(Path(directory) / "ledger.sqlite3")
            before = store.consumed
            registry = build_evidence_registry(store)
            self.assertEqual(store.consumed, before)
            self.assertEqual(registry["top1"]["trial_id"], "trial-15")
            self.assertEqual(
                registry["default_current_candidate_modules"],
                ["no_history_soft_target_listnet", "history_last20_fixed_gate_minus_0_05"],
            )
            history = registry["modules"]["history_last20_fixed_gate_minus_0_05"]
            self.assertEqual(history["classification"], "positive_below_epsilon")
            for name in ("bpr_weights", "multitask_weights"):
                module = registry["modules"][name]
                self.assertEqual(module["classification"], "no_positive_gain")
                self.assertFalse(module["enabled_for_current_search"])
                self.assertTrue(module["eligible_as_new_dataset_anchor"])
                self.assertEqual(module["new_dataset_evidence"], "unobserved_requires_controlled_validation")

    def test_inspect_cli_path_neither_trains_nor_changes_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            state_root = Path(directory) / "state"
            ledger = state_root / "fixture" / "research.sqlite3"
            store = populate_evidence_store(ledger)
            before = store.consumed
            with patch(
                "tracks.agent_a.unified_inspect.fingerprint_dataset",
                return_value={"dataset_fingerprint": FINGERPRINT},
            ):
                result = inspect(Path(directory) / "data", state_root, write_reports=False)
            self.assertEqual(result["mode"], "inspect_only_no_training")
            self.assertEqual(result["budget_before"], result["budget_after"])
            self.assertEqual(result["top1"]["trial_id"], "trial-15")
            self.assertEqual(ResearchStore(ledger, FINGERPRINT).consumed, before)


if __name__ == "__main__":
    unittest.main()
