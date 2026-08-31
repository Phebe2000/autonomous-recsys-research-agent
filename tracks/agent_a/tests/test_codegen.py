import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tracks.agent_a.codegen import (
    CodexCodingProvider,
    CodeGeneratingResearchLoop,
    ProviderResult,
    ProviderInfrastructureError,
    ResearchProposal,
    validate_generated_source,
)
from tracks.agent_a.finalize import score_locked_exposures
from tracks.agent_a.judged_cli import _active, initialize, run as judged_run
from tracks.agent_a.real_executor import RealCandidateExecutor
from tracks.agent_a.safe_submit_check import check_submission
from tracks.agent_a.tests.test_readiness import write_fixture_dataset


GENERATED_SOURCE = '''import numpy as np

def train(X_train, y_train, users_train, train_side, X_valid, y_valid, users_valid, valid_side, dimension, score_validation):
    state = {"keys": X_valid.copy(), "labels": y_valid.copy()}
    scores = predict(X_valid, valid_side, state)
    metrics = score_validation(users_valid, y_valid, scores)
    return {"state": state, "best_step": 1, "history": [{"step": 1, "validation": metrics}]}

def predict(X, side, state):
    scores = np.zeros(len(X), dtype=np.float64)
    for index in range(len(X)):
        matches = np.all(state["keys"] == X[index], axis=1)
        if np.any(matches):
            scores[index] = np.max(state["labels"][matches])
    return scores
'''


class ScriptedCodingProvider:
    def propose(self, context, artifact_dir):
        self.context = context
        return ProviderResult(
            ResearchProposal(
                "Smoke-test a generated validation lookup candidate.",
                "Exercises real generated source, not a configuration alias.",
                "Expected to rank the tiny fixture validation rows perfectly.",
                GENERATED_SOURCE,
                "Compare official validation primary and inspect overfitting risk.",
                "Tiny NumPy fixture only.",
            ),
            "scripted-test-provider",
            "fixture",
            11,
            7,
        )

    def reflect(self, context, artifact_dir):
        return {
            "hypothesis_supported": True,
            "diagnosis": "Fixture lookup worked; this intentionally demonstrates overfitting.",
            "next_action": "Reject lookup behavior for a real benchmark candidate.",
        }, 5, 4


class BrokenCodingProvider:
    def propose(self, context, artifact_dir):
        raise ProviderInfrastructureError("provider unavailable")

    def reflect(self, context, artifact_dir):
        raise AssertionError("reflection should not run")


class GeneratedSourceSafetyTest(unittest.TestCase):
    def test_codex_provider_uses_supported_read_only_cli_flags(self):
        provider = CodexCodingProvider(Path.cwd(), executable="codex", model="fixture")
        source = Path(__file__).parents[1] / "codegen.py"
        text = source.read_text()
        self.assertIn('"--sandbox", "read-only"', text)
        self.assertNotIn('"--ask-for-approval"', text)

    def test_contract_accepts_numpy_candidate_and_rejects_file_or_os_access(self):
        validate_generated_source(GENERATED_SOURCE)
        with self.assertRaisesRegex(ValueError, "imports"):
            validate_generated_source(
                "import os\ndef train(a,b,c,d,e,f,g,h,i,j): return {}\ndef predict(a,b,c): return a\n"
            )
        with self.assertRaisesRegex(ValueError, "forbidden generated name"):
            validate_generated_source(
                "def train(a,b,c,d,e,f,g,h,i,j):\n open('x')\n return {}\n"
                "def predict(a,b,c): return a\n"
            )
        with self.assertRaisesRegex(ValueError, "forbidden generated attribute"):
            validate_generated_source(
                "import numpy as np\n"
                "def train(a,b,c,d,e,f,g,h,i,j): return {\"state\": {\"x\": a}, \"best_step\": 0, \"history\": []}\n"
                "def predict(a,b,c): return np.genfromtxt(\"secret.csv\")\n"
            )


class CodeGeneratingLoopTest(unittest.TestCase):
    def test_judged_dispatch_keeps_screening_controlled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, state_root = root / "data", root / "judged"
            write_fixture_dataset(data_dir)
            initialize(data_dir, state_root)
            with patch(
                "tracks.agent_a.judged_cli.CodexCodingProvider",
                return_value=ScriptedCodingProvider(),
            ):
                report = judged_run(
                    data_dir, state_root, False,
                    provider_model="fixture", target_trials=8,
                )
            _, _, loop = _active(data_dir, state_root)
            self.assertEqual(loop.store.consumed, 8)
            self.assertEqual(
                loop.store.get("trial-07")["method"],
                "train_causal_behavioral_lambdarank_fm_ensemble",
            )
            self.assertEqual(
                loop.store.get("trial-08")["method"],
                "train_causal_behavioral_lambdarank_fm_ensemble",
            )
            self.assertEqual(report["research_mode"], "hybrid_optuna_and_llm_code_generation")

    def test_provider_infrastructure_failure_counts_once_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir, state_root = root / "data", root / "judged"
            write_fixture_dataset(data_dir)
            initialize(data_dir, state_root)
            _, compliance, loop = _active(data_dir, state_root)
            real = RealCandidateExecutor(data_dir, loop.state_dir, loop.store, loop.fingerprint)
            loop.run(real, target_trials=6)
            codegen = CodeGeneratingResearchLoop(loop, compliance, real, BrokenCodingProvider())
            failed = codegen.step()
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(loop.store.consumed, 7)
            self.assertIn("coding_provider_infrastructure", loop.stop_reason())
            stopped = codegen.step()
            self.assertEqual(stopped["status"], "stopped")
            self.assertEqual(loop.store.consumed, 7)

    def test_real_patch_train_reflect_audit_lock_and_final_predict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            state_root = root / "judged"
            write_fixture_dataset(data_dir)
            # Make the baseline's train video signal disagree with public
            # validation so the generated lookup wins the fixture Top-1.
            log_path = data_dir / "log_standard_4_22_to_5_08_pure.csv"
            with log_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fieldnames, rows = reader.fieldnames, list(reader)
            for row in rows:
                if 20220422 <= int(row["date"]) <= 20220428:
                    row["long_view"] = "0" if row["long_view"] != "0" else "1"
            with log_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            initialize(data_dir, state_root)
            _, compliance, loop = _active(data_dir, state_root)
            real = RealCandidateExecutor(data_dir, loop.state_dir, loop.store, loop.fingerprint)
            loop.run(real, target_trials=6)
            provider = ScriptedCodingProvider()
            codegen = CodeGeneratingResearchLoop(loop, compliance, real, provider)
            result = codegen.step()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(loop.store.consumed, 7)
            generated = loop.store.get("trial-07")
            action = generated["config"]["research_action"]
            self.assertEqual(action["change_kind"], "generated_candidate_module")
            self.assertTrue(action["code_diff"].startswith("--- /dev/null"))
            self.assertEqual(loop.store.best_trial()["trial_id"], "trial-07")
            audit = compliance.write_audit(loop.store)
            iteration = audit["iterations"][-1]
            self.assertTrue(any(event["kind"] == "agent_reflection" for event in iteration["events"]))
            self.assertEqual(audit["resource_usage"]["llm_input_tokens"], 16)
            self.assertEqual(audit["resource_usage"]["llm_output_tokens"], 11)

            loop.lock()
            output = root / "submission.csv"
            scored = score_locked_exposures(data_dir, state_root, output, judged=True)
            self.assertEqual(scored["locked_trial_id"], "trial-07")
            self.assertTrue(check_submission(output, data_dir)["schema_valid"])


if __name__ == "__main__":
    unittest.main()
