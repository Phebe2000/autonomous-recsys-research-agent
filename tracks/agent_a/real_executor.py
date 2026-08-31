"""Adapter from unified candidates to existing real validation-only trainers."""

from __future__ import annotations

from pathlib import Path

from data import encode, load

from .auxiliary import load_training_auxiliary, read_training_identities
from .bpr_pipeline import _train_bpr
from .candidate import CandidateSpec
from .history import build_causal_history
from .history_pipeline import _load_baseline, _train_history
from .listwise import group_user_exposures
from .multitask_pipeline import _train_multitask
from .pipeline import train_baseline_candidate, train_listnet_candidate
from .store import ResearchStore


class RealCandidateExecutor:
    """Prepare shared data once and delegate all training mathematics unchanged."""

    def __init__(
        self,
        data_dir: Path,
        state_dir: Path,
        store: ResearchStore,
        fingerprint: str,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.state_dir = Path(state_dir)
        self.store = store
        self.fingerprint = fingerprint
        self.splits = load(str(data_dir))
        self.enc, self.dim = encode(self.splits)
        self._baseline = None
        self._baseline_artifact = None
        self._groups = None
        self._history = None
        self._auxiliary = None

    def _load_shared(self):
        if self._baseline is None:
            self._baseline, _, self._baseline_artifact = _load_baseline(
                self.store, self.dim, self.fingerprint
            )
        if self._groups is None:
            self._groups = group_user_exposures(
                self.enc["train"][2],
                self.enc["train"][1],
                self._baseline.predict(self.enc["train"][0]),
                64,
            )
        return self._baseline, self._groups

    @staticmethod
    def _prior(spec: CandidateSpec) -> dict:
        config = spec.config
        return {
            "schema_version": 1,
            "method": "fm_user_soft_target_listnet_finetune",
            "prior_policy": "configuration_and_method_only",
            "provenance": "unified current-dataset candidate; no transferred artifact or score",
            "initialization": "fresh_validation_best_official_fm_on_current_dataset",
            "seed": config.seed,
            "hyperparameters": {
                "hard_negative_cap": config.listwise.hard_negative_cap,
                "k": config.backbone.latent_dim,
                "lr": config.optimizer.learning_rate,
                "max_epochs": config.resources.max_epochs,
                "metric_weighting": config.listwise.metric_weighting,
                "optimizer": config.optimizer.name,
                "patience_checks": config.resources.patience_checks,
                "score_temperature": config.listwise.score_temperature,
                "target_temperature": config.listwise.target_temperature,
                "update_embeddings": True,
                "user_batch_size": config.resources.user_batch_size,
                "validation_interval": config.resources.validation_interval,
                "warmup_steps": config.optimizer.warmup_steps,
                "weight_decay": config.optimizer.weight_decay,
            },
        }

    def __call__(self, trial, spec: CandidateSpec, module: str, _reporter):
        artifact_dir = self.state_dir / "artifacts" / f"{trial['trial_id']}_unified"
        if module == "baseline":
            _, outcome = train_baseline_candidate(
                self.enc,
                self.dim,
                artifact_dir / "baseline_seed0.npz",
                self.fingerprint,
            )
            return outcome
        baseline, groups = self._load_shared()
        prior = self._prior(spec)
        if module == "listwise":
            _, outcome = train_listnet_candidate(
                self.enc,
                baseline,
                prior,
                artifact_dir / "listnet.npz",
                self.fingerprint,
            )
            return outcome
        legacy = {
            "schema_version": 1,
            "run_id": f"U-{trial['trial_id']}",
            "method": trial["method"],
            "seed": spec.config.seed,
            "listwise_prior": prior,
            "initialization": {
                "source": "fresh_validation_best_official_fm_on_current_dataset",
                "baseline_artifact_sha256": self._baseline_artifact["sha256"],
            },
        }
        if module == "history":
            if self._history is None:
                self._history = build_causal_history(self.data_dir, self.splits, self.enc)
            history = spec.config.history
            legacy["history"] = {
                "last_n": history.last_n,
                "history_dim": history.embedding_dim,
                "history_gate_initial": history.gate,
                "history_lr": history.history_learning_rate,
                "gate_lr": history.gate_learning_rate,
                "history_unfreeze_step": history.unfreeze_step,
                "train_history_embeddings": history.train_embeddings,
                "train_gate": history.train_gate,
            }
            return _train_history(
                trial, self.enc, baseline, groups, self._history,
                legacy, self.state_dir, self.fingerprint,
            )
        if module == "bpr":
            legacy["bpr_weight"] = spec.config.bpr.weight
            return _train_bpr(
                trial, self.enc, baseline, groups, legacy,
                self.state_dir, self.fingerprint,
            )
        if module in {"click", "play"}:
            if self._auxiliary is None:
                identities = read_training_identities(self.data_dir)
                self._auxiliary = load_training_auxiliary(
                    self.data_dir, self.splits, self.enc, identities
                )
            auxiliary = spec.config.auxiliary
            legacy["auxiliary"] = {
                "is_click_weight": auxiliary.is_click_weight,
                "play_time_weight": auxiliary.play_time_weight,
                "head_lr": auxiliary.head_learning_rate,
                "huber_delta": auxiliary.huber_delta,
            }
            return _train_multitask(
                trial, self.enc, baseline, groups, self._auxiliary,
                legacy, self.state_dir, self.fingerprint,
            )
        raise ValueError(f"unsupported real module: {module}")
