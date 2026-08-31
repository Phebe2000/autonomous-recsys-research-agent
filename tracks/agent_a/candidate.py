"""Strict serializable candidate configuration and content-scoped identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
from typing import Any, Mapping, TypeVar

from .fingerprint import canonical_json


CANDIDATE_SCHEMA_VERSION = 3
CODE_VERSION = "agent-a-unified-candidate-v3"
T = TypeVar("T")


def _strict_dataclass(cls: type[T], value: Mapping[str, Any] | T, path: str) -> T:
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    allowed = {item.name for item in fields(cls)}
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unsupported {path} fields: {sorted(extra)}")
    return cls(**dict(value))


@dataclass(frozen=True)
class BackboneConfig:
    kind: str = "fm"
    latent_dim: int = 16

    def validate(self) -> None:
        if self.kind not in {"fm", "lambdarank", "fm_lambdarank_ensemble"}:
            raise ValueError("unsupported candidate backbone")
        if self.latent_dim <= 0:
            raise ValueError("backbone latent dimension must be positive")


@dataclass(frozen=True)
class RankerConfig:
    enabled: bool = False
    feature_schema: str | None = None
    smoothing: float | None = None
    n_estimators: int | None = None
    learning_rate: float | None = None
    num_leaves: int | None = None
    min_child_samples: int | None = None
    validation_interval: int | None = None
    blend_weight: float | None = None

    def validate(self) -> None:
        values = (
            self.feature_schema, self.smoothing, self.n_estimators,
            self.learning_rate, self.num_leaves, self.min_child_samples,
            self.validation_interval,
        )
        if not self.enabled and any(value is not None for value in values):
            raise ValueError("disabled ranker must not carry active settings")
        if self.enabled:
            if self.feature_schema != "causal_behavioral_v1":
                raise ValueError("enabled ranker requires the causal behavioral feature schema")
            numeric = (
                self.smoothing, self.n_estimators, self.learning_rate,
                self.num_leaves, self.min_child_samples, self.validation_interval,
            )
            if any(value is None or value <= 0 for value in numeric):
                raise ValueError("enabled ranker requires positive explicit hyperparameters")
            if self.blend_weight is not None and not 0 < self.blend_weight <= 1:
                raise ValueError("ranker blend weight must be in (0, 1]")


@dataclass(frozen=True)
class ListwiseConfig:
    enabled: bool = True
    score_temperature: float = 1.0
    target_temperature: float = 0.5
    metric_weighting: bool = True
    hard_negative_cap: int = 64

    def validate(self) -> None:
        if self.score_temperature <= 0 or self.target_temperature <= 0:
            raise ValueError("Listwise temperatures must be positive")
        if self.hard_negative_cap < 1:
            raise ValueError("hard_negative_cap must be positive")


@dataclass(frozen=True)
class HistoryConfig:
    enabled: bool = False
    last_n: int | None = None
    gate: float | None = None
    train_embeddings: bool | None = None
    train_gate: bool | None = None
    encoder: str | None = None
    embedding_dim: int | None = None
    history_learning_rate: float | None = None
    gate_learning_rate: float | None = None
    unfreeze_step: int | None = None

    def validate(self) -> None:
        settings = (
            self.last_n, self.gate, self.train_embeddings, self.train_gate,
            self.encoder, self.embedding_dim, self.history_learning_rate,
            self.gate_learning_rate, self.unfreeze_step,
        )
        if not self.enabled and any(value is not None for value in settings):
            raise ValueError("disabled history must not carry active settings")
        if self.enabled:
            if self.last_n is not None and self.last_n <= 0:
                raise ValueError("history last_n must be positive or null for all")
            if self.gate is None or self.train_embeddings is None or self.train_gate is None:
                raise ValueError("enabled history requires gate and trainability settings")
            if self.encoder != "causal_positive_mean_pool" or self.embedding_dim is None or self.embedding_dim <= 0:
                raise ValueError("enabled history requires the supported encoder and positive dimension")
            if self.unfreeze_step is None or self.unfreeze_step < 0:
                raise ValueError("enabled history requires a non-negative unfreeze_step")
            if self.history_learning_rate is not None and self.history_learning_rate <= 0:
                raise ValueError("history learning rate must be positive when specified")
            if self.gate_learning_rate is not None and self.gate_learning_rate <= 0:
                raise ValueError("gate learning rate must be positive when specified")


@dataclass(frozen=True)
class BPRConfig:
    enabled: bool = False
    weight: float | None = None

    def validate(self) -> None:
        if not self.enabled and self.weight is not None:
            raise ValueError("disabled BPR must use weight=null, not weight=0")
        if self.enabled and (self.weight is None or self.weight < 0):
            raise ValueError("enabled BPR requires a non-negative explicit weight")


@dataclass(frozen=True)
class AuxiliaryConfig:
    enabled: bool = False
    is_click_weight: float | None = None
    play_time_weight: float | None = None
    play_transform: str | None = None
    head_learning_rate: float | None = None
    huber_delta: float | None = None

    def validate(self) -> None:
        values = (
            self.is_click_weight, self.play_time_weight, self.play_transform,
            self.head_learning_rate, self.huber_delta,
        )
        if not self.enabled and any(value is not None for value in values):
            raise ValueError("disabled auxiliary tasks must not carry weights or transforms")
        if self.enabled:
            if self.is_click_weight is None or self.play_time_weight is None:
                raise ValueError("enabled auxiliary tasks require explicit weights, including zero")
            if self.is_click_weight < 0 or self.play_time_weight < 0:
                raise ValueError("auxiliary weights must be non-negative")
            if self.head_learning_rate is None or self.head_learning_rate <= 0:
                raise ValueError("enabled auxiliary tasks require a positive head learning rate")
            if self.huber_delta is None or self.huber_delta <= 0:
                raise ValueError("enabled auxiliary tasks require a positive Huber delta")
            if self.play_time_weight > 0 and self.play_transform != "train_log1p_zscore":
                raise ValueError("play_time requires the training-only log1p z-score transform")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    warmup_steps: int = 50

    def validate(self) -> None:
        if self.name != "adamw" or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("unsupported optimizer configuration")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")


@dataclass(frozen=True)
class InitializationConfig:
    source: str = "fresh_validation_best_official_fm_current_dataset"
    artifact_sha256: str | None = None

    def validate(self) -> None:
        if self.source != "fresh_validation_best_official_fm_current_dataset":
            raise ValueError("transferred checkpoints or unsupported initialization are forbidden")


@dataclass(frozen=True)
class ResourceLimits:
    max_epochs: int = 2
    user_batch_size: int = 256
    validation_interval: int = 10
    patience_checks: int = 6
    dataset_trial_cap: int = 50

    def validate(self) -> None:
        if min(self.max_epochs, self.user_batch_size, self.validation_interval, self.patience_checks) <= 0:
            raise ValueError("resource limits must be positive")
        if not 1 <= self.dataset_trial_cap <= 50:
            raise ValueError("dataset trial cap must be in [1, 50]")


@dataclass(frozen=True)
class CandidateConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    ranker: RankerConfig = field(default_factory=RankerConfig)
    listwise: ListwiseConfig = field(default_factory=ListwiseConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    bpr: BPRConfig = field(default_factory=BPRConfig)
    auxiliary: AuxiliaryConfig = field(default_factory=AuxiliaryConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    seed: int = 0
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
    resources: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        for value in (
            self.backbone, self.ranker, self.listwise, self.history, self.bpr,
            self.auxiliary, self.optimizer, self.initialization, self.resources,
        ):
            value.validate()
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        enabled_modules = sum((self.history.enabled, self.bpr.enabled, self.auxiliary.enabled))
        if enabled_modules > 1:
            raise ValueError("history, BPR, and auxiliary tasks cannot be combined in current capabilities")
        if enabled_modules and not self.listwise.enabled:
            raise ValueError("history, BPR, and auxiliary tasks require Listwise")
        ranker_backbone = self.backbone.kind in {"lambdarank", "fm_lambdarank_ensemble"}
        if self.ranker.enabled != ranker_backbone:
            raise ValueError("ranker enabled state must match the selected backbone")
        if (
            self.backbone.kind == "lambdarank"
            and self.ranker.blend_weight not in {None, 1.0}
        ):
            raise ValueError("standalone LambdaRank cannot use an FM blend weight")
        if ranker_backbone and (self.listwise.enabled or enabled_modules):
            raise ValueError("ranker candidates cannot also enable Listwise, History, BPR, or auxiliary tasks")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateConfig":
        allowed = {item.name for item in fields(cls)}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unsupported candidate fields: {sorted(extra)}")
        kwargs = dict(value)
        nested = {
            "backbone": BackboneConfig,
            "ranker": RankerConfig,
            "listwise": ListwiseConfig,
            "history": HistoryConfig,
            "bpr": BPRConfig,
            "auxiliary": AuxiliaryConfig,
            "optimizer": OptimizerConfig,
            "initialization": InitializationConfig,
            "resources": ResourceLimits,
        }
        for key, nested_type in nested.items():
            if key in kwargs:
                kwargs[key] = _strict_dataclass(nested_type, kwargs[key], key)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())


@dataclass(frozen=True)
class CandidateSpec:
    dataset_fingerprint: str
    config: CandidateConfig
    code_version: str = CODE_VERSION
    schema_version: int = CANDIDATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.dataset_fingerprint.startswith("sha256:"):
            raise ValueError("dataset fingerprint must be content-addressed sha256")
        if not self.code_version or self.schema_version <= 0:
            raise ValueError("code and schema versions are required")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateSpec":
        allowed = {item.name for item in fields(cls)}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unsupported candidate spec fields: {sorted(extra)}")
        kwargs = dict(value)
        kwargs["config"] = CandidateConfig.from_mapping(kwargs["config"])
        return cls(**kwargs)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "code_version": self.code_version,
            "dataset_fingerprint": self.dataset_fingerprint,
            "candidate": self.config.to_dict(),
        }

    @property
    def identity(self) -> str:
        return "sha256:" + hashlib.sha256(
            canonical_json(self.identity_payload()).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**self.identity_payload(), "config_identity": self.identity}

    def ledger_config(self) -> dict[str, Any]:
        return {
            "unified_candidate": self.identity_payload(),
            "config_identity": self.identity,
        }
