"""Read-only evidence reconstruction from a fingerprint-scoped research ledger."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from .candidate import (
    AuxiliaryConfig,
    BackboneConfig,
    BPRConfig,
    CandidateConfig,
    CandidateSpec,
    HistoryConfig,
    ListwiseConfig,
    RankerConfig,
)
from .store import ResearchStore


EVIDENCE_SCHEMA_VERSION = 1
EPSILON = 0.002
AUDITED_KUAIRAND_FINGERPRINT = (
    "sha256:6ea8eca058e1ae34c8a3748c4d4a633a853a761ae9420e1c86bd96e2878dd838"
)


def anchor_candidates(fingerprint: str) -> dict[str, CandidateSpec]:
    return {
        "official_fm_baseline": CandidateSpec(
            fingerprint, CandidateConfig(listwise=ListwiseConfig(enabled=False))
        ),
        "no_history_soft_target_listnet": CandidateSpec(fingerprint, CandidateConfig()),
        "history_last20_fixed_gate_minus_0_05": CandidateSpec(
            fingerprint,
            CandidateConfig(
                history=HistoryConfig(
                    enabled=True, last_n=20, gate=-0.05,
                    train_embeddings=False, train_gate=False,
                    encoder="causal_positive_mean_pool", embedding_dim=16,
                    history_learning_rate=None, gate_learning_rate=None,
                    unfreeze_step=0,
                )
            ),
        ),
        "bpr_weights": CandidateSpec(
            fingerprint, CandidateConfig(bpr=BPRConfig(enabled=True, weight=0.01))
        ),
        "multitask_weights": CandidateSpec(
            fingerprint,
            CandidateConfig(
                auxiliary=AuxiliaryConfig(
                    enabled=True, is_click_weight=0.01,
                    play_time_weight=0.0, play_transform=None,
                    head_learning_rate=0.001, huber_delta=1.0,
                )
            ),
        ),
        "causal_behavioral_lambdarank": CandidateSpec(
            fingerprint,
            CandidateConfig(
                backbone=BackboneConfig("lambdarank", 16),
                ranker=RankerConfig(
                    True, "causal_behavioral_v1", 20.0, 160,
                    0.04, 31, 50, 20,
                ),
                listwise=ListwiseConfig(enabled=False),
            ),
        ),
        "fm_lambdarank_ensemble": CandidateSpec(
            fingerprint,
            CandidateConfig(
                backbone=BackboneConfig("fm_lambdarank_ensemble", 16),
                ranker=RankerConfig(
                    True, "causal_behavioral_v1", 20.0, 160,
                    0.04, 31, 50, 20,
                ),
                listwise=ListwiseConfig(enabled=False),
            ),
        ),
    }


def _classification(gain: float | None) -> str:
    if gain is None:
        return "not_observed"
    if gain >= EPSILON:
        return "positive_at_or_above_epsilon"
    if gain > 0:
        return "positive_below_epsilon"
    if gain == 0:
        return "no_positive_gain"
    return "negative"


def _completed(trials: list[dict], predicate: Callable[[dict], bool]) -> list[dict]:
    return [trial for trial in trials if trial["status"] == "completed" and predicate(trial)]


def _module_entry(
    name: str,
    trials: list[dict],
    reference_primary: float,
    anchor: CandidateSpec,
    eligible_new: bool,
    enabled_current: bool,
) -> dict[str, Any]:
    records = [
        {
            "trial_id": trial["trial_id"],
            "method": trial["method"],
            "run_id": trial["config"].get("run_id"),
            "validation": trial["validation"],
            "best_step": trial["result"]["best_step"],
            "gain_vs_no_history_listwise": trial["validation"]["primary"] - reference_primary,
        }
        for trial in trials
    ]
    best = max(records, key=lambda item: (item["validation"]["primary"], -int(item["trial_id"].split("-")[1]))) if records else None
    gain = None if best is None else best["gain_vs_no_history_listwise"]
    return {
        "module": name,
        "observed_on_current_fingerprint": bool(records),
        "claim_scope": "current_fingerprint_only" if records else "unobserved",
        "eligible_as_new_dataset_anchor": eligible_new,
        "new_dataset_evidence": "unobserved_requires_controlled_validation",
        "enabled_for_current_search": enabled_current,
        "anchor_candidate_identity": anchor.identity,
        "classification": _classification(gain),
        "epsilon": EPSILON,
        "best_observation": best,
        "observations": records,
    }


def build_evidence_registry(store: ResearchStore) -> dict[str, Any]:
    """Reconstruct evidence without reserving trials or writing to the ledger."""
    trials = store.trials()
    listwise_trials = _completed(
        trials, lambda trial: trial["method"] == "fm_user_soft_target_listnet_finetune"
    )
    if not listwise_trials:
        raise RuntimeError("no-history Listwise reference is missing")
    reference = min(listwise_trials, key=lambda trial: trial["ordinal"])
    reference_primary = reference["validation"]["primary"]
    anchors = anchor_candidates(store.dataset_fingerprint)

    baseline = _completed(
        trials, lambda trial: trial["method"] == "official_fm_baseline_reproduction"
    )
    history = _completed(
        trials,
        lambda trial: (
            trial["method"] == "fm_user_soft_target_listnet_causal_history_fixed_gate"
            and trial["config"].get("run_id") == "F-02"
        ) or (
            trial["method"] == "fm_user_soft_target_listnet_causal_history_seed_replication"
            and trial["config"].get("paired_design", {}).get("variant") == "history"
        ),
    )
    bpr = _completed(
        trials, lambda trial: trial["method"] == "fm_user_soft_target_listnet_same_user_bpr_regularized"
    )
    multitask = _completed(
        trials, lambda trial: trial["method"] == "fm_user_soft_target_listnet_multitask_auxiliary"
    )
    ranker = _completed(
        trials, lambda trial: trial["method"] in {
            "train_causal_behavioral_lambdarank",
            "train_causal_behavioral_lambdarank_wide",
        }
    )
    ensemble = _completed(
        trials, lambda trial: trial["method"] == "train_causal_behavioral_lambdarank_fm_ensemble"
    )
    modules = {
        "official_fm_baseline": _module_entry(
            "official_fm_baseline", baseline, reference_primary,
            anchors["official_fm_baseline"], False, False,
        ),
        "no_history_soft_target_listnet": _module_entry(
            "no_history_soft_target_listnet", listwise_trials, reference_primary,
            anchors["no_history_soft_target_listnet"], True, True,
        ),
        "history_last20_fixed_gate_minus_0_05": _module_entry(
            "history_last20_fixed_gate_minus_0_05", history, reference_primary,
            anchors["history_last20_fixed_gate_minus_0_05"], True, True,
        ),
        "bpr_weights": _module_entry(
            "bpr_weights", bpr, reference_primary,
            anchors["bpr_weights"], True, False,
        ),
        "multitask_weights": _module_entry(
            "multitask_weights", multitask, reference_primary,
            anchors["multitask_weights"], True, False,
        ),
        "causal_behavioral_lambdarank": _module_entry(
            "causal_behavioral_lambdarank", ranker, reference_primary,
            anchors["causal_behavioral_lambdarank"], True, True,
        ),
        "fm_lambdarank_ensemble": _module_entry(
            "fm_lambdarank_ensemble", ensemble, reference_primary,
            anchors["fm_lambdarank_ensemble"], True, True,
        ),
    }
    top = store.best_trial()
    if top is None:
        raise RuntimeError("ledger has no completed Top-1")
    if store.dataset_fingerprint == AUDITED_KUAIRAND_FINGERPRINT and (
        top["trial_id"] != "trial-15"
        or abs(top["validation"]["primary"] - 0.6015123724937439) > 1e-12
    ):
        raise RuntimeError("ledger Top-1 disagrees with the audited trial-15 evidence")
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "dataset_fingerprint": store.dataset_fingerprint,
        "claim_scope": "observations_do_not_transfer_across_dataset_fingerprints",
        "epsilon": EPSILON,
        "ledger": {"used": store.consumed, "remaining": store.remaining, "maximum": 50},
        "reference": {
            "trial_id": reference["trial_id"],
            "validation": reference["validation"],
        },
        "top1": {
            "trial_id": top["trial_id"],
            "validation": top["validation"],
        },
        "default_current_candidate_modules": [
            "no_history_soft_target_listnet",
            "history_last20_fixed_gate_minus_0_05",
            "causal_behavioral_lambdarank",
            "fm_lambdarank_ensemble",
        ],
        "modules": modules,
    }


def render_evidence_markdown(registry: dict[str, Any]) -> str:
    lines = [
        "# Agent A evidence report",
        "",
        f"Dataset fingerprint: `{registry['dataset_fingerprint']}`",
        "",
        f"Ledger: {registry['ledger']['used']}/{registry['ledger']['maximum']} used; "
        f"{registry['ledger']['remaining']} remaining.",
        "",
        f"Top-1: `{registry['top1']['trial_id']}` with validation primary "
        f"`{registry['top1']['validation']['primary']:.10f}`.",
        "",
        "All observations below are scoped to this fingerprint. New datasets require controlled validation.",
        "",
        "| Module | Classification | Best trial | Primary gain | Current search | New-dataset anchor |",
        "|---|---|---:|---:|---|---|",
    ]
    for module in registry["modules"].values():
        best = module["best_observation"]
        lines.append(
            "| {module} | {classification} | {trial} | {gain} | {current} | {anchor} |".format(
                module=module["module"],
                classification=module["classification"],
                trial="—" if best is None else best["trial_id"],
                gain="—" if best is None else f"{best['gain_vs_no_history_listwise']:+.10f}",
                current=str(module["enabled_for_current_search"]).lower(),
                anchor=str(module["eligible_as_new_dataset_anchor"]).lower(),
            )
        )
    lines.extend(["", "Generated exclusively from the fingerprint-scoped research ledger.", ""])
    return "\n".join(lines)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_evidence_reports(registry: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    json_path = Path(output_dir) / "evidence_registry.json"
    markdown_path = Path(output_dir) / "evidence_report.md"
    _atomic_write(json_path, json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _atomic_write(markdown_path, render_evidence_markdown(registry))
    return json_path, markdown_path
