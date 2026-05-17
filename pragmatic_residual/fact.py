"""Fact/opinion residual for three-class ABSA logits."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .features import (
    PRAGMATIC_FEATURE_NAMES,
    PRAGMATIC_V3_FEATURE_NAMES,
    extract_pragmatic_features,
)


def _value(features: np.ndarray, names: list[str], name: str) -> float:
    try:
        return float(features[names.index(name)])
    except ValueError:
        return 0.0


def _features(sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    return (
        extract_pragmatic_features(
            sample["text_list"],
            sample["aspect_post"],
            feature_set="all",
            pos_list=sample.get("pos"),
        ),
        extract_pragmatic_features(
            sample["text_list"],
            sample["aspect_post"],
            feature_set="all_v3",
            pos_list=sample.get("pos"),
        ),
    )


def fact_residual(sample: dict[str, Any], probs: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Return a neutral-boundary residual for objective fact-like clauses."""
    old, v3 = _features(sample)
    adj = torch.zeros(3, dtype=probs.dtype, device=probs.device)
    actions: list[str] = []

    fact_score = sum(
        _value(old, PRAGMATIC_FEATURE_NAMES, name)
        for name in [
            "fact_listing",
            "there_be_existence",
            "menu_enumeration",
            "objective_description",
        ]
    )
    fact_relation = sum(
        _value(v3, PRAGMATIC_V3_FEATURE_NAMES, name)
        for name in [
            "relation_fact_object_score",
            "relation_object_list_density",
            "relation_low_subjectivity_descriptor",
        ]
    )
    opinion_render = _value(v3, PRAGMATIC_V3_FEATURE_NAMES, "relation_opinion_render_score")
    fact_strength = max(fact_score / 3.0, fact_relation / 3.0)
    base_pred = int(torch.argmax(probs).item())
    top2 = torch.topk(probs, k=2).values
    margin = float(top2[0] - top2[1])
    neutral_boundary = base_pred == 0 and (
        float(probs[2]) >= 0.035 or margin <= 0.78 or fact_strength >= 0.95
    )

    if fact_strength >= 0.45 and opinion_render < 0.30 and neutral_boundary:
        adj[2] += 2.90 * fact_strength
        adj[0] -= 1.10 * fact_strength
        adj[1] -= 0.65 * fact_strength
        actions.append(f"fact_neutral({fact_strength:.2f})")

    return adj, actions
