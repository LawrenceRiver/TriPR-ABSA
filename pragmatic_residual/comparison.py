"""Comparison/external-reference residual for three-class ABSA logits."""

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


def comparison_residual(
    sample: dict[str, Any], probs: torch.Tensor
) -> tuple[torch.Tensor, list[str]]:
    """Return a residual for comparative clauses where the current aspect wins or loses."""
    old, v3 = _features(sample)
    adj = torch.zeros(3, dtype=probs.dtype, device=probs.device)
    actions: list[str] = []

    directional_pos = max(
        _value(old, PRAGMATIC_FEATURE_NAMES, "current_positive_comparison"),
        _value(v3, PRAGMATIC_V3_FEATURE_NAMES, "relation_comparison_current_positive"),
    )
    directional_neg = max(
        _value(old, PRAGMATIC_FEATURE_NAMES, "current_negative_comparison"),
        _value(v3, PRAGMATIC_V3_FEATURE_NAMES, "relation_comparison_current_negative"),
    )
    if directional_pos > 0 and directional_neg > 0:
        return adj, actions
    if directional_pos > 0 or directional_neg > 0:
        comp_pos = directional_pos
        comp_neg = directional_neg
    else:
        comp_pos = _value(old, PRAGMATIC_FEATURE_NAMES, "price_positive") * _value(
            old, PRAGMATIC_FEATURE_NAMES, "external_reference"
        )
        comp_neg = 0.0

    if comp_pos > 0:
        adj[0] += 2.00 * comp_pos
        adj[1] -= 1.05 * comp_pos
        adj[2] -= 0.35 * comp_pos
        actions.append(f"comparison_positive({comp_pos:.2f})")
    if comp_neg > 0:
        adj[1] += 2.25 * comp_neg
        adj[0] -= 1.10 * comp_neg
        adj[2] -= 0.35 * comp_neg
        actions.append(f"comparison_negative({comp_neg:.2f})")

    return adj, actions
