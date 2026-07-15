"""Train-only phrase-intensity residuals for three-class ABSA logits.

This module only consumes a validated phrase prior. It performs no network or
builder work. Class order is 0=positive, 1=negative, 2=neutral.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

NEGATORS = {"not", "no", "never", "nt", "n't", "without"}
CONTRAST = {"but", "however", "though", "although", "yet", "while"}
NON_ASSERTIVE_CUES = {
    "expect",
    "expects",
    "expected",
    "expecting",
    "hope",
    "hoped",
    "hoping",
    "wish",
    "wished",
    "want",
    "wanted",
}
AMBIGUOUS_SINGLE_PRIORS = {"well"}
TRAILING_FUNCTION_WORDS = {"and", "or", "as", "with", "for", "to", "of", "the", "a", "an"}
MAX_EFFECTIVE_DISTANCE = 10
SERVICE_TERMS = {"service", "services", "staff", "waiter", "waitress", "server", "servers"}

DEFAULT_INTENSITY_PARAMS = {
    "scale": 1.0,
    "positive_threshold": 0.4,
    "negative_threshold": 0.2,
    "neutral_suppress": 0.45,
    "strong_positive_cutoff": 0.75,
    "not_strong_positive_value": 0.08,
    "not_weak_positive_factor": 0.35,
    "not_negative_flip": 0.2,
    "mixed_evidence_damping": 1.0,
    "far_decay": 0.55,
}


def norm(token: Any) -> str:
    """Normalize one token for phrase matching."""
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", str(token).lower())


def _prior_error(path: Path, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def load_prior(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate a train-only phrase prior from ``path``."""
    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _prior_error(source, "document must contain valid JSON") from exc

    if not isinstance(data, dict):
        raise _prior_error(source, "document must be an object")
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        raise _prior_error(source, "metadata must be an object")
    if metadata.get("split") != "train":
        raise _prior_error(source, "metadata split must be train")

    raw_prior = data.get("prior")
    if not isinstance(raw_prior, list):
        raise _prior_error(source, "prior must be a list")

    entries: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_prior):
        prefix = f"prior[{index}]"
        if not isinstance(item, dict):
            raise _prior_error(source, f"{prefix} must be an object")

        raw_phrase = item.get("phrase")
        if not isinstance(raw_phrase, str) or not raw_phrase.strip():
            raise _prior_error(source, f"{prefix} phrase must be non-empty")
        polarity = item.get("polarity")
        if isinstance(polarity, bool) or polarity not in (-1, 1):
            raise _prior_error(source, f"{prefix} polarity must be -1 or 1")
        intensity = item.get("intensity")
        if (
            isinstance(intensity, bool)
            or not isinstance(intensity, (int, float))
            or not math.isfinite(float(intensity))
            or not 0.0 < float(intensity) <= 1.0
        ):
            raise _prior_error(source, f"{prefix} intensity must be in (0, 1]")

        parts = [norm(part) for part in raw_phrase.split()]
        parts = [part for part in parts if part]
        if not parts:
            raise _prior_error(source, f"{prefix} phrase must be non-empty after normalization")
        if len(parts) == 1 and parts[0] in AMBIGUOUS_SINGLE_PRIORS:
            continue
        if parts[-1] in TRAILING_FUNCTION_WORDS:
            continue
        if parts[0] in {"the", "a", "an"}:
            parts = parts[1:]
        phrase = " ".join(parts)
        if phrase:
            entries[phrase] = {
                "value": int(polarity) * float(intensity),
                "dimension": item.get("dimension", "other"),
                "subjectivity": item.get("subjectivity", "weaksubj"),
            }
    return entries


def clause_bounds(tokens: list[str], start: int, end: int) -> tuple[int, int]:
    """Return bounds of the contrast/punctuation-delimited aspect clause."""
    left = 0
    right = len(tokens)
    for index in range(start - 1, -1, -1):
        if tokens[index] in {",", ";", "."} or norm(tokens[index]) in CONTRAST:
            left = index + 1
            break
    for index in range(end, len(tokens)):
        if tokens[index] in {",", ";", "."} or norm(tokens[index]) in CONTRAST:
            right = index
            break
    return left, right


def phrase_matches(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    """Find non-normalized phrase spans in normalized ``tokens``."""
    parts = [norm(part) for part in phrase.split()]
    parts = [part for part in parts if part]
    if not parts:
        return []
    return [
        (index, index + len(parts))
        for index in range(0, len(tokens) - len(parts) + 1)
        if tokens[index : index + len(parts)] == parts
    ]


def apply_negation(
    value: float,
    tokens: list[str],
    start: int,
    params: Mapping[str, float],
) -> float:
    """Dampen or weakly flip a prior value under local negation."""
    has_negation = any(tokens[index] in NEGATORS for index in range(max(0, start - 3), start))
    if not has_negation:
        return value
    if value > 0:
        if value >= params["strong_positive_cutoff"]:
            return params["not_strong_positive_value"]
        return value * params["not_weak_positive_factor"]
    if value < 0:
        return abs(value) * params["not_negative_flip"]
    return 0.0


def intensity_score(
    sample: dict[str, Any],
    prior: Mapping[str, Mapping[str, Any]],
    params: Mapping[str, float] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Score the current aspect clause using phrase intensity and negation."""
    resolved_params = {**DEFAULT_INTENSITY_PARAMS, **(params or {})}
    raw_tokens = [str(token).lower() for token in sample["text_list"]]
    tokens = [norm(token) for token in raw_tokens]
    start, end = sample["aspect_post"]
    start = max(0, min(start, len(tokens)))
    end = max(start, min(end, len(tokens)))
    aspect_terms = {token for token in tokens[start:end] if token}
    aspect_terms.update(norm(part) for part in str(sample.get("aspect", "")).split() if norm(part))
    left, right = clause_bounds(raw_tokens, start, end)
    clause = tokens[left:right]

    hits: list[dict[str, Any]] = []
    used: set[int] = set()
    for phrase, info in sorted(prior.items(), key=lambda item: -len(item[0].split())):
        for hit_start, hit_end in phrase_matches(clause, phrase):
            span = set(range(hit_start, hit_end))
            if span & used:
                continue
            used.update(span)
            value = apply_negation(float(info["value"]), clause, hit_start, resolved_params)
            distance = min(abs((left + hit_start) - start), abs((left + hit_end) - end))
            if distance > MAX_EFFECTIVE_DISTANCE:
                continue
            preceding = clause[max(0, hit_start - 3) : hit_start]
            if any(token in NON_ASSERTIVE_CUES for token in preceding):
                continue
            local_context = set(clause[max(0, hit_start - 3) : min(len(clause), hit_end + 4)])
            if local_context & SERVICE_TERMS and not aspect_terms & SERVICE_TERMS:
                continue
            distance_weight = 1.0 if distance <= 4 else resolved_params["far_decay"]
            hits.append(
                {
                    "phrase": phrase,
                    "raw_value": round(float(info["value"]), 4),
                    "value": round(value * distance_weight, 4),
                    "distance": distance,
                    "dimension": info.get("dimension", "other"),
                }
            )

    if not hits:
        return 0.0, []
    positive = sum(hit["value"] for hit in hits if hit["value"] > 0)
    negative = -sum(hit["value"] for hit in hits if hit["value"] < 0)
    score = positive - negative
    if positive > 0 and negative > 0:
        score *= resolved_params["mixed_evidence_damping"]
    return max(-1.5, min(1.5, score)), hits


def residual_from_score(
    score: float,
    params: Mapping[str, float] | None = None,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert one scalar intensity score into a three-class logit residual."""
    resolved_params = {**DEFAULT_INTENSITY_PARAMS, **(params or {})}
    if like is None:
        adjustment = torch.zeros(3)
    else:
        adjustment = torch.zeros(3, dtype=like.dtype, device=like.device)
    absolute_score = abs(score)
    if score >= resolved_params["positive_threshold"]:
        strength = min(1.0, absolute_score)
        adjustment[0] += resolved_params["scale"] * strength
        adjustment[2] -= resolved_params["neutral_suppress"] * strength
        adjustment[1] -= 0.25 * resolved_params["scale"] * strength
    elif score <= -resolved_params["negative_threshold"]:
        strength = min(1.0, absolute_score)
        adjustment[1] += resolved_params["scale"] * strength
        adjustment[2] -= resolved_params["neutral_suppress"] * strength
        adjustment[0] -= 0.25 * resolved_params["scale"] * strength
    return adjustment


def intensity_residual(
    sample: dict[str, Any],
    probs: torch.Tensor,
    prior: Mapping[str, Mapping[str, Any]] | None,
    params: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, list[str], float, list[dict[str, Any]]]:
    """Return residual, action labels, score, and matched prior phrases."""
    if not prior:
        return torch.zeros(3, dtype=probs.dtype, device=probs.device), [], 0.0, []
    score, hits = intensity_score(sample, prior, params)
    adjustment = residual_from_score(score, params, like=probs)
    actions = [f"intensity_score({score:.2f})"] if torch.any(adjustment != 0) else []
    return adjustment, actions, score, hits
