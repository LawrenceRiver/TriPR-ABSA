"""Single entry point for pragmatic residual inference.

The API is deliberately small: any backbone that produces 3-way ABSA logits can
call this module, including TextGT-BERT, BERT-SPC, or an LLM classifier head.
Class order must be 0=positive, 1=negative, 2=neutral.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch

VALID_MODULES = ("fact", "comparison", "intensity")
REQUIRED_SAMPLE_FIELDS = ("text_list", "aspect_post")


def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim == 0 or logits.shape[-1] != 3:
        raise ValueError("logits final dimension must be 3: positive, negative, neutral")


def _validate_single_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if logits.ndim != 1 or logits.shape[0] != 3:
        raise ValueError("single-sample logits must have shape [3]")


def _validate_sample(sample: Mapping[str, Any]) -> None:
    missing = [name for name in REQUIRED_SAMPLE_FIELDS if name not in sample]
    if missing:
        raise ValueError(f"sample is missing required fields: {', '.join(missing)}")
    span = sample["aspect_post"]
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        raise ValueError("sample aspect_post must be a two-item [start, end] span")


def _as_tuple(modules: Iterable[str] | None) -> tuple[str, ...]:
    selected = tuple(VALID_MODULES if modules is None else modules)
    unknown = sorted(set(selected) - set(VALID_MODULES))
    if unknown:
        raise ValueError(f"Unknown pragmatic residual modules: {unknown}")
    return selected


def _apply_selected_modules(
    logits: torch.Tensor,
    sample: dict[str, Any],
    selected: tuple[str, ...],
    prior: Mapping[str, Mapping[str, Any]] | None,
    intensity_params: Mapping[str, float] | None,
    return_details: bool,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    final_logits = logits.clone()
    details: dict[str, Any] = {
        "modules": list(selected),
        "actions": [],
        "residual": torch.zeros_like(logits),
    }

    for module in selected:
        probs = torch.softmax(final_logits, dim=-1)
        if module == "fact":
            from .fact import fact_residual

            adj, actions = fact_residual(sample, probs)
            extra = {}
        elif module == "comparison":
            from .comparison import comparison_residual

            adj, actions = comparison_residual(sample, probs)
            extra = {}
        elif module == "intensity":
            from .intensity import intensity_residual

            adj, actions, score, hits = intensity_residual(sample, probs, prior, intensity_params)
            extra = {"score": score, "hits": hits}
        else:
            raise AssertionError(module)

        final_logits = final_logits + adj
        details["residual"] = details["residual"] + adj
        details["actions"].extend(actions)
        details[module] = {
            "residual": adj.detach().cpu().tolist(),
            "actions": actions,
            **extra,
        }

    if return_details:
        details["residual"] = details["residual"].detach().cpu().tolist()
        return final_logits, details
    return final_logits


def apply_pragmatic_residual(
    logits: torch.Tensor,
    sample: dict[str, Any],
    modules: Iterable[str] | None = None,
    prior: Mapping[str, Mapping[str, Any]] | None = None,
    intensity_params: Mapping[str, float] | None = None,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Apply selected residual modules to one sample's logits."""
    _validate_single_logits(logits)
    _validate_sample(sample)
    selected = _as_tuple(modules)
    return _apply_selected_modules(
        logits,
        sample,
        selected,
        prior,
        intensity_params,
        return_details,
    )


def apply_batch(
    logits: torch.Tensor,
    samples: Sequence[dict[str, Any]],
    modules: Iterable[str] | None = None,
    prior: Mapping[str, Mapping[str, Any]] | None = None,
    intensity_params: Mapping[str, float] | None = None,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, Any]]]:
    """Apply pragmatic residuals to a batch of logits and parsed samples."""
    _validate_logits(logits)
    selected = _as_tuple(modules)
    if logits.ndim != 2 or logits.shape[-1] != 3:
        raise ValueError("batch logits must have shape [batch, 3]")
    if len(samples) != logits.shape[0]:
        raise ValueError("batch logits and samples must have the same length")
    for sample in samples:
        _validate_sample(sample)

    if logits.shape[0] == 0:
        empty = logits.clone()
        if return_details:
            return empty, []
        return empty

    adjusted = []
    all_details = []
    for row_logits, sample in zip(logits, samples):
        result = _apply_selected_modules(
            row_logits,
            sample,
            selected,
            prior,
            intensity_params,
            return_details,
        )
        if return_details:
            final_logits, details = result
            adjusted.append(final_logits)
            all_details.append(details)
        else:
            adjusted.append(result)
    stacked = torch.stack(adjusted, dim=0)
    if return_details:
        return stacked, all_details
    return stacked
