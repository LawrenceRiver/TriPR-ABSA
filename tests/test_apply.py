import pytest
import torch

from pragmatic_residual.apply import (
    _as_tuple,
    _validate_logits,
    _validate_sample,
    apply_batch,
    apply_pragmatic_residual,
)


VALID_SAMPLE = {
    "text_list": ["The", "food", "was", "good", "."],
    "aspect": "food",
    "aspect_post": [1, 2],
}


def test_rejects_unknown_module() -> None:
    with pytest.raises(ValueError, match="Unknown pragmatic residual modules"):
        _as_tuple(["unknown"])


def test_rejects_non_three_class_logits() -> None:
    with pytest.raises(ValueError, match="final dimension must be 3"):
        _validate_logits(torch.zeros(4))


def test_names_missing_sample_field() -> None:
    with pytest.raises(ValueError, match="aspect_post"):
        _validate_sample({"text_list": ["food"]})


def test_accepts_valid_sample() -> None:
    _validate_sample(VALID_SAMPLE)


def test_rejects_batch_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        apply_batch(torch.zeros((2, 3)), [VALID_SAMPLE], modules=())


def test_empty_module_set_is_identity() -> None:
    logits = torch.tensor([0.2, 0.5, 0.3])
    adjusted = apply_pragmatic_residual(logits, VALID_SAMPLE, modules=())
    assert torch.equal(adjusted, logits)
