import json
from pathlib import Path

import pytest
import torch

from pragmatic_residual.intensity import intensity_residual, load_prior

FIXTURE = Path(__file__).parent / "fixtures" / "prior.json"


def test_positive_phrase_increases_positive_logit() -> None:
    prior = load_prior(FIXTURE)
    sample = {
        "text_list": ["The", "food", "was", "excellent", "."],
        "aspect": "food",
        "aspect_post": [1, 2],
    }
    residual, actions, score, hits = intensity_residual(
        sample, torch.tensor([0.4, 0.3, 0.3]), prior
    )
    assert score > 0
    assert residual[0] > 0
    assert actions
    assert hits[0]["phrase"] == "excellent"


def test_negation_dampens_positive_phrase() -> None:
    prior = load_prior(FIXTURE)
    positive = {
        "text_list": ["The", "food", "was", "excellent", "."],
        "aspect": "food",
        "aspect_post": [1, 2],
    }
    negated = {
        "text_list": ["The", "food", "was", "not", "excellent", "."],
        "aspect": "food",
        "aspect_post": [1, 2],
    }
    _, _, positive_score, _ = intensity_residual(positive, torch.tensor([0.4, 0.3, 0.3]), prior)
    _, _, negated_score, _ = intensity_residual(negated, torch.tensor([0.4, 0.3, 0.3]), prior)
    assert negated_score < positive_score


def test_missing_prior_disables_only_intensity() -> None:
    sample = {
        "text_list": ["The", "food", "was", "excellent", "."],
        "aspect": "food",
        "aspect_post": [1, 2],
    }
    residual, actions, score, hits = intensity_residual(sample, torch.tensor([0.4, 0.3, 0.3]), None)
    assert torch.equal(residual, torch.zeros(3))
    assert actions == []
    assert score == 0.0
    assert hits == []


def test_malformed_prior_names_the_file(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"metadata":{"split":"test"},"prior":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed.json.*split must be train"):
        load_prior(malformed)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "document must be an object"),
        ({"metadata": {"split": "train"}, "prior": {}}, "prior must be a list"),
        (
            {
                "metadata": {"split": "train"},
                "prior": [{"phrase": " ", "polarity": 1, "intensity": 0.5}],
            },
            "phrase must be non-empty",
        ),
        (
            {
                "metadata": {"split": "train"},
                "prior": [{"phrase": "good", "polarity": 0, "intensity": 0.5}],
            },
            "polarity must be -1 or 1",
        ),
        (
            {
                "metadata": {"split": "train"},
                "prior": [{"phrase": "good", "polarity": 1, "intensity": 1.1}],
            },
            r"intensity must be in \(0, 1\]",
        ),
    ],
)
def test_prior_validation_is_strict_and_names_source(
    tmp_path: Path, document: object, message: str
) -> None:
    malformed = tmp_path / "strict-prior.json"
    malformed.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"strict-prior.json.*{message}"):
        load_prior(malformed)


def test_package_exports_load_prior() -> None:
    from pragmatic_residual import load_prior as exported_load_prior

    assert exported_load_prior is load_prior


def test_invalid_json_names_source(tmp_path: Path) -> None:
    malformed = tmp_path / "broken-prior.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="broken-prior.json.*valid JSON"):
        load_prior(malformed)
