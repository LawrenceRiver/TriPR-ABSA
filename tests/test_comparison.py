import pytest
import torch

from pragmatic_residual.comparison import comparison_residual


def test_better_elsewhere_is_negative_for_current_food() -> None:
    sample = {
        "text_list": ["I", "have", "had", "better", "food", "elsewhere", "."],
        "aspect": "food",
        "aspect_post": [4, 5],
    }
    residual, actions = comparison_residual(sample, torch.tensor([0.65, 0.20, 0.15]))
    assert residual[1] > 0
    assert residual[0] < 0
    assert actions and actions[0].startswith("comparison_negative")


@pytest.mark.parametrize(
    ("text_list", "aspect_post"),
    [
        (
            ["The", "food", "is", "better", "here", ",", "not", "elsewhere", "."],
            [1, 2],
        ),
        (
            ["The", "food", "got", "better", "after", "eating", "elsewhere", "."],
            [1, 2],
        ),
        (
            [
                "The",
                "food",
                "is",
                "better",
                "now",
                ";",
                "elsewhere",
                "the",
                "service",
                "improved",
                ".",
            ],
            [1, 2],
        ),
        (
            ["I", "have", "not", "had", "better", "food", "elsewhere", "."],
            [5, 6],
        ),
    ],
    ids=[
        "negated-elsewhere",
        "comparative-after-aspect",
        "elsewhere-in-another-clause",
        "negated-comparative",
    ],
)
def test_unrelated_or_negated_elsewhere_has_no_comparison_residual(
    text_list: list[str], aspect_post: list[int]
) -> None:
    sample = {"text_list": text_list, "aspect_post": aspect_post}

    residual, actions = comparison_residual(sample, torch.tensor([0.65, 0.20, 0.15]))

    assert torch.equal(residual, torch.zeros_like(residual))
    assert not any(action.startswith("comparison_negative") for action in actions)


def test_explicit_than_keeps_current_food_as_comparison_winner() -> None:
    sample = {
        "text_list": ["The", "food", "is", "better", "than", "elsewhere", "."],
        "aspect_post": [1, 2],
    }

    residual, actions = comparison_residual(sample, torch.tensor([0.65, 0.20, 0.15]))

    assert residual[0] > 0
    assert residual[1] < 0
    assert actions and actions[0].startswith("comparison_positive")
    assert not any(action.startswith("comparison_negative") for action in actions)
