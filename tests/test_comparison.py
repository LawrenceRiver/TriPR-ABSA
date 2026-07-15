from __future__ import annotations

import pytest
import torch

from pragmatic_residual.comparison import comparison_residual


@pytest.mark.parametrize(
    ("text_list", "aspect_post", "expects_negative"),
    [
        (
            ["This", "restaurant", "served", "better", "food", "in", "Chinatown", "."],
            [4, 5],
            False,
        ),
        (
            [
                "This",
                "restaurant",
                "served",
                "better",
                "food",
                "in",
                "Chinatown",
                "than",
                "its",
                "rivals",
                ".",
            ],
            [4, 5],
            False,
        ),
        (
            ["This", "restaurant", "had", "better", "food", "in", "Chinatown", "."],
            [4, 5],
            False,
        ),
        (
            [
                "We",
                "thought",
                "this",
                "restaurant",
                "had",
                "better",
                "food",
                "in",
                "Chinatown",
                ".",
            ],
            [6, 7],
            False,
        ),
        (
            ["I", "found", "better", "food", "in", "Chinatown", "than", "its", "rivals", "."],
            [3, 4],
            False,
        ),
        (["I", "have", "had", "better", "food", "elsewhere", "."], [4, 5], True),
        (["We", "have", "had", "better", "food", "elsewhere", "."], [4, 5], True),
        (["You", "can", "make", "better", "food", "at", "home", "."], [4, 5], True),
        (["I", "found", "better", "food", "in", "Chinatown", "."], [3, 4], True),
        (["I", "found", "better", "price", "anywhere", "else", "."], [3, 4], True),
    ],
    ids=[
        "entity-served",
        "entity-served-explicit-than",
        "entity-had",
        "unbound-experiencer",
        "experiencer-explicit-than",
        "experiencer-had-elsewhere",
        "we-experiencer-had-elsewhere",
        "experiencer-made-at-home",
        "experiencer-found-in-chinatown",
        "experiencer-found-better-price",
    ],
)
def test_implicit_external_alternative_requires_experiencer_and_safe_cue(
    text_list: list[str], aspect_post: list[int], expects_negative: bool
) -> None:
    sample = {"text_list": text_list, "aspect_post": aspect_post}

    residual, actions = comparison_residual(sample, torch.tensor([0.65, 0.20, 0.15]))

    positive_actions = [action for action in actions if action.startswith("comparison_positive")]
    negative_actions = [action for action in actions if action.startswith("comparison_negative")]
    if expects_negative:
        assert residual[1] > 0
        assert residual[0] < 0
        assert positive_actions == []
        assert len(negative_actions) == 1
    else:
        assert torch.equal(residual, torch.zeros_like(residual))
        assert positive_actions == []
        assert negative_actions == []


@pytest.mark.parametrize(
    ("text_list", "aspect_post"),
    [
        (["The", "best", "food", "in", "Chinatown", "."], [2, 3]),
        (
            [
                "Better",
                "service",
                "is",
                "important",
                "and",
                "the",
                "food",
                "at",
                "home",
                "is",
                "cheap",
                ".",
            ],
            [6, 7],
        ),
    ],
    ids=["positive-superlative", "unrelated-comparative"],
)
def test_superlative_or_unrelated_comparative_has_no_negative_residual(
    text_list: list[str], aspect_post: list[int]
) -> None:
    sample = {"text_list": text_list, "aspect_post": aspect_post}

    residual, actions = comparison_residual(sample, torch.tensor([0.65, 0.20, 0.15]))

    assert torch.equal(residual, torch.zeros_like(residual))
    assert not any(action.startswith("comparison_negative") for action in actions)


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


@pytest.mark.parametrize(
    ("text_list", "aspect_post"),
    [
        (
            ["The", "food", "got", "better", "after", "eating", "at", "home", "."],
            [1, 2],
        ),
        (
            ["The", "food", "is", "better", "now", ";", "at", "home", "we", "cook", "."],
            [1, 2],
        ),
        (
            ["At", "home", "the", "food", "got", "better", "over", "time", "."],
            [3, 4],
        ),
        (
            ["I", "have", "not", "had", "better", "food", "at", "home", "."],
            [5, 6],
        ),
    ],
    ids=[
        "got-better-after-eating-at-home",
        "external-reference-after-punctuation",
        "temporal-got-better-at-home",
        "negated-comparative-at-home",
    ],
)
def test_nondirectional_better_at_home_has_no_negative_comparison(
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


@pytest.mark.parametrize(
    ("text_list", "aspect_post", "expected_action"),
    [
        (["food", "is", "better", "than", "service", "."], [0, 1], "positive"),
        (["service", "is", "better", "than", "the", "food", "."], [5, 6], "negative"),
        (["food", "is", "worse", "than", "service", "."], [0, 1], "negative"),
        (["service", "is", "worse", "than", "the", "food", "."], [5, 6], "positive"),
        (["service", "is", "higher", "than", "the", "food", "."], [5, 6], None),
        (["service", "is", "less", "than", "the", "food", "."], [5, 6], None),
        (["food", "is", "not", "better", "than", "service", "."], [0, 1], None),
        (["service", "is", "not", "worse", "than", "food", "."], [5, 6], None),
        (
            ["The", "price", "anywhere", "else", "is", "better", "than", "the", "price", "."],
            [8, 9],
            "negative",
        ),
        (
            [
                "food",
                "is",
                "better",
                "than",
                "service",
                "and",
                "ambience",
                "is",
                "worse",
                "than",
                "price",
                ".",
            ],
            [10, 11],
            "positive",
        ),
        (
            [
                "food",
                "is",
                "worse",
                "than",
                "service",
                "and",
                "ambience",
                "is",
                "better",
                "than",
                "price",
                ".",
            ],
            [10, 11],
            "negative",
        ),
        (["The", "price", "is", "higher", "than", "anywhere", "else", "."], [1, 2], "negative"),
        (["The", "price", "is", "lower", "than", "anywhere", "else", "."], [1, 2], "positive"),
        (["The", "value", "is", "higher", "than", "anywhere", "else", "."], [1, 2], "positive"),
        (["The", "value", "is", "lower", "than", "anywhere", "else", "."], [1, 2], "negative"),
        (["price", "is", "cheaper", "than", "elsewhere", "."], [0, 1], "positive"),
        (["value", "is", "higher", "than", "price", "."], [4, 5], "positive"),
        (["value", "is", "lower", "than", "price", "."], [4, 5], "negative"),
        (["food", "couldn't", "be", "better", "than", "service", "."], [0, 1], None),
        (["food", "cannot", "be", "better", "than", "service", "."], [0, 1], None),
    ],
    ids=[
        "left-better",
        "right-better",
        "left-worse",
        "right-worse",
        "ambiguous-higher",
        "ambiguous-less",
        "negated-left-better",
        "negated-right-worse",
        "explicit-direction-overrides-weaker-price-cue",
        "nearest-relation-right-worse",
        "nearest-relation-right-better",
        "price-left-higher",
        "price-left-lower",
        "value-left-higher",
        "value-left-lower",
        "price-left-cheaper",
        "price-right-higher",
        "price-right-lower",
        "contracted-negation",
        "cannot-negation",
    ],
)
def test_explicit_than_relation_has_one_direction(
    text_list: list[str], aspect_post: list[int], expected_action: str | None
) -> None:
    sample = {"text_list": text_list, "aspect_post": aspect_post}

    residual, actions = comparison_residual(sample, torch.tensor([0.65, 0.20, 0.15]))

    comparison_actions = [action for action in actions if action.startswith("comparison_")]
    if expected_action == "positive":
        assert residual[0] > 0
        assert residual[1] < 0
        assert comparison_actions == ["comparison_positive(1.00)"]
    elif expected_action == "negative":
        assert residual[1] > 0
        assert residual[0] < 0
        assert comparison_actions == ["comparison_negative(1.00)"]
    else:
        assert torch.equal(residual, torch.zeros_like(residual))
        assert comparison_actions == []


@pytest.mark.parametrize(
    ("negation_tokens", "aspect_post"),
    [
        (["couldn't"], [4, 5]),
        (["couldn’t"], [4, 5]),
        (["could", "n't"], [5, 6]),
    ],
    ids=["ascii-apostrophe", "curly-apostrophe", "split-contraction"],
)
def test_contracted_negation_blocks_implicit_comparison(
    negation_tokens: list[str], aspect_post: list[int]
) -> None:
    text_list = ["I", *negation_tokens, "make", "better", "food", "at", "home", "."]
    sample = {"text_list": text_list, "aspect_post": aspect_post}

    residual, actions = comparison_residual(sample, torch.tensor([0.65, 0.20, 0.15]))

    assert torch.equal(residual, torch.zeros_like(residual))
    assert not any(action.startswith("comparison_") for action in actions)
