import pytest

from pragmatic_residual.features import (
    PRAGMATIC_FEATURE_NAMES,
    PRAGMATIC_V3_FEATURE_NAMES,
    extract_pragmatic_features,
)


def test_combined_all_and_v3_group_keeps_base_features() -> None:
    features = extract_pragmatic_features(
        ["The", "menu", "includes", "sushi", ",", "pasta", "and", "wine", "."],
        [1, 2],
        feature_set="all,fact_relation",
    )

    assert features[PRAGMATIC_FEATURE_NAMES.index("fact_listing")] == 1.0
    relation_idx = PRAGMATIC_V3_FEATURE_NAMES.index("relation_fact_object_score")
    assert features[relation_idx] > 0


def test_comparison_group_excludes_price_domain_feature() -> None:
    features = extract_pragmatic_features(
        ["The", "price", "is", "higher", "than", "anywhere", "else", "."],
        [1, 2],
        feature_set="comparison",
    )

    assert features[PRAGMATIC_FEATURE_NAMES.index("price_positive")] == 0.0


def test_higher_price_is_not_a_positive_domain_fallback() -> None:
    features = extract_pragmatic_features(
        ["The", "price", "is", "higher", "than", "anywhere", "else", "."],
        [1, 2],
        feature_set="domain",
    )

    assert features[PRAGMATIC_FEATURE_NAMES.index("price_positive")] == 0.0
    assert features[PRAGMATIC_FEATURE_NAMES.index("fact_listing")] == 0.0
    assert features[PRAGMATIC_FEATURE_NAMES.index("current_positive_comparison")] == 0.0


@pytest.mark.parametrize(
    ("text_list", "aspect_post", "expected"),
    [
        (["food", "is", "better", "than", "service", "."], [0, 1], (1.0, 0.0)),
        (["service", "is", "better", "than", "food", "."], [4, 5], (0.0, 1.0)),
        (["food", "is", "worse", "than", "service", "."], [0, 1], (0.0, 1.0)),
        (["service", "is", "worse", "than", "food", "."], [4, 5], (1.0, 0.0)),
        (["food", "is", "not", "better", "than", "service", "."], [0, 1], (0.0, 0.0)),
        (["service", "is", "not", "worse", "than", "food", "."], [5, 6], (0.0, 0.0)),
        (
            ["The", "price", "anywhere", "else", "is", "better", "than", "the", "price", "."],
            [8, 9],
            (0.0, 1.0),
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
            (1.0, 0.0),
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
            (0.0, 1.0),
        ),
        (["The", "price", "is", "higher", "than", "anywhere", "else", "."], [1, 2], (0.0, 1.0)),
        (["The", "price", "is", "lower", "than", "anywhere", "else", "."], [1, 2], (1.0, 0.0)),
        (["The", "value", "is", "higher", "than", "anywhere", "else", "."], [1, 2], (1.0, 0.0)),
        (["The", "value", "is", "lower", "than", "anywhere", "else", "."], [1, 2], (0.0, 1.0)),
        (["price", "is", "cheaper", "than", "elsewhere", "."], [0, 1], (1.0, 0.0)),
        (["value", "is", "higher", "than", "price", "."], [4, 5], (1.0, 0.0)),
        (["value", "is", "lower", "than", "price", "."], [4, 5], (0.0, 1.0)),
        (["I", "found", "better", "price", "anywhere", "else", "."], [3, 4], (0.0, 1.0)),
        (["food", "cannot", "be", "better", "than", "service", "."], [0, 1], (0.0, 0.0)),
    ],
    ids=[
        "left-better",
        "right-better",
        "left-worse",
        "right-worse",
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
        "implicit-better-price",
        "cannot-negation",
    ],
)
def test_explicit_relation_direction_is_shared_by_base_and_v3(
    text_list: list[str], aspect_post: list[int], expected: tuple[float, float]
) -> None:
    base = extract_pragmatic_features(text_list, aspect_post, feature_set="all")
    v3 = extract_pragmatic_features(text_list, aspect_post, feature_set="all_v3")

    base_direction = (
        base[PRAGMATIC_FEATURE_NAMES.index("current_positive_comparison")],
        base[PRAGMATIC_FEATURE_NAMES.index("current_negative_comparison")],
    )
    v3_direction = (
        v3[PRAGMATIC_V3_FEATURE_NAMES.index("relation_comparison_current_positive")],
        v3[PRAGMATIC_V3_FEATURE_NAMES.index("relation_comparison_current_negative")],
    )
    assert base_direction == expected
    assert v3_direction == expected
