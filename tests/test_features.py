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


def test_reduced_domain_group_keeps_required_price_feature() -> None:
    features = extract_pragmatic_features(
        ["The", "price", "is", "higher", "than", "anywhere", "else", "."],
        [1, 2],
        feature_set="domain",
    )

    assert features[PRAGMATIC_FEATURE_NAMES.index("price_positive")] == 1.0
    assert features[PRAGMATIC_FEATURE_NAMES.index("fact_listing")] == 0.0
    assert features[PRAGMATIC_FEATURE_NAMES.index("current_positive_comparison")] == 0.0
