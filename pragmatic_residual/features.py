"""Paper-backed pragmatic feature extraction for residual inference.

This module isolates the base and relation-aware feature sets needed by the
fact and comparison residuals. It deliberately has no dependency on the
private training and dataset utilities from the research repository.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np

PRAGMATIC_FEATURE_NAMES = [
    "fact_listing",
    "there_be_existence",
    "menu_enumeration",
    "objective_description",
    "subjective_eval_near",
    "contrast_after_aspect",
    "contrast_before_aspect",
    "negation_near_aspect",
    "negated_positive",
    "negated_negative",
    "current_negative_comparison",
    "current_positive_comparison",
    "aspect_after_than",
    "external_reference",
    "wait_positive",
    "wait_negative",
    "price_positive",
    "price_negative",
]

PRAGMATIC_FEATURE_DIM = len(PRAGMATIC_FEATURE_NAMES)

PRAGMATIC_FEATURE_GROUPS = {
    "fact": [0, 1, 2, 3],
    "fact_refined": [0, 1, 2, 3],
    "scope": [4, 5, 6, 7, 8, 9],
    "comparison": [10, 11, 12, 13],
    "domain": [14, 15, 16, 17],
}

PRAGMATIC_V3_EXTRA_FEATURE_NAMES = [
    "relation_fact_object_score",
    "relation_object_list_density",
    "relation_low_subjectivity_descriptor",
    "relation_opinion_render_score",
    "relation_human_experiencer",
    "relation_beneficiary",
    "relation_same_clause_positive",
    "relation_same_clause_negative",
    "relation_cross_clause_conflict",
    "relation_contrast_balance_neutral",
    "relation_comparison_current_positive",
    "relation_comparison_current_negative",
    "relation_external_location",
    "relation_external_brand",
    "relation_domain_kb_positive",
    "relation_domain_kb_negative",
]

PRAGMATIC_V3_FEATURE_NAMES = PRAGMATIC_FEATURE_NAMES + PRAGMATIC_V3_EXTRA_FEATURE_NAMES
PRAGMATIC_V3_FEATURE_DIM = len(PRAGMATIC_V3_FEATURE_NAMES)

PRAGMATIC_V3_FEATURE_GROUPS = {
    **PRAGMATIC_FEATURE_GROUPS,
    "fact_relation": [18, 19, 20],
    "opinion_render": [21, 22, 23],
    "clause_relation": [24, 25, 26, 27],
    "comparison_relation": [28, 29, 30, 31],
    "domain_kb": [32, 33],
}
PRAGMATIC_V3_FEATURE_GROUPS.update(
    {name: [idx] for idx, name in enumerate(PRAGMATIC_V3_FEATURE_NAMES)}
)


def _norm_token(token: object) -> str:
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", str(token).lower())


def _contains_phrase(tokens: Sequence[str], phrase: str) -> bool:
    phrase_tokens = [_norm_token(token) for token in phrase.split()]
    phrase_tokens = [token for token in phrase_tokens if token]
    if not phrase_tokens:
        return False
    return any(
        list(tokens[start : start + len(phrase_tokens)]) == phrase_tokens
        for start in range(0, len(tokens) - len(phrase_tokens) + 1)
    )


def _count_any(tokens: Sequence[str], words: Iterable[str]) -> int:
    word_set = set(words)
    return sum(1 for token in tokens if token in word_set)


def _aspect_window(tokens: Sequence[str], start: int, end: int, radius: int = 5) -> list[str]:
    left = max(0, start - radius)
    right = min(len(tokens), end + radius)
    return list(tokens[left:right])


def _aspect_text(tokens: Sequence[str], start: int, end: int) -> str:
    return " ".join(tokens[start:end])


def _nearest_index(tokens: Sequence[str], words: Iterable[str], start: int = 0) -> int:
    word_set = set(words)
    for idx in range(start, len(tokens)):
        if tokens[idx] in word_set:
            return idx
    return -1


def _phrase_positions(tokens: Sequence[str], phrase: str) -> list[int]:
    phrase_tokens = [_norm_token(token) for token in phrase.split()]
    phrase_tokens = [token for token in phrase_tokens if token]
    if not phrase_tokens:
        return []
    return [
        idx
        for idx in range(0, len(tokens) - len(phrase_tokens) + 1)
        if list(tokens[idx : idx + len(phrase_tokens)]) == phrase_tokens
    ]


def _first_phrase_position(tokens: Sequence[str], phrases: Iterable[str]) -> int:
    positions = []
    for phrase in phrases:
        positions.extend(_phrase_positions(tokens, phrase))
    return min(positions) if positions else -1


def _clause_bounds(tokens: Sequence[str], start: int, end: int) -> tuple[int, int]:
    markers = {
        ",",
        ";",
        ".",
        "!",
        "?",
        "but",
        "however",
        "though",
        "although",
        "while",
        "yet",
        "except",
    }
    left = 0
    right = len(tokens)
    for idx in range(start - 1, -1, -1):
        if tokens[idx] in markers:
            left = idx + 1
            break
    for idx in range(end, len(tokens)):
        if tokens[idx] in markers:
            right = idx
            break
    return left, right


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(max(denominator, 1e-6))
    return float(numerator) / denominator


def _uses_pragmatic_v3(feature_set: str) -> bool:
    groups = {group.strip() for group in str(feature_set).split(",")}
    return any(group in groups for group in {"all_v3", "v3"}) or any(
        group in PRAGMATIC_V3_FEATURE_GROUPS and group not in PRAGMATIC_FEATURE_GROUPS
        for group in groups
    )


def _uses_refined_fact(feature_set: str) -> bool:
    groups = {group.strip() for group in str(feature_set).split(",")}
    return bool(groups & {"fact_refined", "all_refined_fact"})


def extract_pragmatic_features(
    text_list: Sequence[object],
    aspect_post: Sequence[int],
    feature_set: str = "all",
    pos_list: Sequence[object] | None = None,
) -> np.ndarray:
    """Extract restaurant ABSA cues used by fact and comparison residuals."""
    use_v3 = _uses_pragmatic_v3(feature_set)
    feature_dim = PRAGMATIC_V3_FEATURE_DIM if use_v3 else PRAGMATIC_FEATURE_DIM
    raw_tokens = [_norm_token(token) for token in text_list]
    raw_pos_tags = list(pos_list or [])
    raw_start, raw_end = aspect_post
    raw_start = min(max(raw_start, 0), len(raw_tokens))
    raw_end = min(max(raw_end, raw_start), len(raw_tokens))
    start = sum(1 for token in raw_tokens[:raw_start] if token)
    end = sum(1 for token in raw_tokens[:raw_end] if token)
    tokens = []
    pos_tags = []
    for idx, token in enumerate(raw_tokens):
        if not token:
            continue
        tokens.append(token)
        if idx < len(raw_pos_tags):
            pos_tags.append(raw_pos_tags[idx])
    start = min(max(start, 0), len(tokens))
    end = min(max(end, start), len(tokens))
    aspect = _aspect_text(tokens, start, end)
    window = _aspect_window(tokens, start, end, radius=5)
    left = tokens[:start]
    right = tokens[end:]
    features = np.zeros(feature_dim, dtype="float32")

    listing_verbs = {
        "include",
        "includes",
        "included",
        "including",
        "feature",
        "features",
        "featured",
        "offer",
        "offers",
        "offered",
        "serve",
        "serves",
        "served",
        "serving",
        "carry",
        "carries",
        "available",
    }
    objective_words = {
        "designed",
        "style",
        "restaurant",
        "upstairs",
        "sidewalk",
        "automatically",
        "added",
        "simple",
        "medium",
        "rare",
        "contemporary",
        "japanese",
        "classic",
        "classics",
        "selection",
        "dish",
        "dishes",
        "fare",
    }
    subjective_words = {
        "good",
        "great",
        "excellent",
        "amazing",
        "best",
        "better",
        "worth",
        "nice",
        "wonderful",
        "fresh",
        "clean",
        "cheap",
        "cheaper",
        "free",
        "bad",
        "worse",
        "worst",
        "mediocre",
        "average",
        "rude",
        "snotty",
        "overpriced",
        "expensive",
        "disappointing",
        "disappointingly",
        "stinks",
        "nasty",
        "so-so",
        "decent",
        "ok",
        "okay",
        "horrible",
        "delicious",
        "perfect",
        "slow",
        "thinly",
    }
    positive_words = {
        "good",
        "great",
        "excellent",
        "amazing",
        "best",
        "better",
        "worth",
        "nice",
        "wonderful",
        "fresh",
        "clean",
        "cheap",
        "cheaper",
        "free",
        "delicious",
        "perfect",
        "happy",
        "thrilled",
        "authentic",
        "recommend",
        "reccomend",
    }
    negative_words = {
        "bad",
        "worse",
        "worst",
        "mediocre",
        "average",
        "rude",
        "snotty",
        "overpriced",
        "expensive",
        "disappointing",
        "disappointingly",
        "stinks",
        "nasty",
        "so-so",
        "horrible",
        "complaint",
        "complained",
        "thinly",
        "watered",
        "down",
        "freezing",
        "marginally",
        "annoyed",
        "suspicious",
    }
    contrast_words = {"but", "however", "though", "although", "yet", "while", "except"}
    negators = {
        "not",
        "no",
        "never",
        "nt",
        "without",
        "neither",
        "nor",
        "cant",
        "couldnt",
        "dont",
        "didnt",
        "doesnt",
    }
    comparative_words = {
        "better",
        "worse",
        "best",
        "worst",
        "cheaper",
        "higher",
        "lower",
        "more",
        "less",
    }
    external_phrases = [
        "at home",
        "at a mall",
        "mall food court",
        "anywhere else",
        "other restaurants",
        "other japanese restaurants",
        "local grocery store",
        "big mac",
        "food court",
        "chinatown",
        "by nyc standards",
    ]
    wait_terms = {"wait", "waiting", "line"}
    price_terms = {"price", "prices", "priced", "value", "deal", "bill", "cost", "bucks"}

    if any(token in listing_verbs for token in tokens):
        features[0] = 1.0
    if "there" in tokens and any(token in {"is", "are", "was", "were", "s"} for token in tokens):
        features[1] = 1.0
    comma_count = sum(1 for token in text_list if str(token) == ",")
    if any(
        token in {"menu", "entrees", "desserts", "appetizers", "selection", "fare"}
        for token in tokens
    ) and (
        comma_count >= 1
        or any(token in {"and", "or", "such", "like", "includes"} for token in tokens)
    ):
        features[2] = 1.0
    if _count_any(window, objective_words) > 0 and _count_any(window, subjective_words) == 0:
        features[3] = 1.0

    if _uses_refined_fact(feature_set):
        clause_left, clause_right = _clause_bounds(tokens, start, end)
        same_clause = tokens[clause_left:clause_right]
        clause_subjective = _count_any(
            same_clause, subjective_words | positive_words | negative_words
        )
        clause_listing = _count_any(same_clause, listing_verbs)
        clause_objective = _count_any(same_clause, objective_words)
        clause_menu = _count_any(
            same_clause,
            {
                "menu",
                "menus",
                "entrees",
                "desserts",
                "appetizers",
                "selection",
                "fare",
                "offerings",
            },
        )
        clause_list_markers = _count_any(same_clause, {"and", "or", "like", "including", "with"})
        low_subjectivity_descriptors = {
            "classic",
            "classics",
            "simple",
            "medium",
            "rare",
            "seasonal",
            "contemporary",
            "traditional",
            "japanese",
            "mediterranean",
            "available",
            "automatic",
            "automatically",
            "regular",
            "standard",
        }
        low_subjectivity = _count_any(same_clause + window, low_subjectivity_descriptors) > 0
        punctuation_list = comma_count >= 1 and (clause_listing > 0 or clause_menu > 0)
        object_list_context = punctuation_list or (
            clause_list_markers > 0 and (clause_listing > 0 or clause_menu > 0)
        )
        is_objective_clause = clause_subjective == 0

        features[0] = float(
            clause_listing > 0 and is_objective_clause and (object_list_context or clause_menu > 0)
        )
        features[1] = float(
            "there" in same_clause
            and any(token in {"is", "are", "was", "were", "s"} for token in same_clause)
            and is_objective_clause
        )
        features[2] = float(clause_menu > 0 and object_list_context and is_objective_clause)
        features[3] = float((clause_objective > 0 or low_subjectivity) and is_objective_clause)

    if _count_any(window, subjective_words) > 0:
        features[4] = min(1.0, _count_any(window, subjective_words) / 2.0)
    contrast_idx = _nearest_index(tokens, contrast_words)
    if contrast_idx >= 0:
        if contrast_idx >= end:
            features[5] = 1.0
        elif contrast_idx < start:
            features[6] = 1.0
    if _count_any(window, negators) > 0:
        features[7] = 1.0
    if _count_any(window, negators) > 0 and _count_any(window, positive_words) > 0:
        features[8] = 1.0
    if _count_any(window, negators) > 0 and _count_any(window, negative_words) > 0:
        features[9] = 1.0

    than_idx = _nearest_index(tokens, {"than"})
    better_idx = _nearest_index(tokens, comparative_words)
    external_ref = any(_contains_phrase(tokens, phrase) for phrase in external_phrases)
    if external_ref:
        features[13] = 1.0
    if better_idx >= 0:
        if external_ref and (
            _contains_phrase(tokens, "at home")
            or _contains_phrase(tokens, "at a mall")
            or _contains_phrase(tokens, "food court")
            or _contains_phrase(tokens, "local grocery store")
        ):
            features[10] = 1.0
        if than_idx >= 0 and start > than_idx:
            features[12] = 1.0
            if _count_any(window + right[:4], negative_words) > 0:
                features[10] = 1.0
        if _contains_phrase(tokens, "anywhere else") and _count_any(window, price_terms) > 0:
            features[11] = 1.0
        if _count_any(window, price_terms) > 0 and any(
            token in {"cheap", "cheaper", "low", "lower"} for token in tokens
        ):
            features[11] = 1.0
    if "make" in left and "better" in right and _contains_phrase(tokens, "at home"):
        features[10] = 1.0

    aspect_is_wait = aspect in wait_terms or any(token in wait_terms for token in tokens[start:end])
    aspect_is_price = aspect in price_terms or any(
        token in price_terms for token in tokens[start:end]
    )
    if aspect_is_wait:
        if _count_any(window, {"no", "never", "without"}) > 0:
            features[14] = 1.0
        if _count_any(window, {"long", "slow", "years", "hours", "took", "waited"}) > 0:
            features[15] = 1.0
    if aspect_is_price:
        if (
            _count_any(
                window + tokens,
                {"cheap", "cheaper", "free", "value", "deal", "worth", "low", "lower"},
            )
            > 0
        ):
            features[16] = 1.0
        if _count_any(window + tokens, {"overpriced", "expensive", "high", "higher", "priced"}) > 0:
            features[17] = 1.0
        if _contains_phrase(tokens, "anywhere else") and any(
            token in {"high", "higher"} for token in tokens
        ):
            features[16] = 1.0
            features[17] = 0.0

    if "nt" in tokens:
        features[7] = max(features[7], 1.0)

    if use_v3:
        low_subjectivity_descriptors = {
            "classic",
            "classics",
            "simple",
            "medium",
            "rare",
            "seasonal",
            "contemporary",
            "traditional",
            "japanese",
            "mediterranean",
            "available",
            "automatic",
            "automatically",
            "side",
            "regular",
            "standard",
            "basic",
            "usual",
        }
        intensifiers = {
            "so",
            "very",
            "really",
            "extremely",
            "quite",
            "totally",
            "always",
            "never",
            "definitely",
            "absolutely",
            "perfectly",
            "more",
            "most",
        }
        experiencers = {
            "i",
            "we",
            "me",
            "us",
            "my",
            "our",
            "you",
            "your",
            "everyone",
            "customers",
            "people",
            "diners",
            "dieters",
            "guests",
        }
        beneficiary_markers = {"for", "to"}
        external_location_phrases = [
            "at home",
            "at a mall",
            "mall food court",
            "food court",
            "local grocery store",
            "anywhere else",
            "other restaurants",
            "other places",
            "elsewhere",
            "by nyc standards",
        ]
        external_brand_phrases = ["big mac", "mcdonalds", "mcdonald"]
        clause_left, clause_right = _clause_bounds(tokens, start, end)
        same_clause = tokens[clause_left:clause_right]
        other_clause = tokens[:clause_left] + tokens[clause_right:]
        same_pos = _count_any(same_clause, positive_words)
        same_neg = _count_any(same_clause, negative_words)
        other_pos = _count_any(other_clause, positive_words)
        other_neg = _count_any(other_clause, negative_words)

        if pos_tags and len(pos_tags) >= len(tokens):
            noun_like_count = sum(
                1 for pos in pos_tags[clause_left:clause_right] if str(pos).startswith("NN")
            )
        else:
            object_like_terms = {
                "menu",
                "menus",
                "entrees",
                "desserts",
                "appetizers",
                "selection",
                "offerings",
                "fare",
                "food",
                "dish",
                "dishes",
                "meal",
                "sushi",
                "lasagna",
                "pasta",
                "steak",
                "flan",
                "sopaipillas",
                "cakes",
                "sandwich",
                "sandwiches",
                "wine",
                "wines",
                "decor",
                "candlelight",
                "music",
                "price",
                "prices",
                "bill",
                "wait",
            }
            noun_like_count = sum(1 for token in same_clause if token in object_like_terms)
        comma_markers = sum(1 for token in text_list if str(token) in {",", ";"})
        relation_markers = _count_any(same_clause, {"and", "or", "like", "including"})
        list_markers = relation_markers + (comma_markers if (features[0] or features[2]) else 0)
        if features[0] or features[1] or features[2]:
            object_list_density = min(
                1.0,
                _safe_ratio(noun_like_count + list_markers, max(len(same_clause), 1)) * 3.0,
            )
        elif (
            relation_markers > 0
            and noun_like_count >= 3
            and _count_any(same_clause, subjective_words) == 0
        ):
            object_list_density = min(
                1.0,
                _safe_ratio(noun_like_count + relation_markers, max(len(same_clause), 1)) * 3.0,
            )
        else:
            object_list_density = 0.0
        low_subjectivity = float(
            _count_any(same_clause + window, low_subjectivity_descriptors) > 0
            and _count_any(same_clause, subjective_words) == 0
        )
        fact_object_score = min(
            1.0,
            0.35 * features[0]
            + 0.25 * features[1]
            + 0.25 * features[2]
            + 0.30 * features[3]
            + 0.35 * object_list_density
            + 0.30 * low_subjectivity,
        )

        intensity_hits = _count_any(same_clause + window, intensifiers)
        subjective_hits = _count_any(
            same_clause, subjective_words | positive_words | negative_words
        )
        opinion_render = min(1.0, 0.35 * subjective_hits + 0.20 * intensity_hits)
        human_experiencer = float(_count_any(tokens, experiencers) > 0 and subjective_hits > 0)
        beneficiary = 0.0
        for idx, token in enumerate(tokens[:-1]):
            if token in beneficiary_markers and tokens[idx + 1] in experiencers:
                beneficiary = 1.0
                break
        if _contains_phrase(tokens, "for everyone") or _contains_phrase(
            tokens, "something for everyone"
        ):
            beneficiary = 1.0
        opinion_render = min(1.0, opinion_render + 0.25 * human_experiencer + 0.30 * beneficiary)

        same_clause_positive = min(1.0, 0.5 * same_pos + 0.2 * intensity_hits)
        same_clause_negative = min(
            1.0,
            0.5 * same_neg + 0.2 * _count_any(same_clause, {"not", "no", "never", "nt"}),
        )
        cross_clause_conflict = float(
            (same_pos > 0 and other_neg > 0) or (same_neg > 0 and other_pos > 0)
        )
        contrast_marker_strength = float(any(token in contrast_words for token in tokens))
        semi_contrast = float(
            _contains_phrase(tokens, "even when")
            or _contains_phrase(tokens, "at least")
            or _contains_phrase(tokens, "not quite")
        )
        contrast_balance_neutral = float(
            cross_clause_conflict
            and (contrast_marker_strength or semi_contrast)
            and abs((same_pos + other_pos) - (same_neg + other_neg)) <= 2
        )

        external_location_pos = _first_phrase_position(tokens, external_location_phrases)
        external_brand_pos = _first_phrase_position(tokens, external_brand_phrases)
        external_pos = min(
            [pos for pos in (external_location_pos, external_brand_pos) if pos >= 0],
            default=-1,
        )
        than_pos = _nearest_index(tokens, {"than"})
        comparison_current_positive = 0.0
        comparison_current_negative = 0.0
        if than_pos >= 0 and external_pos >= 0:
            if (
                end <= than_pos
                and external_pos > than_pos
                and _count_any(tokens, {"better", "best", "cheaper", "lower"}) > 0
            ):
                comparison_current_positive = 1.0
            if start > than_pos and _count_any(tokens, {"better", "best"}) > 0:
                comparison_current_negative = 1.0
        if external_pos >= 0 and _count_any(tokens, {"better", "best"}) > 0:
            if (
                _contains_phrase(tokens, "at home")
                or _contains_phrase(tokens, "at a mall")
                or _contains_phrase(tokens, "food court")
                or (than_pos < 0 and external_pos >= end and _contains_phrase(tokens, "elsewhere"))
            ):
                comparison_current_negative = max(comparison_current_negative, 1.0)
        if _contains_phrase(tokens, "make better") and _contains_phrase(tokens, "at home"):
            comparison_current_negative = 1.0
        if (
            _contains_phrase(tokens, "anywhere else")
            and _count_any(tokens, price_terms) > 0
            and _count_any(tokens, {"high", "higher"}) > 0
        ):
            comparison_current_positive = 1.0
            comparison_current_negative = 0.0

        domain_kb_positive = 0.0
        domain_kb_negative = 0.0
        if aspect_is_wait:
            domain_kb_positive = float(
                _count_any(window + tokens, {"no", "never", "without", "quick", "fast", "short"})
                > 0
            )
            domain_kb_negative = float(
                _count_any(window + tokens, {"long", "slow", "waited", "took", "years", "hours"})
                > 0
                and not domain_kb_positive
            )
        if aspect_is_price:
            domain_kb_positive = max(
                domain_kb_positive,
                float(
                    _count_any(
                        window + tokens,
                        {"cheap", "cheaper", "value", "deal", "worth", "low", "reasonable"},
                    )
                    > 0
                ),
            )
            domain_kb_negative = max(
                domain_kb_negative,
                float(
                    _count_any(
                        window + tokens,
                        {"overpriced", "expensive", "pricey", "high", "higher"},
                    )
                    > 0
                    and not comparison_current_positive
                ),
            )
        if _contains_phrase(tokens, "fast food price"):
            domain_kb_positive = 1.0
            domain_kb_negative = 0.0

        features[18] = fact_object_score
        features[19] = object_list_density
        features[20] = low_subjectivity
        features[21] = opinion_render
        features[22] = human_experiencer
        features[23] = beneficiary
        features[24] = same_clause_positive
        features[25] = same_clause_negative
        features[26] = cross_clause_conflict
        features[27] = contrast_balance_neutral
        features[28] = comparison_current_positive
        features[29] = comparison_current_negative
        features[30] = float(external_location_pos >= 0)
        features[31] = float(external_brand_pos >= 0)
        features[32] = domain_kb_positive
        features[33] = domain_kb_negative

    if feature_set in {"all", "all_refined_fact"}:
        selected = set(range(PRAGMATIC_FEATURE_DIM))
    elif feature_set in {"all_v3", "v3"}:
        selected = set(range(PRAGMATIC_V3_FEATURE_DIM))
    else:
        selected = set()
        for group in str(feature_set).split(","):
            group = group.strip()
            if use_v3:
                if group == "all":
                    selected.update(range(PRAGMATIC_FEATURE_DIM))
                else:
                    selected.update(PRAGMATIC_V3_FEATURE_GROUPS.get(group, []))
            else:
                selected.update(PRAGMATIC_FEATURE_GROUPS.get(group, []))
    if selected and len(selected) < feature_dim:
        mask = np.zeros(feature_dim, dtype="float32")
        mask[list(selected)] = 1.0
        features = features * mask

    return features
