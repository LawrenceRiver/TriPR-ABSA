"""Minimal paper-backed features used by fact and comparison residuals."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np

PRAGMATIC_FEATURE_NAMES = [
    "fact_listing",
    "there_be_existence",
    "menu_enumeration",
    "objective_description",
    "current_negative_comparison",
    "current_positive_comparison",
    "aspect_after_than",
    "external_reference",
    "price_positive",
]

PRAGMATIC_FEATURE_GROUPS = {
    "fact": [0, 1, 2, 3],
    "comparison": [4, 5, 6, 7],
    "domain": [8],
}

PRAGMATIC_V3_EXTRA_FEATURE_NAMES = [
    "relation_fact_object_score",
    "relation_object_list_density",
    "relation_low_subjectivity_descriptor",
    "relation_opinion_render_score",
    "relation_comparison_current_positive",
    "relation_comparison_current_negative",
]

PRAGMATIC_V3_FEATURE_NAMES = PRAGMATIC_FEATURE_NAMES + PRAGMATIC_V3_EXTRA_FEATURE_NAMES

PRAGMATIC_V3_FEATURE_GROUPS = {
    **PRAGMATIC_FEATURE_GROUPS,
    "fact_relation": [9, 10, 11],
    "opinion_render": [12],
    "comparison_relation": [13, 14],
}
PRAGMATIC_V3_FEATURE_GROUPS.update(
    {name: [idx] for idx, name in enumerate(PRAGMATIC_V3_FEATURE_NAMES)}
)

_CLAUSE_MARKERS = {
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
_NEGATORS = {
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
    left = 0
    right = len(tokens)
    for idx in range(start - 1, -1, -1):
        if tokens[idx] in _CLAUSE_MARKERS:
            left = idx + 1
            break
    for idx in range(end, len(tokens)):
        if tokens[idx] in _CLAUSE_MARKERS:
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


def _clause_token(token: object) -> str:
    raw = str(token).strip().lower()
    if raw in _CLAUSE_MARKERS:
        return raw
    return _norm_token(token)


def _direct_better_elsewhere(text_list: Sequence[object], aspect_post: Sequence[int]) -> bool:
    """Recognize unnegated comparative -> aspect -> elsewhere within one clause."""
    tokens = [_clause_token(token) for token in text_list]
    raw_start, raw_end = aspect_post
    start = min(max(raw_start, 0), len(tokens))
    end = min(max(raw_end, start), len(tokens))
    clause_left, clause_right = _clause_bounds(tokens, start, end)

    comparative_positions = [
        idx
        for idx in range(max(clause_left, start - 2), start)
        if tokens[idx] in {"better", "best"}
    ]
    if not comparative_positions:
        return False
    comparative_pos = max(comparative_positions)

    elsewhere_positions = [idx for idx in range(end, clause_right) if tokens[idx] == "elsewhere"]
    if not elsewhere_positions:
        return False
    elsewhere_pos = min(elsewhere_positions)

    relation_span = tokens[clause_left : elsewhere_pos + 1]
    return (
        _count_any(relation_span, _NEGATORS) == 0
        and comparative_pos < start <= end <= elsewhere_pos
    )


def extract_pragmatic_features(
    text_list: Sequence[object],
    aspect_post: Sequence[int],
    feature_set: str = "all",
    pos_list: Sequence[object] | None = None,
) -> np.ndarray:
    """Extract the base or V3 features consumed by public residual modules."""
    use_v3 = _uses_pragmatic_v3(feature_set)
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
    window = _aspect_window(tokens, start, end, radius=5)
    left = tokens[:start]
    right = tokens[end:]

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
    price_terms = {"price", "prices", "priced", "value", "deal", "bill", "cost", "bucks"}

    values = {name: 0.0 for name in PRAGMATIC_V3_FEATURE_NAMES}
    values["fact_listing"] = float(any(token in listing_verbs for token in tokens))
    values["there_be_existence"] = float(
        "there" in tokens and any(token in {"is", "are", "was", "were", "s"} for token in tokens)
    )
    comma_count = sum(1 for token in text_list if str(token) == ",")
    values["menu_enumeration"] = float(
        any(
            token in {"menu", "entrees", "desserts", "appetizers", "selection", "fare"}
            for token in tokens
        )
        and (
            comma_count >= 1
            or any(token in {"and", "or", "such", "like", "includes"} for token in tokens)
        )
    )
    values["objective_description"] = float(
        _count_any(window, objective_words) > 0 and _count_any(window, subjective_words) == 0
    )

    than_idx = _nearest_index(tokens, {"than"})
    better_idx = _nearest_index(tokens, comparative_words)
    external_ref = any(_contains_phrase(tokens, phrase) for phrase in external_phrases)
    values["external_reference"] = float(external_ref)
    if better_idx >= 0:
        if external_ref and (
            _contains_phrase(tokens, "at home")
            or _contains_phrase(tokens, "at a mall")
            or _contains_phrase(tokens, "food court")
            or _contains_phrase(tokens, "local grocery store")
        ):
            values["current_negative_comparison"] = 1.0
        if than_idx >= 0 and start > than_idx:
            values["aspect_after_than"] = 1.0
            if _count_any(window + right[:4], negative_words) > 0:
                values["current_negative_comparison"] = 1.0
        if _contains_phrase(tokens, "anywhere else") and _count_any(window, price_terms) > 0:
            values["current_positive_comparison"] = 1.0
        if _count_any(window, price_terms) > 0 and any(
            token in {"cheap", "cheaper", "low", "lower"} for token in tokens
        ):
            values["current_positive_comparison"] = 1.0
    if "make" in left and "better" in right and _contains_phrase(tokens, "at home"):
        values["current_negative_comparison"] = 1.0

    aspect_is_price = any(token in price_terms for token in tokens[start:end])
    if aspect_is_price and (
        _count_any(
            window + tokens,
            {"cheap", "cheaper", "free", "value", "deal", "worth", "low", "lower"},
        )
        > 0
    ):
        values["price_positive"] = 1.0
    if (
        aspect_is_price
        and _contains_phrase(tokens, "anywhere else")
        and any(token in {"high", "higher"} for token in tokens)
    ):
        values["price_positive"] = 1.0

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
        list_markers = relation_markers + (
            comma_markers if values["fact_listing"] or values["menu_enumeration"] else 0
        )
        if values["fact_listing"] or values["there_be_existence"] or values["menu_enumeration"]:
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
            0.35 * values["fact_listing"]
            + 0.25 * values["there_be_existence"]
            + 0.25 * values["menu_enumeration"]
            + 0.30 * values["objective_description"]
            + 0.35 * object_list_density
            + 0.30 * low_subjectivity,
        )

        intensity_hits = _count_any(same_clause + window, intensifiers)
        subjective_hits = _count_any(
            same_clause, subjective_words | positive_words | negative_words
        )
        opinion_render = min(1.0, 0.35 * subjective_hits + 0.20 * intensity_hits)
        human_experiencer = float(_count_any(tokens, experiencers) > 0 and subjective_hits > 0)
        beneficiary = float(
            any(
                token in {"for", "to"} and tokens[idx + 1] in experiencers
                for idx, token in enumerate(tokens[:-1])
            )
            or _contains_phrase(tokens, "for everyone")
            or _contains_phrase(tokens, "something for everyone")
        )
        opinion_render = min(1.0, opinion_render + 0.25 * human_experiencer + 0.30 * beneficiary)

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
            ):
                comparison_current_negative = 1.0
        if _contains_phrase(tokens, "make better") and _contains_phrase(tokens, "at home"):
            comparison_current_negative = 1.0
        if (
            _contains_phrase(tokens, "anywhere else")
            and _count_any(tokens, price_terms) > 0
            and _count_any(tokens, {"high", "higher"}) > 0
        ):
            comparison_current_positive = 1.0
            comparison_current_negative = 0.0
        if than_pos < 0 and _direct_better_elsewhere(text_list, aspect_post):
            comparison_current_negative = 1.0

        values["relation_fact_object_score"] = fact_object_score
        values["relation_object_list_density"] = object_list_density
        values["relation_low_subjectivity_descriptor"] = low_subjectivity
        values["relation_opinion_render_score"] = opinion_render
        values["relation_comparison_current_positive"] = comparison_current_positive
        values["relation_comparison_current_negative"] = comparison_current_negative

    feature_names = PRAGMATIC_V3_FEATURE_NAMES if use_v3 else PRAGMATIC_FEATURE_NAMES
    features = np.array([values[name] for name in feature_names], dtype="float32")

    if feature_set == "all" or feature_set in {"all_v3", "v3"}:
        selected = set(range(len(feature_names)))
    else:
        groups = PRAGMATIC_V3_FEATURE_GROUPS if use_v3 else PRAGMATIC_FEATURE_GROUPS
        selected = set()
        for group in str(feature_set).split(","):
            group = group.strip()
            if use_v3 and group == "all":
                selected.update(range(len(PRAGMATIC_FEATURE_NAMES)))
            else:
                selected.update(groups.get(group, []))
    if selected and len(selected) < len(feature_names):
        mask = np.zeros(len(feature_names), dtype="float32")
        mask[list(selected)] = 1.0
        features = features * mask

    return features
