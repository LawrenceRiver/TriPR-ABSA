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
_RELATION_BOUNDARIES = _CLAUSE_MARKERS | {"and", "or"}
_NEGATORS = {
    "not",
    "no",
    "never",
    "nt",
    "without",
    "neither",
    "nor",
    "cant",
    "cannot",
    "couldnt",
    "dont",
    "didnt",
    "doesnt",
    "arent",
    "hadnt",
    "hasnt",
    "havent",
    "isnt",
    "mustnt",
    "shouldnt",
    "wasnt",
    "werent",
    "wont",
    "wouldnt",
}
_DIRECTIONAL_EXTERNAL_PHRASES = (
    "at home",
    "at a mall",
    "mall food court",
    "anywhere else",
    "other restaurants",
    "other japanese restaurants",
    "other places",
    "local grocery store",
    "big mac",
    "food court",
    "chinatown",
    "by nyc standards",
    "elsewhere",
    "mcdonalds",
    "mcdonald",
)
_COMPARISON_RELATION_CUES = {
    "ate",
    "eaten",
    "experienced",
    "find",
    "found",
    "had",
    "make",
    "made",
    "ordered",
    "tried",
}
_COMPARISON_CUE_WINDOW = 4
_DIRECT_COMPARISON_EXPERIENCERS = {"i", "we", "you"}
_IMPLICIT_SUBJECT_WINDOW = 4
_IMPLICIT_SUBJECT_BRIDGE = {
    "actually",
    "already",
    "also",
    "am",
    "are",
    "can",
    "could",
    "did",
    "do",
    "does",
    "ever",
    "has",
    "have",
    "just",
    "may",
    "might",
    "must",
    "often",
    "once",
    "really",
    "recently",
    "shall",
    "should",
    "sometimes",
    "still",
    "usually",
    "was",
    "were",
    "will",
    "would",
}
_POSITIVE_COMPARATIVES = {"better", "best"}
_NEGATIVE_COMPARATIVES = {"worse", "worst"}
_PRICE_POSITIVE_COMPARATIVES = {"cheaper", "lower"}
_PRICE_NEGATIVE_COMPARATIVES = {"higher"}
_VALUE_POSITIVE_COMPARATIVES = {"higher"}
_VALUE_NEGATIVE_COMPARATIVES = {"lower"}
_EXPLICIT_TARGET_WINDOW = 4
_MONETARY_AMOUNT_ASPECT_TERMS = {"price", "prices", "priced", "bill", "cost", "bucks"}
_VALUE_ASPECT_TERMS = {"value", "values"}
_PRICE_CONTEXT_TERMS = _MONETARY_AMOUNT_ASPECT_TERMS | _VALUE_ASPECT_TERMS | {"deal"}


def _norm_token(token: object) -> str:
    raw = str(token).strip().lower().translate(str.maketrans({"’": "'", "‘": "'"}))
    if raw.endswith("n't"):
        raw = f"{raw[:-3]}nt"
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", raw)


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


def _relation_bounds(tokens: Sequence[str], start: int, end: int) -> tuple[int, int]:
    left = 0
    right = len(tokens)
    for idx in range(start - 1, -1, -1):
        if tokens[idx] in _RELATION_BOUNDARIES:
            left = idx + 1
            break
    for idx in range(end, len(tokens)):
        if tokens[idx] in _RELATION_BOUNDARIES:
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


def _direct_negative_external_comparison(
    text_list: Sequence[object], aspect_post: Sequence[int]
) -> bool:
    """Recognize unnegated superiority -> aspect -> external reference in one clause."""
    tokens = [_clause_token(token) for token in text_list]
    raw_start, raw_end = aspect_post
    start = min(max(raw_start, 0), len(tokens))
    end = min(max(raw_end, start), len(tokens))
    clause_left, clause_right = _relation_bounds(tokens, start, end)
    clause = tokens[clause_left:clause_right]
    if "than" in clause or _count_any(clause, _NEGATORS) > 0:
        return False

    comparative_pos = start - 1
    if comparative_pos < clause_left or tokens[comparative_pos] != "better":
        return False
    cue_positions = [
        idx
        for idx in range(
            max(clause_left, comparative_pos - _COMPARISON_CUE_WINDOW), comparative_pos
        )
        if tokens[idx] in _COMPARISON_RELATION_CUES
    ]
    if not cue_positions:
        return False
    cue_pos = max(cue_positions)
    subject_positions = [
        idx
        for idx in range(max(clause_left, cue_pos - _IMPLICIT_SUBJECT_WINDOW), cue_pos)
        if tokens[idx] in _DIRECT_COMPARISON_EXPERIENCERS
    ]
    if not subject_positions:
        return False
    subject_pos = max(subject_positions)
    if any(token not in _IMPLICIT_SUBJECT_BRIDGE for token in tokens[subject_pos + 1 : cue_pos]):
        return False

    external_spans = [
        (position, position + len(phrase.split()))
        for phrase in _DIRECTIONAL_EXTERNAL_PHRASES
        for position in _phrase_positions(tokens, phrase)
        if end <= position and position + len(phrase.split()) <= clause_right
    ]
    if not external_spans:
        return False
    external_start, external_end = min(external_spans)

    return start <= end <= external_start


def _explicit_comparison_direction(
    text_list: Sequence[object], aspect_post: Sequence[int]
) -> tuple[float, float]:
    """Return positive/negative flags for one clear, unnegated ``than`` relation."""
    tokens = [_clause_token(token) for token in text_list]
    raw_start, raw_end = aspect_post
    start = min(max(raw_start, 0), len(tokens))
    end = min(max(raw_end, start), len(tokens))
    relation_left, relation_right = _relation_bounds(tokens, start, end)
    relation = tokens[relation_left:relation_right]
    if _count_any(relation, _NEGATORS) > 0:
        return 0.0, 0.0

    positive_comparatives = set(_POSITIVE_COMPARATIVES)
    negative_comparatives = set(_NEGATIVE_COMPARATIVES)
    aspect_tokens = tokens[start:end]
    if any(token in _MONETARY_AMOUNT_ASPECT_TERMS for token in aspect_tokens):
        positive_comparatives.update(_PRICE_POSITIVE_COMPARATIVES)
        negative_comparatives.update(_PRICE_NEGATIVE_COMPARATIVES)
    elif any(token in _VALUE_ASPECT_TERMS for token in aspect_tokens):
        positive_comparatives.update(_VALUE_POSITIVE_COMPARATIVES)
        negative_comparatives.update(_VALUE_NEGATIVE_COMPARATIVES)
    clear_comparatives = positive_comparatives | negative_comparatives

    candidates: list[tuple[int, int, int, bool]] = []
    for than_pos in range(relation_left, relation_right):
        if tokens[than_pos] != "than":
            continue
        comparative_positions = [
            idx for idx in range(relation_left, than_pos) if tokens[idx] in clear_comparatives
        ]
        if not comparative_positions:
            continue
        comparative_pos = max(comparative_positions)
        if end <= comparative_pos and comparative_pos - end <= _EXPLICIT_TARGET_WINDOW:
            aspect_is_left_target = True
            distance = comparative_pos - end
        elif start > than_pos and start - than_pos <= _EXPLICIT_TARGET_WINDOW:
            aspect_is_left_target = False
            distance = start - than_pos
        else:
            continue
        candidates.append((distance, comparative_pos, than_pos, aspect_is_left_target))

    if candidates:
        _, comparative_pos, _, aspect_is_left_target = min(candidates)
        positive_comparative = tokens[comparative_pos] in positive_comparatives
        current_is_positive = positive_comparative == aspect_is_left_target
        return (1.0, 0.0) if current_is_positive else (0.0, 1.0)

    return 0.0, 0.0


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
    direct_negative_comparison = _direct_negative_external_comparison(text_list, aspect_post)
    explicit_positive, explicit_negative = _explicit_comparison_direction(text_list, aspect_post)
    window = _aspect_window(tokens, start, end, radius=5)

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
    price_terms = _PRICE_CONTEXT_TERMS

    values = {name: 0.0 for name in PRAGMATIC_V3_FEATURE_NAMES}
    values["current_positive_comparison"] = explicit_positive
    values["current_negative_comparison"] = explicit_negative
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
        if than_idx >= 0 and start > than_idx:
            values["aspect_after_than"] = 1.0
        if (
            than_idx < 0
            and _count_any(window, price_terms) > 0
            and any(token in {"cheap", "cheaper", "low", "lower"} for token in tokens)
        ):
            values["current_positive_comparison"] = 1.0
    if direct_negative_comparison:
        values["current_negative_comparison"] = 1.0
    if explicit_positive or explicit_negative:
        values["current_positive_comparison"] = explicit_positive
        values["current_negative_comparison"] = explicit_negative

    aspect_is_price = any(token in _MONETARY_AMOUNT_ASPECT_TERMS for token in tokens[start:end])
    if (
        aspect_is_price
        and than_idx < 0
        and (
            _count_any(
                window + tokens,
                {"cheap", "cheaper", "free", "value", "deal", "worth", "low", "lower"},
            )
            > 0
        )
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

        comparison_current_positive = explicit_positive
        comparison_current_negative = explicit_negative
        if direct_negative_comparison:
            comparison_current_negative = 1.0
        if explicit_positive or explicit_negative:
            comparison_current_positive = explicit_positive
            comparison_current_negative = explicit_negative

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
