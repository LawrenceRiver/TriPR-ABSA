import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "reported_metrics.json"

METRIC_DEFINITION = {
    "accuracy": "correct predictions divided by all test examples",
    "macro_f1": "unweighted mean of per-class F1",
}

EXPECTED_EXPERIMENTS = [
    {
        "id": "restaurant_strategy",
        "dataset": "SemEval-2014 Restaurant",
        "split": "test",
        "aggregation": "mean of three selected checkpoints",
        "source": (
            "internal project evaluation: strategy ablation table and selected-checkpoint "
            "summary"
        ),
        "limitations": (
            "Reported result from 3 selected checkpoints; not independently rerun during release "
            "preparation and not a state-of-the-art claim."
        ),
        "rows": [
            ("baseline", 0.8382, 0.7480),
            ("fact", 0.8406, 0.7522),
            ("comparison", 0.8394, 0.7499),
            ("intensity", 0.8388, 0.7488),
            ("fact+comparison", 0.8418, 0.7541),
            ("all", 0.8424, 0.7547),
        ],
    },
    {
        "id": "laptop_strategy",
        "dataset": "SemEval-2014 Laptop",
        "split": "test",
        "aggregation": "single checkpoint; no cross-checkpoint aggregation",
        "source": "internal project evaluation: Laptop cross-domain strategy ablation",
        "limitations": (
            "Single-checkpoint preliminary cross-domain evidence with mixed, small changes; not "
            "evidence of universal improvement and not a state-of-the-art claim."
        ),
        "rows": [
            ("baseline", 0.8022, 0.7701),
            ("fact", 0.7991, 0.7676),
            ("comparison", 0.8038, 0.7724),
            ("intensity", 0.8022, 0.7703),
            ("fact+comparison", 0.8006, 0.7700),
            ("all", 0.8022, 0.7714),
        ],
    },
    {
        "id": "twitter_strategy",
        "dataset": "Twitter ABSA",
        "split": "test",
        "aggregation": "single checkpoint; no cross-checkpoint aggregation",
        "source": "internal project evaluation: Twitter cross-domain strategy ablation",
        "limitations": (
            "Single-checkpoint preliminary cross-domain evidence: fact is best; all is "
            "approximately neutral in macro-F1 while accuracy decreases; not evidence of "
            "universal improvement and not a state-of-the-art claim."
        ),
        "rows": [
            ("baseline", 0.7578, 0.7450),
            ("fact", 0.7592, 0.7463),
            ("comparison", 0.7563, 0.7439),
            ("intensity", 0.7548, 0.7447),
            ("fact+comparison", 0.7578, 0.7452),
            ("all", 0.7548, 0.7449),
        ],
    },
]


def test_reported_metrics_have_provenance() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["class_order"] == ["positive", "negative", "neutral"]
    assert [experiment["id"] for experiment in data["experiments"]] == [
        expected["id"] for expected in EXPECTED_EXPERIMENTS
    ]

    for experiment, expected in zip(data["experiments"], EXPECTED_EXPERIMENTS):
        assert experiment["dataset"]
        assert experiment["dataset"] == expected["dataset"]
        assert experiment["split"] == expected["split"]
        assert experiment["aggregation"] == expected["aggregation"]
        assert experiment["source"]
        assert experiment["source"] == expected["source"]
        assert experiment["limitations"]
        assert experiment["limitations"] == expected["limitations"]
        assert experiment["metric_definition"] == METRIC_DEFINITION
        assert experiment["release_status"] == ("reported, not rerun during release preparation")
        assert experiment["rows"]

        assert [row["name"] for row in experiment["rows"]] == [row[0] for row in expected["rows"]]
        assert {row["name"]: (row["accuracy"], row["macro_f1"]) for row in experiment["rows"]} == {
            name: (accuracy, macro_f1) for name, accuracy, macro_f1 in expected["rows"]
        }

        for row in experiment["rows"]:
            assert 0 <= row["accuracy"] <= 1
            assert 0 <= row["macro_f1"] <= 1


def test_primary_result_matches_course_paper() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    primary = next(item for item in data["experiments"] if item["id"] == "restaurant_strategy")
    final = next(row for row in primary["rows"] if row["name"] == "all")
    assert final["accuracy"] == 0.8424
    assert final["macro_f1"] == 0.7547
