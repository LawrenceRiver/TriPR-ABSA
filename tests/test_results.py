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
            "internal project evaluation: strategy ablation table and selected-checkpoint summary"
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

EXPECTED_MULTI_BACKBONE = {
    "id": "multi_backbone",
    "dataset": "SemEval-2014 Restaurant",
    "split": "test",
    "aggregation": "reported model evaluations; most rows use one checkpoint",
    "source": "internal project evaluation: multi-backbone compatibility table",
    "limitations": "Adapter-compatibility evidence, not a multi-seed model leaderboard.",
    "metric_definition": METRIC_DEFINITION,
    "release_status": "reported, not rerun during release preparation",
    "rows": [
        {
            "name": "text-gt-bert",
            "accuracy": 0.8647,
            "macro_f1": 0.8088,
            "best_strategy": "all",
            "residual_accuracy": 0.8702,
            "residual_macro_f1": 0.8171,
        },
        {
            "name": "text-gt",
            "accuracy": 0.8186,
            "macro_f1": 0.7419,
            "best_strategy": "fact+comparison",
            "residual_accuracy": 0.8257,
            "residual_macro_f1": 0.7617,
        },
        {
            "name": "text-tg",
            "accuracy": 0.8195,
            "macro_f1": 0.7260,
            "best_strategy": "fact+comparison",
            "residual_accuracy": 0.8329,
            "residual_macro_f1": 0.7610,
        },
        {
            "name": "text-gin",
            "accuracy": 0.8034,
            "macro_f1": 0.7339,
            "best_strategy": "fact+comparison",
            "residual_accuracy": 0.8097,
            "residual_macro_f1": 0.7530,
        },
        {
            "name": "text-transformer",
            "accuracy": 0.8097,
            "macro_f1": 0.7153,
            "best_strategy": "all",
            "residual_accuracy": 0.8195,
            "residual_macro_f1": 0.7474,
        },
        {
            "name": "gnn-transformer",
            "accuracy": 0.8088,
            "macro_f1": 0.7046,
            "best_strategy": "all",
            "residual_accuracy": 0.8275,
            "residual_macro_f1": 0.7433,
        },
        {
            "name": "transformer-gnn",
            "accuracy": 0.8070,
            "macro_f1": 0.7284,
            "best_strategy": "all",
            "residual_accuracy": 0.8123,
            "residual_macro_f1": 0.7494,
        },
        {
            "name": "parallel-gt",
            "accuracy": 0.8275,
            "macro_f1": 0.7514,
            "best_strategy": "fact+comparison",
            "residual_accuracy": 0.8365,
            "residual_macro_f1": 0.7737,
        },
    ],
}


def test_reported_metrics_have_provenance() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert data["schema_version"] == 2
    assert data["class_order"] == ["positive", "negative", "neutral"]
    assert [experiment["id"] for experiment in data["experiments"]] == [
        expected["id"] for expected in EXPECTED_EXPERIMENTS
    ] + [EXPECTED_MULTI_BACKBONE["id"]]

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

    assert data["experiments"][-1] == EXPECTED_MULTI_BACKBONE


def test_primary_result_matches_course_paper() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    primary = next(item for item in data["experiments"] if item["id"] == "restaurant_strategy")
    final = next(row for row in primary["rows"] if row["name"] == "all")
    assert final["accuracy"] == 0.8424
    assert final["macro_f1"] == 0.7547


def test_reported_paper_tables_preserve_ablation_and_error_analysis() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    tables = data["paper_tables"]

    assert set(tables) == {
        "restaurant_main_comparison",
        "restaurant_module_ablation",
        "restaurant_baseline_errors",
    }

    main = tables["restaurant_main_comparison"]
    assert [
        (
            row["name"],
            row["accuracy_mean"],
            row["accuracy_std"],
            row["macro_f1_mean"],
            row["macro_f1_std"],
            row["delta_macro_f1"],
        )
        for row in main["rows"]
    ] == [
        ("Text-Transformer", 0.8022, 0.0021, 0.6914, 0.0101, -0.0566),
        ("Text-GT", 0.8067, 0.0078, 0.7023, 0.0165, -0.0457),
        ("BERT-SPC", 0.8389, 0.0101, 0.7588, 0.0156, 0.0108),
        ("AEN-BERT", 0.8034, 0.0118, 0.6826, 0.0186, -0.0654),
        ("LCF-BERT", 0.8332, 0.0078, 0.7482, 0.0142, 0.0001),
        ("TextGT-BERT", 0.8383, 0.0176, 0.7480, 0.0395, None),
        ("TextGT-BERT + fact", 0.8406, 0.0209, 0.7522, 0.0445, 0.0042),
        ("TextGT-BERT + comparison", 0.8394, 0.0172, 0.7499, 0.0384, 0.0019),
        ("TextGT-BERT + intensity", 0.8388, 0.0194, 0.7488, 0.0424, 0.0008),
        ("TextGT-BERT + fact+comparison", 0.8418, 0.0205, 0.7541, 0.0434, 0.0060),
        ("TextGT-BERT + all", 0.8424, 0.0223, 0.7547, 0.0465, 0.0067),
    ]

    ablation = tables["restaurant_module_ablation"]
    assert [
        (
            row["name"],
            row["delta_macro_f1"],
            row["changed"],
            row["fixed"],
            row["broken"],
        )
        for row in ablation["rows"]
    ] == [
        ("fact", 0.0042, 4.33, 3.33, 0.67),
        ("comparison", 0.0019, 1.33, 1.33, 0.00),
        ("intensity", 0.0008, 2.67, 1.33, 0.67),
        ("fact+comparison", 0.0060, 5.67, 4.67, 0.67),
        ("all", 0.0067, 8.00, 6.00, 1.33),
    ]

    errors = tables["restaurant_baseline_errors"]
    assert errors["total_errors"] == 151
    assert sum(row["count"] for row in errors["error_pairs"]) == 151
    assert errors["error_pairs"][0] == {
        "gold": "neutral",
        "predicted": "positive",
        "count": 51,
        "percentage": 33.8,
    }
    assert [(row["name"], row["count"]) for row in errors["error_buckets"]] == [
        ("far_context", 114),
        ("multi_aspect", 112),
        ("long_sentence", 41),
        ("negation", 40),
        ("contrast", 24),
        ("intensifier", 16),
        ("comparative", 13),
        ("plain", 13),
        ("hedge", 9),
        ("weak_praise", 5),
    ]
    assert sum(row["count"] for row in errors["error_buckets"]) > errors["total_errors"]
