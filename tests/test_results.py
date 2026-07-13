import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "reported_metrics.json"


def test_reported_metrics_have_provenance() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    for experiment in data["experiments"]:
        assert experiment["dataset"]
        assert experiment["source"]
        assert experiment["limitations"]
        assert experiment["metric_definition"] == {
            "accuracy": "correct predictions divided by all test examples",
            "macro_f1": "unweighted mean of per-class F1",
        }
        assert experiment["release_status"] == ("reported, not rerun during release preparation")
        assert experiment["rows"]
        for row in experiment["rows"]:
            assert 0 <= row["accuracy"] <= 1
            assert 0 <= row["macro_f1"] <= 1


def test_primary_result_matches_course_paper() -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    primary = next(item for item in data["experiments"] if item["id"] == "restaurant_strategy")
    final = next(row for row in primary["rows"] if row["name"] == "all")
    assert final["accuracy"] == 0.8424
    assert final["macro_f1"] == 0.7547
