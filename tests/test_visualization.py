import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reported_metrics.json"
PUBLIC_FIGURES = ["cross-domain.png", "multi-backbone.png", "strategy-comparison.png"]


def run_visualization(results: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "visualize_results.py"),
            "--results",
            str(results),
            "--output-dir",
            str(output_dir),
            "--dpi",
            "100",
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_visualization_entrypoint_generates_public_figures(tmp_path: Path) -> None:
    completed = run_visualization(RESULTS, tmp_path)
    assert completed.returncode == 0, completed.stderr
    generated = sorted(path.name for path in tmp_path.glob("*.png"))
    assert {"strategy-comparison.png", "cross-domain.png", "multi-backbone.png"}.issubset(
        set(generated)
    )
    assert generated == PUBLIC_FIGURES
    for name in generated:
        with Image.open(tmp_path / name) as image:
            assert image.width >= 600
            assert image.height >= 350


def test_visualization_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output_dir in (first, second):
        completed = run_visualization(RESULTS, output_dir)
        assert completed.returncode == 0, completed.stderr

    assert {name: (first / name).read_bytes() for name in PUBLIC_FIGURES} == {
        name: (second / name).read_bytes() for name in PUBLIC_FIGURES
    }


def test_visualizations_use_json_scores_and_strategy_labels(tmp_path: Path) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    baseline = tmp_path / "baseline"
    completed = run_visualization(RESULTS, baseline)
    assert completed.returncode == 0, completed.stderr

    cross_domain_data = json.loads(json.dumps(data))
    for experiment in cross_domain_data["experiments"][:3]:
        experiment["rows"][-1]["macro_f1"] += 0.001
    cross_domain_results = tmp_path / "cross-domain-results.json"
    cross_domain_results.write_text(json.dumps(cross_domain_data), encoding="utf-8")
    cross_domain = tmp_path / "cross-domain"
    completed = run_visualization(cross_domain_results, cross_domain)
    assert completed.returncode == 0, completed.stderr
    assert (cross_domain / "cross-domain.png").read_bytes() != (
        baseline / "cross-domain.png"
    ).read_bytes()
    assert (cross_domain / "multi-backbone.png").read_bytes() == (
        baseline / "multi-backbone.png"
    ).read_bytes()

    strategy_data = json.loads(json.dumps(data))
    strategy_data["experiments"][-1]["rows"][0]["best_strategy"] = "fact+comparison"
    strategy_results = tmp_path / "strategy-results.json"
    strategy_results.write_text(json.dumps(strategy_data), encoding="utf-8")
    strategy = tmp_path / "strategy"
    completed = run_visualization(strategy_results, strategy)
    assert completed.returncode == 0, completed.stderr
    assert (strategy / "multi-backbone.png").read_bytes() != (
        baseline / "multi-backbone.png"
    ).read_bytes()
    assert (strategy / "cross-domain.png").read_bytes() == (
        baseline / "cross-domain.png"
    ).read_bytes()
