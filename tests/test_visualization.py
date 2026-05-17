import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_visualization_entrypoint_generates_public_figures(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "visualize_results.py"),
            "--results",
            str(ROOT / "results" / "reported_metrics.json"),
            "--output-dir",
            str(tmp_path),
            "--dpi",
            "100",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    generated = sorted(path.name for path in tmp_path.glob("*.png"))
    assert generated == ["strategy-comparison.png"]
    with Image.open(tmp_path / "strategy-comparison.png") as image:
        assert image.width >= 600
        assert image.height >= 350
