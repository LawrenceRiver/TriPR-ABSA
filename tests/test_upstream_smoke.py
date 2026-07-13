from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_upstream_cli_help_loads() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "train.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--model_name" in completed.stdout
