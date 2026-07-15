import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "path",
    [
        "pragmatic_residual/__pycache__/features.cpython-39.pyc",
        "scripts/visualize_results.pyo",
        "tests/test_apply.pyd",
        ".DS_Store",
        "assets/.DS_Store",
        ".env",
        ".env.local",
        "private-key.pem",
        "credentials-local.json",
        "secrets-development.json",
    ],
)
def test_generated_local_files_are_ignored(path: str) -> None:
    completed = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", path],
        check=False,
        cwd=ROOT,
    )

    assert completed.returncode == 0, f"expected generated local file to be ignored: {path}"
