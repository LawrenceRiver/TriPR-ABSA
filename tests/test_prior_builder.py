import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]


def load_builder() -> ModuleType:
    path = ROOT / "scripts" / "build_phrase_prior.py"
    spec = importlib.util.spec_from_file_location("build_phrase_prior", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_help_has_offline_default() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_phrase_prior.py"), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--provider" in completed.stdout
    assert "offline" in completed.stdout


def test_fixture_contains_no_secret_field() -> None:
    data = json.loads((ROOT / "tests" / "fixtures" / "prior.json").read_text())
    assert not any("key" in name.lower() for name in data)


def test_offline_builder_uses_train_split_only(tmp_path: Path) -> None:
    output = tmp_path / "prior.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_phrase_prior.py"),
            "--train-file",
            str(ROOT / "tests" / "fixtures" / "train.json"),
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["metadata"]["split"] == "train"
    assert data["metadata"]["provider"] == "offline"
    assert data["metadata"]["uses_test_text_or_label"] is False
    assert any(item["phrase"] == "excellent" for item in data["prior"])
    assert "DEEPSEEK_API_KEY" not in output.read_text(encoding="utf-8")


def test_deepseek_provider_requires_environment_key(tmp_path: Path) -> None:
    environment = {key: value for key, value in os.environ.items() if key != "DEEPSEEK_API_KEY"}
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_phrase_prior.py"),
            "--provider",
            "deepseek",
            "--train-file",
            str(ROOT / "tests" / "fixtures" / "train.json"),
            "--output",
            str(tmp_path / "prior.json"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "DEEPSEEK_API_KEY is required when provider=deepseek" in completed.stderr


def test_upstream_parser_expands_each_aspect_without_labels() -> None:
    builder = load_builder()
    samples = builder.load_train_samples(ROOT / "tests" / "fixtures" / "train.json")
    assert samples == [
        {
            "text_list": ["The", "food", "was", "excellent", "."],
            "pos": ["DT", "NN", "VBD", "JJ", "."],
            "aspect": "food",
            "aspect_post": [1, 2],
        }
    ]
    assert "polarity" not in samples[0]


def test_offline_builder_never_calls_http(tmp_path: Path, monkeypatch) -> None:
    builder = load_builder()

    def unexpected_http(*args, **kwargs):
        raise AssertionError("offline provider attempted an HTTP call")

    monkeypatch.setattr(builder, "post_chat_completion", unexpected_http)
    output = tmp_path / "offline.json"
    assert (
        builder.main(
            [
                "--train-file",
                str(ROOT / "tests" / "fixtures" / "train.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    phrases = {item["phrase"] for item in json.loads(output.read_text())["prior"]}
    assert phrases == {"excellent"}


def test_extracted_phrases_do_not_cross_punctuation() -> None:
    builder = load_builder()
    candidates = builder.extract_candidates(
        [
            {
                "text_list": ["fairly", ".", "priced", "service"],
                "pos": ["RB", ".", "VBN", "NN"],
                "aspect": "service",
                "aspect_post": [3, 4],
            }
        ]
    )
    assert "fairly priced" not in {item["phrase"] for item in candidates}


def test_http_retry_count_is_bounded(monkeypatch) -> None:
    builder = load_builder()
    attempts = 0

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def read() -> bytes:
            return b'{"choices": []}'

    def flaky_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise urllib.error.URLError("temporary")
        return Response()

    monkeypatch.setattr(builder.urllib.request, "urlopen", flaky_urlopen)
    monkeypatch.setattr(builder.time, "sleep", lambda seconds: None)
    result = builder.post_chat_completion(
        "secret",
        "https://example.invalid/chat/completions",
        {"model": "test"},
        timeout=1,
        max_retries=2,
    )
    assert result == {"choices": []}
    assert attempts == 3


def test_deepseek_writes_partial_after_each_batch(tmp_path: Path, monkeypatch) -> None:
    builder = load_builder()
    output = tmp_path / "partial.json"
    candidates = [
        {"phrase": "excellent", "count": 1},
        {"phrase": "awful", "count": 1},
    ]
    observed_partial = []

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr(builder, "extract_candidates", lambda *args, **kwargs: candidates)

    def fake_call(api_key, api_base, model, batch, timeout, max_retries):
        assert api_key == "secret"
        if output.exists():
            observed_partial.append(json.loads(output.read_text(encoding="utf-8")))
        phrase = batch[0]["phrase"]
        return [
            {
                "phrase": phrase,
                "is_quality_or_intensity_prior": True,
                "polarity": 1 if phrase == "excellent" else -1,
                "intensity": 0.9,
                "dimension": "quality",
            }
        ]

    monkeypatch.setattr(builder, "call_deepseek", fake_call)
    assert (
        builder.main(
            [
                "--provider",
                "deepseek",
                "--train-file",
                str(ROOT / "tests" / "fixtures" / "train.json"),
                "--output",
                str(output),
                "--batch-size",
                "1",
            ]
        )
        == 0
    )
    assert observed_partial[0]["metadata"]["partial"] is True
    assert "excellent" in {item["phrase"] for item in observed_partial[0]["prior"]}
    assert json.loads(output.read_text())["metadata"]["partial"] is False


def test_resume_skips_phrases_in_partial_payload(tmp_path: Path, monkeypatch) -> None:
    builder = load_builder()
    output = tmp_path / "resume.json"
    output.write_text(
        json.dumps(
            {
                "metadata": {"split": "train", "provider": "deepseek", "partial": True},
                "prior": [
                    {
                        "phrase": "excellent",
                        "polarity": 1,
                        "intensity": 0.9,
                        "dimension": "quality",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    candidates = [
        {"phrase": "excellent", "count": 1},
        {"phrase": "awful", "count": 1},
    ]
    requested = []

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr(builder, "extract_candidates", lambda *args, **kwargs: candidates)

    def fake_call(api_key, api_base, model, batch, timeout, max_retries):
        requested.extend(item["phrase"] for item in batch)
        return [
            {
                "phrase": "awful",
                "is_quality_or_intensity_prior": True,
                "polarity": -1,
                "intensity": 0.9,
                "dimension": "quality",
            }
        ]

    monkeypatch.setattr(builder, "call_deepseek", fake_call)
    assert (
        builder.main(
            [
                "--provider",
                "deepseek",
                "--train-file",
                str(ROOT / "tests" / "fixtures" / "train.json"),
                "--output",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    assert requested == ["awful"]
    assert {item["phrase"] for item in json.loads(output.read_text())["prior"]} == {
        "awful",
        "excellent",
    }


def test_final_prior_drops_contextual_and_invalid_annotations() -> None:
    builder = load_builder()
    cleaned = builder.clean_prior_entries(
        [
            {"phrase": "excellent", "polarity": 1, "intensity": 0.9},
            {"phrase": "context", "polarity": 0, "intensity": 0.5},
            {
                "phrase": "ignore",
                "polarity": -1,
                "intensity": 0.5,
                "is_quality_or_intensity_prior": False,
            },
            {"phrase": "too-much", "polarity": 1, "intensity": 1.5},
        ]
    )
    assert [item["phrase"] for item in cleaned] == ["excellent"]
