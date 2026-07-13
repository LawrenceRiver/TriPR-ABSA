import importlib.util
import io
import json
import os
import ssl
import subprocess
import sys
import urllib.error
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEEPSEEK_CANDIDATES = [
    {"phrase": "excellent", "count": 1},
    {"phrase": "awful", "count": 1},
]


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


def test_fixture_contains_no_secret_key_or_value_recursively() -> None:
    data = json.loads((ROOT / "tests" / "fixtures" / "prior.json").read_text())

    def assert_secret_free(value) -> None:
        if isinstance(value, dict):
            for name, child in value.items():
                assert "key" not in name.lower()
                assert_secret_free(child)
        elif isinstance(value, list):
            for child in value:
                assert_secret_free(child)
        elif isinstance(value, str):
            lowered = value.lower()
            assert not any(
                marker in lowered
                for marker in ("api_key", "api-key", "bearer ", "password", "secret", "token")
            )
            assert not lowered.startswith("sk-")

    assert_secret_free(data)


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


def test_http_errors_do_not_echo_response_or_request_secrets(monkeypatch, capsys) -> None:
    builder = load_builder()
    credential = "ENV_CREDENTIAL_MUST_NOT_LEAK"
    response_body_secret = "BODY_SECRET_MUST_NOT_LEAK"
    response_header_secret = "HEADER_SECRET_MUST_NOT_LEAK"
    request_secret = "REQUEST_SECRET_MUST_NOT_LEAK"
    attempts = 0

    monkeypatch.setenv("DEEPSEEK_API_KEY", credential)

    def rejected_urlopen(request, timeout):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(
            f"https://example.invalid/chat?token={request_secret}",
            500,
            credential,
            {"Authorization": response_header_secret},
            io.BytesIO(response_body_secret.encode("utf-8")),
        )

    monkeypatch.setattr(builder.urllib.request, "urlopen", rejected_urlopen)
    monkeypatch.setattr(builder.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="HTTP 500") as raised:
        builder.post_chat_completion(
            credential,
            "https://example.invalid/chat",
            {"model": "test"},
            timeout=1,
            max_retries=1,
        )
    rendered = f"{raised.value}\n{capsys.readouterr().err}"
    assert attempts == 2
    assert credential not in rendered
    assert response_body_secret not in rendered
    assert response_header_secret not in rendered
    assert request_secret not in rendered


@pytest.mark.parametrize(
    ("failure_kind", "category"), [("url", "network error"), ("tls", "TLS error")]
)
def test_remote_network_reasons_do_not_echo_environment_credential(
    monkeypatch, capsys, failure_kind: str, category: str
) -> None:
    builder = load_builder()
    credential = "ENV_CREDENTIAL_MUST_NOT_LEAK"
    monkeypatch.setenv("DEEPSEEK_API_KEY", credential)

    def rejected_urlopen(request, timeout):
        if failure_kind == "url":
            raise urllib.error.URLError(credential)
        raise ssl.SSLError(credential)

    monkeypatch.setattr(builder.urllib.request, "urlopen", rejected_urlopen)
    monkeypatch.setattr(builder.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match=category) as raised:
        builder.post_chat_completion(
            credential,
            "https://example.invalid/chat",
            {"model": "test"},
            timeout=1,
            max_retries=1,
        )
    captured = capsys.readouterr()
    rendered = f"{raised.value}\n{captured.out}\n{captured.err}"
    assert credential not in rendered


def test_deepseek_writes_partial_after_each_batch(tmp_path: Path, monkeypatch) -> None:
    builder = load_builder()
    output = tmp_path / "partial.json"
    candidates = DEEPSEEK_CANDIDATES
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
    assert {item["phrase"] for item in observed_partial[0]["prior"]} == {"excellent"}
    assert observed_partial[0]["completed_phrases"] == ["excellent"]
    assert "awful" not in observed_partial[0]["completed_phrases"]
    assert json.loads(output.read_text())["metadata"]["partial"] is False


def deepseek_args(builder: ModuleType, output: Path, *extra: str):
    return builder.create_parser().parse_args(
        [
            "--provider",
            "deepseek",
            "--train-file",
            str(ROOT / "tests" / "fixtures" / "train.json"),
            "--output",
            str(output),
            *extra,
        ]
    )


def checkpoint_payload(
    builder: ModuleType,
    args,
    candidates,
    *,
    prior=None,
    completed_phrases=None,
    completed_batches: int = 0,
):
    return {
        "metadata": {
            "split": "train",
            "provider": "deepseek",
            "model": args.model,
            "candidate_mode": args.candidate_mode,
            "uses_test_text_or_label": False,
            "partial": True,
            "api_base": args.api_base,
            "api_key_env": args.api_key_env,
            "completed_batches": completed_batches,
            "candidate_digest": builder.checkpoint_digest(candidates, args),
        },
        "candidates": candidates,
        "completed_phrases": completed_phrases or [],
        "prior": prior or [],
    }


def accepted_entry(phrase: str, polarity: int) -> dict:
    return {
        "phrase": phrase,
        "is_quality_or_intensity_prior": True,
        "polarity": polarity,
        "intensity": 0.9,
        "dimension": "quality",
        "source": "deepseek",
    }


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("partial", False, "partial must be true"),
        ("provider", "offline", "provider must be deepseek"),
        ("model", "stale-model", "model does not match current run"),
        ("api_base", "https://stale.invalid", "api-base does not match current run"),
        ("candidate_mode", "cue", "candidate-mode does not match current run"),
        ("candidate_digest", "0" * 64, "candidate digest does not match current run"),
    ],
)
def test_resume_rejects_stale_or_mismatched_checkpoint(
    tmp_path: Path, monkeypatch, field: str, invalid, message: str
) -> None:
    builder = load_builder()
    output = tmp_path / "stale-checkpoint.json"
    args = deepseek_args(builder, output, "--resume")
    payload = checkpoint_payload(builder, args, DEEPSEEK_CANDIDATES)
    payload["metadata"][field] = invalid
    output.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr(builder, "extract_candidates", lambda *args, **kwargs: DEEPSEEK_CANDIDATES)
    with pytest.raises(ValueError, match=rf"stale-checkpoint.json.*{message}"):
        builder.build_prior(args)


def test_resume_filters_restored_entries_to_current_candidates(tmp_path: Path, monkeypatch) -> None:
    builder = load_builder()
    output = tmp_path / "resume.json"
    args = deepseek_args(builder, output, "--resume")
    payload = checkpoint_payload(
        builder,
        args,
        DEEPSEEK_CANDIDATES,
        prior=[accepted_entry("excellent", 1), accepted_entry("ghost", -1)],
        completed_phrases=["excellent", "ghost"],
        completed_batches=2,
    )
    output.write_text(json.dumps(payload), encoding="utf-8")
    requested = []

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr(builder, "extract_candidates", lambda *args, **kwargs: DEEPSEEK_CANDIDATES)

    def fake_call(api_key, api_base, model, batch, timeout, max_retries):
        requested.extend(item["phrase"] for item in batch)
        return [accepted_entry("awful", -1)]

    monkeypatch.setattr(builder, "call_deepseek", fake_call)
    builder.build_prior(args)
    assert requested == ["awful"]
    final = json.loads(output.read_text())
    assert {item["phrase"] for item in final["prior"]} == {
        "awful",
        "excellent",
    }
    assert final["metadata"]["completed_batches"] == 3


def test_resume_skips_rejected_but_completed_phrases(tmp_path: Path, monkeypatch) -> None:
    builder = load_builder()
    output = tmp_path / "rejected-completed.json"
    args = deepseek_args(builder, output, "--resume")
    payload = checkpoint_payload(
        builder,
        args,
        DEEPSEEK_CANDIDATES,
        prior=[],
        completed_phrases=["excellent"],
        completed_batches=4,
    )
    output.write_text(json.dumps(payload), encoding="utf-8")
    requested = []

    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setattr(builder, "extract_candidates", lambda *args, **kwargs: DEEPSEEK_CANDIDATES)

    def fake_call(api_key, api_base, model, batch, timeout, max_retries):
        requested.extend(item["phrase"] for item in batch)
        return [accepted_entry("awful", -1)]

    monkeypatch.setattr(builder, "call_deepseek", fake_call)
    builder.build_prior(args)
    assert requested == ["awful"]
    final = json.loads(output.read_text())
    assert final["completed_phrases"] == ["excellent", "awful"]
    assert final["metadata"]["completed_batches"] == 5


def test_checkpoint_digest_is_order_independent_and_config_bound(tmp_path: Path) -> None:
    builder = load_builder()
    args = deepseek_args(builder, tmp_path / "prior.json")
    digest = builder.checkpoint_digest(DEEPSEEK_CANDIDATES, args)
    assert digest == builder.checkpoint_digest(list(reversed(DEEPSEEK_CANDIDATES)), args)
    args.model = "different-model"
    assert digest != builder.checkpoint_digest(DEEPSEEK_CANDIDATES, args)


def test_atomic_replacement_failure_preserves_previous_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    builder = load_builder()
    output = tmp_path / "checkpoint.json"
    previous = {"metadata": {"partial": True}, "completed_phrases": ["excellent"]}
    output.write_text(json.dumps(previous), encoding="utf-8")

    def fail_replace(source, destination):
        assert Path(source).parent == output.parent
        assert Path(destination) == output
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(builder.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated replacement failure"):
        builder.atomic_write_json(output, {"metadata": {"partial": False}})
    assert json.loads(output.read_text(encoding="utf-8")) == previous
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_atomic_write_fsyncs_parent_directory_after_replace(tmp_path: Path, monkeypatch) -> None:
    builder = load_builder()
    output = tmp_path / "checkpoint.json"
    events = []
    directory_fd = None
    real_open = builder.os.open
    real_fsync = builder.os.fsync
    real_close = builder.os.close
    real_replace = builder.os.replace

    def tracking_replace(source, destination):
        events.append("replace")
        return real_replace(source, destination)

    def tracking_open(path, flags, *args):
        nonlocal directory_fd
        descriptor = real_open(path, flags, *args)
        if Path(path) == output.parent:
            directory_fd = descriptor
            events.append("open-directory")
        return descriptor

    def tracking_fsync(descriptor):
        if descriptor == directory_fd:
            events.append("fsync-directory")
        return real_fsync(descriptor)

    def tracking_close(descriptor):
        if descriptor == directory_fd:
            events.append("close-directory")
        return real_close(descriptor)

    monkeypatch.setattr(builder.os, "replace", tracking_replace)
    monkeypatch.setattr(builder.os, "open", tracking_open)
    monkeypatch.setattr(builder.os, "fsync", tracking_fsync)
    monkeypatch.setattr(builder.os, "close", tracking_close)
    builder.atomic_write_json(output, {"metadata": {"partial": True}})
    assert events == ["replace", "open-directory", "fsync-directory", "close-directory"]
    assert json.loads(output.read_text(encoding="utf-8")) == {"metadata": {"partial": True}}


def test_final_prior_drops_contextual_and_invalid_annotations() -> None:
    builder = load_builder()
    cleaned = builder.clean_prior_entries(
        [
            accepted_entry("excellent", 1),
            {
                "phrase": "context",
                "is_quality_or_intensity_prior": True,
                "polarity": 0,
                "intensity": 0.5,
            },
            {
                "phrase": "ignore",
                "polarity": -1,
                "intensity": 0.5,
                "is_quality_or_intensity_prior": False,
            },
            {
                "phrase": "too-much",
                "is_quality_or_intensity_prior": True,
                "polarity": 1,
                "intensity": 1.5,
            },
        ]
    )
    assert [item["phrase"] for item in cleaned] == ["excellent"]


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("is_quality_or_intensity_prior", 1),
        ("is_quality_or_intensity_prior", "true"),
        ("polarity", True),
        ("polarity", 1.0),
        ("polarity", "1"),
        ("intensity", True),
        ("intensity", "0.9"),
        ("intensity", 10**1000),
        ("intensity", float("nan")),
        ("intensity", float("inf")),
        ("phrase", 123),
        ("phrase", ""),
        ("phrase", "   "),
    ],
)
def test_provider_annotations_require_exact_types(field: str, invalid) -> None:
    builder = load_builder()
    entry = accepted_entry("excellent", 1)
    entry[field] = invalid
    assert builder.clean_prior_entries([entry]) == []


def test_provider_annotation_requires_explicit_true_prior_flag() -> None:
    builder = load_builder()
    entry = accepted_entry("excellent", 1)
    del entry["is_quality_or_intensity_prior"]
    assert builder.clean_prior_entries([entry]) == []
