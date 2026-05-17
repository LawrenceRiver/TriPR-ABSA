#!/usr/bin/env python3
"""Build a train-only phrase prior, offline by default.

The optional DeepSeek defaults were verified against the official API docs at
https://api-docs.deepseek.com/ on 2026-07-13: model ``deepseek-v4-flash`` and
endpoint ``https://api.deepseek.com/chat/completions``. DeepSeek receives only
train-derived phrase candidates and aggregate counts, never dataset labels or
complete rows. Credentials are read from an environment variable at runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_BASE = "https://api.deepseek.com/chat/completions"
DEFAULT_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEFAULT_BATCH_SIZE = 30

SEED_PRIORS = {
    "decent": (1, 0.38, "general_opinion", "weaksubj"),
    "good": (1, 0.62, "general_opinion", "weaksubj"),
    "great": (1, 0.90, "general_opinion", "strongsubj"),
    "excellent": (1, 0.95, "general_opinion", "strongsubj"),
    "amazing": (1, 0.96, "general_opinion", "strongsubj"),
    "delicious": (1, 0.90, "taste", "strongsubj"),
    "tasty": (1, 0.74, "taste", "weaksubj"),
    "fresh": (1, 0.72, "freshness", "weaksubj"),
    "clean": (1, 0.64, "ambience", "weaksubj"),
    "fairly priced": (1, 0.62, "value", "weaksubj"),
    "reasonable": (1, 0.55, "value", "weaksubj"),
    "bad": (-1, 0.72, "general_opinion", "strongsubj"),
    "awful": (-1, 0.90, "general_opinion", "strongsubj"),
    "horrible": (-1, 0.96, "general_opinion", "strongsubj"),
    "mediocre": (-1, 0.52, "general_opinion", "weaksubj"),
    "disappointing": (-1, 0.76, "general_opinion", "strongsubj"),
    "disappointed": (-1, 0.76, "general_opinion", "strongsubj"),
    "overpriced": (-1, 0.78, "value", "strongsubj"),
    "expensive": (-1, 0.58, "value", "weaksubj"),
    "watered down": (-1, 0.82, "drink_quality", "weaksubj"),
    "stale": (-1, 0.78, "freshness", "strongsubj"),
    "dry": (-1, 0.64, "texture", "weaksubj"),
    "cold": (-1, 0.55, "temperature", "weaksubj"),
    "rude": (-1, 0.82, "service", "strongsubj"),
    "slow": (-1, 0.62, "service", "weaksubj"),
}

STOP_CANDIDATE_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "with",
    "in",
    "on",
    "at",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "this",
    "that",
    "these",
    "those",
    "not",
    "no",
    "never",
    "but",
    "however",
    "though",
    "although",
    "yet",
    "while",
}
DESCRIPTOR_POS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS", "VBN", "VBG", "NN", "NNS"}
CUE_WORDS = set(SEED_PRIORS) | {"better", "worse", "best", "worst", "ok", "okay"}


def norm(token: Any) -> str:
    return re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", str(token).lower())


def canonical_phrase(phrase: Any) -> str:
    return " ".join(part for part in (norm(item) for item in str(phrase).split()) if part)


def _source_error(path: Path, message: str) -> ValueError:
    return ValueError(f"{path}: {message}")


def load_train_samples(path: str | Path) -> list[dict[str, Any]]:
    """Parse upstream JSON directly and expand one label-free sample per aspect."""
    source = Path(path)
    try:
        records = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise _source_error(source, "train file must contain valid JSON") from exc
    if not isinstance(records, list):
        raise _source_error(source, "train document must be a list")

    samples: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise _source_error(source, f"record[{record_index}] must be an object")
        tokens = record.get("token")
        pos = record.get("pos")
        aspects = record.get("aspects")
        if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
            raise _source_error(source, f"record[{record_index}].token must be a string list")
        if not isinstance(pos, list) or len(pos) != len(tokens):
            raise _source_error(source, f"record[{record_index}].pos must align with token")
        if not isinstance(aspects, list):
            raise _source_error(source, f"record[{record_index}].aspects must be a list")

        for aspect_index, aspect in enumerate(aspects):
            prefix = f"record[{record_index}].aspects[{aspect_index}]"
            if not isinstance(aspect, dict):
                raise _source_error(source, f"{prefix} must be an object")
            start = aspect.get("from")
            end = aspect.get("to")
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or not 0 <= start < end <= len(tokens)
            ):
                raise _source_error(source, f"{prefix} must have a valid from/to span")
            term = aspect.get("term")
            if isinstance(term, list) and all(isinstance(item, str) for item in term):
                aspect_text = " ".join(term)
            elif isinstance(term, str):
                aspect_text = term
            else:
                raise _source_error(source, f"{prefix}.term must be a string or string list")
            samples.append(
                {
                    "text_list": list(tokens),
                    "pos": [str(tag) for tag in pos],
                    "aspect": aspect_text,
                    "aspect_post": [start, end],
                }
            )
    return samples


def phrase_in_tokens(tokens: list[str], phrase: str) -> bool:
    parts = canonical_phrase(phrase).split()
    return bool(parts) and any(
        tokens[index : index + len(parts)] == parts
        for index in range(0, len(tokens) - len(parts) + 1)
    )


def valid_candidate_phrase(phrase: str) -> bool:
    parts = canonical_phrase(phrase).split()
    if not parts or all(part in STOP_CANDIDATE_WORDS for part in parts):
        return False
    if parts[0] in STOP_CANDIDATE_WORDS or parts[-1] in STOP_CANDIDATE_WORDS:
        return False
    return all(len(part) > 1 for part in parts)


def extract_candidates(
    samples: Sequence[dict[str, Any]],
    candidate_mode: str = "auto",
) -> list[dict[str, Any]]:
    """Extract compact phrase/count candidates from train aspect windows."""
    counts: Counter[str] = Counter()
    for sample in samples:
        tokens = [norm(token) for token in sample["text_list"]]
        pos = sample["pos"]
        start, end = sample["aspect_post"]
        left = max(0, start - 6)
        right = min(len(tokens), end + 8)
        window = tokens[left:right]
        window_pos = pos[left:right]
        has_cue = any(phrase_in_tokens(tokens, phrase) for phrase in CUE_WORDS)
        if candidate_mode == "cue" and not has_cue:
            continue

        for token, tag in zip(window, window_pos):
            if valid_candidate_phrase(token) and any(
                str(tag).startswith(kind) for kind in DESCRIPTOR_POS
            ):
                counts[token] += 1
        for width in (2, 3):
            for index in range(0, len(window) - width + 1):
                phrase_tokens = window[index : index + width]
                if any(not token for token in phrase_tokens):
                    continue
                phrase = canonical_phrase(" ".join(phrase_tokens))
                if valid_candidate_phrase(phrase):
                    counts[phrase] += 1
    return [
        {"phrase": phrase, "count": count}
        for phrase, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def clean_prior_entries(entries: Sequence[Any]) -> list[dict[str, Any]]:
    """Return only entries accepted by the strict runtime prior loader."""
    cleaned: dict[str, dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict) or not item.get("is_quality_or_intensity_prior", True):
            continue
        phrase = canonical_phrase(item.get("phrase", ""))
        if not phrase:
            continue
        try:
            polarity = int(item.get("polarity", 0))
            intensity = float(item.get("intensity", 0.0))
        except (TypeError, ValueError):
            continue
        if polarity not in (-1, 1) or not math.isfinite(intensity) or not 0.0 < intensity <= 1.0:
            continue
        entry = {
            "phrase": phrase,
            "polarity": polarity,
            "intensity": intensity,
            "dimension": str(item.get("dimension", "other")),
            "subjectivity": str(item.get("subjectivity", "weaksubj")),
        }
        if "source" in item:
            entry["source"] = str(item["source"])
        if "count" in item and isinstance(item["count"], int):
            entry["count"] = item["count"]
        cleaned[phrase] = entry
    return [cleaned[phrase] for phrase in sorted(cleaned)]


def offline_prior(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resolve only built-in seed phrases that actually occur in train candidates."""
    candidate_counts = {item["phrase"]: item["count"] for item in candidates}
    entries = []
    for phrase, (polarity, intensity, dimension, subjectivity) in SEED_PRIORS.items():
        if phrase not in candidate_counts:
            continue
        entries.append(
            {
                "phrase": phrase,
                "polarity": polarity,
                "intensity": intensity,
                "dimension": dimension,
                "subjectivity": subjectivity,
                "source": "seed",
                "count": candidate_counts[phrase],
            }
        )
    return clean_prior_entries(entries)


def parse_json_content(content: str) -> Any:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        for opening, closing in (("{", "}"), ("[", "]")):
            start = content.find(opening)
            end = content.rfind(closing)
            if start >= 0 and end > start:
                return json.loads(content[start : end + 1])
        raise


def post_chat_completion(
    api_key: str,
    api_base: str,
    payload: dict[str, Any],
    timeout: int,
    max_retries: int,
) -> dict[str, Any]:
    """Post one request with at most ``max_retries`` retries after the first attempt."""
    request = urllib.request.Request(
        api_base,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    last_error: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            if not isinstance(result, dict):
                raise ValueError("DeepSeek response must be a JSON object")
            return result
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            last_error = RuntimeError(f"DeepSeek request failed: HTTP {exc.code} {detail}")
            if 400 <= exc.code < 500 and exc.code not in {408, 409, 429}:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, ssl.SSLError, ValueError) as exc:
            last_error = exc
        if attempt >= max_retries:
            break
        delay = min(2**attempt, 12) + 0.25
        print(
            f"[retry] DeepSeek request failed on attempt {attempt + 1}/"
            f"{max_retries + 1}: {last_error}; sleep {delay:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)
    raise RuntimeError(
        f"DeepSeek request failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def call_deepseek(
    api_key: str,
    api_base: str,
    model: str,
    batch: Sequence[dict[str, Any]],
    timeout: int,
    max_retries: int,
) -> list[Any]:
    """Annotate train-derived phrases without sending rows, text, or labels."""
    prompt = {
        "task": "Annotate train-derived phrases for an auxiliary quality/intensity prior.",
        "privacy": "No dataset labels or complete dataset rows are included.",
        "schema": {
            "phrase": "string",
            "is_quality_or_intensity_prior": "boolean",
            "polarity": "-1 negative, 0 contextual, 1 positive",
            "intensity": "number from 0.0 to 1.0",
            "subjectivity": "objective|weaksubj|strongsubj",
            "dimension": "string",
        },
        "rules": [
            "Return strict JSON with an annotations list and no markdown.",
            "Mark contextual phrases with polarity 0 or is_quality_or_intensity_prior false.",
            "The prior is auxiliary and must not decide a dataset label by itself.",
        ],
        "items": [{"phrase": item["phrase"], "count": item.get("count", 0)} for item in batch],
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Create compact JSON phrase lexicons for NLP research. Return JSON only."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    data = post_chat_completion(api_key, api_base, payload, timeout, max_retries)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("DeepSeek response is missing choices[0].message.content") from exc
    parsed = parse_json_content(content)
    annotations = parsed.get("annotations") if isinstance(parsed, dict) else parsed
    if not isinstance(annotations, list):
        raise ValueError("DeepSeek response must contain an annotations list")
    return annotations


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a train-only phrase prior (provider defaults to offline).",
        epilog="DeepSeek API defaults verified at https://api-docs.deepseek.com/ on 2026-07-13.",
    )
    parser.add_argument("--provider", choices=("offline", "deepseek"), default="offline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-mode", choices=("auto", "cue"), default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def _metadata(args: argparse.Namespace, partial: bool) -> dict[str, Any]:
    metadata = {
        "split": "train",
        "provider": args.provider,
        "model": args.model if args.provider == "deepseek" else None,
        "candidate_mode": args.candidate_mode,
    }
    metadata.update(
        {
            "uses_test_text_or_label": False,
            "partial": partial,
            "api_base": args.api_base if args.provider == "deepseek" else None,
            "api_key_env": args.api_key_env if args.provider == "deepseek" else None,
        }
    )
    return metadata


def _write_payload(
    output: Path,
    args: argparse.Namespace,
    candidates: Sequence[dict[str, Any]],
    prior: Sequence[dict[str, Any]],
    partial: bool,
    completed_batches: int,
) -> None:
    metadata = _metadata(args, partial)
    metadata["completed_batches"] = completed_batches
    payload = {
        "metadata": metadata,
        "candidates": list(candidates),
        "prior": clean_prior_entries(prior),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _resume_prior(output: Path, enabled: bool) -> list[dict[str, Any]]:
    if not enabled or not output.exists():
        return []
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{output}: resume payload must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{output}: resume payload must be an object")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict) or metadata.get("split") != "train":
        raise ValueError(f"{output}: resume payload split must be train")
    prior = payload.get("prior")
    if not isinstance(prior, list):
        raise ValueError(f"{output}: resume payload prior must be a list")
    return clean_prior_entries(prior)


def build_prior(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    api_key: str | None = None
    if args.provider == "deepseek":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError(f"{args.api_key_env} is required when provider=deepseek")

    samples = load_train_samples(args.train_file)
    candidates = extract_candidates(samples, candidate_mode=args.candidate_mode)
    base_prior = offline_prior(candidates)
    if args.provider == "offline":
        _write_payload(args.output, args, candidates, base_prior, False, 0)
        return base_prior

    prior_by_phrase = {entry["phrase"]: entry for entry in base_prior}
    previous = _resume_prior(args.output, args.resume)
    prior_by_phrase.update({entry["phrase"]: entry for entry in previous})
    seen = {entry["phrase"] for entry in previous}
    remaining = [candidate for candidate in candidates if candidate["phrase"] not in seen]
    allowed_phrases = {candidate["phrase"] for candidate in candidates}
    completed_batches = 0
    for start in range(0, len(remaining), args.batch_size):
        batch = remaining[start : start + args.batch_size]
        annotations = call_deepseek(
            api_key,
            args.api_base,
            args.model,
            batch,
            args.timeout,
            args.max_retries,
        )
        for entry in clean_prior_entries(annotations):
            if entry["phrase"] in allowed_phrases:
                entry["source"] = "deepseek"
                prior_by_phrase[entry["phrase"]] = entry
        completed_batches += 1
        _write_payload(
            args.output,
            args,
            candidates,
            list(prior_by_phrase.values()),
            True,
            completed_batches,
        )

    final_prior = clean_prior_entries(list(prior_by_phrase.values()))
    _write_payload(args.output, args, candidates, final_prior, False, completed_batches)
    return final_prior


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    prior = build_prior(args)
    print(f"wrote {args.output}")
    print(f"prior_entries={len(prior)} provider={args.provider}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
