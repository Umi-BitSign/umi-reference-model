from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import rfc8785


class CanonicalJSONError(ValueError):
    """Raised when a protocol JSON object is not valid canonical JSON input."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return rfc8785.dumps(value)
    except (rfc8785.CanonicalizationError, TypeError, ValueError) as exc:
        raise CanonicalJSONError(f"value cannot be RFC 8785 encoded: {exc}") from exc


def canonical_json_sha256(value: Any, *, domain: bytes = b"") -> str:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(canonical_json_bytes(value))
    return digest.hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path, *, maximum_bytes: int = 16 * 1024 * 1024) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CanonicalJSONError(f"cannot read JSON file {path}: {exc}") from exc
    if not raw or len(raw) > maximum_bytes:
        raise CanonicalJSONError(f"JSON file size must be in [1, {maximum_bytes}] bytes: {path}")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CanonicalJSONError(f"non-finite JSON number: {token}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise CanonicalJSONError(f"JSON file is not UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CanonicalJSONError(f"invalid JSON in {path}: {exc}") from exc
    return value


def write_canonical_json(path: Path, value: Any) -> None:
    encoded = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
