"""Declared generation settings for the warm SHuBERT CPU serving boundary.

Settings come only from hash-verified bundle metadata and the sealed runtime.
The wrapper preserves an older bundle's runtime and inference bytes in place.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from bundle_verification import read_bundle_metadata

DEFAULT_GENERATION = {"num_beams": 5, "max_length": 2048}
OPTIONAL_GENERATION = frozenset({"length_penalty", "no_repeat_ngram_size"})


def validate_generation(value: object) -> dict:
    if (
        not isinstance(value, dict)
        or not set(DEFAULT_GENERATION).issubset(value)
        or set(value) - (set(DEFAULT_GENERATION) | OPTIONAL_GENERATION)
    ):
        raise ValueError("unsupported CPU generation settings")
    for key in DEFAULT_GENERATION:
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if "length_penalty" in value and (
        type(value["length_penalty"]) not in (int, float)
        or not math.isfinite(value["length_penalty"])
    ):
        raise ValueError("length_penalty must be a finite number")
    if "no_repeat_ngram_size" in value and (
        type(value["no_repeat_ngram_size"]) is not int or value["no_repeat_ngram_size"] < 0
    ):
        raise ValueError("no_repeat_ngram_size must be a nonnegative integer")
    return dict(value)


def _object(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate bundle configuration key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("bundle configuration must be an object")
    return value


def declared_generation(bundle: Path, *, bundle_sha256: str) -> dict:
    metadata = read_bundle_metadata(
        bundle,
        expected_bundle_sha256=bundle_sha256,
        names=("MODEL_IDENTITY.json", "fs_router.json"),
    )
    declarations = []
    identity = _object(metadata["MODEL_IDENTITY.json"]) if "MODEL_IDENTITY.json" in metadata else {}
    for key in ("generation", "baseline_generation_overrides"):
        if key in identity:
            declarations.append(identity[key])
    if "fs_router.json" in metadata:
        router = _object(metadata["fs_router.json"])
        if "baseline_generate_overrides" in router:
            declarations.append(router["baseline_generate_overrides"])
    declared = {}
    for declaration in declarations:
        if not isinstance(declaration, dict):
            raise ValueError("generation declaration must be an object")
        validate_generation(DEFAULT_GENERATION | declaration)
        for key, value in declaration.items():
            if key in declared and declared[key] != value:
                raise ValueError("bundle generation declarations disagree")
            declared[key] = value
    return validate_generation(DEFAULT_GENERATION | declared)


class ConfiguredGenerator:
    """Pass sealed kwargs to the existing translator's original generate call.

    The warm SHuBERT runtime uses its translator only for ``generate`` after
    loading. Keep the original translator alive and do not mutate its methods
    or Transformers global configuration. Request features and token IDs pass
    through unchanged; the bundle's declared decoding parameters take priority.
    """

    def __init__(self, translator, settings: dict):
        self._generate = translator.generate
        if not callable(self._generate):
            raise ValueError("unsupported SHuBERT translator generate interface")
        self._settings = validate_generation(settings)

    def generate(self, **kwargs):
        return self._generate(**(kwargs | self._settings))
