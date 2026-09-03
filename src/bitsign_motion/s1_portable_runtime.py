from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
from torch import Tensor

from .canonical import canonical_json_bytes, canonical_json_sha256
from .holistic_container import (
    ALLOWED_BOOTSTRAP_AUDIO_CODECS,
    ALLOWED_BOOTSTRAP_VIDEO_PROFILES,
    EXPECTED_FFMPEG_VERSION,
    EXPECTED_MEDIAPIPE_VERSION,
    EXPECTED_MODEL_SHA256,
    MAXIMUM_BOOTSTRAP_CANDIDATE_FRAMES,
    MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE,
    MAXIMUM_COMPLETION_BYTES,
    MAXIMUM_DERIVED_LONG_SIDE,
    MAXIMUM_DERIVED_SHORT_SIDE,
    MAXIMUM_DOCKER_DIAGNOSTIC_BYTES,
    MAXIMUM_MODEL_BYTES,
    MAXIMUM_RAW_ARTIFACT_BYTES,
    MAXIMUM_RECEIPT_BYTES,
    MAXIMUM_SOURCE_LONG_SIDE,
    MAXIMUM_SOURCE_SHORT_SIDE,
    MAXIMUM_SOURCE_VIDEO_BYTES,
    TARGET_RATE,
)
from .mediapipe_mapping import (
    DEFAULT_MEDIAPIPE_MAPPING_POLICY,
    MEDIAPIPE_MAPPING_PROFILE,
    MEDIAPIPE_MAPPING_STATUS,
)
from .motion_artifact import (
    DEFAULT_MOTION_FEATURE_POLICY,
    GLOBAL_FIELDS,
    MOTION_FRAME_COUNT,
    MOTION_GLOBAL_WIDTH,
    MOTION_POINT_COUNT,
    MOTION_POINT_WIDTH,
    POINT_FIELDS,
    S1_TARGET_FEATURE_PROFILE,
)
from .portable_model import EOS_TOKEN_ID, MOTION_FEATURE_DIM, PortableS1, PortableS1Config
from .s1_decode_tokenizer import (
    S1_TOKENIZER_MAXIMUM_MODEL_BYTES,
    S1_TOKENIZER_MAXIMUM_RECORD_BYTES,
    S1DecodeTokenizer,
    S1DecodeTokenizerError,
)
from .s1_state_digest import S1_TENSOR_SET_DIGEST_PROFILE, s1_tensor_set_sha256

S1_PORTABLE_SCHEMA: Final = "umi-s1-portable/1"
S1_PORTABLE_MANIFEST_SCHEMA: Final = "umi-s1-portable-manifest/1"
S1_PORTABLE_CONFIG_SCHEMA: Final = "umi-s1-portable-config/1"
S1_PORTABLE_PREPROCESSING_SCHEMA: Final = "umi-s1-preprocessing-contract/1"
S1_PORTABLE_RIGHTS_SCHEMA: Final = "umi-s1-portable-rights/1"
S1_PORTABLE_MULTI_SOURCE_RIGHTS_SCHEMA: Final = "umi-s1-portable-multi-source-rights/1"
S1_PORTABLE_TENSOR_SCHEMA: Final = "umi-s1-portable-tensors/1"
S1_TEXT_POSTPROCESS_SCHEMA: Final = "umi-s1-text-postprocess/1"
S1_RAW_TEXT_POSTPROCESS_REVISION: Final = "raw-tokenizer-output/1"
S1_WORD_PREFIX_SELECTION_SCHEMA: Final = "umi-s1-word-prefix-selection/1"
S1_VALIDATION_WORD_PREFIX_REVISION: Final = "validation-whitespace-word-prefix/1"
S1_AUTHORITY_WORD_PREFIX_REVISION: Final = "training-authority-whitespace-word-prefix/1"
S1_EXPERIMENT_BOOTSTRAP_CLAIM_PROFILE: Final = (
    "fleurs-prefix-contrastive050-epoch8-validation-only/1"
)
S1_PUBLIC_FINETUNE_CLAIM_PROFILE: Final = "public-s1-finetune/1"
S1_PORTABLE_MODEL_CLASS: Final = "bitsign_motion.portable_model.PortableS1"
S1_PORTABLE_RUNTIME_REVISION: Final = "s1-pytorch-beam-runtime/1"
S1_PORTABLE_BEAM_WIDTH: Final = 2
S1_PORTABLE_MAXIMUM_DECODE_TOKENS: Final = 24
S1_PORTABLE_NO_REPEAT_NGRAM_SIZE: Final = 3

_EXPERIMENT_BOOTSTRAP_MODEL_STATE_SHA256: Final = (
    "137b2733af1803d853a87396c31e5332f4170a4b5c5beffdfb246c60bb681521"
)
_EXPERIMENT_BOOTSTRAP_RELEASE_IDENTITY_SHA256: Final = (
    "48292a46e555bf4a1cb63788d3d4b78a7e8fb4c4d99cc67d68bd37aa12ddc9b3"
)
_EXPERIMENT_BOOTSTRAP_TOKENIZER_MODEL_SHA256: Final = (
    "7d8abdec60dab3a2a1969c1a7f194bdaa26f4c23884b23a204365b1535cf6783"
)
_EXPERIMENT_BOOTSTRAP_TOKENIZER_RECORD_SHA256: Final = (
    "eaaa5055b60f756c4e93befbec725344bfd3395da7d84a5ae2da78d3d24f57c0"
)
_EXPERIMENT_BOOTSTRAP_RIGHTS_SHA256: Final = (
    "adc1dc03f23568498759d2f14c35b55969973d2c1a7d43164ef03d2a77850634"
)

_IDENTITY_DOMAIN = b"umi-s1-portable-v1\0"
_MANIFEST_DOMAIN = b"umi-s1-portable-manifest-v1\0"
_WORD_PREFIX_SELECTION_DOMAIN = b"umi-s1-word-prefix-selection-v1\0"
_PUBLIC_FINETUNE_RELEASE_IDENTITY_DOMAIN = b"umi-public-s1-finetune-release-identity-v1\0"
_PUBLIC_FINETUNE_RELEASE_IDENTITY_SCHEMA: Final = "umi-public-s1-finetune-release-identity/1"
_PUBLIC_FINETUNE_RELEASE_CLAIM_BOUNDARY: Final = (
    "FLEURS-validation-selected public bootstrap candidate only. This is not untouched "
    "confirmation, UMI activation evidence, interpreter equivalence, accessibility "
    "certification, or proof of useful real-world accuracy."
)
_EXPECTED_FILES: Final = (
    "bundle-manifest.json",
    "inference-identity.json",
    "model-config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer.model",
)
_MAXIMUM_FILE_BYTES: Final = {
    "bundle-manifest.json": 1024 * 1024,
    "inference-identity.json": 2 * 1024 * 1024,
    "model-config.json": 256 * 1024,
    "model.safetensors": 64 * 1024 * 1024,
    "tokenizer.json": S1_TOKENIZER_MAXIMUM_RECORD_BYTES,
    "tokenizer.model": S1_TOKENIZER_MAXIMUM_MODEL_BYTES,
}
_LOWER_SHA256: Final = frozenset("0123456789abcdef")
_RUNTIME_SOURCE_MODULES: Final = (
    "__init__.py",
    "amd64_holistic_container.py",
    "canonical.py",
    "constants.py",
    "holistic_container.py",
    "holistic_motion.py",
    "mediapipe_mapping.py",
    "motion_artifact.py",
    "portable_model.py",
    "s1_decode_tokenizer.py",
    "s1_portable_runtime.py",
    "s1_state_digest.py",
    "umi_reference_backend.py",
)
_RIGHTS_FIELDS: Final = {
    "schema",
    "source_id",
    "source_version",
    "source_license_id",
    "license_sha256",
    "terms_sha256",
    "source_use_policy_sha256",
    "attribution_notice_sha256",
    "source_entry_sha256",
    "rights_as_of",
    "public_weight_eligible",
    "public_data_eligible",
    "runtime_code_license",
    "intended_public_weight_license",
    "redistribution_blocked_pending_final_rights_review",
    "release_rights_decision_sha256",
    "upstream_rights_sha256",
    "claim_boundary",
}
_MULTI_SOURCE_RIGHTS_FIELDS: Final = {
    "schema",
    "sources",
    "rights_as_of",
    "runtime_code_license",
    "intended_public_weight_license",
    "redistribution_blocked_pending_final_rights_review",
    "release_rights_decision_sha256",
    "upstream_rights_sha256",
    "claim_boundary",
}
_MULTI_SOURCE_RIGHTS_ENTRY_FIELDS: Final = {
    "source_id",
    "source_version",
    "source_name",
    "source_entry_sha256",
    "license_id",
    "license_sha256",
    "terms_sha256",
    "source_use_policy_sha256",
    "attribution_notice",
    "attribution_notice_sha256",
    "public_weight_eligible",
    "public_data_eligible",
    "raw_data_release",
    "public_data_release",
}


class S1PortableError(RuntimeError):
    """Raised when a portable S1 inference artifact is unsafe or inconsistent."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and set(value) <= _LOWER_SHA256
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise S1PortableError(f"{label} must be lowercase hexadecimal SHA-256")
    return cast(str, value)


def _require_image_digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or not _is_sha256(value.removeprefix("sha256:"))
    ):
        raise S1PortableError(f"{label} must be an immutable sha256 image digest")
    return value


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise S1PortableError(f"{label} has an unexpected field set")
    return value


def _strict_json(payload: bytes, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    if not 1 <= len(payload) <= maximum_bytes:
        raise S1PortableError(f"{label} violates its byte ceiling")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise S1PortableError(f"{label} contains a duplicate object member")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                S1PortableError(f"{label} contains a non-finite number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise S1PortableError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise S1PortableError(f"{label} is not a canonical JSON object")
    return value


def _source_sha256(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise S1PortableError(f"required source file is unavailable: {path.name}") from exc
    if not payload or len(payload) > 4 * 1024 * 1024:
        raise S1PortableError(f"required source file violates its byte ceiling: {path.name}")
    return _sha256(payload)


def _runtime_contract() -> dict[str, Any]:
    root = Path(__file__).resolve(strict=True).parent
    modules = [
        {
            "module": (
                "bitsign_motion"
                if filename == "__init__.py"
                else f"bitsign_motion.{Path(filename).stem}"
            ),
            "source_sha256": _source_sha256(root / filename),
        }
        for filename in _RUNTIME_SOURCE_MODULES
    ]
    return {
        "revision": S1_PORTABLE_RUNTIME_REVISION,
        "python_api": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
        "torch_api_version": str(torch.__version__).split("+", 1)[0],
        "numpy_version": importlib.metadata.version("numpy"),
        "rfc8785_version": importlib.metadata.version("rfc8785"),
        "safetensors_version": importlib.metadata.version("safetensors"),
        "tensor_format": S1_PORTABLE_TENSOR_SCHEMA,
        "decoding": {
            "algorithm": "beam-search",
            "beam_width": S1_PORTABLE_BEAM_WIDTH,
            "default_maximum_decode_tokens": S1_PORTABLE_MAXIMUM_DECODE_TOKENS,
            "score": "cumulative-log-probability",
            "length_normalization": False,
            "suppressed_token_ids": [0, 1],
            "eos_token_id": 2,
            "no_repeat_ngram_size": S1_PORTABLE_NO_REPEAT_NGRAM_SIZE,
            "tie_break": "lexicographically-smallest-token-sequence",
        },
        "modules": modules,
    }


def build_s1_preprocessing_contract(
    *,
    extractor_image_sha256: str,
    amd64_extractor_image_sha256: str | None = None,
    extractor_model_sha256: str,
    mapping_profile: str,
    mapping_policy_sha256: str,
    motion_feature_profile: str,
    motion_feature_policy_sha256: str,
    _source_records: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind the bootstrap extractor and, when supplied, the reference-miner frontend."""

    image = _require_image_digest(
        extractor_image_sha256
        if extractor_image_sha256.startswith("sha256:")
        else f"sha256:{extractor_image_sha256}",
        "extractor image",
    )
    amd64_image = (
        None
        if amd64_extractor_image_sha256 is None
        else _require_image_digest(
            amd64_extractor_image_sha256
            if amd64_extractor_image_sha256.startswith("sha256:")
            else f"sha256:{amd64_extractor_image_sha256}",
            "AMD64 extractor image",
        )
    )
    if (
        extractor_model_sha256 != EXPECTED_MODEL_SHA256
        or mapping_profile != MEDIAPIPE_MAPPING_PROFILE
        or mapping_policy_sha256
        != canonical_json_sha256(DEFAULT_MEDIAPIPE_MAPPING_POLICY.as_dict())
        or motion_feature_profile != S1_TARGET_FEATURE_PROFILE
        or motion_feature_policy_sha256
        != canonical_json_sha256(DEFAULT_MOTION_FEATURE_POLICY.as_dict())
    ):
        raise S1PortableError("preprocessing inputs differ from the current S1 candidate")
    if _source_records is None:
        package_root = Path(__file__).resolve(strict=True).parents[2]
        sources = {
            "container_host_source_sha256": _source_sha256(
                package_root / "src" / "bitsign_motion" / "holistic_container.py"
            ),
            "container_worker_source_sha256": _source_sha256(
                package_root / "docker" / "mediapipe-holistic" / "worker.py"
            ),
            "container_requirements_sha256": _source_sha256(
                package_root / "docker" / "mediapipe-holistic" / "requirements.lock"
            ),
            "landmark_mapping_source_sha256": _source_sha256(
                package_root / "src" / "bitsign_motion" / "mediapipe_mapping.py"
            ),
            "motion_composition_source_sha256": _source_sha256(
                package_root / "src" / "bitsign_motion" / "motion_artifact.py"
            ),
        }
        if amd64_image is not None:
            sources.update(
                {
                    "amd64_container_host_source_sha256": _source_sha256(
                        package_root / "src" / "bitsign_motion" / "amd64_holistic_container.py"
                    ),
                    "amd64_container_worker_source_sha256": _source_sha256(
                        package_root / "docker" / "mediapipe-holistic" / "worker.amd64.py"
                    ),
                    "amd64_container_requirements_sha256": _source_sha256(
                        package_root / "docker" / "mediapipe-holistic" / "requirements.amd64.lock"
                    ),
                }
            )
    else:
        sources = dict(_source_records)
    images = {"linux/arm64": image}
    if amd64_image is not None:
        images["linux/amd64"] = amd64_image
    has_reference_miner_frontend = amd64_image is not None
    return {
        "schema": S1_PORTABLE_PREPROCESSING_SCHEMA,
        "status": (
            "component-test-reference-miner"
            if has_reference_miner_frontend
            else "ex-203-candidate-only"
        ),
        "canonical_or_release_quality": False,
        "supported_oci_images": images,
        "dependencies": {
            "ffmpeg": EXPECTED_FFMPEG_VERSION,
            "mediapipe": EXPECTED_MEDIAPIPE_VERSION,
            "numpy": "2.5.2",
            "holistic_task_model_sha256": extractor_model_sha256,
        },
        "sources": sources,
        "mapping": {
            "profile": mapping_profile,
            "status": MEDIAPIPE_MAPPING_STATUS,
            "policy_sha256": mapping_policy_sha256,
        },
        "motion": {
            "feature_profile": motion_feature_profile,
            "feature_policy_sha256": motion_feature_policy_sha256,
            "shape": [MOTION_FRAME_COUNT, MOTION_FEATURE_DIM],
            "dtype": "float32",
            "frame_mask_shape": [MOTION_FRAME_COUNT],
            "frame_mask_dtype": "int32",
            "sample_rate_hz": TARGET_RATE,
            "maximum_frames": MOTION_FRAME_COUNT,
            "point_count": MOTION_POINT_COUNT,
            "point_width": MOTION_POINT_WIDTH,
            "point_fields": list(POINT_FIELDS),
            "global_width": MOTION_GLOBAL_WIDTH,
            "global_fields": list(GLOBAL_FIELDS),
            "padding": "positive-zero-with-frame-mask-zero",
        },
        "video": {
            "clip_duration_us": [2_000_000, 15_000_000],
            "clip_interval": "half-open-[start,end)",
            "whole_video_duration": "ffprobe-format-duration-as-exact-decimal-rational",
            "timestamp_microsecond_rounding": "nearest-integer-ties-toward-negative-infinity",
            "clip_end_tolerance": "source-metadata-rounding-at-most-one-nominal-frame-v1",
            "frame_selection": "8hz-equal-clip-bin-center-nearest-ffprobe-pts-v2",
            "frame_tie_break": ("earlier-presentation-timestamp-then-earlier-interval-frame-index"),
            "candidate_probe_policy": (
                "absolute-stream-time-with-one-frame-start-and-bframe-plus-one-end-guard-v1"
            ),
            "display_orientation": "ffmpeg-default-autorotation-before-rgb24-extraction",
            "mirroring": ("caller-declares-input-mirrored-and-ffmpeg-hflip-runs-before-extraction"),
            "aspect_policy": "apply-source-sar-then-nearest-bounded-even-per-axis",
            "source_long_side_maximum": MAXIMUM_SOURCE_LONG_SIDE,
            "source_short_side_maximum": MAXIMUM_SOURCE_SHORT_SIDE,
            "derived_long_side_maximum": MAXIMUM_DERIVED_LONG_SIDE,
            "derived_short_side_maximum": MAXIMUM_DERIVED_SHORT_SIDE,
            "source_frame_rate_maximum": str(MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE),
            "candidate_frame_maximum": MAXIMUM_BOOTSTRAP_CANDIDATE_FRAMES,
            "allowed_video_profiles": [
                list(value) for value in sorted(ALLOWED_BOOTSTRAP_VIDEO_PROFILES)
            ],
            "allowed_audio_codecs": sorted(ALLOWED_BOOTSTRAP_AUDIO_CODECS),
            "audio_policy": "ignored-by-ffmpeg-an",
        },
        "ceilings": {
            "source_video_bytes": MAXIMUM_SOURCE_VIDEO_BYTES,
            "task_model_bytes": MAXIMUM_MODEL_BYTES,
            "raw_artifact_bytes": MAXIMUM_RAW_ARTIFACT_BYTES,
            "receipt_bytes": MAXIMUM_RECEIPT_BYTES,
            "completion_stdout_bytes": MAXIMUM_COMPLETION_BYTES,
            "diagnostic_stderr_bytes": MAXIMUM_DOCKER_DIAGNOSTIC_BYTES,
            "container_memory_bytes": 4 * 1024**3,
            "container_memory_swap_bytes": 4 * 1024**3,
            "container_processes": 128,
            "container_cpus": 4,
            "container_tmpfs_bytes": 512 * 1024**2,
            "container_nofile": 256,
            "container_timeout_seconds": [1, 3_600],
            "default_container_timeout_seconds": 180,
            "ffprobe_timeout_seconds": 120,
            "ffmpeg_timeout_seconds": 120,
        },
        "claim_boundary": (
            (
                "The ARM64 bootstrap and AMD64 reference-miner frontends are bound to exact "
                "images, but their normalized motion tensors are not bit-equivalent. The "
                "AMD64 image is supported for component testing and baseline mining only. "
                "This profile is not release-quality, cross-runtime equivalence, or UMI "
                "activation evidence."
            )
            if has_reference_miner_frontend
            else (
                "This ARM64 bootstrap preprocessing profile is an EX-203 candidate. It has not "
                "established release-quality or cross-runtime landmark equivalence. No "
                "linux/amd64 extractor image is supported by this revision."
            )
        ),
    }


def build_s1_rights_record(
    *,
    source: Mapping[str, Any],
    runtime_code_license: str,
    intended_public_weight_license: str,
    redistribution_blocked_pending_final_rights_review: bool,
    release_rights_decision_sha256: str | None,
    upstream_rights: Mapping[str, Any],
) -> dict[str, Any]:
    required_source = {
        "source_id",
        "source_version",
        "license_id",
        "license_sha256",
        "terms_sha256",
        "source_use_policy_sha256",
        "attribution_notice_sha256",
        "source_entry_sha256",
        "rights_as_of",
        "public_weight_eligible",
        "public_data_eligible",
    }
    if not required_source <= set(source):
        raise S1PortableError("source rights record is incomplete")
    if (
        not isinstance(runtime_code_license, str)
        or not runtime_code_license
        or not isinstance(intended_public_weight_license, str)
        or not intended_public_weight_license
        or type(redistribution_blocked_pending_final_rights_review) is not bool
    ):
        raise S1PortableError("public weight rights values are invalid")
    if redistribution_blocked_pending_final_rights_review:
        if release_rights_decision_sha256 is not None:
            raise S1PortableError("blocked weight redistribution cannot name a release decision")
    elif source["public_weight_eligible"] is not True:
        raise S1PortableError("weight redistribution requires eligible source rights")
    else:
        _require_sha256(release_rights_decision_sha256, "release rights decision")
    record: dict[str, Any] = {
        "schema": S1_PORTABLE_RIGHTS_SCHEMA,
        "source_id": source["source_id"],
        "source_version": source["source_version"],
        "source_license_id": source["license_id"],
        "license_sha256": source["license_sha256"],
        "terms_sha256": source["terms_sha256"],
        "source_use_policy_sha256": source["source_use_policy_sha256"],
        "attribution_notice_sha256": source["attribution_notice_sha256"],
        "source_entry_sha256": source["source_entry_sha256"],
        "rights_as_of": source["rights_as_of"],
        "public_weight_eligible": source["public_weight_eligible"],
        "public_data_eligible": source["public_data_eligible"],
        "runtime_code_license": runtime_code_license,
        "intended_public_weight_license": intended_public_weight_license,
        "redistribution_blocked_pending_final_rights_review": (
            redistribution_blocked_pending_final_rights_review
        ),
        "release_rights_decision_sha256": release_rights_decision_sha256,
        "upstream_rights_sha256": canonical_json_sha256(dict(upstream_rights)),
        "claim_boundary": (
            "The portable artifact carries weights and tokenizer data only. It grants no right "
            "to redistribute source videos or annotations. Runtime code and model-weight "
            "distribution remain governed by the recorded licenses and final review state."
        ),
    }
    _validate_rights(record)
    return record


def build_s1_multi_source_rights_record(
    *,
    sources: Sequence[Mapping[str, Any]],
    rights_as_of: str,
    runtime_code_license: str,
    intended_public_weight_license: str,
    redistribution_blocked_pending_final_rights_review: bool,
    release_rights_decision_sha256: str | None,
    upstream_rights: Mapping[str, Any],
) -> dict[str, Any]:
    """Embed readable rights and attribution for every source behind public weights."""

    rows = [dict(source) for source in sources]
    if not rows or [row.get("source_id") for row in rows] != sorted(
        row.get("source_id") for row in rows if isinstance(row.get("source_id"), str)
    ):
        raise S1PortableError("portable source rights must be nonempty and sorted")
    if len({row.get("source_id") for row in rows}) != len(rows):
        raise S1PortableError("portable source rights contain duplicate sources")
    if (
        not isinstance(rights_as_of, str)
        or not rights_as_of
        or not isinstance(runtime_code_license, str)
        or not runtime_code_license
        or not isinstance(intended_public_weight_license, str)
        or not intended_public_weight_license
        or type(redistribution_blocked_pending_final_rights_review) is not bool
    ):
        raise S1PortableError("public multi-source weight rights values are invalid")
    if redistribution_blocked_pending_final_rights_review:
        if release_rights_decision_sha256 is not None:
            raise S1PortableError("blocked weight redistribution cannot name a release decision")
    else:
        _require_sha256(release_rights_decision_sha256, "release rights decision")
    record: dict[str, Any] = {
        "schema": S1_PORTABLE_MULTI_SOURCE_RIGHTS_SCHEMA,
        "sources": rows,
        "rights_as_of": rights_as_of,
        "runtime_code_license": runtime_code_license,
        "intended_public_weight_license": intended_public_weight_license,
        "redistribution_blocked_pending_final_rights_review": (
            redistribution_blocked_pending_final_rights_review
        ),
        "release_rights_decision_sha256": release_rights_decision_sha256,
        "upstream_rights_sha256": canonical_json_sha256(dict(upstream_rights)),
        "claim_boundary": (
            "The portable artifact carries weights, tokenizer data, and readable attribution "
            "for every model source. It grants no right to redistribute source videos or "
            "annotations. Runtime code and model-weight distribution remain governed by every "
            "recorded source license and the final review state."
        ),
    }
    _validate_rights(record)
    return record


def _validate_rights(value: object) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("schema") == S1_PORTABLE_MULTI_SOURCE_RIGHTS_SCHEMA:
        rights = _exact_dict(value, _MULTI_SOURCE_RIGHTS_FIELDS, "portable multi-source rights")
        raw_sources = rights["sources"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise S1PortableError("portable multi-source rights must name sources")
        sources = [
            _exact_dict(source, _MULTI_SOURCE_RIGHTS_ENTRY_FIELDS, "portable source rights")
            for source in raw_sources
        ]
        source_ids: list[str] = []
        for source in sources:
            for field in (
                "source_entry_sha256",
                "license_sha256",
                "terms_sha256",
                "source_use_policy_sha256",
                "attribution_notice_sha256",
            ):
                _require_sha256(source[field], f"portable source rights {field}")
            for field in ("source_id", "source_version", "source_name", "license_id"):
                if not isinstance(source[field], str) or not source[field]:
                    raise S1PortableError(f"portable source rights {field} is invalid")
            notice = source["attribution_notice"]
            if (
                not isinstance(notice, str)
                or not notice
                or _sha256(notice.encode("utf-8")) != source["attribution_notice_sha256"]
            ):
                raise S1PortableError("portable source attribution notice differs")
            for field in (
                "public_weight_eligible",
                "public_data_eligible",
                "raw_data_release",
                "public_data_release",
            ):
                if type(source[field]) is not bool:
                    raise S1PortableError(f"portable source rights {field} must be boolean")
            if (
                source["public_weight_eligible"] is not True
                or source["public_data_eligible"] is not False
                or source["raw_data_release"] is not False
                or source["public_data_release"] is not False
            ):
                raise S1PortableError("portable source is not eligible for weight-only release")
            source_ids.append(cast(str, source["source_id"]))
        if source_ids != sorted(source_ids) or len(source_ids) != len(set(source_ids)):
            raise S1PortableError("portable source rights order or uniqueness differs")
        for field in (
            "rights_as_of",
            "runtime_code_license",
            "intended_public_weight_license",
            "claim_boundary",
        ):
            if not isinstance(rights[field], str) or not rights[field]:
                raise S1PortableError(f"portable rights {field} is invalid")
        _require_sha256(rights["upstream_rights_sha256"], "portable rights upstream review")
        blocked = rights["redistribution_blocked_pending_final_rights_review"]
        if type(blocked) is not bool:
            raise S1PortableError("portable rights blocked state must be boolean")
        decision = rights["release_rights_decision_sha256"]
        if blocked:
            if decision is not None:
                raise S1PortableError("blocked portable rights cannot name a release decision")
        else:
            _require_sha256(decision, "portable rights release decision")
        return rights
    rights = _exact_dict(value, _RIGHTS_FIELDS, "portable rights")
    if rights["schema"] != S1_PORTABLE_RIGHTS_SCHEMA:
        raise S1PortableError("portable rights schema is unsupported")
    for field in (
        "license_sha256",
        "terms_sha256",
        "source_use_policy_sha256",
        "attribution_notice_sha256",
        "source_entry_sha256",
        "upstream_rights_sha256",
    ):
        _require_sha256(rights[field], f"portable rights {field}")
    for field in (
        "source_id",
        "source_version",
        "source_license_id",
        "rights_as_of",
        "runtime_code_license",
        "intended_public_weight_license",
        "claim_boundary",
    ):
        if not isinstance(rights[field], str) or not rights[field]:
            raise S1PortableError(f"portable rights {field} is invalid")
    for field in (
        "public_weight_eligible",
        "public_data_eligible",
        "redistribution_blocked_pending_final_rights_review",
    ):
        if type(rights[field]) is not bool:
            raise S1PortableError(f"portable rights {field} must be boolean")
    decision = rights["release_rights_decision_sha256"]
    if rights["redistribution_blocked_pending_final_rights_review"] is True:
        if decision is not None:
            raise S1PortableError("blocked portable rights cannot name a release decision")
    else:
        if rights["public_weight_eligible"] is not True:
            raise S1PortableError("cleared portable rights require public weight eligibility")
        _require_sha256(decision, "portable rights release decision")
    return rights


def _raw_text_postprocess_contract() -> dict[str, Any]:
    return {
        "schema": S1_TEXT_POSTPROCESS_SCHEMA,
        "revision": S1_RAW_TEXT_POSTPROCESS_REVISION,
        "operation": "identity",
        "selected_from_validation": False,
        "claim_boundary": (
            "No text cleanup is applied. Any later deterministic postprocess must be selected "
            "on validation data, assigned a new revision, and change the inference identity."
        ),
    }


def _fraction_pair(value: object, label: str) -> tuple[int, int]:
    record = _exact_dict(value, {"numerator", "denominator"}, label)
    numerator = record["numerator"]
    denominator = record["denominator"]
    if (
        not isinstance(numerator, str)
        or not isinstance(denominator, str)
        or not numerator.isascii()
        or not denominator.isascii()
    ):
        raise S1PortableError(f"{label} must use decimal integer strings")
    try:
        numerator_value = int(numerator)
        denominator_value = int(denominator)
    except ValueError as exc:
        raise S1PortableError(f"{label} must use decimal integer strings") from exc
    if (
        numerator != str(numerator_value)
        or denominator != str(denominator_value)
        or numerator_value < 0
        or denominator_value <= 0
        or numerator_value > denominator_value
        or math.gcd(numerator_value, denominator_value) != 1
    ):
        raise S1PortableError(f"{label} is not a reduced score in [0,1]")
    return numerator_value, denominator_value


def seal_s1_validation_word_prefix_selection(
    unsigned_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal and validate validation-only selection evidence for the portable identity."""

    record = dict(unsigned_selection)
    if "content_sha256" in record:
        raise S1PortableError("unsigned word-prefix selection already has a content digest")
    record["content_sha256"] = canonical_json_sha256(record, domain=_WORD_PREFIX_SELECTION_DOMAIN)
    build_s1_validation_word_prefix_contract(record)
    return record


def build_s1_validation_word_prefix_contract(
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a word-prefix cap selected only from the frozen validation predictions."""

    record = dict(selection)
    fields = {
        "schema",
        "partition",
        "normalization_revision",
        "source_prediction_set_sha256",
        "sample_count",
        "candidate_maximum_words",
        "candidate_scores",
        "selected_maximum_words",
        "selection_rule",
        "selected_arm",
        "selected_epoch",
        "test_partition_opened",
        "content_sha256",
    }
    _exact_dict(record, fields, "word-prefix selection")
    supplied_digest = _require_sha256(record["content_sha256"], "word-prefix selection content")
    unsigned = dict(record)
    del unsigned["content_sha256"]
    if canonical_json_sha256(unsigned, domain=_WORD_PREFIX_SELECTION_DOMAIN) != supplied_digest:
        raise S1PortableError("word-prefix selection content digest differs")
    candidates = record["candidate_maximum_words"]
    scores = record["candidate_scores"]
    if (
        record["schema"] != S1_WORD_PREFIX_SELECTION_SCHEMA
        or record["partition"] != "fleurs_val"
        or not isinstance(record["normalization_revision"], str)
        or not record["normalization_revision"]
        or type(record["sample_count"]) is not int
        or record["sample_count"] < 1
        or not isinstance(candidates, list)
        or not candidates
        or any(type(value) is not int or not 1 <= value <= 128 for value in candidates)
        or candidates != sorted(set(candidates))
        or not isinstance(scores, list)
        or len(scores) != len(candidates)
        or record["selection_rule"] != "greatest-score-then-smallest-cap-v1"
        or not isinstance(record["selected_arm"], str)
        or not record["selected_arm"]
        or type(record["selected_epoch"]) is not int
        or record["selected_epoch"] < 1
        or record["test_partition_opened"] is not False
    ):
        raise S1PortableError("word-prefix selection authority is invalid")
    _require_sha256(record["source_prediction_set_sha256"], "validation prediction set")
    observed: list[tuple[int, tuple[int, int]]] = []
    for index, (maximum, raw_score) in enumerate(zip(candidates, scores, strict=True)):
        score = _exact_dict(
            raw_score,
            {"maximum_words", "mean_normalized_score"},
            f"word-prefix candidate {index}",
        )
        if score["maximum_words"] != maximum:
            raise S1PortableError("word-prefix candidate ordering differs")
        observed.append(
            (
                maximum,
                _fraction_pair(
                    score["mean_normalized_score"], f"word-prefix candidate {index} score"
                ),
            )
        )
    selected_cap, (selected_numerator, selected_denominator) = observed[0]
    for maximum, (numerator, denominator) in observed[1:]:
        left = numerator * selected_denominator
        right = selected_numerator * denominator
        if left > right or (left == right and maximum < selected_cap):
            selected_cap = maximum
            selected_numerator = numerator
            selected_denominator = denominator
    if record["selected_maximum_words"] != selected_cap:
        raise S1PortableError("word-prefix selected cap does not reproduce")
    return {
        "schema": S1_TEXT_POSTPROCESS_SCHEMA,
        "revision": S1_VALIDATION_WORD_PREFIX_REVISION,
        "operation": "whitespace-word-prefix",
        "maximum_words": selected_cap,
        "selected_from_validation": True,
        "selection": record,
        "claim_boundary": (
            "The cap was selected from the frozen FLEURS validation predictions before the "
            "test partition opened. It bounds repetitive bootstrap output and is not UMI "
            "activation or production-quality evidence."
        ),
    }


def build_s1_authority_word_prefix_contract(
    *,
    training_authority_sha256: str,
    training_authority_content_sha256: str,
    final_report_content_sha256: str,
    maximum_words: int,
) -> dict[str, Any]:
    """Bind the deploy cap that an experiment fixed before optimization began."""

    authority_sha256 = _require_sha256(training_authority_sha256, "training authority")
    authority_content_sha256 = _require_sha256(
        training_authority_content_sha256, "training authority content"
    )
    final_content_sha256 = _require_sha256(final_report_content_sha256, "final report content")
    if type(maximum_words) is not int or maximum_words != 8:
        raise S1PortableError("authority-bound word-prefix cap must equal the deploy profile")
    return {
        "schema": S1_TEXT_POSTPROCESS_SCHEMA,
        "revision": S1_AUTHORITY_WORD_PREFIX_REVISION,
        "operation": "whitespace-word-prefix",
        "maximum_words": maximum_words,
        "selected_from_validation": False,
        "basis": "training-target-policy-fixed-before-optimization",
        "authority": {
            "schema": "umi-s1-training-word-prefix-authority/1",
            "training_authority_sha256": authority_sha256,
            "training_authority_content_sha256": authority_content_sha256,
            "final_report_content_sha256": final_content_sha256,
            "training_target_policy": "deploy-aligned-whole-word-prefix-v1",
            "maximum_training_output_words": maximum_words,
        },
        "claim_boundary": (
            "The eight-word cap was fixed in the content-addressed training authority before "
            "optimization. It was not selected after inspecting validation or test output and "
            "is not evidence of UMI activation or production quality."
        ),
    }


def _validate_text_postprocess(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise S1PortableError("text postprocess must be an object")
    if value.get("revision") == S1_RAW_TEXT_POSTPROCESS_REVISION:
        record = _exact_dict(
            value,
            {"schema", "revision", "operation", "selected_from_validation", "claim_boundary"},
            "text postprocess",
        )
        if record != _raw_text_postprocess_contract():
            raise S1PortableError("raw text postprocess policy differs")
        return record
    if value.get("revision") == S1_VALIDATION_WORD_PREFIX_REVISION:
        record = _exact_dict(
            value,
            {
                "schema",
                "revision",
                "operation",
                "maximum_words",
                "selected_from_validation",
                "selection",
                "claim_boundary",
            },
            "text postprocess",
        )
        expected = build_s1_validation_word_prefix_contract(
            cast(Mapping[str, Any], record["selection"])
        )
        if record != expected:
            raise S1PortableError("word-prefix text postprocess policy differs")
        return record
    if value.get("revision") == S1_AUTHORITY_WORD_PREFIX_REVISION:
        record = _exact_dict(
            value,
            {
                "schema",
                "revision",
                "operation",
                "maximum_words",
                "selected_from_validation",
                "basis",
                "authority",
                "claim_boundary",
            },
            "text postprocess",
        )
        authority = _exact_dict(
            record["authority"],
            {
                "schema",
                "training_authority_sha256",
                "training_authority_content_sha256",
                "final_report_content_sha256",
                "training_target_policy",
                "maximum_training_output_words",
            },
            "text postprocess authority",
        )
        expected = build_s1_authority_word_prefix_contract(
            training_authority_sha256=cast(str, authority["training_authority_sha256"]),
            training_authority_content_sha256=cast(
                str, authority["training_authority_content_sha256"]
            ),
            final_report_content_sha256=cast(str, authority["final_report_content_sha256"]),
            maximum_words=cast(int, record["maximum_words"]),
        )
        if record != expected:
            raise S1PortableError("authority-bound word-prefix text postprocess policy differs")
        return record
    raise S1PortableError("text postprocess policy is unsupported by this runtime")


def _apply_text_postprocess(text: str, policy: Mapping[str, Any]) -> str:
    record = _validate_text_postprocess(policy)
    if record["revision"] == S1_RAW_TEXT_POSTPROCESS_REVISION:
        return text
    return " ".join(text.split()[: cast(int, record["maximum_words"])])


def _claim_fraction(value: object, label: str) -> Fraction:
    record = _exact_dict(value, {"numerator", "denominator"}, label)
    if not isinstance(record["numerator"], str) or not isinstance(record["denominator"], str):
        raise S1PortableError(f"{label} must use string integer fields")
    try:
        result = Fraction(int(record["numerator"]), int(record["denominator"]))
    except (ValueError, ZeroDivisionError) as exc:
        raise S1PortableError(f"{label} is not an exact rational") from exc
    if record != {"numerator": str(result.numerator), "denominator": str(result.denominator)}:
        raise S1PortableError(f"{label} is not reduced canonical rational form")
    return result


def _claim_fraction_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _public_finetune_claim_boundary(
    evidence: object,
    *,
    model_state_sha256: str,
    upstream_release_identity_sha256: str,
    tokenizer_model_sha256: str,
    tokenizer_record_sha256: str,
    text_postprocess: Mapping[str, Any],
    rights_sha256: str,
) -> str:
    identity = _exact_dict(
        evidence,
        {
            "schema",
            "candidate",
            "gate",
            "tokenizer",
            "text_pretraining",
            "motion_sources",
            "rights_review_sha256",
            "rights_record_sha256",
            "base_preprocessing",
            "release_contents",
            "claim_boundary",
            "content_sha256",
        },
        "public fine-tune release identity",
    )
    supplied_content = _require_sha256(
        identity["content_sha256"], "public fine-tune release identity content"
    )
    unsigned_identity = dict(identity)
    del unsigned_identity["content_sha256"]
    if (
        identity["schema"] != _PUBLIC_FINETUNE_RELEASE_IDENTITY_SCHEMA
        or canonical_json_sha256(unsigned_identity, domain=_PUBLIC_FINETUNE_RELEASE_IDENTITY_DOMAIN)
        != supplied_content
        or canonical_json_sha256(identity) != upstream_release_identity_sha256
        or identity["claim_boundary"] != _PUBLIC_FINETUNE_RELEASE_CLAIM_BOUNDARY
    ):
        raise S1PortableError("public fine-tune release identity differs")

    candidate = _exact_dict(
        identity["candidate"],
        {
            "training_authority_sha256",
            "training_authority_file_sha256",
            "training_authority_content_sha256",
            "final_report_sha256",
            "final_report_content_sha256",
            "selected_epoch",
            "selected_epoch_report_content_sha256",
            "selected_validation_sha256",
            "selected_validation_content_sha256",
            "selected_checkpoint_manifest_sha256",
            "selected_checkpoint_metadata_sha256",
            "selected_model_state_sha256",
        },
        "public fine-tune candidate identity",
    )
    for field in (
        "training_authority_sha256",
        "training_authority_file_sha256",
        "training_authority_content_sha256",
        "final_report_sha256",
        "final_report_content_sha256",
        "selected_epoch_report_content_sha256",
        "selected_validation_sha256",
        "selected_validation_content_sha256",
        "selected_checkpoint_manifest_sha256",
        "selected_checkpoint_metadata_sha256",
        "selected_model_state_sha256",
    ):
        _require_sha256(candidate[field], f"public fine-tune candidate {field}")
    selected_epoch = candidate["selected_epoch"]
    if type(selected_epoch) is not int or selected_epoch < 1:
        raise S1PortableError("public fine-tune selected epoch is invalid")
    if candidate["selected_model_state_sha256"] != model_state_sha256:
        raise S1PortableError("public fine-tune model state differs")

    tokenizer = _exact_dict(
        identity["tokenizer"],
        {"binding_sha256", "model_sha256", "record_sha256", "training_report_sha256"},
        "public fine-tune tokenizer identity",
    )
    for field in tokenizer:
        _require_sha256(tokenizer[field], f"public fine-tune tokenizer {field}")
    if (
        tokenizer["model_sha256"] != tokenizer_model_sha256
        or tokenizer["record_sha256"] != tokenizer_record_sha256
    ):
        raise S1PortableError("public fine-tune tokenizer differs")

    postprocess = _validate_text_postprocess(dict(text_postprocess))
    postprocess_authority = _exact_dict(
        postprocess.get("authority"),
        {
            "schema",
            "training_authority_sha256",
            "training_authority_content_sha256",
            "final_report_content_sha256",
            "training_target_policy",
            "maximum_training_output_words",
        },
        "public fine-tune postprocess authority",
    )
    if (
        postprocess_authority["training_authority_sha256"] != candidate["training_authority_sha256"]
        or postprocess_authority["training_authority_content_sha256"]
        != candidate["training_authority_content_sha256"]
        or postprocess_authority["final_report_content_sha256"]
        != candidate["final_report_content_sha256"]
    ):
        raise S1PortableError("public fine-tune postprocess authority differs")

    _require_sha256(identity["rights_review_sha256"], "public fine-tune rights review")
    if identity["rights_record_sha256"] != rights_sha256:
        raise S1PortableError("public fine-tune rights record differs")

    release_contents = _exact_dict(
        identity["release_contents"],
        {
            "safe_tensor_weights",
            "tokenizer_artifacts",
            "raw_training_or_validation_data",
            "source_video_or_annotations",
            "prediction_plaintext",
        },
        "public fine-tune release contents",
    )
    if release_contents != {
        "safe_tensor_weights": True,
        "tokenizer_artifacts": True,
        "raw_training_or_validation_data": False,
        "source_video_or_annotations": False,
        "prediction_plaintext": False,
    }:
        raise S1PortableError("public fine-tune release contains non-portable data")

    text_pretraining = identity["text_pretraining"]
    if (
        not isinstance(text_pretraining, dict)
        or text_pretraining.get("release_lineage_allowed") is not True
        or text_pretraining.get("motion_encoder_invocation_count") != 0
        or text_pretraining.get("cross_attention_invocation_count") != 0
    ):
        raise S1PortableError("public fine-tune text-pretraining lineage differs")
    for field in (
        "training_authority_sha256",
        "training_authority_content_sha256",
        "final_report_sha256",
        "final_report_content_sha256",
        "selected_checkpoint_manifest_sha256",
        "selected_checkpoint_metadata_sha256",
        "selected_model_state_sha256",
        "training_authority_file_sha256",
        "corpus_manifest_sha256",
        "corpus_manifest_content_sha256",
        "source_entry_sha256",
    ):
        _require_sha256(text_pretraining.get(field), f"public text pretraining {field}")

    motion_sources = _exact_dict(
        identity["motion_sources"], {"fleurs", "two_m_flores"}, "public motion sources"
    )
    fleurs = _exact_dict(
        motion_sources["fleurs"],
        {
            "manifest_sha256",
            "manifest_content_sha256",
            "source_entry_sha256",
            "source_ledger_sha256",
            "training_rows_loaded",
            "validation_rows_loaded",
            "test_inference_rows",
        },
        "public FLEURS source",
    )
    two_m = _exact_dict(
        motion_sources["two_m_flores"],
        {
            "manifest_sha256",
            "manifest_content_sha256",
            "binding_content_sha256",
            "source_entry_sha256",
            "source_ledger_sha256",
            "sample_count",
            "devtest_rows_loaded",
            "evaluation_rows_loaded",
        },
        "public 2M-Flores source",
    )
    for label, record, fields in (
        (
            "FLEURS",
            fleurs,
            (
                "manifest_sha256",
                "manifest_content_sha256",
                "source_entry_sha256",
                "source_ledger_sha256",
            ),
        ),
        (
            "2M-Flores",
            two_m,
            (
                "manifest_sha256",
                "manifest_content_sha256",
                "binding_content_sha256",
                "source_entry_sha256",
                "source_ledger_sha256",
            ),
        ),
    ):
        for field in fields:
            _require_sha256(record[field], f"public {label} {field}")
    for label, value in (
        ("FLEURS training rows", fleurs["training_rows_loaded"]),
        ("FLEURS validation rows", fleurs["validation_rows_loaded"]),
        ("2M-Flores samples", two_m["sample_count"]),
    ):
        if type(value) is not int or value < 1:
            raise S1PortableError(f"{label} must be positive")
    if (
        fleurs["test_inference_rows"] != 0
        or two_m["devtest_rows_loaded"] != 0
        or two_m["evaluation_rows_loaded"] != 0
    ):
        raise S1PortableError("public fine-tune evidence used test, devtest, or evaluation rows")

    gate_result = _exact_dict(
        identity["gate"],
        {
            "passed",
            "real_motion_score",
            "zero_motion_score",
            "deranged_motion_score",
            "real_minus_zero",
            "real_minus_deranged",
            "prediction_count",
            "unique_real_hypotheses",
            "maximum_real_hypothesis_multiplicity",
            "gate",
        },
        "public fine-tune gate result",
    )
    configured_gate = _exact_dict(
        gate_result["gate"],
        {
            "minimum_real_score",
            "minimum_control_advantage",
            "minimum_unique_real_hypotheses",
            "maximum_real_hypothesis_multiplicity",
            "reject_empty_hypotheses",
            "reject_unk_token_id",
        },
        "public fine-tune configured gate",
    )
    real = _claim_fraction(gate_result["real_motion_score"], "public real-motion score")
    zero = _claim_fraction(gate_result["zero_motion_score"], "public zero-motion score")
    deranged = _claim_fraction(gate_result["deranged_motion_score"], "public deranged-motion score")
    real_minus_zero = _claim_fraction(
        gate_result["real_minus_zero"], "public real-minus-zero effect"
    )
    real_minus_deranged = _claim_fraction(
        gate_result["real_minus_deranged"], "public real-minus-deranged effect"
    )
    minimum_real = _claim_fraction(configured_gate["minimum_real_score"], "minimum real score")
    minimum_advantage = _claim_fraction(
        configured_gate["minimum_control_advantage"], "minimum control advantage"
    )
    prediction_count = gate_result["prediction_count"]
    unique = gate_result["unique_real_hypotheses"]
    multiplicity = gate_result["maximum_real_hypothesis_multiplicity"]
    minimum_unique = configured_gate["minimum_unique_real_hypotheses"]
    maximum_multiplicity = configured_gate["maximum_real_hypothesis_multiplicity"]
    if (
        gate_result["passed"] is not True
        or not 0 <= zero <= 1
        or not 0 <= deranged <= 1
        or not 0 <= real <= 1
        or real_minus_zero != real - zero
        or real_minus_deranged != real - deranged
        or real < minimum_real
        or real_minus_zero < minimum_advantage
        or real_minus_deranged < minimum_advantage
        or type(prediction_count) is not int
        or prediction_count != fleurs["validation_rows_loaded"]
        or type(unique) is not int
        or type(minimum_unique) is not int
        or not 1 <= minimum_unique <= unique <= prediction_count
        or type(multiplicity) is not int
        or type(maximum_multiplicity) is not int
        or not 1 <= multiplicity <= maximum_multiplicity
        or configured_gate["reject_empty_hypotheses"] is not True
        or configured_gate["reject_unk_token_id"] != 3
    ):
        raise S1PortableError("public fine-tune metric or diversity evidence differs")

    return (
        "This is a low-accuracy, FLEURS-validation-selected public bootstrap and miner "
        "replacement target, not a production-quality ASL translator. On its bound "
        f"{prediction_count}-sample FLEURS validation diagnostic, exact normalized scores "
        f"were real motion {_claim_fraction_text(real)}, zero motion "
        f"{_claim_fraction_text(zero)}, and deranged motion "
        f"{_claim_fraction_text(deranged)}; exact real-motion advantages were "
        f"{_claim_fraction_text(real_minus_zero)} over zero and "
        f"{_claim_fraction_text(real_minus_deranged)} over deranged motion. It produced "
        f"{unique} unique real-motion hypotheses with maximum multiplicity {multiplicity}. "
        "The bound gate rejects non-ok, empty, and UNK-bearing predictions. No FLEURS test "
        "inference or 2M-Flores devtest/evaluation inference was performed. This does not "
        "establish expected production performance, UMI activation, interpreter equivalence, "
        "accessibility certification, or production-quality translation."
    )


def _portable_claim_boundary(
    profile: str | None,
    *,
    model_state_sha256: str,
    upstream_release_identity_sha256: str,
    tokenizer_model_sha256: str,
    tokenizer_record_sha256: str,
    text_postprocess: Mapping[str, Any],
    rights_sha256: str,
    claim_boundary_evidence: Mapping[str, Any] | None = None,
) -> str:
    if profile is None:
        if claim_boundary_evidence is not None:
            raise S1PortableError("generic portable claim does not accept external evidence")
        return (
            "This is an integration fixture and replacement target, not a usable ASL "
            "translator. On the bound fixed-validation diagnostic, zero motion outscored "
            "real motion, so the checkpoint has not established useful motion grounding. "
            "Its exact ARM64 training frontend and AMD64 reference-miner frontend are not "
            "tensor-equivalent. The AMD64 path is a component-test baseline, not UMI "
            "activation evidence, accessibility certification, or production-quality "
            "translation."
        )
    if profile == S1_PUBLIC_FINETUNE_CLAIM_PROFILE:
        if claim_boundary_evidence is None:
            raise S1PortableError("public fine-tune claim requires release evidence")
        return _public_finetune_claim_boundary(
            claim_boundary_evidence,
            model_state_sha256=model_state_sha256,
            upstream_release_identity_sha256=upstream_release_identity_sha256,
            tokenizer_model_sha256=tokenizer_model_sha256,
            tokenizer_record_sha256=tokenizer_record_sha256,
            text_postprocess=text_postprocess,
            rights_sha256=rights_sha256,
        )
    if profile != S1_EXPERIMENT_BOOTSTRAP_CLAIM_PROFILE:
        raise S1PortableError("portable claim-boundary profile is unsupported")
    if claim_boundary_evidence is not None:
        raise S1PortableError("fixed bootstrap claim does not accept external evidence")
    expected_postprocess = build_s1_authority_word_prefix_contract(
        training_authority_sha256=(
            "f57a189360f80f7c88cfc435df033c5ef8e0c2f7084fa6fe835f45f5f57a54be"
        ),
        training_authority_content_sha256=(
            "6e33714687a942df27d0c3729ede68eb0d68e857aea17cb27b1e9f135c70343a"
        ),
        final_report_content_sha256=(
            "fa81389b82a95edd671719157c2f06decd3f684bf9a724ef0ac6e687e9154735"
        ),
        maximum_words=8,
    )
    if (
        model_state_sha256 != _EXPERIMENT_BOOTSTRAP_MODEL_STATE_SHA256
        or upstream_release_identity_sha256 != _EXPERIMENT_BOOTSTRAP_RELEASE_IDENTITY_SHA256
        or tokenizer_model_sha256 != _EXPERIMENT_BOOTSTRAP_TOKENIZER_MODEL_SHA256
        or tokenizer_record_sha256 != _EXPERIMENT_BOOTSTRAP_TOKENIZER_RECORD_SHA256
        or dict(text_postprocess) != expected_postprocess
        or rights_sha256 != _EXPERIMENT_BOOTSTRAP_RIGHTS_SHA256
    ):
        raise S1PortableError("portable claim-boundary evidence differs")
    return (
        "This is a low-accuracy validation-only FLEURS-ASL bootstrap and miner replacement "
        "target, not a usable ASL translator. Its bound 285-sample validation diagnostic "
        "scored real motion about 8.12%, only about 0.16 percentage points above zero and "
        "deranged motion. No FLEURS test result is claimed. Some historical training or "
        "validation-diagnostic source bytes named by the bound records are unavailable, so "
        "original-code reproduction is incomplete; checkpoint and record lineage plus "
        "current-runtime validation are the verification scope. This does not establish "
        "robust motion grounding, expected production performance, UMI activation, "
        "accessibility certification, or production-quality translation. The ARM64 "
        "training and AMD64 reference-miner frontends are not tensor-equivalent; AMD64 is "
        "component-test only."
    )


def _validate_preprocessing(value: object) -> dict[str, Any]:
    fields = {
        "schema",
        "status",
        "canonical_or_release_quality",
        "supported_oci_images",
        "dependencies",
        "sources",
        "mapping",
        "motion",
        "video",
        "ceilings",
        "claim_boundary",
    }
    record = _exact_dict(value, fields, "preprocessing contract")
    if not isinstance(record["supported_oci_images"], dict):
        raise S1PortableError("OCI images must be an object")
    images = cast(dict[str, Any], record["supported_oci_images"])
    if set(images) not in ({"linux/arm64"}, {"linux/arm64", "linux/amd64"}):
        raise S1PortableError("OCI images have an unexpected platform set")
    dependencies = _exact_dict(
        record["dependencies"],
        {"ffmpeg", "mediapipe", "numpy", "holistic_task_model_sha256"},
        "preprocessing dependencies",
    )
    base_source_fields = {
        "container_host_source_sha256",
        "container_worker_source_sha256",
        "container_requirements_sha256",
        "landmark_mapping_source_sha256",
        "motion_composition_source_sha256",
    }
    amd64_source_fields = {
        "amd64_container_host_source_sha256",
        "amd64_container_worker_source_sha256",
        "amd64_container_requirements_sha256",
    }
    sources = _exact_dict(
        record["sources"],
        base_source_fields | (amd64_source_fields if "linux/amd64" in images else set()),
        "preprocessing sources",
    )
    mapping = _exact_dict(record["mapping"], {"profile", "status", "policy_sha256"}, "mapping")
    motion = _exact_dict(
        record["motion"],
        {
            "feature_profile",
            "feature_policy_sha256",
            "shape",
            "dtype",
            "frame_mask_shape",
            "frame_mask_dtype",
            "sample_rate_hz",
            "maximum_frames",
            "point_count",
            "point_width",
            "point_fields",
            "global_width",
            "global_fields",
            "padding",
        },
        "motion contract",
    )
    if (
        record["schema"] != S1_PORTABLE_PREPROCESSING_SCHEMA
        or record["status"]
        != (
            "component-test-reference-miner" if "linux/amd64" in images else "ex-203-candidate-only"
        )
        or record["canonical_or_release_quality"] is not False
        or dependencies
        != {
            "ffmpeg": EXPECTED_FFMPEG_VERSION,
            "mediapipe": EXPECTED_MEDIAPIPE_VERSION,
            "numpy": "2.5.2",
            "holistic_task_model_sha256": EXPECTED_MODEL_SHA256,
        }
        or mapping
        != {
            "profile": MEDIAPIPE_MAPPING_PROFILE,
            "status": MEDIAPIPE_MAPPING_STATUS,
            "policy_sha256": canonical_json_sha256(DEFAULT_MEDIAPIPE_MAPPING_POLICY.as_dict()),
        }
        or motion
        != {
            "feature_profile": S1_TARGET_FEATURE_PROFILE,
            "feature_policy_sha256": canonical_json_sha256(DEFAULT_MOTION_FEATURE_POLICY.as_dict()),
            "shape": [MOTION_FRAME_COUNT, MOTION_FEATURE_DIM],
            "dtype": "float32",
            "frame_mask_shape": [MOTION_FRAME_COUNT],
            "frame_mask_dtype": "int32",
            "sample_rate_hz": TARGET_RATE,
            "maximum_frames": MOTION_FRAME_COUNT,
            "point_count": MOTION_POINT_COUNT,
            "point_width": MOTION_POINT_WIDTH,
            "point_fields": list(POINT_FIELDS),
            "global_width": MOTION_GLOBAL_WIDTH,
            "global_fields": list(GLOBAL_FIELDS),
            "padding": "positive-zero-with-frame-mask-zero",
        }
    ):
        raise S1PortableError("preprocessing contract differs from the fixed S1 candidate")
    _require_image_digest(images["linux/arm64"], "ARM64 extractor image")
    if "linux/amd64" in images:
        _require_image_digest(images["linux/amd64"], "AMD64 extractor image")
    for name, digest in sources.items():
        _require_sha256(digest, f"preprocessing source {name}")
    required_claim = "not bit-equivalent" if "linux/amd64" in images else "No linux/amd64"
    if (
        not isinstance(record["claim_boundary"], str)
        or required_claim not in record["claim_boundary"]
    ):
        raise S1PortableError("preprocessing claim boundary is incomplete")
    if not isinstance(record["video"], dict) or not isinstance(record["ceilings"], dict):
        raise S1PortableError("preprocessing media ceilings are invalid")
    expected = build_s1_preprocessing_contract(
        extractor_image_sha256=images["linux/arm64"],
        amd64_extractor_image_sha256=images.get("linux/amd64"),
        extractor_model_sha256=dependencies["holistic_task_model_sha256"],
        mapping_profile=mapping["profile"],
        mapping_policy_sha256=mapping["policy_sha256"],
        motion_feature_profile=motion["feature_profile"],
        motion_feature_policy_sha256=motion["feature_policy_sha256"],
        _source_records=cast(Mapping[str, str], sources),
    )
    if record != expected:
        raise S1PortableError("preprocessing media or resource contract differs")
    return record


def require_s1_preprocessing_platform(preprocessing: Mapping[str, Any], platform_id: str) -> str:
    record = _validate_preprocessing(preprocessing)
    images = cast(Mapping[str, str], record["supported_oci_images"])
    image = images.get(platform_id)
    if image is None:
        raise S1PortableError(f"preprocessing platform is unsupported: {platform_id}")
    return image


def _fixed_config_record(model: PortableS1, state_sha256: str) -> dict[str, Any]:
    return {
        "schema": S1_PORTABLE_CONFIG_SCHEMA,
        "model_class": S1_PORTABLE_MODEL_CLASS,
        "model_config": model.config.as_dict(),
        "parameter_count": model.parameter_count,
        "state_digest_profile": S1_TENSOR_SET_DIGEST_PROFILE,
        "model_state_sha256": state_sha256,
        "tensor_names": sorted(model.state_dict()),
        "input_contract": {
            "motion_shape": [MOTION_FRAME_COUNT, MOTION_FEATURE_DIM],
            "motion_dtype": "float32",
            "frame_mask_shape": [MOTION_FRAME_COUNT],
            "frame_mask_dtype": "int32",
            "maximum_output_tokens": PortableS1Config().max_output_tokens,
        },
    }


def _validate_config(record: object) -> dict[str, Any]:
    value = _exact_dict(
        record,
        {
            "schema",
            "model_class",
            "model_config",
            "parameter_count",
            "state_digest_profile",
            "model_state_sha256",
            "tensor_names",
            "input_contract",
        },
        "portable model config",
    )
    if (
        value["schema"] != S1_PORTABLE_CONFIG_SCHEMA
        or value["model_class"] != S1_PORTABLE_MODEL_CLASS
    ):
        raise S1PortableError("portable model config schema or class is unsupported")
    if value["model_config"] != PortableS1Config().as_dict():
        raise S1PortableError("portable model config is not the frozen public S1 config")
    model = PortableS1(PortableS1Config())
    if (
        value["parameter_count"] != model.parameter_count
        or value["state_digest_profile"] != S1_TENSOR_SET_DIGEST_PROFILE
        or value["tensor_names"] != sorted(model.state_dict())
        or value["input_contract"]
        != {
            "motion_shape": [MOTION_FRAME_COUNT, MOTION_FEATURE_DIM],
            "motion_dtype": "float32",
            "frame_mask_shape": [MOTION_FRAME_COUNT],
            "frame_mask_dtype": "int32",
            "maximum_output_tokens": PortableS1Config().max_output_tokens,
        }
    ):
        raise S1PortableError("portable model contract differs from fixed S1")
    _require_sha256(value["model_state_sha256"], "portable model state")
    return value


def _validate_safetensors_header(
    payload: bytes,
    *,
    expected_tensor_names: set[str],
    expected_state_sha256: str,
) -> None:
    if len(payload) < 10:
        raise S1PortableError("portable tensor payload is truncated")
    header_size = int.from_bytes(payload[:8], "little")
    if not 2 <= header_size <= 1024 * 1024 or 8 + header_size > len(payload):
        raise S1PortableError("portable safetensors header is invalid")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise S1PortableError("portable safetensors header contains a duplicate member")
            result[key] = value
        return result

    try:
        header = json.loads(
            payload[8 : 8 + header_size],
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                S1PortableError(f"portable safetensors header contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise S1PortableError("portable safetensors header is not strict JSON") from exc
    if not isinstance(header, dict) or set(header) != expected_tensor_names | {"__metadata__"}:
        raise S1PortableError("portable safetensors header names another tensor set")
    if header["__metadata__"] != {
        "schema": S1_PORTABLE_TENSOR_SCHEMA,
        "model": "S1",
        "model_state_sha256": expected_state_sha256,
    }:
        raise S1PortableError("portable safetensors metadata differs")


def _read_bundle_files(root: Path) -> dict[str, bytes]:
    path = Path(os.path.abspath(root))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise S1PortableError("portable bundle root cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            path.is_symlink()
            or not stat.S_ISDIR(opened.st_mode)
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or tuple(sorted(os.listdir(descriptor))) != _EXPECTED_FILES
        ):
            raise S1PortableError("portable bundle root field set is invalid")
        payloads: dict[str, bytes] = {}
        for name in _EXPECTED_FILES:
            file_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                file_descriptor = os.open(name, file_flags, dir_fd=descriptor)
            except OSError as exc:
                raise S1PortableError(f"portable bundle file cannot be opened: {name}") from exc
            try:
                initial = os.fstat(file_descriptor)
                maximum = _MAXIMUM_FILE_BYTES[name]
                if (
                    not stat.S_ISREG(initial.st_mode)
                    or initial.st_nlink != 1
                    or not 1 <= initial.st_size <= maximum
                ):
                    raise S1PortableError(f"portable bundle file violates its contract: {name}")
                chunks: list[bytes] = []
                observed = 0
                while True:
                    chunk = os.read(file_descriptor, min(1024 * 1024, maximum + 1 - observed))
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > maximum:
                        raise S1PortableError(f"portable bundle file exceeds its ceiling: {name}")
                    chunks.append(chunk)
                final = os.fstat(file_descriptor)
                if observed != initial.st_size or (
                    initial.st_dev,
                    initial.st_ino,
                    initial.st_size,
                    initial.st_mtime_ns,
                ) != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
                    raise S1PortableError(f"portable bundle file changed while reading: {name}")
                payloads[name] = b"".join(chunks)
            finally:
                os.close(file_descriptor)
        return payloads
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:  # pragma: no cover
                raise OSError("write made no progress")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise S1PortableError(f"portable bundle file cannot be published: {path.name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_input(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise S1PortableError(f"cannot read {label}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 1 <= len(payload) <= maximum_bytes
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise S1PortableError(f"{label} violates its input contract")
    return payload


def export_s1_portable_bundle(
    model: PortableS1,
    output: Path,
    *,
    tokenizer_model_path: Path,
    tokenizer_record_path: Path,
    preprocessing: Mapping[str, Any],
    rights: Mapping[str, Any],
    upstream_release_identity_sha256: str,
    text_postprocess: Mapping[str, Any] | None = None,
    claim_boundary_profile: str | None = None,
    claim_boundary_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically publish an independently copyable, pickle-free S1 runtime bundle."""

    if type(model) is not PortableS1 or model.config != PortableS1Config():
        raise S1PortableError("portable export requires the frozen public PortableS1 model")
    _require_sha256(upstream_release_identity_sha256, "upstream release identity")
    preprocessing_record = _validate_preprocessing(dict(preprocessing))
    rights_record = _validate_rights(dict(rights))
    text_postprocess_record = _validate_text_postprocess(
        _raw_text_postprocess_contract() if text_postprocess is None else dict(text_postprocess)
    )
    if claim_boundary_profile not in (
        None,
        S1_EXPERIMENT_BOOTSTRAP_CLAIM_PROFILE,
        S1_PUBLIC_FINETUNE_CLAIM_PROFILE,
    ):
        raise S1PortableError("portable claim-boundary profile is unsupported")
    if output.exists() or output.is_symlink():
        raise S1PortableError("portable bundle destination already exists")
    parent = output.parent.resolve(strict=True)
    lock = parent / f".{output.name}.portable-s1.lock"
    try:
        lock.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise S1PortableError("another portable S1 export owns the destination") from exc
    candidate = Path(tempfile.mkdtemp(prefix=f".{output.name}.candidate-", dir=parent))
    candidate.chmod(0o700)
    published = False
    try:
        tokenizer_model = _read_input(
            tokenizer_model_path,
            maximum_bytes=S1_TOKENIZER_MAXIMUM_MODEL_BYTES,
            label="tokenizer model",
        )
        tokenizer_record = _read_input(
            tokenizer_record_path,
            maximum_bytes=S1_TOKENIZER_MAXIMUM_RECORD_BYTES,
            label="tokenizer record",
        )
        tokenizer_model_sha256 = _sha256(tokenizer_model)
        tokenizer_record_sha256 = _sha256(tokenizer_record)
        try:
            tokenizer = S1DecodeTokenizer.from_bytes(
                record_bytes=tokenizer_record,
                model_bytes=tokenizer_model,
                expected_record_sha256=tokenizer_record_sha256,
                expected_model_sha256=tokenizer_model_sha256,
            )
        except S1DecodeTokenizerError as exc:
            raise S1PortableError("tokenizer cannot enter the portable bundle") from exc

        state = {
            name: tensor.detach().to(device="cpu").contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
        if any(not torch.isfinite(tensor).all().item() for tensor in state.values()):
            raise S1PortableError("portable model contains a non-finite tensor")
        state_sha256 = s1_tensor_set_sha256(state)
        claim_boundary = _portable_claim_boundary(
            claim_boundary_profile,
            model_state_sha256=state_sha256,
            upstream_release_identity_sha256=upstream_release_identity_sha256,
            tokenizer_model_sha256=tokenizer.model_sha256,
            tokenizer_record_sha256=tokenizer.record_sha256,
            text_postprocess=text_postprocess_record,
            rights_sha256=canonical_json_sha256(rights_record),
            claim_boundary_evidence=claim_boundary_evidence,
        )
        config = _fixed_config_record(model, state_sha256)
        config_bytes = canonical_json_bytes(config)
        tensor_bytes = save_safetensors(
            state,
            metadata={
                "schema": S1_PORTABLE_TENSOR_SCHEMA,
                "model": "S1",
                "model_state_sha256": state_sha256,
            },
        )
        if len(tensor_bytes) > _MAXIMUM_FILE_BYTES["model.safetensors"]:
            raise S1PortableError("portable tensor payload exceeds its byte ceiling")
        identity: dict[str, Any] = {
            "schema": S1_PORTABLE_SCHEMA,
            "upstream_release_identity_sha256": upstream_release_identity_sha256,
            "model": {
                "name": "S1",
                "class": S1_PORTABLE_MODEL_CLASS,
                "config_sha256": _sha256(config_bytes),
                "checkpoint_sha256": _sha256(tensor_bytes),
                "checkpoint_size_bytes": len(tensor_bytes),
                "state_digest_profile": S1_TENSOR_SET_DIGEST_PROFILE,
                "state_sha256": state_sha256,
            },
            "tokenizer": {
                "model_sha256": tokenizer.model_sha256,
                "model_size_bytes": len(tokenizer_model),
                "record_sha256": tokenizer.record_sha256,
                "record_size_bytes": len(tokenizer_record),
                "normalization_revision": tokenizer.normalization_revision,
            },
            "preprocessing": preprocessing_record,
            "runtime": _runtime_contract(),
            "text_postprocess": text_postprocess_record,
            "rights": rights_record,
            "claim_boundary": claim_boundary,
        }
        identity["inference_revision"] = canonical_json_sha256(identity, domain=_IDENTITY_DOMAIN)
        identity_bytes = canonical_json_bytes(identity)
        payloads = {
            "inference-identity.json": identity_bytes,
            "model-config.json": config_bytes,
            "model.safetensors": tensor_bytes,
            "tokenizer.json": tokenizer_record,
            "tokenizer.model": tokenizer_model,
        }
        manifest: dict[str, Any] = {
            "schema": S1_PORTABLE_MANIFEST_SCHEMA,
            "inference_revision": identity["inference_revision"],
            "files": [
                {"name": name, "sha256": _sha256(payloads[name]), "size_bytes": len(payloads[name])}
                for name in sorted(payloads)
            ],
        }
        manifest["content_sha256"] = canonical_json_sha256(manifest, domain=_MANIFEST_DOMAIN)
        payloads["bundle-manifest.json"] = canonical_json_bytes(manifest)
        for name in _EXPECTED_FILES:
            _write_exclusive(candidate / name, payloads[name])
        for path in candidate.iterdir():
            path.chmod(0o600)
        candidate_loaded = load_s1_portable_bundle(
            candidate,
            expected_inference_revision=cast(str, identity["inference_revision"]),
        )
        if candidate_loaded.identity != identity:
            raise S1PortableError("candidate portable identity differs after reload")
        del candidate_loaded
        candidate.rename(output)
        published = True
        loaded = load_s1_portable_bundle(
            output,
            expected_inference_revision=cast(str, identity["inference_revision"]),
        )
        if loaded.identity != identity:
            raise S1PortableError("published portable identity differs after reload")
        return {
            "schema": S1_PORTABLE_SCHEMA,
            "inference_revision": identity["inference_revision"],
            "manifest_sha256": _sha256(payloads["bundle-manifest.json"]),
            "size_bytes": sum(len(value) for value in payloads.values()),
            "model_state_sha256": state_sha256,
        }
    finally:
        if not published:
            shutil.rmtree(candidate, ignore_errors=True)
        try:
            lock.rmdir()
        except OSError:
            pass


@dataclass(frozen=True, slots=True)
class S1PortableInference:
    token_ids: tuple[int, ...]
    raw_text: str
    text: str
    text_postprocess_revision: str


@dataclass(slots=True)
class S1PortableRuntime:
    _model: PortableS1
    tokenizer: S1DecodeTokenizer
    identity: dict[str, Any]
    manifest: dict[str, Any]
    device: torch.device

    @property
    def inference_revision(self) -> str:
        return cast(str, self.identity["inference_revision"])

    @property
    def preprocessing(self) -> Mapping[str, Any]:
        return cast(Mapping[str, Any], self.identity["preprocessing"])

    def model_copy_for_export(self) -> PortableS1:
        """Return an isolated, frozen CPU copy for trusted format exporters."""

        with torch.no_grad():
            model = PortableS1(self._model.config)
            state = {
                name: tensor.detach().to(device="cpu").contiguous().clone()
                for name, tensor in self._model.state_dict().items()
            }
            model.load_state_dict(state, strict=True)
        expected = cast(Mapping[str, Any], self.identity["model"])["state_sha256"]
        if s1_tensor_set_sha256(model.state_dict()) != expected:
            raise S1PortableError("portable export model copy differs from its identity")
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def require_preprocessing_platform(self, platform_id: str) -> str:
        return require_s1_preprocessing_platform(self.preprocessing, platform_id)

    @torch.inference_mode()
    def infer_motion(
        self,
        motion: NDArray[np.float32] | Tensor,
        frame_mask: NDArray[np.int32] | Tensor,
        *,
        max_new_tokens: int | None = None,
    ) -> S1PortableInference:
        motion_tensor = torch.as_tensor(motion)
        mask_tensor = torch.as_tensor(frame_mask)
        if motion_tensor.shape != (MOTION_FRAME_COUNT, MOTION_FEATURE_DIM):
            raise S1PortableError("motion must have shape [120,1184]")
        if mask_tensor.shape != (MOTION_FRAME_COUNT,):
            raise S1PortableError("frame_mask must have shape [120]")
        if motion_tensor.dtype != torch.float32 or mask_tensor.dtype != torch.int32:
            raise S1PortableError("motion and frame_mask must use float32 and int32")
        if not torch.isfinite(motion_tensor).all().item():
            raise S1PortableError("motion contains a non-finite value")
        if not torch.all((mask_tensor == 0) | (mask_tensor == 1)).item():
            raise S1PortableError("frame_mask must contain only zero and one")
        valid = int(mask_tensor.sum().item())
        if (
            valid < 1
            or not torch.all(mask_tensor[:valid] == 1).item()
            or not torch.all(mask_tensor[valid:] == 0).item()
        ):
            raise S1PortableError("frame_mask must contain one nonempty contiguous prefix")
        padded = motion_tensor[valid:]
        if padded.numel() and (
            not torch.all(padded == 0).item() or torch.signbit(padded).any().item()
        ):
            raise S1PortableError("padded motion rows must be positive zero")
        decode_tokens = (
            S1_PORTABLE_MAXIMUM_DECODE_TOKENS if max_new_tokens is None else max_new_tokens
        )
        generated = self._model.beam_decode(
            motion_tensor.unsqueeze(0).to(self.device),
            mask_tensor.unsqueeze(0).to(self.device),
            max_new_tokens=decode_tokens,
            beam_width=S1_PORTABLE_BEAM_WIDTH,
            no_repeat_ngram_size=S1_PORTABLE_NO_REPEAT_NGRAM_SIZE,
        )[0]
        generated_ids = tuple(int(value) for value in generated.to(device="cpu").tolist())
        try:
            eos_index = generated_ids.index(EOS_TOKEN_ID)
        except ValueError:
            token_ids = generated_ids
        else:
            token_ids = generated_ids[: eos_index + 1]
        try:
            raw_text = self.tokenizer.decode(token_ids)
        except S1DecodeTokenizerError as exc:
            raise S1PortableError("generated S1 tokens cannot be decoded safely") from exc
        postprocess = cast(Mapping[str, Any], self.identity["text_postprocess"])
        text = _apply_text_postprocess(raw_text, postprocess)
        return S1PortableInference(
            token_ids=token_ids,
            raw_text=raw_text,
            text=text,
            text_postprocess_revision=cast(str, postprocess["revision"]),
        )


def load_s1_portable_bundle(
    root: Path,
    *,
    expected_inference_revision: str,
    device: str | torch.device = "cpu",
) -> S1PortableRuntime:
    """Load fixed S1 code and safe tensors; metadata is never imported or executed."""

    expected_revision = _require_sha256(expected_inference_revision, "expected inference revision")
    payloads = _read_bundle_files(root)
    manifest = _strict_json(
        payloads["bundle-manifest.json"],
        maximum_bytes=_MAXIMUM_FILE_BYTES["bundle-manifest.json"],
        label="portable manifest",
    )
    _exact_dict(manifest, {"schema", "inference_revision", "files", "content_sha256"}, "manifest")
    supplied_manifest_digest = _require_sha256(manifest["content_sha256"], "manifest content")
    unsigned_manifest = dict(manifest)
    del unsigned_manifest["content_sha256"]
    if (
        manifest["schema"] != S1_PORTABLE_MANIFEST_SCHEMA
        or manifest["inference_revision"] != expected_revision
        or canonical_json_sha256(unsigned_manifest, domain=_MANIFEST_DOMAIN)
        != supplied_manifest_digest
    ):
        raise S1PortableError("portable manifest identity or content digest differs")
    files = manifest["files"]
    expected_inventory = [
        {"name": name, "sha256": _sha256(payloads[name]), "size_bytes": len(payloads[name])}
        for name in sorted(set(_EXPECTED_FILES) - {"bundle-manifest.json"})
    ]
    if files != expected_inventory:
        raise S1PortableError("portable manifest file inventory differs")

    identity = _strict_json(
        payloads["inference-identity.json"],
        maximum_bytes=_MAXIMUM_FILE_BYTES["inference-identity.json"],
        label="inference identity",
    )
    _exact_dict(
        identity,
        {
            "schema",
            "inference_revision",
            "upstream_release_identity_sha256",
            "model",
            "tokenizer",
            "preprocessing",
            "runtime",
            "text_postprocess",
            "rights",
            "claim_boundary",
        },
        "inference identity",
    )
    supplied_revision = _require_sha256(identity["inference_revision"], "inference revision")
    unsigned_identity = dict(identity)
    del unsigned_identity["inference_revision"]
    if (
        identity["schema"] != S1_PORTABLE_SCHEMA
        or supplied_revision != expected_revision
        or canonical_json_sha256(unsigned_identity, domain=_IDENTITY_DOMAIN) != supplied_revision
        or identity["runtime"] != _runtime_contract()
    ):
        raise S1PortableError("portable inference identity or runtime differs")
    _require_sha256(identity["upstream_release_identity_sha256"], "upstream release identity")
    _validate_preprocessing(identity["preprocessing"])
    _validate_text_postprocess(identity["text_postprocess"])
    _validate_rights(identity["rights"])

    config = _strict_json(
        payloads["model-config.json"],
        maximum_bytes=_MAXIMUM_FILE_BYTES["model-config.json"],
        label="portable model config",
    )
    _validate_config(config)
    model_identity = _exact_dict(
        identity["model"],
        {
            "name",
            "class",
            "config_sha256",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
            "state_digest_profile",
            "state_sha256",
        },
        "model identity",
    )
    if (
        model_identity["name"] != "S1"
        or model_identity["class"] != S1_PORTABLE_MODEL_CLASS
        or model_identity["config_sha256"] != _sha256(payloads["model-config.json"])
        or model_identity["checkpoint_sha256"] != _sha256(payloads["model.safetensors"])
        or model_identity["checkpoint_size_bytes"] != len(payloads["model.safetensors"])
        or model_identity["state_digest_profile"] != S1_TENSOR_SET_DIGEST_PROFILE
        or model_identity["state_sha256"] != config["model_state_sha256"]
    ):
        raise S1PortableError("portable model identity differs from its files")
    expected_model = PortableS1(PortableS1Config())
    expected_state = expected_model.state_dict()
    _validate_safetensors_header(
        payloads["model.safetensors"],
        expected_tensor_names=set(expected_state),
        expected_state_sha256=cast(str, model_identity["state_sha256"]),
    )
    try:
        tensors = load_safetensors(payloads["model.safetensors"])
    except Exception as exc:
        raise S1PortableError("portable tensor payload is not valid safetensors") from exc
    if set(tensors) != set(expected_state):
        raise S1PortableError("portable tensor name set differs from S1")
    for name, expected in expected_state.items():
        observed = tensors[name]
        if (
            observed.dtype != expected.dtype
            or observed.shape != expected.shape
            or not torch.isfinite(observed).all().item()
        ):
            raise S1PortableError(f"portable tensor contract differs: {name}")
    state_sha256 = s1_tensor_set_sha256(tensors)
    if state_sha256 != model_identity["state_sha256"]:
        raise S1PortableError("portable tensor-set digest differs")
    try:
        expected_model.load_state_dict(tensors, strict=True)
    except (RuntimeError, ValueError) as exc:
        raise S1PortableError("portable tensors cannot load into fixed S1") from exc
    if s1_tensor_set_sha256(expected_model.state_dict()) != state_sha256:
        raise S1PortableError("portable tied tensors do not reproduce after loading")

    tokenizer_identity = _exact_dict(
        identity["tokenizer"],
        {
            "model_sha256",
            "model_size_bytes",
            "record_sha256",
            "record_size_bytes",
            "normalization_revision",
        },
        "tokenizer identity",
    )
    if tokenizer_identity["model_size_bytes"] != len(
        payloads["tokenizer.model"]
    ) or tokenizer_identity["record_size_bytes"] != len(payloads["tokenizer.json"]):
        raise S1PortableError("tokenizer size binding differs")
    try:
        tokenizer = S1DecodeTokenizer.from_bytes(
            record_bytes=payloads["tokenizer.json"],
            model_bytes=payloads["tokenizer.model"],
            expected_record_sha256=_require_sha256(
                tokenizer_identity["record_sha256"], "tokenizer record"
            ),
            expected_model_sha256=_require_sha256(
                tokenizer_identity["model_sha256"], "tokenizer model"
            ),
        )
    except S1DecodeTokenizerError as exc:
        raise S1PortableError("portable tokenizer is invalid") from exc
    if tokenizer.normalization_revision != tokenizer_identity["normalization_revision"]:
        raise S1PortableError("tokenizer normalization revision differs")
    selected_device = torch.device(device)
    model = expected_model.eval().to(selected_device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return S1PortableRuntime(
        _model=model,
        tokenizer=tokenizer,
        identity=identity,
        manifest=manifest,
        device=selected_device,
    )
