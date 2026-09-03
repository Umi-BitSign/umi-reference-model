from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

from .canonical import CanonicalJSONError, canonical_json_bytes, canonical_json_sha256

SELECTION_LEDGER_SCHEMA: Final = "umi-s1-public-selection-ledger/1"
SELECTION_LEDGER_FILENAME: Final = "umi-s1-baseline-v0-selection-ledger.json"
MOTION_ABLATION_SCHEMA: Final = "umi-s1-public-motion-ablation/1"
MOTION_ABLATION_FILENAME: Final = "umi-s1-baseline-v0-motion-ablation-evidence.json"
RIGHTS_EVIDENCE_SCHEMA: Final = "umi-s1-public-rights-decision/1"
RIGHTS_EVIDENCE_FILENAME: Final = "umi-s1-baseline-v0-rights-decision.json"
RELEASE_E2E_SCHEMA: Final = "umi-s1-public-release-e2e/1"
RELEASE_E2E_FILENAME: Final = "umi-s1-baseline-v0-release-e2e-evidence.json"
RELEASE_E2E_RUN_SCHEMA: Final = "umi-s1-release-e2e-run/1"

EVIDENCE_FILES: Final = {
    "selection-ledger": SELECTION_LEDGER_FILENAME,
    "motion-ablation-evidence": MOTION_ABLATION_FILENAME,
    "rights-evidence": RIGHTS_EVIDENCE_FILENAME,
    "release-e2e-evidence": RELEASE_E2E_FILENAME,
}

_SELECTION_DOMAIN: Final = b"umi-s1-public-selection-ledger-v1\0"
_MOTION_DOMAIN: Final = b"umi-s1-public-motion-ablation-v1\0"
_RIGHTS_DOMAIN: Final = b"umi-s1-public-rights-decision-v1\0"
_E2E_DOMAIN: Final = b"umi-s1-public-release-e2e-v1\0"
_PRIVATE_E2E_DOMAIN: Final = b"umi-s1-release-e2e-run-v1\0"
_MOTION_SAMPLE_SET_DOMAIN: Final = b"umi-s1-validation-motion-ablation-samples-v1\0"
_MOTION_RANK_DOMAIN: Final = b"umi-s1-validation-motion-ablation-rank-v1\0"
_MOTION_PERMUTATION_DOMAIN: Final = b"umi-s1-validation-motion-ablation-permutation-v1\0"
_DECODER_TOKEN_IDS_DOMAIN: Final = b"umi-s1-validation-decoder-comparison-token-ids-v1\0"
_CHECKPOINT_TOKEN_IDS_DOMAIN: Final = b"umi-s1-validation-checkpoint-sweep-token-ids-v1\0"
_GUIDANCE_TOKEN_IDS_DOMAIN: Final = b"umi-s1-validation-motion-guidance-token-ids-v1\0"
_DECODER_SAMPLE_SET_DOMAIN: Final = b"umi-s1-validation-decoder-comparison-samples-v1\0"
_GUIDANCE_SAMPLE_SET_DOMAIN: Final = b"umi-s1-validation-motion-guidance-samples-v1\0"
_DECODER_PREDICTION_DOMAIN: Final = b"umi-s1-validation-decoder-comparison-prediction-v1\0"
_DECODER_PREDICTION_SET_DOMAIN: Final = b"umi-s1-validation-decoder-comparison-predictions-v1\0"
_CHECKPOINT_PREDICTION_DOMAIN: Final = b"umi-s1-validation-checkpoint-sweep-prediction-v1\0"
_CHECKPOINT_PREDICTION_SET_DOMAIN: Final = b"umi-s1-validation-checkpoint-sweep-predictions-v1\0"
_GUIDANCE_PREDICTION_DOMAIN: Final = b"umi-s1-validation-motion-guidance-prediction-v1\0"
_GUIDANCE_PREDICTION_SET_DOMAIN: Final = b"umi-s1-validation-motion-guidance-predictions-v1\0"
_MOTION_PREDICTION_DOMAIN: Final = b"umi-s1-validation-motion-ablation-prediction-v1\0"
_MOTION_PREDICTION_SET_DOMAIN: Final = b"umi-s1-validation-motion-ablation-predictions-v1\0"
_GUIDANCE_BASE_INFERENCE_REVISION: Final = (
    "bf4ae743776afdc3df6c5f661ae14fdd7d59d589bc6ffeb375a2f9c98fc9932b"
)
_EVALUATED_BEAM_INFERENCE_REVISION: Final = (
    "42b095d27c5405ea2bb875a65b98b5db7e767a1816eaea0126a9fc6592ffd60d"
)
_DIRECT_EVALUATION_TRANSFER: Final = "same-model-state-tokenizer-runtime-and-decoder-contract"
_SOURCE_REBIND_EVALUATION_TRANSFER: Final = (
    "predecessor-validation-preserved-model-config-tokenizer-decoder-"
    "and-materialized-motion-semantics"
)
_PRIVATE_REPORT_DOMAINS: Final = {
    "umi-s1-validation-decoder-comparison/1": (b"umi-s1-validation-decoder-comparison-v1\0"),
    "umi-s1-validation-checkpoint-sweep/1": (b"umi-s1-validation-checkpoint-sweep-v1\0"),
    "umi-s1-validation-motion-guidance-sweep/1": (b"umi-s1-validation-motion-guidance-sweep-v1\0"),
    "umi-s1-validation-motion-ablation/2": (b"umi-s1-validation-motion-ablation-v1\0"),
    "umi-s1-final-rights-review/1": b"umi-s1-final-rights-review-v1\0",
    RELEASE_E2E_RUN_SCHEMA: _PRIVATE_E2E_DOMAIN,
}

_SELECTION_PRIVATE_SCHEMAS: Final = (
    "umi-s1-validation-decoder-comparison/1",
    "umi-s1-validation-checkpoint-sweep/1",
    "umi-s1-validation-motion-guidance-sweep/1",
)
_SELECTION_PRIVATE_ROLES: Final = (
    "decoder_comparison",
    "checkpoint_sweep",
    "motion_guidance_sweep",
)
_CONDITION_ORDER: Final = ("real_motion", "zero_motion", "deranged_motion")
_CHECKPOINT_EPOCHS: Final = (19, 20, 21, 22, 25, 30, 35)
_GUIDANCE_SCALES: Final = (
    {"hexadecimal": "0x0.0p+0", "ieee754_binary64_be": "0000000000000000"},
    {"hexadecimal": "0x1.0000000000000p-2", "ieee754_binary64_be": "3fd0000000000000"},
    {"hexadecimal": "0x1.0000000000000p-1", "ieee754_binary64_be": "3fe0000000000000"},
    {"hexadecimal": "0x1.0000000000000p+0", "ieee754_binary64_be": "3ff0000000000000"},
    {"hexadecimal": "0x1.0000000000000p+1", "ieee754_binary64_be": "4000000000000000"},
    {"hexadecimal": "0x1.0000000000000p+2", "ieee754_binary64_be": "4010000000000000"},
    {"hexadecimal": "0x1.0000000000000p+3", "ieee754_binary64_be": "4020000000000000"},
)
_RELEASE_ID: Final = "umi-s1-baseline-v0"
_SAMPLE_COUNT: Final = 285
_TASK_MODEL_SHA256: Final = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8"
_RIGHTS_SOURCE_DECISIONS: Final = (
    {
        "source_id": "fleurs-asl-v1",
        "source_version": "kaggle-v2-2024-08-29",
        "license_id": "CC-BY-SA-4.0",
        "public_data_eligible": False,
        "public_weight_eligible": True,
        "public_data_release": False,
        "raw_data_release": False,
    },
    {
        "source_id": "fsboard-v3",
        "source_version": "kaggle-v13-2025-07-26",
        "license_id": "CC-BY-4.0",
        "public_data_eligible": False,
        "public_weight_eligible": True,
        "public_data_release": False,
        "raw_data_release": False,
    },
)
_SELECTION_SOURCE: Final = {
    "source_id": "fleurs-asl-v1",
    "source_version": "kaggle-v2-2024-08-29",
    "license_id": "CC-BY-SA-4.0",
    "public_data_eligible": False,
}
_MAXIMUM_PRIVATE_REPORT_BYTES: Final = 64 * 1024 * 1024
_MAXIMUM_PUBLIC_REPORT_BYTES: Final = 1024 * 1024
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_GIT_REVISION: Final = re.compile(r"[0-9a-f]{40}")
_UTC_TIMESTAMP: Final = re.compile(
    r"20[0-9]{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)
_NUMERIC_VERSION: Final = re.compile(r"[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}")
_DOCKER_VERSION: Final = re.compile(
    r"client=[0-9]{1,4}(?:\.[0-9]{1,4}){1,3};"
    r"server=[0-9]{1,4}(?:\.[0-9]{1,4}){1,3}"
)
_APOSTROPHES: Final = frozenset(("'", "\u2019"))
_SOURCE_ROW_VALUE: Final = re.compile(
    r"(?:sample|signer|document|sentence)[_-][0-9]+", re.IGNORECASE
)
_FORBIDDEN_PUBLIC_KEYS: Final = frozenset(
    {
        "samples",
        "sample_set_sha256",
        "sample_set_commitment_sha256",
        "sample_set_commitments",
        "sample_id",
        "signer_id",
        "document_id",
        "sentence_id",
        "reference",
        "references",
        "reference_sha256",
        "reference_word_count",
        "motion_artifact_sha256",
        "motion_float32_sha256",
        "frame_mask_int32_sha256",
        "valid_frames",
        "hypothesis",
        "raw_hypothesis",
        "normalized_hypothesis",
        "hypothesis_sha256",
        "raw_hypothesis_sha256",
        "normalized_hypothesis_sha256",
        "token_ids",
        "generated_token_ids",
        "generated_token_ids_sha256",
        "edit_distance",
        "predictions",
        "prediction_sha256",
        "prediction_set_sha256",
        "test_source_sha256",
    }
)

_SELECTION_CLAIM_BOUNDARY: Final = (
    "Aggregate-only record of post-test-open validation choices. The reported runs used "
    "MPS inference over bound ARM64-materialized motion tensors. They are not Linux/AMD64 "
    "end-to-end quality evidence, untouched test evidence, UMI activation evidence, "
    "accessibility certification, or production-quality translation evidence. The record "
    "contains no per-sample source annotations, references, tensor identities, token IDs, "
    "hypotheses, or edit distances. Full private reports are bound only by their canonical "
    "content and file digests. Their aggregate scores require legally obtained source data "
    "and the bound evaluators for independent reproduction. When the evaluated and "
    "released revisions differ, the release candidate records the predecessor revision "
    "and the exact transfer basis; the aggregate quality run was not relabeled as a fresh "
    "evaluation of the source-rebound release revision."
)
_MOTION_CLAIM_BOUNDARY: Final = (
    "Aggregate-only, post-test-open validation diagnostic using MPS inference over bound "
    "ARM64-materialized motion tensors. Zero motion achieved a higher aggregate score than "
    "real motion. Permuted motion scored below real motion. The checkpoint reacts to motion, "
    "while this diagnostic does not establish useful motion grounding. This is not "
    "Linux/AMD64 end-to-end quality evidence, untouched test evidence, UMI activation "
    "evidence, accessibility certification, or production-quality translation evidence."
)
_RIGHTS_CLAIM_BOUNDARY: Final = (
    "Public projection of a project-owner release decision. It is not legal advice or a "
    "third-party license warranty. It authorizes the recorded model-weight distribution and "
    "does not authorize redistribution of source videos, source annotations, or the "
    "MediaPipe task model."
)
_E2E_CLAIM_BOUNDARY: Final = (
    "Functional Linux/AMD64 release-path evidence for one locally built extractor image and "
    "one rights-cleared private video fixture. It establishes raw-video execution, bounded "
    "success and failure responses, response sealing, post-reveal decryption, and cleanup for "
    "the bound run. It reports no translation quality, cross-host image equivalence, UMI "
    "activation evidence, accessibility certification, or reward guarantee."
)


class S1ReleaseEvidenceError(ValueError):
    """Raised when release evidence is unsafe, incomplete, or inconsistent."""


def _normalize_text(value: str) -> str:
    lowercased = unicodedata.normalize("NFKC", value).lower()
    normalized: list[str] = []
    for index, character in enumerate(lowercased):
        if unicodedata.category(character)[0] in ("L", "N"):
            normalized.append(character)
            continue
        internal_apostrophe = (
            character in _APOSTROPHES
            and index > 0
            and index + 1 < len(lowercased)
            and unicodedata.category(lowercased[index - 1])[0] in ("L", "N")
            and unicodedata.category(lowercased[index + 1])[0] in ("L", "N")
        )
        normalized.append(character if internal_apostrophe else " ")
    return " ".join("".join(normalized).split())


def _strict_equal(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if set(cast(dict[object, object], actual)) != set(expected):
            return False
        actual_dict = cast(dict[object, object], actual)
        return all(_strict_equal(actual_dict[key], value) for key, value in expected.items())
    if isinstance(expected, list):
        actual_list = cast(list[object], actual)
        return len(actual_list) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual_list, expected, strict=True)
        )
    if isinstance(expected, tuple):
        actual_tuple = cast(tuple[object, ...], actual)
        return len(actual_tuple) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual_tuple, expected, strict=True)
        )
    return actual == expected


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or not set(value) <= _LOWER_HEX
    ):
        raise S1ReleaseEvidenceError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def _require_git_revision(value: object, label: str) -> str:
    if not isinstance(value, str) or _GIT_REVISION.fullmatch(value) is None:
        raise S1ReleaseEvidenceError(f"{label} must be a full lowercase Git revision")
    return value


def _parse_utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _UTC_TIMESTAMP.fullmatch(value) is None:
        raise S1ReleaseEvidenceError(f"{label} is not a canonical UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise S1ReleaseEvidenceError(f"{label} is not a calendar-valid UTC timestamp") from exc


def _fraction(value: object, label: str) -> Fraction:
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise S1ReleaseEvidenceError(f"{label} is not an exact rational")
    numerator = value["numerator"]
    denominator = value["denominator"]
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        raise S1ReleaseEvidenceError(f"{label} is not an exact rational")
    try:
        result = Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError) as exc:
        raise S1ReleaseEvidenceError(f"{label} is not an exact rational") from exc
    if _fraction_record(result) != value:
        raise S1ReleaseEvidenceError(f"{label} is not a reduced exact rational")
    return result


def _fraction_record(value: Fraction) -> dict[str, str]:
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _strict_json(
    path: Path,
    *,
    label: str,
    maximum_bytes: int,
) -> tuple[dict[str, Any], bytes]:
    direct = Path(os.path.abspath(path))
    try:
        path_state = direct.lstat()
    except OSError as exc:
        raise S1ReleaseEvidenceError(f"{label} is unavailable") from exc
    if (
        stat.S_ISLNK(path_state.st_mode)
        or not stat.S_ISREG(path_state.st_mode)
        or path_state.st_nlink != 1
        or not 1 <= path_state.st_size <= maximum_bytes
    ):
        raise S1ReleaseEvidenceError(f"{label} violates its file contract")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(direct, flags)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            chunks: list[bytes] = []
            remaining = maximum_bytes + 1
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise S1ReleaseEvidenceError(f"{label} is unavailable") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (path_state.st_dev, path_state.st_ino) != (before.st_dev, before.st_ino)
        or not 1 <= len(raw) <= maximum_bytes
        or len(raw) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise S1ReleaseEvidenceError(f"{label} violates its file contract")

    return _strict_json_bytes(raw, label=label, maximum_bytes=maximum_bytes), raw


def _strict_json_bytes(raw: bytes, *, label: str, maximum_bytes: int) -> dict[str, Any]:
    if not 1 <= len(raw) <= maximum_bytes:
        raise S1ReleaseEvidenceError(f"{label} violates its byte ceiling")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise S1ReleaseEvidenceError(f"{label} contains a duplicate object member")
            result[key] = item
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                S1ReleaseEvidenceError(f"{label} contains a non-finite number: {token}")
            ),
        )
        canonical = canonical_json_bytes(value)
    except S1ReleaseEvidenceError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        CanonicalJSONError,
        RecursionError,
        ValueError,
    ) as exc:
        raise S1ReleaseEvidenceError(f"{label} is not strict canonical JSON") from exc
    if not isinstance(value, dict) or canonical != raw:
        raise S1ReleaseEvidenceError(f"{label} is not a canonical JSON object")
    return value


def _private_report(path: Path, expected_schema: str) -> tuple[dict[str, Any], str]:
    report, raw = _strict_json(
        path,
        label=f"private {expected_schema} report",
        maximum_bytes=_MAXIMUM_PRIVATE_REPORT_BYTES,
    )
    if report.get("schema") != expected_schema:
        raise S1ReleaseEvidenceError("private report schema differs")
    supplied = _require_sha256(report.get("content_sha256"), "private report content")
    unsigned = dict(report)
    del unsigned["content_sha256"]
    if canonical_json_sha256(unsigned, domain=_PRIVATE_REPORT_DOMAINS[expected_schema]) != supplied:
        raise S1ReleaseEvidenceError("private report content digest differs")
    return report, hashlib.sha256(raw).hexdigest()


def _binding(role: str, report: dict[str, Any], file_sha256: str) -> dict[str, str]:
    return {
        "role": role,
        "schema": cast(str, report["schema"]),
        "content_sha256": _require_sha256(report["content_sha256"], f"{role} content"),
        "file_sha256": _require_sha256(file_sha256, f"{role} file"),
    }


def _assert_public_safe(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN_PUBLIC_KEYS:
                raise S1ReleaseEvidenceError(f"public evidence contains forbidden field: {key}")
            _assert_public_safe(item)
    elif isinstance(value, list):
        for item in value:
            _assert_public_safe(item)
    elif isinstance(value, str) and _SOURCE_ROW_VALUE.search(value):
        raise S1ReleaseEvidenceError("public evidence contains a source-row identifier")


def _verify_content(
    report: object,
    *,
    schema: str,
    domain: bytes,
    fields: set[str],
) -> tuple[dict[str, Any], str]:
    if not isinstance(report, dict) or set(report) != fields:
        raise S1ReleaseEvidenceError(f"{schema} has an unexpected field set")
    if report["schema"] != schema:
        raise S1ReleaseEvidenceError(f"{schema} identity differs")
    supplied = _require_sha256(report["content_sha256"], f"{schema} content")
    unsigned = dict(report)
    del unsigned["content_sha256"]
    if canonical_json_sha256(unsigned, domain=domain) != supplied:
        raise S1ReleaseEvidenceError(f"{schema} content digest differs")
    _assert_public_safe(report)
    return report, supplied


def _validate_binding(value: object, role: str, schema: str) -> None:
    if (
        not isinstance(value, dict)
        or set(value) != {"role", "schema", "content_sha256", "file_sha256"}
        or value["role"] != role
        or value["schema"] != schema
    ):
        raise S1ReleaseEvidenceError(f"{role} private-report binding differs")
    _require_sha256(value["content_sha256"], f"{role} content")
    _require_sha256(value["file_sha256"], f"{role} file")


def _aggregate_result(
    row: object,
    *,
    identity_key: str,
    identity_value: object,
    sample_count: int,
) -> dict[str, Any]:
    if not isinstance(row, dict) or not _strict_equal(row.get(identity_key), identity_value):
        raise S1ReleaseEvidenceError(f"private aggregate result {identity_value!r} differs")
    if type(row.get("sample_count")) is not int or row.get("sample_count") != sample_count:
        raise S1ReleaseEvidenceError("private aggregate sample count differs")
    mean_wer = _fraction(row.get("mean_wer"), "private aggregate mean WER")
    mean_score = _fraction(row.get("mean_normalized_score"), "private aggregate mean score")
    unique = row.get("unique_normalized_hypotheses")
    if (
        mean_wer < 0
        or not 0 <= mean_score <= 1
        or type(unique) is not int
        or not 1 <= unique <= sample_count
    ):
        raise S1ReleaseEvidenceError("private aggregate result is outside its range")
    return {
        identity_key: identity_value,
        "sample_count": sample_count,
        "mean_wer": _fraction_record(mean_wer),
        "mean_normalized_score": _fraction_record(mean_score),
        "unique_output_count": unique,
    }


def _result_score(row: dict[str, Any], label: str) -> Fraction:
    return _fraction(row["mean_normalized_score"], label)


def _private_test_use_is_validation_only(report: dict[str, Any]) -> bool:
    data_use = report.get("data_use")
    schema = report.get("schema")
    schema_fields: dict[str, object]
    if schema == "umi-s1-validation-decoder-comparison/1":
        schema_fields = {
            "candidate_decoder_was_chosen_after_test_open": True,
            "test_partition_was_opened_before_this_comparison": True,
            "test_motion_tensors_loaded_by_this_comparison": False,
        }
        schema_fields[
            "canonical_materialization_test_metadata_may_be_parsed_during_authority_validation"
        ] = True
    elif schema == "umi-s1-validation-checkpoint-sweep/1":
        schema_fields = {
            "comparison_was_defined_after_the_test_partition_was_opened": True,
            "test_motion_tensors_loaded_by_this_sweep": False,
            "canonical_materialization_test_metadata_may_be_parsed_for_source_authority": True,
        }
    elif schema == "umi-s1-validation-motion-guidance-sweep/1":
        schema_fields = {
            "guidance_scales_were_chosen_after_test_open": True,
            "test_partition_was_opened_before_this_sweep": True,
            "test_motion_tensors_loaded_by_this_sweep": False,
        }
        schema_fields[
            "canonical_materialization_test_metadata_may_be_parsed_during_authority_validation"
        ] = True
    elif schema == "umi-s1-validation-motion-ablation/2":
        schema_fields = {
            "ablation_was_defined_after_the_test_partition_was_opened": True,
            "test_motion_tensors_loaded_by_this_ablation": False,
            "canonical_materialization_test_metadata_may_be_parsed_for_source_authority": True,
        }
    else:
        return False
    expected = {
        **schema_fields,
        "edit_distance_reproduction_requirement": (
            "legally obtained bound FLEURS references matching every reference_sha256"
        ),
        "edit_distances_independently_reproducible_without_bound_source_references": False,
        "plaintext_source_references_in_report": False,
        "test_evaluation_invocation_count": 0,
        "test_references_enter_inference_scoring_or_selection": False,
        "untouched_test_claim": False,
    }
    return _strict_equal(data_use, expected)


def _private_sample_keys(
    report: dict[str, Any],
) -> tuple[list[tuple[str, str, int | None]], list[dict[str, Any]]]:
    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != _SAMPLE_COUNT:
        raise S1ReleaseEvidenceError("private sample inventory differs")
    keys: list[tuple[str, str, int | None]] = []
    for row in samples:
        if not isinstance(row, dict) or set(row) != {
            "sample_id",
            "reference_sha256",
            "reference_word_count",
            "motion_artifact_sha256",
            "motion_float32_sha256",
            "frame_mask_int32_sha256",
            "valid_frames",
        }:
            raise S1ReleaseEvidenceError("private sample row is invalid")
        sample_id = row.get("sample_id")
        reference = row.get("reference_sha256")
        word_count = row.get("reference_word_count")
        if (
            not isinstance(sample_id, str)
            or not sample_id
            or any(ord(character) < 0x20 for character in sample_id)
            or len(sample_id.encode("utf-8")) > 512
            or type(word_count) is not int
            or word_count < 1
            or type(row["valid_frames"]) is not int
            or not 1 <= row["valid_frames"] <= 120
        ):
            raise S1ReleaseEvidenceError("private sample identity is invalid")
        for name in (
            "reference_sha256",
            "motion_artifact_sha256",
            "motion_float32_sha256",
            "frame_mask_int32_sha256",
        ):
            _require_sha256(row[name], f"private sample {name}")
        keys.append((sample_id, cast(str, reference), word_count))
    if [key[0] for key in keys] != sorted({key[0] for key in keys}):
        raise S1ReleaseEvidenceError("private sample identities are not sorted and unique")
    schema = report.get("schema")
    sample_domain = (
        _GUIDANCE_SAMPLE_SET_DOMAIN
        if schema == "umi-s1-validation-motion-guidance-sweep/1"
        else _DECODER_SAMPLE_SET_DOMAIN
    )
    if canonical_json_sha256(samples, domain=sample_domain) != report.get("sample_set_sha256"):
        raise S1ReleaseEvidenceError("private sample-set digest differs")
    return keys, cast(list[dict[str, Any]], samples)


def _recompute_private_result(
    row: object,
    *,
    sample_keys: list[tuple[str, str, int | None]],
    label: str,
    token_domain: bytes | None = None,
    maximum_tokens: int | None = None,
    prediction_domain: bytes | None = None,
    prediction_set_domain: bytes | None = None,
    prediction_context: dict[str, Any] | None = None,
    prediction_set_context: dict[str, Any] | None = None,
    result_identity_fields: set[str] | None = None,
    enforce_beam_constraints: bool = False,
) -> tuple[Fraction, Fraction, int, list[dict[str, Any]]]:
    token_contract = (
        token_domain,
        maximum_tokens,
        prediction_domain,
        prediction_set_domain,
        prediction_context,
        prediction_set_context,
        result_identity_fields,
    )
    if any(value is None for value in token_contract) and any(
        value is not None for value in token_contract
    ):
        raise S1ReleaseEvidenceError("private token-validation contract is incomplete")
    if not isinstance(row, dict):
        raise S1ReleaseEvidenceError(f"{label} private result is invalid")
    predictions = row.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(sample_keys):
        raise S1ReleaseEvidenceError(f"{label} private predictions differ")
    if token_domain is not None:
        prediction_set = {
            **cast(dict[str, Any], prediction_set_context),
            "predictions": predictions,
        }
        if row.get("prediction_set_sha256") != canonical_json_sha256(
            prediction_set, domain=cast(bytes, prediction_set_domain)
        ):
            raise S1ReleaseEvidenceError(f"{label} prediction-set digest differs")
    if token_domain is not None and set(row) != {
        "sample_count",
        "mean_wer",
        "mean_normalized_score",
        "unique_normalized_hypotheses",
        "prediction_set_sha256",
        "predictions",
        *cast(set[str], result_identity_fields),
    }:
        raise S1ReleaseEvidenceError(f"{label} result field set differs")
    wers: list[Fraction] = []
    scores: list[Fraction] = []
    hypotheses: set[str] = set()
    checked: list[dict[str, Any]] = []
    for index, (prediction, expected) in enumerate(zip(predictions, sample_keys, strict=True)):
        if not isinstance(prediction, dict):
            raise S1ReleaseEvidenceError(f"{label} prediction {index} is invalid")
        if token_domain is not None:
            expected_fields = {
                "sample_id",
                "status",
                "generated_token_ids",
                "generated_token_ids_sha256",
                "raw_hypothesis",
                "raw_hypothesis_sha256",
                "hypothesis",
                "hypothesis_sha256",
                "normalized_hypothesis",
                "normalized_hypothesis_sha256",
                "reference_sha256",
                "edit_distance",
                "reference_word_count",
                "wer",
                "normalized_score",
                "prediction_sha256",
            }
            if set(prediction) != expected_fields:
                raise S1ReleaseEvidenceError(f"{label} prediction field set differs")
            unsigned_prediction = dict(prediction)
            prediction_sha256 = unsigned_prediction.pop("prediction_sha256")
            if prediction_sha256 != canonical_json_sha256(
                {**cast(dict[str, Any], prediction_context), **unsigned_prediction},
                domain=cast(bytes, prediction_domain),
            ):
                raise S1ReleaseEvidenceError(f"{label} prediction digest differs")
        sample_id, reference, sample_word_count = expected
        word_count = prediction.get("reference_word_count")
        edit_distance = prediction.get("edit_distance")
        hypothesis = prediction.get("normalized_hypothesis")
        if (
            prediction.get("sample_id") != sample_id
            or prediction.get("reference_sha256") != reference
            or prediction.get("status") != "ok"
            or type(word_count) is not int
            or word_count < 1
            or (sample_word_count is not None and word_count != sample_word_count)
            or type(edit_distance) is not int
            or edit_distance < 0
            or not isinstance(hypothesis, str)
        ):
            raise S1ReleaseEvidenceError(f"{label} prediction identity differs")
        if token_domain is not None:
            token_ids = prediction.get("generated_token_ids")
            if (
                not isinstance(token_ids, list)
                or len(token_ids) != maximum_tokens
                or any(
                    type(token_id) is not int or not 0 <= token_id < 4096 for token_id in token_ids
                )
                or 1 in token_ids
                or prediction.get("generated_token_ids_sha256")
                != canonical_json_sha256(token_ids, domain=token_domain)
            ):
                raise S1ReleaseEvidenceError(f"{label} generated token record differs")
            if 2 in token_ids:
                eos_index = token_ids.index(2)
                if any(token_id != 0 for token_id in token_ids[eos_index + 1 :]):
                    raise S1ReleaseEvidenceError(f"{label} has non-padding tokens after EOS")
                before_eos = token_ids[:eos_index]
            else:
                before_eos = token_ids
            if enforce_beam_constraints:
                if any(token_id in (0, 1) for token_id in before_eos):
                    raise S1ReleaseEvidenceError(f"{label} emitted a suppressed token")
                trigrams = [
                    tuple(before_eos[offset : offset + 3])
                    for offset in range(max(0, len(before_eos) - 2))
                ]
                if len(trigrams) != len(set(trigrams)):
                    raise S1ReleaseEvidenceError(f"{label} repeats a generated trigram")
            raw_hypothesis = prediction.get("raw_hypothesis")
            capped_hypothesis = (
                " ".join(raw_hypothesis.split()[:8]) if isinstance(raw_hypothesis, str) else None
            )
            if (
                not isinstance(raw_hypothesis, str)
                or not isinstance(capped_hypothesis, str)
                or len(capped_hypothesis.encode("utf-8")) > 4096
                or prediction.get("raw_hypothesis_sha256")
                != hashlib.sha256(raw_hypothesis.encode("utf-8")).hexdigest()
                or prediction.get("hypothesis") != capped_hypothesis
                or prediction.get("hypothesis_sha256")
                != hashlib.sha256(capped_hypothesis.encode("utf-8")).hexdigest()
                or hypothesis != _normalize_text(capped_hypothesis)
                or prediction.get("normalized_hypothesis_sha256")
                != hashlib.sha256(hypothesis.encode("utf-8")).hexdigest()
            ):
                raise S1ReleaseEvidenceError(f"{label} prediction text differs")
        wer = Fraction(edit_distance, word_count)
        score = max(Fraction(0), Fraction(1) - wer)
        if (
            _fraction(prediction.get("wer"), f"{label} prediction WER") != wer
            or _fraction(prediction.get("normalized_score"), f"{label} prediction score") != score
        ):
            raise S1ReleaseEvidenceError(f"{label} prediction arithmetic differs")
        wers.append(wer)
        scores.append(score)
        hypotheses.add(hypothesis)
        checked.append(prediction)
    divisor = len(sample_keys)
    mean_wer = sum(wers, Fraction(0)) / divisor
    mean_score = sum(scores, Fraction(0)) / divisor
    unique = len(hypotheses)
    if (
        _fraction(row.get("mean_wer"), f"{label} private mean WER") != mean_wer
        or _fraction(row.get("mean_normalized_score"), f"{label} private mean score") != mean_score
        or row.get("unique_normalized_hypotheses") != unique
        or type(row.get("sample_count")) is not int
        or row.get("sample_count") != divisor
    ):
        raise S1ReleaseEvidenceError(f"{label} private aggregate does not reproduce")
    return mean_wer, mean_score, unique, checked


def _validate_private_motion_inventory(
    report: dict[str, Any],
) -> tuple[list[tuple[str, str, int | None]], dict[str, str]]:
    samples = report.get("samples")
    if not isinstance(samples, list) or len(samples) != _SAMPLE_COUNT:
        raise S1ReleaseEvidenceError("motion ablation sample inventory differs")
    sample_ids: list[str] = []
    sample_keys: list[tuple[str, str, int | None]] = []
    for row in samples:
        if not isinstance(row, dict) or set(row) != {
            "sample_id",
            "reference_sha256",
            "motion_float32_sha256",
            "frame_mask_int32_sha256",
            "valid_frames",
        }:
            raise S1ReleaseEvidenceError("motion ablation sample row is invalid")
        sample_id = row["sample_id"]
        if not isinstance(sample_id, str):
            raise S1ReleaseEvidenceError("motion ablation sample ID is invalid")
        try:
            sample_id_bytes = sample_id.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise S1ReleaseEvidenceError("motion ablation sample ID is invalid") from exc
        if not 1 <= len(sample_id_bytes) <= 512 or any(
            ord(character) < 0x20 for character in sample_id
        ):
            raise S1ReleaseEvidenceError("motion ablation sample ID is invalid")
        reference = _require_sha256(row["reference_sha256"], "motion reference")
        _require_sha256(row["motion_float32_sha256"], "motion tensor")
        _require_sha256(row["frame_mask_int32_sha256"], "motion frame mask")
        valid_frames = row["valid_frames"]
        if type(valid_frames) is not int or not 1 <= valid_frames <= 120:
            raise S1ReleaseEvidenceError("motion ablation valid-frame count is invalid")
        sample_ids.append(sample_id)
        sample_keys.append((sample_id, reference, None))
    if sample_ids != sorted(set(sample_ids)):
        raise S1ReleaseEvidenceError("motion ablation sample IDs are not sorted and unique")
    sample_set_sha256 = _require_sha256(report.get("sample_set_sha256"), "motion sample set")
    if canonical_json_sha256(samples, domain=_MOTION_SAMPLE_SET_DOMAIN) != sample_set_sha256:
        raise S1ReleaseEvidenceError("motion ablation sample-set digest differs")
    source = report.get("source")
    if (
        not isinstance(source, dict)
        or set(source)
        != {
            "profile",
            "authority",
            "validation_sample_count",
            "ablation_sample_set_sha256",
        }
        or source["profile"] != "selected-v0-fleurs-validation"
        or not isinstance(source["authority"], dict)
        or source["validation_sample_count"] != _SAMPLE_COUNT
        or source["ablation_sample_set_sha256"] != sample_set_sha256
    ):
        raise S1ReleaseEvidenceError("motion ablation source identity differs")

    permutation = report.get("permutation")
    if not isinstance(permutation, dict) or set(permutation) != {
        "method",
        "sample_set_sha256",
        "rank_domain",
        "ranked_sample_ids",
        "assignments",
        "content_sha256",
    }:
        raise S1ReleaseEvidenceError("motion ablation permutation is invalid")
    permutation_digest = _require_sha256(
        permutation["content_sha256"], "motion permutation content"
    )
    unsigned_permutation = dict(permutation)
    del unsigned_permutation["content_sha256"]
    if (
        canonical_json_sha256(unsigned_permutation, domain=_MOTION_PERMUTATION_DOMAIN)
        != permutation_digest
        or permutation["method"] != "sha256-ranked-single-cycle-left-rotation/1"
        or permutation["sample_set_sha256"] != sample_set_sha256
        or permutation["rank_domain"] != _MOTION_RANK_DOMAIN.rstrip(b"\0").decode("ascii")
        or not isinstance(permutation["ranked_sample_ids"], list)
        or not isinstance(permutation["assignments"], list)
    ):
        raise S1ReleaseEvidenceError("motion ablation permutation does not reproduce")
    ranked = permutation["ranked_sample_ids"]
    if any(not isinstance(sample_id, str) for sample_id in ranked):
        raise S1ReleaseEvidenceError("motion ablation permutation IDs are invalid")
    seed = bytes.fromhex(sample_set_sha256)
    expected_ranked = sorted(
        sample_ids,
        key=lambda sample_id: (
            hashlib.sha256(_MOTION_RANK_DOMAIN + seed + sample_id.encode("utf-8")).digest(),
            sample_id,
        ),
    )
    if ranked != expected_ranked:
        raise S1ReleaseEvidenceError("motion ablation permutation rank order differs")
    donors = {
        target: expected_ranked[(index + 1) % len(expected_ranked)]
        for index, target in enumerate(expected_ranked)
    }
    expected_assignments = [
        {"sample_id": sample_id, "motion_sample_id": donors[sample_id]} for sample_id in sample_ids
    ]
    if permutation["assignments"] != expected_assignments or any(
        target == donor for target, donor in donors.items()
    ):
        raise S1ReleaseEvidenceError("motion ablation permutation is not a derangement")
    return sample_keys, donors


def _validate_private_motion_prediction_integrity(
    row: object,
    *,
    label: str,
    vocabulary_size: int,
) -> None:
    result_fields = {
        "condition",
        "sample_count",
        "mean_wer",
        "mean_normalized_score",
        "unique_normalized_hypotheses",
        "prediction_set_sha256",
        "predictions",
    }
    if not isinstance(row, dict) or set(row) != result_fields:
        raise S1ReleaseEvidenceError(f"{label} result field set differs")
    predictions = row["predictions"]
    if not isinstance(predictions, list) or row["prediction_set_sha256"] != (
        canonical_json_sha256(predictions, domain=_MOTION_PREDICTION_SET_DOMAIN)
    ):
        raise S1ReleaseEvidenceError(f"{label} prediction-set digest differs")
    prediction_fields = {
        "sample_id",
        "motion_sample_id",
        "status",
        "token_ids",
        "raw_hypothesis",
        "raw_hypothesis_sha256",
        "hypothesis",
        "hypothesis_sha256",
        "normalized_hypothesis",
        "normalized_hypothesis_sha256",
        "reference_sha256",
        "edit_distance",
        "reference_word_count",
        "wer",
        "normalized_score",
        "prediction_sha256",
    }
    for prediction in predictions:
        if not isinstance(prediction, dict) or set(prediction) != prediction_fields:
            raise S1ReleaseEvidenceError(f"{label} prediction field set differs")
        unsigned = dict(prediction)
        supplied = unsigned.pop("prediction_sha256")
        if supplied != canonical_json_sha256(unsigned, domain=_MOTION_PREDICTION_DOMAIN):
            raise S1ReleaseEvidenceError(f"{label} prediction digest differs")
        token_ids = prediction["token_ids"]
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != 24
            or any(
                type(token_id) is not int or not 0 <= token_id < vocabulary_size
                for token_id in token_ids
            )
            or 1 in token_ids
        ):
            raise S1ReleaseEvidenceError(f"{label} token sequence differs")
        if 2 in token_ids:
            eos_index = token_ids.index(2)
            if any(token_id != 0 for token_id in token_ids[eos_index + 1 :]):
                raise S1ReleaseEvidenceError(f"{label} has tokens after EOS")
            generated = token_ids[:eos_index]
        else:
            if 0 in token_ids:
                raise S1ReleaseEvidenceError(f"{label} generated padding before EOS")
            generated = token_ids
        trigrams = [
            tuple(generated[offset : offset + 3]) for offset in range(max(0, len(generated) - 2))
        ]
        raw = prediction["raw_hypothesis"]
        capped = " ".join(raw.split()[:8]) if isinstance(raw, str) else None
        if not isinstance(raw, str) or not isinstance(capped, str):
            raise S1ReleaseEvidenceError(f"{label} prediction text differs")
        try:
            raw_bytes = raw.encode("utf-8")
            capped_bytes = capped.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise S1ReleaseEvidenceError(f"{label} prediction is not UTF-8") from exc
        limited = len(capped_bytes) > 4096
        hypothesis = "" if limited else capped
        normalized = _normalize_text(hypothesis)
        if (
            len(trigrams) != len(set(trigrams))
            or prediction["status"] != ("hypothesis_utf8_limit" if limited else "ok")
            or prediction["raw_hypothesis_sha256"] != hashlib.sha256(raw_bytes).hexdigest()
            or prediction["hypothesis"] != hypothesis
            or prediction["hypothesis_sha256"]
            != hashlib.sha256(hypothesis.encode("utf-8")).hexdigest()
            or prediction["normalized_hypothesis"] != normalized
            or prediction["normalized_hypothesis_sha256"]
            != hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        ):
            raise S1ReleaseEvidenceError(f"{label} prediction text differs")


def _identity_projection(value: object, names: tuple[str, ...], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise S1ReleaseEvidenceError(f"{label} identity is missing")
    projected = {name: value.get(name) for name in names}
    if any(item is None for item in projected.values()):
        raise S1ReleaseEvidenceError(f"{label} identity is incomplete")
    return projected


def _paired_score_counts(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> tuple[int, int, int]:
    left_better = equal = right_better = 0
    for left_row, right_row in zip(left, right, strict=True):
        left_score = _fraction(left_row["normalized_score"], "paired left score")
        right_score = _fraction(right_row["normalized_score"], "paired right score")
        if left_score > right_score:
            left_better += 1
        elif left_score == right_score:
            equal += 1
        else:
            right_better += 1
    return left_better, equal, right_better


def _prediction_semantics(prediction: dict[str, Any]) -> tuple[object, ...]:
    return tuple(
        prediction.get(name)
        for name in (
            "sample_id",
            "status",
            "generated_token_ids",
            "raw_hypothesis",
            "hypothesis",
            "normalized_hypothesis",
            "reference_sha256",
            "reference_word_count",
            "edit_distance",
            "wer",
            "normalized_score",
        )
    )


_BEAM_CONTRACT_FIELDS: Final = (
    "beam_width",
    "eos_token_id",
    "length_normalization",
    "maximum_decode_tokens",
    "maximum_hypothesis_utf8_bytes",
    "maximum_output_words",
    "metric",
    "no_repeat_ngram_size",
    "normalization_revision",
    "score",
    "suppressed_token_ids",
    "tie_break",
    "word_prefix_input",
)


def _beam_contract(value: object, label: str, *, algorithm: str = "beam-search") -> dict[str, Any]:
    contract = _identity_projection(value, _BEAM_CONTRACT_FIELDS, label)
    if not isinstance(value, dict) or value.get("algorithm") != algorithm:
        raise S1ReleaseEvidenceError(f"{label} decoder algorithm differs")
    if "finished_tail_token_id" in value and value["finished_tail_token_id"] != 0:
        raise S1ReleaseEvidenceError(f"{label} decoder tail token differs")
    expected = {
        "beam_width": 2,
        "eos_token_id": 2,
        "length_normalization": False,
        "maximum_decode_tokens": 24,
        "maximum_hypothesis_utf8_bytes": 4096,
        "maximum_output_words": 8,
        "metric": "exact-single-reference-wer-normalized-score",
        "no_repeat_ngram_size": 3,
        "score": "cumulative-log-probability",
        "suppressed_token_ids": [0, 1],
        "tie_break": "lexicographically-smallest-token-sequence",
        "word_prefix_input": "raw-tokenizer-output",
    }
    if any(contract[name] != item for name, item in expected.items()):
        raise S1ReleaseEvidenceError(f"{label} decoder contract differs")
    return contract


def _validate_greedy_contract(value: object, *, normalization_revision: str) -> None:
    expected = {
        "algorithm": "greedy-search",
        "arm": "published_greedy",
        "eos_token_id": 2,
        "finished_tail_token_id": 0,
        "maximum_decode_tokens": 128,
        "maximum_hypothesis_utf8_bytes": 4096,
        "maximum_output_words": 8,
        "metric": "exact-single-reference-wer-normalized-score",
        "normalization_revision": normalization_revision,
        "role": "published-baseline",
        "word_prefix_input": "raw-tokenizer-output",
    }
    if value != expected:
        raise S1ReleaseEvidenceError("published greedy decoder contract differs")


def project_selection_ledger(
    decoder_report_path: Path,
    checkpoint_report_path: Path,
    guidance_report_path: Path,
    *,
    release_inference_revision: str,
) -> dict[str, Any]:
    release_revision = _require_sha256(release_inference_revision, "release inference revision")
    loaded = [
        _private_report(path, schema)
        for path, schema in zip(
            (decoder_report_path, checkpoint_report_path, guidance_report_path),
            _SELECTION_PRIVATE_SCHEMAS,
            strict=True,
        )
    ]
    decoder, checkpoint, guidance = (item[0] for item in loaded)
    private_bindings = [
        _binding(role, report, file_digest)
        for role, (report, file_digest) in zip(_SELECTION_PRIVATE_ROLES, loaded, strict=True)
    ]
    reports = (decoder, checkpoint, guidance)
    if any(
        report.get("partition") != "fleurs_val"
        or report.get("sample_count") != _SAMPLE_COUNT
        or not _private_test_use_is_validation_only(report)
        for report in reports
    ):
        raise S1ReleaseEvidenceError("selection input is not fixed validation-only evidence")
    decoder_samples, decoder_inventory = _private_sample_keys(decoder)
    checkpoint_samples, checkpoint_inventory = _private_sample_keys(checkpoint)
    guidance_samples, guidance_inventory = _private_sample_keys(guidance)
    if (
        checkpoint_samples != decoder_samples
        or guidance_samples != decoder_samples
        or checkpoint_inventory != decoder_inventory
        or guidance_inventory != decoder_inventory
    ):
        raise S1ReleaseEvidenceError("selection reports use different sample inventories")
    tokenizer_fields = ("model_sha256", "record_sha256", "normalization_revision")
    source_fields = ("source_id", "source_version", "license_id", "public_data_eligible")
    decoder_tokenizer = _identity_projection(
        decoder.get("tokenizer"), tokenizer_fields, "decoder tokenizer"
    )
    decoder_source = _identity_projection(decoder.get("source"), source_fields, "decoder source")
    for report, label in ((checkpoint, "checkpoint"), (guidance, "guidance")):
        if (
            _identity_projection(report.get("tokenizer"), tokenizer_fields, f"{label} tokenizer")
            != decoder_tokenizer
            or _identity_projection(report.get("source"), source_fields, f"{label} source")
            != decoder_source
        ):
            raise S1ReleaseEvidenceError("selection reports bind different authorities")

    decoder_results = decoder.get("results")
    if not isinstance(decoder_results, list) or len(decoder_results) != 2:
        raise S1ReleaseEvidenceError("decoder result inventory differs")
    decoder_public: list[dict[str, Any]] = []
    decoder_predictions: list[list[dict[str, Any]]] = []
    for row, (arm, maximum_tokens) in zip(
        decoder_results,
        (("published_greedy", 128), ("candidate_beam", 24)),
        strict=True,
    ):
        _mean_wer, _mean_score, _unique, predictions = _recompute_private_result(
            row,
            sample_keys=decoder_samples,
            label=f"decoder {arm}",
            token_domain=_DECODER_TOKEN_IDS_DOMAIN,
            maximum_tokens=maximum_tokens,
            prediction_domain=_DECODER_PREDICTION_DOMAIN,
            prediction_set_domain=_DECODER_PREDICTION_SET_DOMAIN,
            prediction_context={"arm": arm},
            prediction_set_context={"arm": arm},
            result_identity_fields={"arm"},
            enforce_beam_constraints=arm == "candidate_beam",
        )
        decoder_predictions.append(predictions)
        decoder_public.append(
            _aggregate_result(
                row,
                identity_key="arm",
                identity_value=arm,
                sample_count=_SAMPLE_COUNT,
            )
        )
    decoder_paired = decoder.get("paired_comparison")
    if not isinstance(decoder_paired, dict):
        raise S1ReleaseEvidenceError("decoder paired comparison is missing")
    baseline_score = _result_score(decoder_public[0], "decoder baseline score")
    candidate_score = _result_score(decoder_public[1], "decoder candidate score")
    if baseline_score <= 0:
        raise S1ReleaseEvidenceError("decoder baseline score must be positive")
    decoder_delta = candidate_score - baseline_score
    better = decoder_paired.get("candidate_better_sample_count")
    equal = decoder_paired.get("equal_sample_count")
    worse = decoder_paired.get("candidate_worse_sample_count")
    reproduced_better, reproduced_equal, reproduced_worse = _paired_score_counts(
        decoder_predictions[1], decoder_predictions[0]
    )
    if (
        any(type(value) is not int or value < 0 for value in (better, equal, worse))
        or better + equal + worse != _SAMPLE_COUNT
        or (better, equal, worse) != (reproduced_better, reproduced_equal, reproduced_worse)
        or _fraction(decoder_paired.get("candidate_minus_baseline"), "decoder delta")
        != decoder_delta
        or _fraction(
            decoder_paired.get("relative_improvement_over_baseline"),
            "decoder relative improvement",
        )
        != decoder_delta / baseline_score
        or decoder_delta <= 0
    ):
        raise S1ReleaseEvidenceError("decoder comparison aggregate does not reproduce")

    checkpoint_results = checkpoint.get("results")
    if not isinstance(checkpoint_results, list) or len(checkpoint_results) != len(
        _CHECKPOINT_EPOCHS
    ):
        raise S1ReleaseEvidenceError("checkpoint result inventory differs")
    checkpoint_public: list[dict[str, Any]] = []
    checkpoint_states: list[str] = []
    checkpoint_predictions: list[list[dict[str, Any]]] = []
    for row, epoch in zip(checkpoint_results, _CHECKPOINT_EPOCHS, strict=True):
        _mean_wer, _mean_score, _unique, predictions = _recompute_private_result(
            row,
            sample_keys=decoder_samples,
            label=f"checkpoint epoch {epoch}",
            token_domain=_CHECKPOINT_TOKEN_IDS_DOMAIN,
            maximum_tokens=24,
            prediction_domain=_CHECKPOINT_PREDICTION_DOMAIN,
            prediction_set_domain=_CHECKPOINT_PREDICTION_SET_DOMAIN,
            prediction_context={
                "epoch": epoch,
            },
            prediction_set_context={
                "epoch": epoch,
                "checkpoint": row.get("checkpoint") if isinstance(row, dict) else None,
            },
            result_identity_fields={"epoch", "checkpoint"},
            enforce_beam_constraints=True,
        )
        checkpoint_predictions.append(predictions)
        summary = _aggregate_result(
            row,
            identity_key="epoch",
            identity_value=epoch,
            sample_count=_SAMPLE_COUNT,
        )
        private_checkpoint = row.get("checkpoint") if isinstance(row, dict) else None
        if not isinstance(private_checkpoint, dict):
            raise S1ReleaseEvidenceError("checkpoint identity is missing")
        checkpoint_states.append(
            _require_sha256(
                private_checkpoint.get("model_state_sha256"),
                f"epoch {epoch} model state",
            )
        )
        checkpoint_public.append(summary)
    selected_checkpoint = max(
        checkpoint_public,
        key=lambda row: (_result_score(row, "checkpoint score"), -cast(int, row["epoch"])),
    )
    checkpoint_selection = checkpoint.get("selection")
    selected_state = checkpoint_states[
        _CHECKPOINT_EPOCHS.index(cast(int, selected_checkpoint["epoch"]))
    ]
    selected_checkpoint_predictions = checkpoint_predictions[
        _CHECKPOINT_EPOCHS.index(cast(int, selected_checkpoint["epoch"]))
    ]
    if (
        not isinstance(checkpoint_selection, dict)
        or checkpoint_selection.get("selected_epoch") != selected_checkpoint["epoch"]
        or checkpoint_selection.get("selected_model_state_sha256") != selected_state
        or checkpoint_selection.get("baseline_epoch") != 20
        or checkpoint_selection.get("candidate_has_higher_observed_validation_mean") is not False
        or [_prediction_semantics(prediction) for prediction in selected_checkpoint_predictions]
        != [_prediction_semantics(prediction) for prediction in decoder_predictions[1]]
    ):
        raise S1ReleaseEvidenceError("checkpoint selection does not reproduce")

    guidance_results = guidance.get("results")
    scales = guidance.get("guidance_scales")
    if (
        not isinstance(guidance_results, list)
        or not isinstance(scales, list)
        or len(guidance_results) != 7
        or len(scales) != 7
        or scales != list(_GUIDANCE_SCALES)
    ):
        raise S1ReleaseEvidenceError("guidance result inventory differs")
    guidance_public: list[dict[str, Any]] = []
    guidance_predictions: list[list[dict[str, Any]]] = []
    for index, (row, scale) in enumerate(zip(guidance_results, scales, strict=True)):
        if not isinstance(scale, dict) or set(scale) != {
            "hexadecimal",
            "ieee754_binary64_be",
        }:
            raise S1ReleaseEvidenceError("guidance scale identity differs")
        summary = _aggregate_result(
            row,
            identity_key="scale_index",
            identity_value=index,
            sample_count=_SAMPLE_COUNT,
        )
        if not isinstance(row, dict) or row.get("guidance_scale") != scale:
            raise S1ReleaseEvidenceError("guidance result scale differs")
        _mean_wer, _mean_score, _unique, predictions = _recompute_private_result(
            row,
            sample_keys=decoder_samples,
            label=f"guidance scale {index}",
            token_domain=_GUIDANCE_TOKEN_IDS_DOMAIN,
            maximum_tokens=24,
            prediction_domain=_GUIDANCE_PREDICTION_DOMAIN,
            prediction_set_domain=_GUIDANCE_PREDICTION_SET_DOMAIN,
            prediction_context={"guidance_scale": scale},
            prediction_set_context={"guidance_scale": scale},
            result_identity_fields={"scale_index", "guidance_scale"},
            enforce_beam_constraints=True,
        )
        guidance_predictions.append(predictions)
        summary["scale"] = dict(scale)
        guidance_public.append(summary)
    selected_guidance = max(
        guidance_public,
        key=lambda row: (_result_score(row, "guidance score"), -cast(int, row["scale_index"])),
    )
    guidance_selection = guidance.get("selection")
    scale_zero_parity = guidance.get("scale_zero_parity")
    if (
        not isinstance(guidance_selection, dict)
        or guidance_selection.get("selected_scale_index") != selected_guidance["scale_index"]
        or selected_guidance["scale_index"] != 0
        or not isinstance(scale_zero_parity, dict)
        or scale_zero_parity.get("all_token_identical") is not True
        or scale_zero_parity.get("exact_token_match_count") != _SAMPLE_COUNT
        or any(
            _prediction_semantics(left) != _prediction_semantics(right)
            for left, right in zip(guidance_predictions[0], decoder_predictions[1], strict=True)
        )
    ):
        raise S1ReleaseEvidenceError("guidance selection or scale-zero parity differs")

    candidate = decoder.get("candidate")
    checkpoint_identity = decoder.get("checkpoint")
    tokenizer = decoder.get("tokenizer")
    guidance_checkpoint = guidance.get("checkpoint")
    if (
        not isinstance(candidate, dict)
        or not isinstance(checkpoint_identity, dict)
        or not isinstance(tokenizer, dict)
        or not isinstance(guidance_checkpoint, dict)
        or candidate.get("runtime_revision") != "s1-pytorch-beam-runtime/1"
        or checkpoint_identity.get("model_state_sha256") != selected_state
        or guidance_checkpoint.get("model_state_sha256") != selected_state
        or _result_score(selected_checkpoint, "selected checkpoint score") != candidate_score
        or _result_score(selected_guidance, "selected guidance score") != candidate_score
    ):
        raise S1ReleaseEvidenceError("selection reports do not bind one candidate")
    private_decoders = decoder.get("decoders")
    if not isinstance(private_decoders, list) or len(private_decoders) != 2:
        raise S1ReleaseEvidenceError("decoder contract inventory differs")
    selected_contract = _beam_contract(private_decoders[1], "decoder comparison")
    _validate_greedy_contract(
        private_decoders[0],
        normalization_revision=cast(str, decoder_tokenizer["normalization_revision"]),
    )
    if (
        private_decoders[1].get("arm") != "candidate_beam"
        or private_decoders[1].get("role") != "candidate-replacement"
        or selected_contract["normalization_revision"]
        != decoder_tokenizer["normalization_revision"]
    ):
        raise S1ReleaseEvidenceError("candidate decoder identity differs")
    guidance_decoder = guidance.get("decoder")
    guidance_base = (
        guidance_decoder.get("base_runtime") if isinstance(guidance_decoder, dict) else None
    )
    if (
        _beam_contract(checkpoint.get("decoder"), "checkpoint sweep") != selected_contract
        or _beam_contract(
            guidance_decoder,
            "guidance sweep",
            algorithm="motion-guided-beam-search",
        )
        != selected_contract
        or not isinstance(guidance_base, dict)
        or set(guidance_base) != {"inference_revision", "runtime_revision"}
        or guidance_base.get("inference_revision") != _GUIDANCE_BASE_INFERENCE_REVISION
        or guidance_base.get("runtime_revision") != candidate.get("runtime_revision")
    ):
        raise S1ReleaseEvidenceError("selection reports bind different decoder contracts")
    evaluated_revision = _require_sha256(
        candidate.get("inference_revision"), "evaluated inference revision"
    )
    if evaluated_revision != _EVALUATED_BEAM_INFERENCE_REVISION:
        raise S1ReleaseEvidenceError("evaluated inference revision differs")
    evaluation_transfer_basis = (
        _DIRECT_EVALUATION_TRANSFER
        if evaluated_revision == release_revision
        else _SOURCE_REBIND_EVALUATION_TRANSFER
    )
    source = decoder.get("source")
    if not isinstance(source, dict):
        raise S1ReleaseEvidenceError("decoder source identity is missing")
    source_identity = {
        "source_id": source.get("source_id"),
        "source_version": source.get("source_version"),
        "license_id": source.get("license_id"),
        "public_data_eligible": source.get("public_data_eligible"),
    }
    if not _strict_equal(source_identity, _SELECTION_SOURCE):
        raise S1ReleaseEvidenceError("selection source identity differs")
    decoder_runtime = decoder.get("runtime")
    checkpoint_runtime = checkpoint.get("runtime")
    guidance_runtime = guidance.get("runtime")
    guidance_device = (
        guidance_runtime.get("device_identity") if isinstance(guidance_runtime, dict) else None
    )
    if (
        not isinstance(decoder_runtime, dict)
        or decoder_runtime.get("resolved_device") != "mps"
        or not isinstance(checkpoint_runtime, dict)
        or checkpoint_runtime.get("resolved_device") != "mps"
        or checkpoint_runtime.get("machine_architecture") != "arm64"
        or not isinstance(guidance_device, dict)
        or guidance_device.get("backend") != "mps"
        or guidance_device.get("machine") != "arm64"
    ):
        raise S1ReleaseEvidenceError("selection runtime is not the bound MPS/ARM64 context")

    report: dict[str, Any] = {
        "schema": SELECTION_LEDGER_SCHEMA,
        "release_id": _RELEASE_ID,
        "purpose": "aggregate post-test-open validation selection ledger",
        "partition": "fleurs_val",
        "sample_count": _SAMPLE_COUNT,
        "source": source_identity,
        "evaluation_context": {
            "test_partition_already_open": True,
            "test_evaluation_invocation_count": 0,
            "inference_backend": "mps",
            "machine_architecture": "arm64",
            "input_materialization": "bound-arm64-validation-motion-tensors",
            "linux_amd64_end_to_end_quality_evidence": False,
        },
        "private_reports": private_bindings,
        "release_candidate": {
            "release_inference_revision": release_revision,
            "evaluated_inference_revision": evaluated_revision,
            "runtime_revision": candidate["runtime_revision"],
            "selected_epoch": selected_checkpoint["epoch"],
            "selected_model_state_sha256": selected_state,
            "tokenizer_model_sha256": _require_sha256(
                tokenizer.get("model_sha256"), "tokenizer model"
            ),
            "tokenizer_record_sha256": _require_sha256(
                tokenizer.get("record_sha256"), "tokenizer record"
            ),
            "evaluation_transfer_basis": evaluation_transfer_basis,
        },
        "decoder_selection": {
            "profiles": [
                "published-greedy-max128-eight-word-cap",
                "beam2-max24-no-repeat3-eight-word-cap",
            ],
            "results": decoder_public,
            "paired_counts": {
                "candidate_better": better,
                "equal": equal,
                "candidate_worse": worse,
            },
            "candidate_minus_baseline": _fraction_record(decoder_delta),
            "relative_improvement_over_baseline": _fraction_record(decoder_delta / baseline_score),
            "selected_profile": "beam2-max24-no-repeat3-eight-word-cap",
        },
        "checkpoint_selection": {
            "rule": "highest-exact-validation-mean-then-lowest-epoch",
            "results": checkpoint_public,
            "selected_epoch": selected_checkpoint["epoch"],
            "selected_model_state_sha256": selected_state,
        },
        "guidance_selection": {
            "rule": "highest-exact-validation-mean-then-lowest-scale",
            "results": guidance_public,
            "selected_scale_index": selected_guidance["scale_index"],
            "scale_zero_matches_portable_decoder": True,
            "scale_zero_match_count": _SAMPLE_COUNT,
        },
        "public_data": {
            "aggregate_only": True,
            "source_rows_included": False,
            "per_example_outputs_included": False,
        },
        "claim_boundary": _SELECTION_CLAIM_BOUNDARY,
    }
    report["content_sha256"] = canonical_json_sha256(report, domain=_SELECTION_DOMAIN)
    validate_selection_ledger(report, expected_inference_revision=release_revision)
    return report


def _validate_summary_result(
    value: object,
    *,
    identity_key: str,
    expected_identity: object,
    sample_count: int,
) -> Fraction:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            identity_key,
            "sample_count",
            "mean_wer",
            "mean_normalized_score",
            "unique_output_count",
        }
        or not _strict_equal(value[identity_key], expected_identity)
        or type(value["sample_count"]) is not int
        or value["sample_count"] != sample_count
        or type(value["unique_output_count"]) is not int
        or not 1 <= value["unique_output_count"] <= sample_count
    ):
        raise S1ReleaseEvidenceError("aggregate result differs")
    if _fraction(value["mean_wer"], "aggregate mean WER") < 0:
        raise S1ReleaseEvidenceError("aggregate mean WER is negative")
    score = _fraction(value["mean_normalized_score"], "aggregate mean score")
    if not 0 <= score <= 1:
        raise S1ReleaseEvidenceError("aggregate mean score is outside [0, 1]")
    return score


def validate_selection_ledger(
    value: object,
    *,
    expected_inference_revision: str,
) -> str:
    report, supplied = _verify_content(
        value,
        schema=SELECTION_LEDGER_SCHEMA,
        domain=_SELECTION_DOMAIN,
        fields={
            "schema",
            "release_id",
            "purpose",
            "partition",
            "sample_count",
            "source",
            "evaluation_context",
            "private_reports",
            "release_candidate",
            "decoder_selection",
            "checkpoint_selection",
            "guidance_selection",
            "public_data",
            "claim_boundary",
            "content_sha256",
        },
    )
    if (
        report["release_id"] != _RELEASE_ID
        or report["purpose"] != "aggregate post-test-open validation selection ledger"
        or report["partition"] != "fleurs_val"
        or type(report["sample_count"]) is not int
        or report["sample_count"] != _SAMPLE_COUNT
        or report["claim_boundary"] != _SELECTION_CLAIM_BOUNDARY
        or not _strict_equal(
            report["evaluation_context"],
            {
                "test_partition_already_open": True,
                "test_evaluation_invocation_count": 0,
                "inference_backend": "mps",
                "machine_architecture": "arm64",
                "input_materialization": "bound-arm64-validation-motion-tensors",
                "linux_amd64_end_to_end_quality_evidence": False,
            },
        )
        or not _strict_equal(
            report["public_data"],
            {
                "aggregate_only": True,
                "source_rows_included": False,
                "per_example_outputs_included": False,
            },
        )
    ):
        raise S1ReleaseEvidenceError("selection ledger identity differs")
    source = report["source"]
    if (
        not isinstance(source, dict)
        or set(source) != {"source_id", "source_version", "license_id", "public_data_eligible"}
        or any(
            not isinstance(source[name], str)
            for name in ("source_id", "source_version", "license_id")
        )
        or source["public_data_eligible"] is not False
        or not _strict_equal(source, _SELECTION_SOURCE)
    ):
        raise S1ReleaseEvidenceError("selection source projection differs")
    bindings = report["private_reports"]
    if not isinstance(bindings, list) or len(bindings) != 3:
        raise S1ReleaseEvidenceError("selection private-report bindings differ")
    for binding, role, schema in zip(
        bindings, _SELECTION_PRIVATE_ROLES, _SELECTION_PRIVATE_SCHEMAS, strict=True
    ):
        _validate_binding(binding, role, schema)
    candidate = report["release_candidate"]
    candidate_fields = {
        "release_inference_revision",
        "evaluated_inference_revision",
        "runtime_revision",
        "selected_epoch",
        "selected_model_state_sha256",
        "tokenizer_model_sha256",
        "tokenizer_record_sha256",
        "evaluation_transfer_basis",
    }
    direct_evaluation = isinstance(candidate, dict) and candidate.get(
        "evaluated_inference_revision"
    ) == candidate.get("release_inference_revision")
    transfer_is_valid = (
        direct_evaluation
        and isinstance(candidate, dict)
        and candidate.get("evaluation_transfer_basis") == _DIRECT_EVALUATION_TRANSFER
    ) or (
        not direct_evaluation
        and isinstance(candidate, dict)
        and candidate.get("evaluated_inference_revision") == _EVALUATED_BEAM_INFERENCE_REVISION
        and candidate.get("evaluation_transfer_basis") == _SOURCE_REBIND_EVALUATION_TRANSFER
    )
    if (
        not isinstance(candidate, dict)
        or set(candidate) != candidate_fields
        or candidate["release_inference_revision"]
        != _require_sha256(expected_inference_revision, "expected inference revision")
        or candidate["runtime_revision"] != "s1-pytorch-beam-runtime/1"
        or type(candidate["selected_epoch"]) is not int
        or candidate["selected_epoch"] != 20
        or not transfer_is_valid
    ):
        raise S1ReleaseEvidenceError("selection release candidate differs")
    for name in (
        "evaluated_inference_revision",
        "selected_model_state_sha256",
        "tokenizer_model_sha256",
        "tokenizer_record_sha256",
    ):
        _require_sha256(candidate[name], f"selection {name}")

    decoder = report["decoder_selection"]
    decoder_fields = {
        "profiles",
        "results",
        "paired_counts",
        "candidate_minus_baseline",
        "relative_improvement_over_baseline",
        "selected_profile",
    }
    expected_profiles = [
        "published-greedy-max128-eight-word-cap",
        "beam2-max24-no-repeat3-eight-word-cap",
    ]
    if (
        not isinstance(decoder, dict)
        or set(decoder) != decoder_fields
        or decoder["profiles"] != expected_profiles
        or decoder["selected_profile"] != expected_profiles[1]
        or not isinstance(decoder["results"], list)
        or len(decoder["results"]) != 2
    ):
        raise S1ReleaseEvidenceError("decoder selection differs")
    decoder_scores = [
        _validate_summary_result(
            row,
            identity_key="arm",
            expected_identity=arm,
            sample_count=_SAMPLE_COUNT,
        )
        for row, arm in zip(decoder["results"], ("published_greedy", "candidate_beam"), strict=True)
    ]
    delta = decoder_scores[1] - decoder_scores[0]
    if decoder_scores[0] <= 0:
        raise S1ReleaseEvidenceError("decoder baseline score must be positive")
    counts = decoder["paired_counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != {"candidate_better", "equal", "candidate_worse"}
        or any(type(item) is not int or item < 0 for item in counts.values())
        or sum(cast(dict[str, int], counts).values()) != _SAMPLE_COUNT
        or _fraction(decoder["candidate_minus_baseline"], "decoder delta") != delta
        or _fraction(decoder["relative_improvement_over_baseline"], "decoder relative improvement")
        != delta / decoder_scores[0]
        or delta <= 0
    ):
        raise S1ReleaseEvidenceError("decoder aggregate comparison differs")

    checkpoint = report["checkpoint_selection"]
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint) != {"rule", "results", "selected_epoch", "selected_model_state_sha256"}
        or checkpoint["rule"] != "highest-exact-validation-mean-then-lowest-epoch"
        or not isinstance(checkpoint["results"], list)
        or len(checkpoint["results"]) != len(_CHECKPOINT_EPOCHS)
    ):
        raise S1ReleaseEvidenceError("checkpoint selection differs")
    checkpoint_scores: list[tuple[Fraction, int]] = []
    for row, epoch in zip(checkpoint["results"], _CHECKPOINT_EPOCHS, strict=True):
        score = _validate_summary_result(
            row,
            identity_key="epoch",
            expected_identity=epoch,
            sample_count=_SAMPLE_COUNT,
        )
        checkpoint_scores.append((score, epoch))
    best_score, best_epoch = max(checkpoint_scores, key=lambda item: (item[0], -item[1]))
    best_state = _require_sha256(
        checkpoint["selected_model_state_sha256"], "selected checkpoint state"
    )
    if (
        checkpoint["selected_epoch"] != best_epoch
        or checkpoint["selected_model_state_sha256"] != best_state
        or best_epoch != candidate["selected_epoch"]
        or best_state != candidate["selected_model_state_sha256"]
        or best_score != decoder_scores[1]
    ):
        raise S1ReleaseEvidenceError("checkpoint aggregate winner differs")

    guidance = report["guidance_selection"]
    if (
        not isinstance(guidance, dict)
        or set(guidance)
        != {
            "rule",
            "results",
            "selected_scale_index",
            "scale_zero_matches_portable_decoder",
            "scale_zero_match_count",
        }
        or guidance["rule"] != "highest-exact-validation-mean-then-lowest-scale"
        or guidance["scale_zero_matches_portable_decoder"] is not True
        or type(guidance["scale_zero_match_count"]) is not int
        or guidance["scale_zero_match_count"] != _SAMPLE_COUNT
        or not isinstance(guidance["results"], list)
        or len(guidance["results"]) != 7
    ):
        raise S1ReleaseEvidenceError("guidance selection differs")
    guidance_scores: list[tuple[Fraction, int]] = []
    for index, row in enumerate(guidance["results"]):
        if not isinstance(row, dict) or set(row) != {
            "scale_index",
            "scale",
            "sample_count",
            "mean_wer",
            "mean_normalized_score",
            "unique_output_count",
        }:
            raise S1ReleaseEvidenceError("guidance aggregate result differs")
        scale = row["scale"]
        if (
            not isinstance(scale, dict)
            or set(scale) != {"hexadecimal", "ieee754_binary64_be"}
            or not isinstance(scale["hexadecimal"], str)
            or not isinstance(scale["ieee754_binary64_be"], str)
            or re.fullmatch(r"[0-9a-f]{16}", scale["ieee754_binary64_be"]) is None
            or scale != _GUIDANCE_SCALES[index]
        ):
            raise S1ReleaseEvidenceError("guidance scale differs")
        summary = dict(row)
        del summary["scale"]
        score = _validate_summary_result(
            summary,
            identity_key="scale_index",
            expected_identity=index,
            sample_count=_SAMPLE_COUNT,
        )
        guidance_scores.append((score, index))
    guidance_best = max(guidance_scores, key=lambda item: (item[0], -item[1]))
    if (
        guidance["selected_scale_index"] != guidance_best[1]
        or guidance_best[1] != 0
        or guidance_best[0] != decoder_scores[1]
    ):
        raise S1ReleaseEvidenceError("guidance aggregate winner differs")
    return supplied


def load_selection_ledger(path: Path, *, expected_inference_revision: str) -> dict[str, Any]:
    report, _raw = _strict_json(
        path, label="public selection ledger", maximum_bytes=_MAXIMUM_PUBLIC_REPORT_BYTES
    )
    validate_selection_ledger(report, expected_inference_revision=expected_inference_revision)
    return report


def load_selection_ledger_bytes(raw: bytes, *, expected_inference_revision: str) -> dict[str, Any]:
    report = _strict_json_bytes(
        raw, label="public selection ledger", maximum_bytes=_MAXIMUM_PUBLIC_REPORT_BYTES
    )
    validate_selection_ledger(report, expected_inference_revision=expected_inference_revision)
    return report


def _prediction_score(row: object, label: str) -> Fraction:
    if not isinstance(row, dict):
        raise S1ReleaseEvidenceError(f"{label} prediction is invalid")
    return _fraction(row.get("normalized_score"), f"{label} score")


def _paired_counts(
    left: list[object], right: list[object], *, left_name: str, right_name: str
) -> dict[str, int]:
    if len(left) != len(right):
        raise S1ReleaseEvidenceError("motion paired inventories differ")
    left_better = equal = right_better = 0
    for index, (left_row, right_row) in enumerate(zip(left, right, strict=True)):
        left_score = _prediction_score(left_row, f"{left_name} {index}")
        right_score = _prediction_score(right_row, f"{right_name} {index}")
        if left_score > right_score:
            left_better += 1
        elif left_score == right_score:
            equal += 1
        else:
            right_better += 1
    return {
        f"{left_name}_better": left_better,
        "equal": equal,
        f"{right_name}_better": right_better,
    }


def project_motion_ablation(
    private_report_path: Path,
    *,
    release_inference_revision: str,
) -> dict[str, Any]:
    release_revision = _require_sha256(release_inference_revision, "release inference revision")
    private, file_digest = _private_report(
        private_report_path, "umi-s1-validation-motion-ablation/2"
    )
    if (
        private.get("partition") != "fleurs_val"
        or private.get("sample_count") != _SAMPLE_COUNT
        or private.get("test_samples_loaded") is not False
        or not _private_test_use_is_validation_only(private)
        or not isinstance(private.get("source"), dict)
        or not isinstance(private.get("implementation"), dict)
    ):
        raise S1ReleaseEvidenceError("motion ablation is not regenerated strict evidence")
    runtime = private.get("runtime")
    device_identity = runtime.get("device_identity") if isinstance(runtime, dict) else None
    if (
        not isinstance(runtime, dict)
        or runtime.get("resolved_device") != "mps"
        or not isinstance(device_identity, dict)
        or device_identity.get("backend") != "mps"
        or device_identity.get("machine") != "arm64"
    ):
        raise S1ReleaseEvidenceError("motion ablation runtime is not MPS/ARM64")
    conditions = private.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 3:
        raise S1ReleaseEvidenceError("motion ablation condition inventory differs")
    sample_keys, deranged_donors = _validate_private_motion_inventory(private)
    motion_contract = _beam_contract(private.get("decoder"), "motion ablation")
    checkpoint = private.get("checkpoint")
    tokenizer = private.get("tokenizer")
    model_config = checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
    vocabulary_size = (
        model_config.get("vocabulary_size") if isinstance(model_config, dict) else None
    )
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(tokenizer, dict)
        or type(vocabulary_size) is not int
        or not 3 <= vocabulary_size <= 65536
    ):
        raise S1ReleaseEvidenceError("motion ablation model identity is missing")
    public_conditions: list[dict[str, Any]] = []
    private_predictions: dict[str, list[object]] = {}
    reference_word_counts: list[int] | None = None
    for row, name in zip(conditions, _CONDITION_ORDER, strict=True):
        _validate_private_motion_prediction_integrity(
            row,
            label=f"motion condition {name}",
            vocabulary_size=vocabulary_size,
        )
        _mean_wer, _mean_score, _unique, verified_predictions = _recompute_private_result(
            row,
            sample_keys=sample_keys,
            label=f"motion condition {name}",
        )
        for prediction, (sample_id, _reference, _word_count) in zip(
            verified_predictions, sample_keys, strict=True
        ):
            expected_motion_sample_id = (
                deranged_donors[sample_id] if name == "deranged_motion" else sample_id
            )
            if prediction.get("motion_sample_id") != expected_motion_sample_id:
                raise S1ReleaseEvidenceError(
                    f"motion condition {name} uses the wrong motion sample"
                )
        observed_word_counts = [
            cast(int, prediction["reference_word_count"]) for prediction in verified_predictions
        ]
        if reference_word_counts is None:
            reference_word_counts = observed_word_counts
        elif observed_word_counts != reference_word_counts:
            raise S1ReleaseEvidenceError("motion conditions use different reference word counts")
        summary = _aggregate_result(
            row,
            identity_key="condition",
            identity_value=name,
            sample_count=_SAMPLE_COUNT,
        )
        private_predictions[name] = cast(list[object], verified_predictions)
        public_conditions.append(summary)
    real_score, zero_score, deranged_score = [
        _result_score(row, f"{row['condition']} score") for row in public_conditions
    ]
    if not zero_score > real_score > deranged_score:
        raise S1ReleaseEvidenceError("motion ablation ordering differs from release disclosure")
    effects = {
        "real_minus_zero": _fraction_record(real_score - zero_score),
        "real_minus_deranged": _fraction_record(real_score - deranged_score),
    }
    private_effects = private.get("effects")
    if not isinstance(private_effects, dict) or any(
        _fraction(private_effects.get(name), f"private {name}")
        != _fraction(value, f"public {name}")
        for name, value in effects.items()
    ):
        raise S1ReleaseEvidenceError("motion ablation effects do not reproduce")
    if motion_contract["normalization_revision"] != tokenizer.get("normalization_revision"):
        raise S1ReleaseEvidenceError("motion ablation tokenizer contract differs")
    report: dict[str, Any] = {
        "schema": MOTION_ABLATION_SCHEMA,
        "release_id": _RELEASE_ID,
        "purpose": "aggregate post-test-open validation motion diagnostic",
        "partition": "fleurs_val",
        "sample_count": _SAMPLE_COUNT,
        "private_report": _binding("motion_ablation", private, file_digest),
        "release_candidate": {
            "release_inference_revision": release_revision,
            "selected_model_state_sha256": _require_sha256(
                checkpoint.get("model_state_sha256"), "motion ablation model state"
            ),
            "tokenizer_model_sha256": _require_sha256(
                tokenizer.get("model_sha256"), "motion ablation tokenizer model"
            ),
            "tokenizer_record_sha256": _require_sha256(
                tokenizer.get("record_sha256"), "motion ablation tokenizer record"
            ),
            "decoder_profile": "beam2-max24-no-repeat3-eight-word-cap",
        },
        "evaluation_context": {
            "test_partition_already_open": True,
            "test_evaluation_invocation_count": 0,
            "inference_backend": "mps",
            "machine_architecture": "arm64",
            "input_materialization": "bound-arm64-validation-motion-tensors",
            "linux_amd64_end_to_end_quality_evidence": False,
        },
        "conditions": public_conditions,
        "effects": effects,
        "paired_counts": {
            "real_vs_zero": _paired_counts(
                private_predictions["real_motion"],
                private_predictions["zero_motion"],
                left_name="real",
                right_name="zero",
            ),
            "real_vs_deranged": _paired_counts(
                private_predictions["real_motion"],
                private_predictions["deranged_motion"],
                left_name="real",
                right_name="deranged",
            ),
        },
        "conclusions": {
            "zero_motion_has_higher_aggregate_score_than_real_motion": True,
            "real_motion_has_higher_aggregate_score_than_deranged_motion": True,
            "useful_motion_grounding_established": False,
        },
        "public_data": {
            "aggregate_only": True,
            "source_rows_included": False,
            "per_example_outputs_included": False,
        },
        "claim_boundary": _MOTION_CLAIM_BOUNDARY,
    }
    report["content_sha256"] = canonical_json_sha256(report, domain=_MOTION_DOMAIN)
    validate_motion_ablation(report, expected_inference_revision=release_inference_revision)
    return report


def validate_motion_ablation(
    value: object,
    *,
    expected_inference_revision: str,
) -> str:
    report, supplied = _verify_content(
        value,
        schema=MOTION_ABLATION_SCHEMA,
        domain=_MOTION_DOMAIN,
        fields={
            "schema",
            "release_id",
            "purpose",
            "partition",
            "sample_count",
            "private_report",
            "release_candidate",
            "evaluation_context",
            "conditions",
            "effects",
            "paired_counts",
            "conclusions",
            "public_data",
            "claim_boundary",
            "content_sha256",
        },
    )
    if (
        report["release_id"] != _RELEASE_ID
        or report["purpose"] != "aggregate post-test-open validation motion diagnostic"
        or report["partition"] != "fleurs_val"
        or type(report["sample_count"]) is not int
        or report["sample_count"] != _SAMPLE_COUNT
        or report["claim_boundary"] != _MOTION_CLAIM_BOUNDARY
        or not _strict_equal(
            report["evaluation_context"],
            {
                "test_partition_already_open": True,
                "test_evaluation_invocation_count": 0,
                "inference_backend": "mps",
                "machine_architecture": "arm64",
                "input_materialization": "bound-arm64-validation-motion-tensors",
                "linux_amd64_end_to_end_quality_evidence": False,
            },
        )
        or not _strict_equal(
            report["public_data"],
            {
                "aggregate_only": True,
                "source_rows_included": False,
                "per_example_outputs_included": False,
            },
        )
    ):
        raise S1ReleaseEvidenceError("motion ablation identity differs")
    _validate_binding(
        report["private_report"],
        "motion_ablation",
        "umi-s1-validation-motion-ablation/2",
    )
    candidate = report["release_candidate"]
    if (
        not isinstance(candidate, dict)
        or set(candidate)
        != {
            "release_inference_revision",
            "selected_model_state_sha256",
            "tokenizer_model_sha256",
            "tokenizer_record_sha256",
            "decoder_profile",
        }
        or candidate["release_inference_revision"]
        != _require_sha256(expected_inference_revision, "expected inference revision")
        or candidate["decoder_profile"] != "beam2-max24-no-repeat3-eight-word-cap"
    ):
        raise S1ReleaseEvidenceError("motion release candidate differs")
    for name in (
        "selected_model_state_sha256",
        "tokenizer_model_sha256",
        "tokenizer_record_sha256",
    ):
        _require_sha256(candidate[name], f"motion {name}")
    conditions = report["conditions"]
    if not isinstance(conditions, list) or len(conditions) != 3:
        raise S1ReleaseEvidenceError("motion condition inventory differs")
    scores = [
        _validate_summary_result(
            row,
            identity_key="condition",
            expected_identity=name,
            sample_count=_SAMPLE_COUNT,
        )
        for row, name in zip(conditions, _CONDITION_ORDER, strict=True)
    ]
    real_score, zero_score, deranged_score = scores
    effects = report["effects"]
    if (
        not isinstance(effects, dict)
        or set(effects) != {"real_minus_zero", "real_minus_deranged"}
        or _fraction(effects["real_minus_zero"], "real minus zero") != real_score - zero_score
        or _fraction(effects["real_minus_deranged"], "real minus deranged")
        != real_score - deranged_score
        or not zero_score > real_score > deranged_score
    ):
        raise S1ReleaseEvidenceError("motion aggregate effects differ")
    paired = report["paired_counts"]
    expected_count_fields = {
        "real_vs_zero": {"real_better", "equal", "zero_better"},
        "real_vs_deranged": {"real_better", "equal", "deranged_better"},
    }
    if not isinstance(paired, dict) or set(paired) != set(expected_count_fields):
        raise S1ReleaseEvidenceError("motion paired counts differ")
    for comparison, fields in expected_count_fields.items():
        counts = paired[comparison]
        if (
            not isinstance(counts, dict)
            or set(counts) != fields
            or any(type(item) is not int or item < 0 for item in counts.values())
            or sum(cast(dict[str, int], counts).values()) != _SAMPLE_COUNT
        ):
            raise S1ReleaseEvidenceError("motion paired counts differ")
    if not _strict_equal(
        report["conclusions"],
        {
            "zero_motion_has_higher_aggregate_score_than_real_motion": True,
            "real_motion_has_higher_aggregate_score_than_deranged_motion": True,
            "useful_motion_grounding_established": False,
        },
    ):
        raise S1ReleaseEvidenceError("motion conclusions differ")
    return supplied


def load_motion_ablation(path: Path, *, expected_inference_revision: str) -> dict[str, Any]:
    report, _raw = _strict_json(
        path,
        label="public motion ablation",
        maximum_bytes=_MAXIMUM_PUBLIC_REPORT_BYTES,
    )
    validate_motion_ablation(report, expected_inference_revision=expected_inference_revision)
    return report


def load_motion_ablation_bytes(raw: bytes, *, expected_inference_revision: str) -> dict[str, Any]:
    report = _strict_json_bytes(
        raw, label="public motion ablation", maximum_bytes=_MAXIMUM_PUBLIC_REPORT_BYTES
    )
    validate_motion_ablation(report, expected_inference_revision=expected_inference_revision)
    return report


def project_rights_evidence(private_review_path: Path) -> dict[str, Any]:
    private, file_digest = _private_report(private_review_path, "umi-s1-final-rights-review/1")
    release_terms = private.get("release_terms")
    private_sources = private.get("sources")
    if not isinstance(release_terms, dict) or not isinstance(private_sources, list):
        raise S1ReleaseEvidenceError("private rights decision is incomplete")
    terms = {
        "review_class": release_terms.get("review_class"),
        "runtime_code_license": release_terms.get("runtime_code_license"),
        "weight_license": release_terms.get("weight_license"),
        "public_weight_redistribution_approved": release_terms.get(
            "public_weight_redistribution_approved"
        ),
        "raw_source_data_redistribution_approved": release_terms.get(
            "raw_source_data_redistribution_approved"
        ),
        "source_annotations_redistribution_approved": release_terms.get(
            "source_annotations_redistribution_approved"
        ),
        "mediapipe_task_redistribution_approved": release_terms.get(
            "mediapipe_task_redistribution_approved"
        ),
    }
    sources: list[dict[str, Any]] = []
    for source in private_sources:
        if not isinstance(source, dict):
            raise S1ReleaseEvidenceError("private rights source is invalid")
        sources.append(
            {
                "source_id": source.get("source_id"),
                "source_version": source.get("source_version"),
                "license_id": source.get("license_id"),
                "public_data_eligible": source.get("public_data_eligible"),
                "public_weight_eligible": source.get("public_weight_eligible"),
                "public_data_release": source.get("public_data_release"),
                "raw_data_release": source.get("raw_data_release"),
            }
        )
    report: dict[str, Any] = {
        "schema": RIGHTS_EVIDENCE_SCHEMA,
        "release_id": _RELEASE_ID,
        "decision_authority": private.get("decision_authority"),
        "private_review": _binding("final_rights_review", private, file_digest),
        "release_terms": terms,
        "sources": sources,
        "claim_boundary": _RIGHTS_CLAIM_BOUNDARY,
    }
    report["content_sha256"] = canonical_json_sha256(report, domain=_RIGHTS_DOMAIN)
    validate_rights_evidence(report)
    return report


def validate_rights_evidence(value: object) -> str:
    report, supplied = _verify_content(
        value,
        schema=RIGHTS_EVIDENCE_SCHEMA,
        domain=_RIGHTS_DOMAIN,
        fields={
            "schema",
            "release_id",
            "decision_authority",
            "private_review",
            "release_terms",
            "sources",
            "claim_boundary",
            "content_sha256",
        },
    )
    if (
        report["release_id"] != _RELEASE_ID
        or report["decision_authority"] != "project-owner-release-decision/1"
        or report["claim_boundary"] != _RIGHTS_CLAIM_BOUNDARY
    ):
        raise S1ReleaseEvidenceError("public rights identity differs")
    _validate_binding(
        report["private_review"],
        "final_rights_review",
        "umi-s1-final-rights-review/1",
    )
    terms = report["release_terms"]
    if not _strict_equal(
        terms,
        {
            "review_class": "project-release-decision-not-legal-opinion",
            "runtime_code_license": "Apache-2.0",
            "weight_license": "CC-BY-SA-4.0",
            "public_weight_redistribution_approved": True,
            "raw_source_data_redistribution_approved": False,
            "source_annotations_redistribution_approved": False,
            "mediapipe_task_redistribution_approved": False,
        },
    ):
        raise S1ReleaseEvidenceError("public rights terms differ")
    sources = report["sources"]
    source_fields = {
        "source_id",
        "source_version",
        "license_id",
        "public_data_eligible",
        "public_weight_eligible",
        "public_data_release",
        "raw_data_release",
    }
    if not isinstance(sources, list) or len(sources) != len(_RIGHTS_SOURCE_DECISIONS):
        raise S1ReleaseEvidenceError("public rights sources are missing")
    identities: list[tuple[str, str]] = []
    for source in sources:
        if (
            not isinstance(source, dict)
            or set(source) != source_fields
            or any(
                not isinstance(source[name], str)
                for name in ("source_id", "source_version", "license_id")
            )
            or source["public_data_eligible"] is not False
            or source["public_weight_eligible"] is not True
            or source["public_data_release"] is not False
            or source["raw_data_release"] is not False
        ):
            raise S1ReleaseEvidenceError("public rights source differs")
        identities.append((source["source_id"], source["source_version"]))
    if identities != sorted(set(identities)):
        raise S1ReleaseEvidenceError("public rights sources are not sorted and unique")
    if not _strict_equal(sources, list(_RIGHTS_SOURCE_DECISIONS)):
        raise S1ReleaseEvidenceError("public rights source inventory differs")
    return supplied


def load_rights_evidence(path: Path) -> dict[str, Any]:
    report, _raw = _strict_json(
        path, label="public rights evidence", maximum_bytes=_MAXIMUM_PUBLIC_REPORT_BYTES
    )
    validate_rights_evidence(report)
    return report


def load_rights_evidence_bytes(raw: bytes) -> dict[str, Any]:
    report = _strict_json_bytes(
        raw, label="public rights evidence", maximum_bytes=_MAXIMUM_PUBLIC_REPORT_BYTES
    )
    validate_rights_evidence(report)
    return report


def _validate_e2e_common(report: dict[str, Any]) -> None:
    if report.get("release_id") != _RELEASE_ID or report.get("status") != "passed":
        raise S1ReleaseEvidenceError("release E2E identity differs")
    base = _require_sha256(report.get("base_inference_revision"), "E2E base inference")
    derived = _require_sha256(report.get("derived_inference_revision"), "E2E derived inference")
    if base == derived:
        raise S1ReleaseEvidenceError("release E2E did not bind a local extractor image")
    _require_git_revision(report.get("umi_git_revision"), "E2E UMI revision")
    _require_git_revision(
        report.get("tested_reference_model_git_revision"),
        "E2E tested reference-model revision",
    )
    started = _parse_utc_timestamp(report.get("started_at_utc"), "E2E start")
    finished = _parse_utc_timestamp(report.get("finished_at_utc"), "E2E finish")
    if finished <= started:
        raise S1ReleaseEvidenceError("release E2E timestamps differ")
    runtime = report.get("runtime")
    runtime_fields = {
        "host_operating_system",
        "host_architecture",
        "container_platform",
        "model_device",
        "python_version",
        "torch_version",
        "numpy_version",
        "safetensors_version",
        "bittensor_version",
        "docker_engine_version",
        "extractor_image_id",
        "mediapipe_task_model_sha256",
    }
    version_fields = (
        "python_version",
        "torch_version",
        "numpy_version",
        "safetensors_version",
        "bittensor_version",
    )
    if (
        not isinstance(runtime, dict)
        or set(runtime) != runtime_fields
        or any(not isinstance(item, str) or not item for item in runtime.values())
        or any(_NUMERIC_VERSION.fullmatch(runtime[name]) is None for name in version_fields)
        or _DOCKER_VERSION.fullmatch(runtime["docker_engine_version"]) is None
        or runtime["host_operating_system"] != "Linux"
        or runtime["host_architecture"] != "x86_64"
        or runtime["container_platform"] != "linux/amd64"
        or runtime["model_device"] != "cpu"
        or runtime["mediapipe_task_model_sha256"] != _TASK_MODEL_SHA256
        or re.fullmatch(r"sha256:[0-9a-f]{64}", runtime["extractor_image_id"]) is None
    ):
        raise S1ReleaseEvidenceError("release E2E runtime differs")
    required_execution = {
        "request_count": 2,
        "valid_video_status": "ok",
        "invalid_video_status": "error",
        "invalid_video_error_code": "inference_failed",
        "signed_envelopes_verified": True,
        "plaintext_unavailable_before_reveal": True,
        "decryption_after_reveal_verified": True,
        "response_bindings_verified": True,
        "maximum_output_words": 8,
        "output_limit_passed": True,
        "temporary_job_cleanup_passed": True,
        "extractor_container_cleanup_passed": True,
        "video_fixture_distributed": False,
        "translation_quality_measured": False,
    }
    if not _strict_equal(report.get("execution"), required_execution):
        raise S1ReleaseEvidenceError("release E2E execution result differs")
    if not _strict_equal(
        report.get("timeouts_seconds"),
        {
            "inner_model_hard_deadline": 150,
            "outer_inference_timeout": 180,
            "outer_admission_timeout": 10,
            "outer_lifecycle_timeout": 60,
        },
    ):
        raise S1ReleaseEvidenceError("release E2E timeout settings differ")


def validate_private_release_e2e(value: object) -> str:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "release_id",
        "status",
        "base_inference_revision",
        "derived_inference_revision",
        "tested_reference_model_git_revision",
        "umi_git_revision",
        "started_at_utc",
        "finished_at_utc",
        "runtime",
        "timeouts_seconds",
        "fixture",
        "execution",
        "evidence",
        "content_sha256",
    }:
        raise S1ReleaseEvidenceError("private release E2E field set differs")
    if value["schema"] != RELEASE_E2E_RUN_SCHEMA:
        raise S1ReleaseEvidenceError("private release E2E schema differs")
    supplied = _require_sha256(value["content_sha256"], "private E2E content")
    unsigned = dict(value)
    del unsigned["content_sha256"]
    if canonical_json_sha256(unsigned, domain=_PRIVATE_E2E_DOMAIN) != supplied:
        raise S1ReleaseEvidenceError("private release E2E content digest differs")
    _validate_e2e_common(value)
    fixture = value["fixture"]
    if (
        not isinstance(fixture, dict)
        or set(fixture)
        != {
            "fixture_class",
            "video_sha256",
            "rights_cleared_for_private_testing",
            "distributed",
        }
        or fixture["fixture_class"] != "rights-cleared-private-video"
        or fixture["rights_cleared_for_private_testing"] is not True
        or fixture["distributed"] is not False
    ):
        raise S1ReleaseEvidenceError("private release E2E fixture differs")
    _require_sha256(fixture["video_sha256"], "private E2E video")
    bindings = value["evidence"]
    required_bindings = {
        "extractor_build_record_content_sha256",
        "extractor_build_record_file_sha256",
        "valid_wire_response_sha256",
        "invalid_wire_response_sha256",
        "post_reveal_plaintext_set_sha256",
        "run_log_sha256",
    }
    if not isinstance(bindings, dict) or set(bindings) != required_bindings:
        raise S1ReleaseEvidenceError("private release E2E bindings differ")
    for name, digest in bindings.items():
        _require_sha256(digest, f"private E2E {name}")
    return supplied


def seal_private_release_e2e(capture: object) -> dict[str, Any]:
    if not isinstance(capture, dict) or "schema" in capture or "content_sha256" in capture:
        raise S1ReleaseEvidenceError("release E2E capture must be an unsealed object")
    report = {"schema": RELEASE_E2E_RUN_SCHEMA, **capture}
    report["content_sha256"] = canonical_json_sha256(report, domain=_PRIVATE_E2E_DOMAIN)
    validate_private_release_e2e(report)
    return report


def build_unsealed_release_e2e_capture(
    *,
    base_inference_revision: str,
    derived_inference_revision: str,
    tested_reference_model_git_revision: str,
    umi_git_revision: str,
    started_at_utc: str,
    finished_at_utc: str,
    runtime: dict[str, Any],
    timeouts_seconds: dict[str, int],
    fixture: dict[str, Any],
    execution: dict[str, Any],
    evidence: dict[str, str],
) -> dict[str, Any]:
    """Build and validate the exact unsealed capture consumed by the sealing CLI."""

    capture: dict[str, Any] = {
        "release_id": _RELEASE_ID,
        "status": "passed",
        "base_inference_revision": base_inference_revision,
        "derived_inference_revision": derived_inference_revision,
        "tested_reference_model_git_revision": tested_reference_model_git_revision,
        "umi_git_revision": umi_git_revision,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "runtime": dict(runtime),
        "timeouts_seconds": dict(timeouts_seconds),
        "fixture": dict(fixture),
        "execution": dict(execution),
        "evidence": dict(evidence),
    }
    seal_private_release_e2e(capture)
    return capture


def project_release_e2e(private_run_path: Path) -> dict[str, Any]:
    private, file_digest = _private_report(private_run_path, RELEASE_E2E_RUN_SCHEMA)
    validate_private_release_e2e(private)
    report = {
        "schema": RELEASE_E2E_SCHEMA,
        "release_id": private["release_id"],
        "status": private["status"],
        "base_inference_revision": private["base_inference_revision"],
        "derived_inference_revision": private["derived_inference_revision"],
        "tested_reference_model_git_revision": private["tested_reference_model_git_revision"],
        "umi_git_revision": private["umi_git_revision"],
        "private_run": _binding("release_e2e", private, file_digest),
        "test_fixture": {
            "fixture_class": "rights-cleared-private-video",
            "distributed": False,
            "source_identity_published": False,
        },
        "started_at_utc": private["started_at_utc"],
        "finished_at_utc": private["finished_at_utc"],
        "runtime": private["runtime"],
        "timeouts_seconds": private["timeouts_seconds"],
        "execution": private["execution"],
        "claim_boundary": _E2E_CLAIM_BOUNDARY,
    }
    report["content_sha256"] = canonical_json_sha256(report, domain=_E2E_DOMAIN)
    validate_release_e2e(report)
    return report


def validate_release_e2e(value: object) -> str:
    report, supplied = _verify_content(
        value,
        schema=RELEASE_E2E_SCHEMA,
        domain=_E2E_DOMAIN,
        fields={
            "schema",
            "release_id",
            "status",
            "base_inference_revision",
            "derived_inference_revision",
            "tested_reference_model_git_revision",
            "umi_git_revision",
            "private_run",
            "test_fixture",
            "started_at_utc",
            "finished_at_utc",
            "runtime",
            "timeouts_seconds",
            "execution",
            "claim_boundary",
            "content_sha256",
        },
    )
    if report["claim_boundary"] != _E2E_CLAIM_BOUNDARY:
        raise S1ReleaseEvidenceError("release E2E claim boundary differs")
    _validate_e2e_common(report)
    _validate_binding(report["private_run"], "release_e2e", RELEASE_E2E_RUN_SCHEMA)
    if not _strict_equal(
        report["test_fixture"],
        {
            "fixture_class": "rights-cleared-private-video",
            "distributed": False,
            "source_identity_published": False,
        },
    ):
        raise S1ReleaseEvidenceError("release E2E test-fixture disclosure differs")
    return supplied


def load_release_e2e(path: Path) -> dict[str, Any]:
    report, _raw = _strict_json(
        path, label="public release E2E evidence", maximum_bytes=_MAXIMUM_PUBLIC_REPORT_BYTES
    )
    validate_release_e2e(report)
    return report


def load_release_e2e_bytes(raw: bytes) -> dict[str, Any]:
    report = _strict_json_bytes(
        raw, label="public release E2E evidence", maximum_bytes=_MAXIMUM_PUBLIC_REPORT_BYTES
    )
    validate_release_e2e(report)
    return report


def _write_exclusive(path: Path, report: dict[str, Any], *, mode: int = 0o644) -> None:
    payload = canonical_json_bytes(report)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            mode,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        raise S1ReleaseEvidenceError(f"cannot publish release evidence: {path.name}") from exc


def _load_capture(path: Path) -> dict[str, Any]:
    capture, _raw = _strict_json(
        path, label="release E2E capture", maximum_bytes=_MAXIMUM_PUBLIC_REPORT_BYTES
    )
    return capture


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Project private S1 evidence into aggregate-only public release records"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    selection = subparsers.add_parser("selection")
    selection.add_argument("--decoder-report", type=Path, required=True)
    selection.add_argument("--checkpoint-report", type=Path, required=True)
    selection.add_argument("--guidance-report", type=Path, required=True)
    selection.add_argument("--release-inference-revision", required=True)
    selection.add_argument("--output", type=Path, required=True)

    motion = subparsers.add_parser("motion-ablation")
    motion.add_argument("--report", type=Path, required=True)
    motion.add_argument("--release-inference-revision", required=True)
    motion.add_argument("--output", type=Path, required=True)

    rights = subparsers.add_parser("rights")
    rights.add_argument("--review", type=Path, required=True)
    rights.add_argument("--output", type=Path, required=True)

    e2e_run = subparsers.add_parser("seal-e2e-run")
    e2e_run.add_argument("--capture", type=Path, required=True)
    e2e_run.add_argument("--output", type=Path, required=True)

    e2e = subparsers.add_parser("release-e2e")
    e2e.add_argument("--run-report", type=Path, required=True)
    e2e.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "selection":
            report = project_selection_ledger(
                arguments.decoder_report,
                arguments.checkpoint_report,
                arguments.guidance_report,
                release_inference_revision=arguments.release_inference_revision,
            )
        elif arguments.command == "motion-ablation":
            report = project_motion_ablation(
                arguments.report,
                release_inference_revision=arguments.release_inference_revision,
            )
        elif arguments.command == "rights":
            report = project_rights_evidence(arguments.review)
        elif arguments.command == "seal-e2e-run":
            report = seal_private_release_e2e(_load_capture(arguments.capture))
        else:
            report = project_release_e2e(arguments.run_report)
        _write_exclusive(
            arguments.output,
            report,
            mode=0o600 if arguments.command == "seal-e2e-run" else 0o644,
        )
    except (OSError, S1ReleaseEvidenceError) as exc:
        print(f"release evidence failed: {exc}", file=sys.stderr)
        return 2
    print(report["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
