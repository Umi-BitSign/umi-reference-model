from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import io
import os
import re
import stat
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Final, cast

from . import s1_portable_runtime as portable
from .canonical import CanonicalJSONError, canonical_json_bytes, canonical_json_sha256
from .s1_portable_runtime import S1PortableError, load_s1_portable_bundle

PUBLIC_S1_FINETUNE_RELEASE_IDENTITY_SCHEMA: Final = "umi-public-s1-finetune-release-identity/1"
PUBLIC_S1_FINETUNE_RELEASE_REVIEW_SCHEMA: Final = "umi-public-s1-finetune-release-review/1"
PUBLIC_S1_FINETUNE_RELEASE_MANIFEST_SCHEMA: Final = "umi-public-s1-finetune-release/1"
PUBLIC_S1_FINETUNE_INTAKE_POLICY_SCHEMA: Final = "umi-s1-public-finetune-intake-policy/1"
PUBLIC_S1_FINETUNE_INTAKE_SCHEMA: Final = "umi-s1-public-finetune-intake/1"
PUBLIC_S1_FINETUNE_INTAKE_RESULT_SCHEMA: Final = "umi-s1-public-finetune-intake-result/1"

PUBLIC_S1_FINETUNE_EVIDENCE_FILENAME: Final = "umi-s1-public-finetune-v1-evidence.json"
PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME: Final = "umi-s1-public-finetune-v1-portable.zip"

_IDENTITY_DOMAIN: Final = b"umi-public-s1-finetune-release-identity-v1\0"
_REVIEW_DOMAIN: Final = b"umi-public-s1-finetune-release-review-v1\0"
_MANIFEST_DOMAIN: Final = b"umi-public-s1-finetune-release-v1\0"
_POLICY_DOMAIN: Final = b"umi-s1-public-finetune-intake-policy-v1\0"
_INTAKE_DOMAIN: Final = b"umi-s1-public-finetune-intake-v1\0"
_EXPECTED_SOURCE_ROOT: Final = (
    "portable",
    "release-identity.json",
    "release-manifest.json",
    "rights-review.json",
)
_EXPECTED_INTAKE_ROOT: Final = (
    PUBLIC_S1_FINETUNE_EVIDENCE_FILENAME,
    PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME,
)
_EXPECTED_SOURCE_IDS: Final = (
    "facebook/2M-Flores-ASL",
    "fleurs-asl-v1",
    "fsboard-v3",
    "google-research-datasets/taskmaster/TM-1-2019",
)
_MAXIMUM_OUTER_JSON_BYTES: Final = 2 * 1024 * 1024
_MAXIMUM_POLICY_JSON_BYTES: Final = 1024 * 1024
_MAXIMUM_INTAKE_JSON_BYTES: Final = 1024 * 1024
_MAXIMUM_PORTABLE_BYTES: Final = 80 * 1024 * 1024
_MAXIMUM_ARCHIVE_BYTES: Final = _MAXIMUM_PORTABLE_BYTES + 1024 * 1024
_FIXED_ZIP_TIME: Final = (1980, 1, 1, 0, 0, 0)
_DIGEST: Final = re.compile(r"[0-9a-f]{64}")
_TIMESTAMP: Final = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_SOURCE_ROW_VALUE: Final = re.compile(
    r"(?:sample|signer|document|sentence)[_-][0-9]+", re.IGNORECASE
)
_FORBIDDEN_PUBLIC_KEYS: Final = frozenset(
    {
        "samples",
        "sample_id",
        "signer_id",
        "document_id",
        "sentence_id",
        "reference",
        "references",
        "reference_sha256",
        "motion_artifact_sha256",
        "motion_float32_sha256",
        "frame_mask_int32_sha256",
        "hypothesis",
        "raw_hypothesis",
        "normalized_hypothesis",
        "hypothesis_sha256",
        "token_ids",
        "generated_token_ids",
        "edit_distance",
        "predictions",
        "prediction_sha256",
        "prediction_set_sha256",
        "source_video",
        "source_annotations",
    }
)
_RIGHTS_DECISION_AUTHORITY: Final = "project-owner-release-decision/1"
_RIGHTS_REVIEW_CLASS: Final = "project-release-decision-not-legal-opinion"
_RUNTIME_CODE_LICENSE: Final = "Apache-2.0"
_WEIGHT_LICENSE: Final = "CC-BY-SA-4.0"
_RIGHTS_RECORD_CLAIM_BOUNDARY: Final = (
    "The portable artifact carries weights, tokenizer data, and readable attribution "
    "for every model source. It grants no right to redistribute source videos or "
    "annotations. Runtime code and model-weight distribution remain governed by every "
    "recorded source license and the final review state."
)
_TEXT_PRETRAINING_CLAIM: Final = (
    "Taskmaster was used only for decoder-side English language pretraining."
)
_RIGHTS_CLAIM_BOUNDARY: Final = (
    "Project weight-only release decision, not legal advice or a third-party license "
    "warranty. It does not release source video, annotations, conversations, "
    "validation references, predictions, or extraction model weights."
)
_INTAKE_CLAIM_BOUNDARY: Final = (
    "Aggregate-only public projection of a completed public-s1-finetune/1 release. It "
    "contains the release identity, owner rights decision, readable source attribution, "
    "and a deterministic archive of the six-file portable runtime. It contains no "
    "predictions, references, source rows, source video, source annotations, training "
    "checkpoints, or raw training data. The reported scores remain validation diagnostics, "
    "not untouched confirmation, UMI activation evidence, interpreter equivalence, "
    "accessibility certification, or expected production accuracy."
)


class PublicS1ReleaseIntakeError(ValueError):
    """Raised when a public S1 release cannot cross the public intake boundary."""


@dataclass(frozen=True, slots=True)
class _ValidatedPortable:
    payloads: dict[str, bytes]
    identity: dict[str, Any]
    inference_revision: str
    manifest_sha256: str
    size_bytes: int
    model_state_sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedSource:
    release_identity: dict[str, Any]
    release_identity_bytes: bytes
    rights_review: dict[str, Any]
    rights_review_bytes: bytes
    release_manifest: dict[str, Any]
    release_manifest_bytes: bytes
    portable: _ValidatedPortable


def _fraction(value: object, label: str) -> Fraction:
    record = _exact_dict(value, {"numerator", "denominator"}, label)
    numerator = record["numerator"]
    denominator = record["denominator"]
    if not isinstance(numerator, str) or not isinstance(denominator, str):
        raise PublicS1ReleaseIntakeError(f"{label} is not an exact rational")
    try:
        result = Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError) as exc:
        raise PublicS1ReleaseIntakeError(f"{label} is not an exact rational") from exc
    if record != {"numerator": str(result.numerator), "denominator": str(result.denominator)}:
        raise PublicS1ReleaseIntakeError(f"{label} is not reduced canonical rational form")
    return result


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise PublicS1ReleaseIntakeError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PublicS1ReleaseIntakeError(f"{label} has an unexpected field set")
    return value


def _direct_directory(
    path: Path,
    *,
    expected: tuple[str, ...],
    label: str,
    expected_identity: tuple[int, int] | None = None,
) -> tuple[Path, tuple[int, int]]:
    direct = Path(os.path.abspath(path))
    try:
        if direct.resolve(strict=True) != direct:
            raise PublicS1ReleaseIntakeError(f"{label} contains a symlink")
    except OSError as exc:
        raise PublicS1ReleaseIntakeError(f"{label} cannot be resolved safely") from exc
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = direct.lstat()
        descriptor = os.open(direct, flags)
    except OSError as exc:
        raise PublicS1ReleaseIntakeError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        identity = (opened.st_dev, opened.st_ino)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISDIR(opened.st_mode)
            or (before.st_dev, before.st_ino) != identity
            or (expected_identity is not None and identity != expected_identity)
            or tuple(sorted(os.listdir(descriptor))) != expected
        ):
            raise PublicS1ReleaseIntakeError(f"{label} inventory differs")
    finally:
        os.close(descriptor)
    return direct, identity


def _read_regular(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    direct = Path(os.path.abspath(path))
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        before = direct.lstat()
        descriptor = os.open(direct, flags)
    except OSError as exc:
        raise PublicS1ReleaseIntakeError(f"{label} cannot be opened safely") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
            or not 1 <= opened.st_size <= maximum_bytes
        ):
            raise PublicS1ReleaseIntakeError(f"{label} violates its file contract")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum_bytes:
                raise PublicS1ReleaseIntakeError(f"{label} exceeds its byte ceiling")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if observed != opened.st_size or (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise PublicS1ReleaseIntakeError(f"{label} changed while being read")
    return b"".join(chunks)


def _canonical_file(path: Path, *, label: str, maximum_bytes: int) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(path, maximum_bytes=maximum_bytes, label=label)
    try:
        record = portable._strict_json(payload, maximum_bytes=maximum_bytes, label=label)
    except S1PortableError as exc:
        raise PublicS1ReleaseIntakeError(f"{label} is not strict canonical JSON") from exc
    return record, payload


def _verify_content(
    record: Mapping[str, Any], *, schema: str, domain: bytes, fields: set[str], label: str
) -> str:
    value = _exact_dict(record, fields, label)
    if value["schema"] != schema:
        raise PublicS1ReleaseIntakeError(f"{label} schema differs")
    supplied = _require_digest(value["content_sha256"], f"{label} content")
    unsigned = dict(value)
    del unsigned["content_sha256"]
    if canonical_json_sha256(unsigned, domain=domain) != supplied:
        raise PublicS1ReleaseIntakeError(f"{label} content digest differs")
    return supplied


def _parse_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or _TIMESTAMP.fullmatch(value) is None:
        raise PublicS1ReleaseIntakeError(f"{label} must be RFC 3339 UTC without fractions")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise PublicS1ReleaseIntakeError(f"{label} is not a calendar-valid timestamp") from exc
    if parsed > datetime.now(UTC):
        raise PublicS1ReleaseIntakeError(f"{label} cannot be future-dated")
    return parsed


def _assert_aggregate_safe(value: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _FORBIDDEN_PUBLIC_KEYS:
                raise PublicS1ReleaseIntakeError(
                    f"public S1 intake contains forbidden field: {key}"
                )
            _assert_aggregate_safe(item)
    elif isinstance(value, list):
        for item in value:
            _assert_aggregate_safe(item)
    elif isinstance(value, str) and _SOURCE_ROW_VALUE.search(value):
        raise PublicS1ReleaseIntakeError("public S1 intake contains a source-row identifier")


def _validate_intake_policy(policy: Mapping[str, Any]) -> None:
    _verify_content(
        policy,
        schema=PUBLIC_S1_FINETUNE_INTAKE_POLICY_SCHEMA,
        domain=_POLICY_DOMAIN,
        fields={
            "schema",
            "release_profile",
            "source_release",
            "gate",
            "tokenizer",
            "base_preprocessing",
            "sources",
            "runtime_code_license",
            "weight_license",
            "content_sha256",
        },
        label="public S1 intake policy",
    )
    if (
        policy["release_profile"] != portable.S1_PUBLIC_FINETUNE_CLAIM_PROFILE
        or policy["runtime_code_license"] != _RUNTIME_CODE_LICENSE
        or policy["weight_license"] != _WEIGHT_LICENSE
    ):
        raise PublicS1ReleaseIntakeError("public S1 intake policy identity differs")
    source_release = _exact_dict(
        policy["source_release"],
        {
            "release_identity_sha256",
            "rights_review_sha256",
            "release_manifest_sha256",
        },
        "public S1 intake source-release policy",
    )
    for field, value in source_release.items():
        _require_digest(value, f"public S1 intake source-release {field}")
    gate = _exact_dict(
        policy["gate"],
        {
            "minimum_real_score",
            "minimum_control_advantage",
            "minimum_unique_real_hypotheses",
            "maximum_real_hypothesis_multiplicity",
            "reject_empty_hypotheses",
            "reject_unk_token_id",
        },
        "public S1 intake gate",
    )
    minimum_real = _fraction(gate["minimum_real_score"], "minimum real score")
    minimum_advantage = _fraction(gate["minimum_control_advantage"], "minimum control advantage")
    if (
        not 0 < minimum_real <= 1
        or not 0 < minimum_advantage <= 1
        or type(gate["minimum_unique_real_hypotheses"]) is not int
        or gate["minimum_unique_real_hypotheses"] < 1
        or type(gate["maximum_real_hypothesis_multiplicity"]) is not int
        or gate["maximum_real_hypothesis_multiplicity"] < 1
        or gate["reject_empty_hypotheses"] is not True
        or gate["reject_unk_token_id"] != 3
    ):
        raise PublicS1ReleaseIntakeError("public S1 intake gate is unsafe")
    tokenizer = _exact_dict(
        policy["tokenizer"],
        {"binding_sha256", "model_sha256", "record_sha256", "training_report_sha256"},
        "public S1 intake tokenizer",
    )
    base = _exact_dict(
        policy["base_preprocessing"],
        {"inference_revision", "identity_sha256", "preprocessing_sha256"},
        "public S1 intake base preprocessing",
    )
    for label, record in (("tokenizer", tokenizer), ("base preprocessing", base)):
        for field, value in record.items():
            _require_digest(value, f"public S1 intake {label} {field}")
    sources = policy["sources"]
    if not isinstance(sources, list) or len(sources) != len(_EXPECTED_SOURCE_IDS):
        raise PublicS1ReleaseIntakeError("public S1 intake source policy differs")
    expected_source_fields = {
        "source_id",
        "source_name",
        "source_version",
        "license_id",
        "license_sha256",
        "terms_sha256",
        "source_use_policy_sha256",
        "attribution_notice_sha256",
        "source_entry_sha256",
    }
    rows = [_exact_dict(row, expected_source_fields, "public S1 intake source") for row in sources]
    if tuple(row["source_id"] for row in rows) != _EXPECTED_SOURCE_IDS:
        raise PublicS1ReleaseIntakeError("public S1 intake source identities differ")
    for row in rows:
        for field in ("source_id", "source_version", "license_id"):
            if not isinstance(row[field], str) or not row[field]:
                raise PublicS1ReleaseIntakeError(f"public S1 intake source {field} is invalid")
        source_name = row["source_name"]
        if (
            not isinstance(source_name, str)
            or not source_name
            or source_name != source_name.strip()
            or len(source_name.encode("utf-8")) > 512
            or not all(character.isprintable() for character in source_name)
        ):
            raise PublicS1ReleaseIntakeError("public S1 intake source name is invalid")
        for field in expected_source_fields - {
            "source_id",
            "source_name",
            "source_version",
            "license_id",
        }:
            _require_digest(row[field], f"public S1 intake source {field}")
    _assert_aggregate_safe(policy)


def load_public_s1_finetune_intake_policy(path: Path) -> dict[str, Any]:
    """Load one canonical, externally reviewed release-intake policy."""

    policy, _payload = _canonical_file(
        path, label="public S1 intake policy", maximum_bytes=_MAXIMUM_POLICY_JSON_BYTES
    )
    _validate_intake_policy(policy)
    return policy


def _validate_gate_and_policy(identity: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    if identity["gate"]["gate"] != policy["gate"]:
        raise PublicS1ReleaseIntakeError("candidate gate differs from public intake policy")
    if identity["tokenizer"] != policy["tokenizer"]:
        raise PublicS1ReleaseIntakeError("candidate tokenizer differs from public intake policy")
    if identity["base_preprocessing"] != policy["base_preprocessing"]:
        raise PublicS1ReleaseIntakeError(
            "candidate base preprocessing differs from public intake policy"
        )
    candidate = cast(Mapping[str, Any], identity["candidate"])
    text = cast(Mapping[str, Any], identity["text_pretraining"])
    if (
        candidate["training_authority_sha256"] != candidate["training_authority_file_sha256"]
        or text["training_authority_sha256"] != text["training_authority_file_sha256"]
        or type(text["selected_epoch"]) is not int
        or text["selected_epoch"] < 1
    ):
        raise PublicS1ReleaseIntakeError("candidate training-authority linkage differs")
    gate = cast(Mapping[str, Any], identity["gate"])
    real = _fraction(gate["real_motion_score"], "real-motion score")
    zero = _fraction(gate["zero_motion_score"], "zero-motion score")
    deranged = _fraction(gate["deranged_motion_score"], "deranged-motion score")
    count = gate["prediction_count"]
    unique = gate["unique_real_hypotheses"]
    multiplicity = gate["maximum_real_hypothesis_multiplicity"]
    if (
        not real > zero
        or not real > deranged
        or type(count) is not int
        or type(unique) is not int
        or type(multiplicity) is not int
        or not 1 <= unique <= count
        or not (count + unique - 1) // unique <= multiplicity <= count - unique + 1
    ):
        raise PublicS1ReleaseIntakeError("candidate aggregate gate evidence is infeasible")


def _validate_source_policy(review: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    projected: list[dict[str, Any]] = []
    for source in cast(list[Mapping[str, Any]], review["sources"]):
        projected.append(
            {
                field: source[field]
                for field in (
                    "source_id",
                    "source_name",
                    "source_version",
                    "license_id",
                    "license_sha256",
                    "terms_sha256",
                    "source_use_policy_sha256",
                    "attribution_notice_sha256",
                    "source_entry_sha256",
                )
            }
        )
        notice = source["attribution_notice"]
        if (
            not isinstance(notice, str)
            or not notice
            or len(notice.encode("utf-8")) > 8192
            or any(ord(character) < 32 for character in notice)
        ):
            raise PublicS1ReleaseIntakeError("public S1 source attribution is invalid")
    if projected != policy["sources"]:
        raise PublicS1ReleaseIntakeError("candidate source rights differ from intake policy")


def _validate_release_policy(
    *,
    policy: Mapping[str, Any],
    identity_bytes: bytes,
    review_bytes: bytes,
    manifest_bytes: bytes,
) -> None:
    expected = {
        "release_identity_sha256": _sha256(identity_bytes),
        "rights_review_sha256": _sha256(review_bytes),
        "release_manifest_sha256": _sha256(manifest_bytes),
    }
    if policy["source_release"] != expected:
        raise PublicS1ReleaseIntakeError("source release differs from trusted intake policy")


def _portable_directory(root: Path) -> _ValidatedPortable:
    try:
        first = portable._read_bundle_files(root)
        identity = portable._strict_json(
            first["inference-identity.json"],
            maximum_bytes=portable._MAXIMUM_FILE_BYTES["inference-identity.json"],
            label="public S1 portable identity",
        )
        revision = _require_digest(identity.get("inference_revision"), "portable revision")
        loaded = load_s1_portable_bundle(root, expected_inference_revision=revision)
        second = portable._read_bundle_files(root)
    except (OSError, RuntimeError, S1PortableError, ValueError) as exc:
        if isinstance(exc, PublicS1ReleaseIntakeError):
            raise
        raise PublicS1ReleaseIntakeError("public S1 portable bundle failed strict loading") from exc
    if first != second or loaded.identity != identity:
        raise PublicS1ReleaseIntakeError("public S1 portable bundle changed during intake")
    size_bytes = sum(len(payload) for payload in first.values())
    if size_bytes > _MAXIMUM_PORTABLE_BYTES:
        raise PublicS1ReleaseIntakeError("public S1 portable bundle exceeds its intake ceiling")
    model = _exact_dict(
        identity.get("model"),
        {
            "name",
            "class",
            "config_sha256",
            "checkpoint_sha256",
            "checkpoint_size_bytes",
            "state_digest_profile",
            "state_sha256",
        },
        "public S1 portable model identity",
    )
    model_state_sha256 = _require_digest(model["state_sha256"], "portable model state")
    del loaded
    return _ValidatedPortable(
        payloads=first,
        identity=identity,
        inference_revision=revision,
        manifest_sha256=_sha256(first["bundle-manifest.json"]),
        size_bytes=size_bytes,
        model_state_sha256=model_state_sha256,
    )


def _portable_payloads(payloads: Mapping[str, bytes]) -> _ValidatedPortable:
    if tuple(sorted(payloads)) != tuple(sorted(portable._EXPECTED_FILES)):
        raise PublicS1ReleaseIntakeError("portable archive inventory differs")
    with tempfile.TemporaryDirectory(prefix="umi-public-s1-intake-") as temporary:
        root = Path(temporary)
        for name in portable._EXPECTED_FILES:
            destination = root / name
            destination.write_bytes(payloads[name])
            destination.chmod(0o600)
        return _portable_directory(root)


def _archive_bytes(payloads: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in portable._EXPECTED_FILES:
            info = zipfile.ZipInfo(name, _FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payloads[name])
    return output.getvalue()


def _archive_payloads(path: Path) -> tuple[dict[str, bytes], bytes]:
    raw = _read_regular(path, maximum_bytes=_MAXIMUM_ARCHIVE_BYTES, label="portable archive")
    try:
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != list(portable._EXPECTED_FILES):
                raise PublicS1ReleaseIntakeError("portable archive inventory or ordering differs")
            if archive.comment:
                raise PublicS1ReleaseIntakeError("portable archive comment is forbidden")
            for info in infos:
                maximum = portable._MAXIMUM_FILE_BYTES[info.filename]
                if (
                    info.compress_type != zipfile.ZIP_STORED
                    or info.date_time != _FIXED_ZIP_TIME
                    or info.create_system != 3
                    or info.external_attr != 0o100644 << 16
                    or info.flag_bits != 0
                    or info.extra
                    or info.comment
                    or not 1 <= info.file_size <= maximum
                    or info.compress_size != info.file_size
                ):
                    raise PublicS1ReleaseIntakeError(
                        f"portable archive entry differs: {info.filename}"
                    )
            payloads = {info.filename: archive.read(info) for info in infos}
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, PublicS1ReleaseIntakeError):
            raise
        raise PublicS1ReleaseIntakeError("portable archive is invalid") from exc
    if raw != _archive_bytes(payloads):
        raise PublicS1ReleaseIntakeError("portable archive is not byte-deterministic")
    return payloads, raw


def _validate_release_identity_shape(identity: Mapping[str, Any]) -> None:
    _exact_dict(
        identity.get("text_pretraining"),
        {
            "training_authority_sha256",
            "training_authority_content_sha256",
            "final_report_sha256",
            "final_report_content_sha256",
            "selected_checkpoint_manifest_sha256",
            "selected_checkpoint_metadata_sha256",
            "selected_model_state_sha256",
            "selected_epoch",
            "release_lineage_allowed",
            "motion_encoder_invocation_count",
            "cross_attention_invocation_count",
            "claim",
            "training_authority_file_sha256",
            "corpus_manifest_sha256",
            "corpus_manifest_content_sha256",
            "source_entry_sha256",
        },
        "public S1 text-pretraining identity",
    )
    if identity["text_pretraining"]["claim"] != _TEXT_PRETRAINING_CLAIM:
        raise PublicS1ReleaseIntakeError("public S1 text-pretraining claim differs")
    base = _exact_dict(
        identity.get("base_preprocessing"),
        {"inference_revision", "identity_sha256", "preprocessing_sha256"},
        "public S1 base preprocessing",
    )
    for field, value in base.items():
        _require_digest(value, f"public S1 base preprocessing {field}")


def _validate_rights_review(
    review: Mapping[str, Any],
    *,
    identity: Mapping[str, Any],
    rights: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    _verify_content(
        review,
        schema=PUBLIC_S1_FINETUNE_RELEASE_REVIEW_SCHEMA,
        domain=_REVIEW_DOMAIN,
        fields={
            "schema",
            "effective_at",
            "decision_authority",
            "review_class",
            "candidate",
            "sources",
            "release_terms",
            "claim_boundary",
            "content_sha256",
        },
        label="public S1 rights review",
    )
    _parse_timestamp(review["effective_at"], "public S1 rights effective_at")
    if (
        review["decision_authority"] != _RIGHTS_DECISION_AUTHORITY
        or review["review_class"] != _RIGHTS_REVIEW_CLASS
        or review["claim_boundary"] != _RIGHTS_CLAIM_BOUNDARY
    ):
        raise PublicS1ReleaseIntakeError("public S1 rights decision boundary differs")
    candidate = _exact_dict(
        review["candidate"],
        {
            "training_authority_sha256",
            "training_authority_content_sha256",
            "final_report_sha256",
            "final_report_content_sha256",
            "selected_epoch",
            "selected_model_state_sha256",
            "selected_checkpoint_manifest_sha256",
        },
        "public S1 rights candidate",
    )
    identity_candidate = cast(Mapping[str, Any], identity["candidate"])
    if any(candidate[field] != identity_candidate[field] for field in candidate):
        raise PublicS1ReleaseIntakeError("public S1 rights candidate differs from release identity")
    sources = review["sources"]
    if not isinstance(sources, list) or sources != rights.get("sources"):
        raise PublicS1ReleaseIntakeError("public S1 rights sources differ from portable rights")
    source_ids = tuple(row.get("source_id") for row in sources if isinstance(row, Mapping))
    if source_ids != _EXPECTED_SOURCE_IDS or len(source_ids) != len(sources):
        raise PublicS1ReleaseIntakeError("public S1 rights source set differs")
    terms = _exact_dict(
        review["release_terms"],
        {
            "public_weight_redistribution_approved",
            "weight_license",
            "runtime_code_license",
            "raw_source_data_redistribution_approved",
            "source_annotations_redistribution_approved",
            "tokenizer_artifacts_included",
        },
        "public S1 release terms",
    )
    if terms != {
        "public_weight_redistribution_approved": True,
        "weight_license": _WEIGHT_LICENSE,
        "runtime_code_license": _RUNTIME_CODE_LICENSE,
        "raw_source_data_redistribution_approved": False,
        "source_annotations_redistribution_approved": False,
        "tokenizer_artifacts_included": True,
    }:
        raise PublicS1ReleaseIntakeError("public S1 release terms differ from portable rights")
    review_sha256 = canonical_json_sha256(dict(review))
    if (
        identity["rights_review_sha256"] != review_sha256
        or rights.get("rights_as_of") != review["effective_at"]
        or rights.get("redistribution_blocked_pending_final_rights_review") is not False
        or rights.get("runtime_code_license") != _RUNTIME_CODE_LICENSE
        or rights.get("intended_public_weight_license") != _WEIGHT_LICENSE
        or rights.get("release_rights_decision_sha256") != review_sha256
        or rights.get("upstream_rights_sha256") != review_sha256
        or rights.get("claim_boundary") != _RIGHTS_RECORD_CLAIM_BOUNDARY
    ):
        raise PublicS1ReleaseIntakeError("public S1 rights decision linkage differs")
    by_id = {cast(str, row["source_id"]): row for row in sources}
    motion_sources = cast(Mapping[str, Mapping[str, Any]], identity["motion_sources"])
    text = cast(Mapping[str, Any], identity["text_pretraining"])
    if (
        by_id["fleurs-asl-v1"]["source_entry_sha256"]
        != motion_sources["fleurs"]["source_entry_sha256"]
        or by_id["facebook/2M-Flores-ASL"]["source_entry_sha256"]
        != motion_sources["two_m_flores"]["source_entry_sha256"]
        or by_id["google-research-datasets/taskmaster/TM-1-2019"]["source_entry_sha256"]
        != text["source_entry_sha256"]
    ):
        raise PublicS1ReleaseIntakeError("public S1 source lineage differs from release identity")
    _validate_source_policy(review, policy)


def _validate_outer_documents(
    *,
    identity: dict[str, Any],
    identity_bytes: bytes,
    review: dict[str, Any],
    review_bytes: bytes,
    manifest: dict[str, Any],
    manifest_bytes: bytes,
    model: _ValidatedPortable,
    policy: Mapping[str, Any],
) -> None:
    _validate_release_policy(
        policy=policy,
        identity_bytes=identity_bytes,
        review_bytes=review_bytes,
        manifest_bytes=manifest_bytes,
    )
    _verify_content(
        identity,
        schema=PUBLIC_S1_FINETUNE_RELEASE_IDENTITY_SCHEMA,
        domain=_IDENTITY_DOMAIN,
        fields={
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
        label="public S1 release identity",
    )
    _validate_release_identity_shape(identity)
    portable_identity = model.identity
    upstream_identity_sha256 = canonical_json_sha256(identity)
    if portable_identity.get("upstream_release_identity_sha256") != upstream_identity_sha256:
        raise PublicS1ReleaseIntakeError("portable bundle names another release identity")
    rights = portable_identity.get("rights")
    if (
        not isinstance(rights, dict)
        or rights.get("schema") != portable.S1_PORTABLE_MULTI_SOURCE_RIGHTS_SCHEMA
        or identity["rights_record_sha256"] != canonical_json_sha256(rights)
    ):
        raise PublicS1ReleaseIntakeError("portable multi-source rights linkage differs")
    if identity["base_preprocessing"]["preprocessing_sha256"] != canonical_json_sha256(
        portable_identity["preprocessing"]
    ):
        raise PublicS1ReleaseIntakeError("portable preprocessing differs from release identity")
    tokenizer = cast(Mapping[str, Any], portable_identity["tokenizer"])
    try:
        expected_claim = portable._public_finetune_claim_boundary(
            identity,
            model_state_sha256=model.model_state_sha256,
            upstream_release_identity_sha256=upstream_identity_sha256,
            tokenizer_model_sha256=cast(str, tokenizer["model_sha256"]),
            tokenizer_record_sha256=cast(str, tokenizer["record_sha256"]),
            text_postprocess=cast(Mapping[str, Any], portable_identity["text_postprocess"]),
            rights_sha256=canonical_json_sha256(rights),
        )
    except S1PortableError as exc:
        raise PublicS1ReleaseIntakeError(
            "public S1 release identity fails its portable claim boundary"
        ) from exc
    if portable_identity.get("claim_boundary") != expected_claim:
        raise PublicS1ReleaseIntakeError("portable claim differs from public release evidence")
    _validate_gate_and_policy(identity, policy)
    _validate_rights_review(review, identity=identity, rights=rights, policy=policy)

    _verify_content(
        manifest,
        schema=PUBLIC_S1_FINETUNE_RELEASE_MANIFEST_SCHEMA,
        domain=_MANIFEST_DOMAIN,
        fields={
            "schema",
            "release_identity",
            "rights_review",
            "portable",
            "raw_data_included",
            "content_sha256",
        },
        label="public S1 release manifest",
    )
    expected_identity = {
        "sha256": _sha256(identity_bytes),
        "content_sha256": identity["content_sha256"],
        "size_bytes": len(identity_bytes),
    }
    expected_review = {
        "sha256": _sha256(review_bytes),
        "content_sha256": review["content_sha256"],
        "size_bytes": len(review_bytes),
    }
    expected_portable = {
        "schema": portable.S1_PORTABLE_SCHEMA,
        "inference_revision": model.inference_revision,
        "manifest_sha256": model.manifest_sha256,
        "size_bytes": model.size_bytes,
        "model_state_sha256": model.model_state_sha256,
    }
    if (
        manifest["release_identity"] != expected_identity
        or manifest["rights_review"] != expected_review
        or manifest["portable"] != expected_portable
        or manifest["raw_data_included"] is not False
        or not manifest_bytes
    ):
        raise PublicS1ReleaseIntakeError("public S1 release manifest linkage differs")
    _assert_aggregate_safe(identity)
    _assert_aggregate_safe(review)
    _assert_aggregate_safe(manifest)


def _source_root(root: Path, *, policy: Mapping[str, Any]) -> _ValidatedSource:
    source, source_identity = _direct_directory(
        root, expected=_EXPECTED_SOURCE_ROOT, label="public S1 source root"
    )
    identity, identity_bytes = _canonical_file(
        source / "release-identity.json",
        label="public S1 release identity",
        maximum_bytes=_MAXIMUM_OUTER_JSON_BYTES,
    )
    review, review_bytes = _canonical_file(
        source / "rights-review.json",
        label="public S1 rights review",
        maximum_bytes=_MAXIMUM_OUTER_JSON_BYTES,
    )
    manifest, manifest_bytes = _canonical_file(
        source / "release-manifest.json",
        label="public S1 release manifest",
        maximum_bytes=_MAXIMUM_OUTER_JSON_BYTES,
    )
    model = _portable_directory(source / "portable")
    _validate_outer_documents(
        identity=identity,
        identity_bytes=identity_bytes,
        review=review,
        review_bytes=review_bytes,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        model=model,
        policy=policy,
    )
    if (
        _read_regular(
            source / "release-identity.json",
            maximum_bytes=_MAXIMUM_OUTER_JSON_BYTES,
            label="public S1 release identity",
        )
        != identity_bytes
        or _read_regular(
            source / "rights-review.json",
            maximum_bytes=_MAXIMUM_OUTER_JSON_BYTES,
            label="public S1 rights review",
        )
        != review_bytes
        or _read_regular(
            source / "release-manifest.json",
            maximum_bytes=_MAXIMUM_OUTER_JSON_BYTES,
            label="public S1 release manifest",
        )
        != manifest_bytes
    ):
        raise PublicS1ReleaseIntakeError("public S1 source records changed during intake")
    _direct_directory(
        source,
        expected=_EXPECTED_SOURCE_ROOT,
        label="public S1 source root",
        expected_identity=source_identity,
    )
    return _ValidatedSource(
        release_identity=identity,
        release_identity_bytes=identity_bytes,
        rights_review=review,
        rights_review_bytes=review_bytes,
        release_manifest=manifest,
        release_manifest_bytes=manifest_bytes,
        portable=model,
    )


def _projection(
    source: _ValidatedSource, *, archive_bytes: bytes, policy: Mapping[str, Any]
) -> dict[str, Any]:
    archive = {
        "filename": PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME,
        "format": "zip-stored-deterministic-v1",
        "inference_revision": source.portable.inference_revision,
        "bundle_manifest_sha256": source.portable.manifest_sha256,
        "bundle_size_bytes": source.portable.size_bytes,
        "model_state_sha256": source.portable.model_state_sha256,
        "archive_sha256": _sha256(archive_bytes),
        "archive_size_bytes": len(archive_bytes),
    }
    report: dict[str, Any] = {
        "schema": PUBLIC_S1_FINETUNE_INTAKE_SCHEMA,
        "source_release": {
            "release_identity_sha256": _sha256(source.release_identity_bytes),
            "release_identity_content_sha256": source.release_identity["content_sha256"],
            "rights_review_sha256": _sha256(source.rights_review_bytes),
            "rights_review_content_sha256": source.rights_review["content_sha256"],
            "release_manifest_sha256": _sha256(source.release_manifest_bytes),
            "release_manifest_content_sha256": source.release_manifest["content_sha256"],
        },
        "intake_policy": dict(policy),
        "release_identity": source.release_identity,
        "rights_review": source.rights_review,
        "release_manifest": source.release_manifest,
        "portable_archive": archive,
        "public_data": {
            "aggregate_only": True,
            "source_rows_included": False,
            "per_example_outputs_included": False,
            "raw_source_data_included": False,
        },
        "claim_boundary": _INTAKE_CLAIM_BOUNDARY,
    }
    _assert_aggregate_safe(report)
    report["content_sha256"] = canonical_json_sha256(report, domain=_INTAKE_DOMAIN)
    return report


def _validate_projection(
    report: dict[str, Any],
    *,
    portable_model: _ValidatedPortable,
    archive_bytes: bytes,
    expected_policy: Mapping[str, Any],
) -> str:
    content = _verify_content(
        report,
        schema=PUBLIC_S1_FINETUNE_INTAKE_SCHEMA,
        domain=_INTAKE_DOMAIN,
        fields={
            "schema",
            "source_release",
            "intake_policy",
            "release_identity",
            "rights_review",
            "release_manifest",
            "portable_archive",
            "public_data",
            "claim_boundary",
            "content_sha256",
        },
        label="public S1 intake evidence",
    )
    if report["claim_boundary"] != _INTAKE_CLAIM_BOUNDARY:
        raise PublicS1ReleaseIntakeError("public S1 intake claim boundary differs")
    if report["public_data"] != {
        "aggregate_only": True,
        "source_rows_included": False,
        "per_example_outputs_included": False,
        "raw_source_data_included": False,
    }:
        raise PublicS1ReleaseIntakeError("public S1 intake data boundary differs")
    policy = _exact_dict(
        report["intake_policy"],
        {
            "schema",
            "release_profile",
            "source_release",
            "gate",
            "tokenizer",
            "base_preprocessing",
            "sources",
            "runtime_code_license",
            "weight_license",
            "content_sha256",
        },
        "projected public S1 intake policy",
    )
    _validate_intake_policy(policy)
    if canonical_json_bytes(policy) != canonical_json_bytes(dict(expected_policy)):
        raise PublicS1ReleaseIntakeError("embedded intake policy differs from trusted policy")
    identity = _exact_dict(
        report["release_identity"],
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
        "projected release identity",
    )
    review = cast(dict[str, Any], report["rights_review"])
    manifest = cast(dict[str, Any], report["release_manifest"])
    identity_bytes = canonical_json_bytes(identity)
    review_bytes = canonical_json_bytes(review)
    manifest_bytes = canonical_json_bytes(manifest)
    source_release = _exact_dict(
        report["source_release"],
        {
            "release_identity_sha256",
            "release_identity_content_sha256",
            "rights_review_sha256",
            "rights_review_content_sha256",
            "release_manifest_sha256",
            "release_manifest_content_sha256",
        },
        "public S1 source-release binding",
    )
    expected_source = {
        "release_identity_sha256": _sha256(identity_bytes),
        "release_identity_content_sha256": identity["content_sha256"],
        "rights_review_sha256": _sha256(review_bytes),
        "rights_review_content_sha256": review["content_sha256"],
        "release_manifest_sha256": _sha256(manifest_bytes),
        "release_manifest_content_sha256": manifest["content_sha256"],
    }
    if source_release != expected_source:
        raise PublicS1ReleaseIntakeError("public S1 source-release binding differs")
    _validate_outer_documents(
        identity=identity,
        identity_bytes=identity_bytes,
        review=review,
        review_bytes=review_bytes,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        model=portable_model,
        policy=policy,
    )
    archive = _exact_dict(
        report["portable_archive"],
        {
            "filename",
            "format",
            "inference_revision",
            "bundle_manifest_sha256",
            "bundle_size_bytes",
            "model_state_sha256",
            "archive_sha256",
            "archive_size_bytes",
        },
        "public S1 portable archive binding",
    )
    expected_archive = {
        "filename": PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME,
        "format": "zip-stored-deterministic-v1",
        "inference_revision": portable_model.inference_revision,
        "bundle_manifest_sha256": portable_model.manifest_sha256,
        "bundle_size_bytes": portable_model.size_bytes,
        "model_state_sha256": portable_model.model_state_sha256,
        "archive_sha256": _sha256(archive_bytes),
        "archive_size_bytes": len(archive_bytes),
    }
    if archive != expected_archive:
        raise PublicS1ReleaseIntakeError("public S1 portable archive binding differs")
    _assert_aggregate_safe(report)
    return content


def _write_exclusive(path: Path, payload: bytes, *, mode: int = 0o644) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, mode)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:  # pragma: no cover
                raise OSError("public S1 intake write made no progress")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        raise PublicS1ReleaseIntakeError(f"cannot write public S1 intake: {path.name}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _rename_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing path."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        rename = libc.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 0x00000001)
    else:  # pragma: no cover - release hosts are macOS or Linux
        raise PublicS1ReleaseIntakeError("this host lacks atomic no-replace directory publication")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise PublicS1ReleaseIntakeError("public S1 intake destination already exists")
        raise PublicS1ReleaseIntakeError("cannot atomically publish public S1 intake") from OSError(
            error_number, os.strerror(error_number)
        )


def _cleanup_unpublished(root: Path, *, expected_identity: tuple[int, int]) -> None:
    """Remove only our private, unpublished temporary directory and known files."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(root, flags)
        opened = os.fstat(descriptor)
        names = tuple(os.listdir(descriptor))
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != expected_identity
            or not set(names).issubset(_EXPECTED_INTAKE_ROOT)
        ):
            return
        for name in names:
            os.unlink(name, dir_fd=descriptor)
        after = os.fstat(descriptor)
        if (after.st_dev, after.st_ino) != expected_identity:
            return
    except OSError:
        return
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        current = root.lstat()
        if (
            not stat.S_ISLNK(current.st_mode)
            and stat.S_ISDIR(current.st_mode)
            and (current.st_dev, current.st_ino) == expected_identity
        ):
            os.rmdir(root)
    except OSError:
        pass


def verify_public_s1_finetune_intake(root: Path, *, policy_path: Path) -> dict[str, Any]:
    """Verify a projected public release and its deterministic portable archive."""

    policy = load_public_s1_finetune_intake_policy(policy_path)
    intake, intake_identity = _direct_directory(
        root, expected=_EXPECTED_INTAKE_ROOT, label="public S1 intake root"
    )
    report, report_bytes = _canonical_file(
        intake / PUBLIC_S1_FINETUNE_EVIDENCE_FILENAME,
        label="public S1 intake evidence",
        maximum_bytes=_MAXIMUM_INTAKE_JSON_BYTES,
    )
    payloads, archive_bytes = _archive_payloads(intake / PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME)
    model = _portable_payloads(payloads)
    content_sha256 = _validate_projection(
        report,
        portable_model=model,
        archive_bytes=archive_bytes,
        expected_policy=policy,
    )
    _direct_directory(
        intake,
        expected=_EXPECTED_INTAKE_ROOT,
        label="public S1 intake root",
        expected_identity=intake_identity,
    )
    return {
        "schema": PUBLIC_S1_FINETUNE_INTAKE_RESULT_SCHEMA,
        "intake_policy_sha256": _sha256(canonical_json_bytes(policy)),
        "intake_policy_content_sha256": policy["content_sha256"],
        "inference_revision": model.inference_revision,
        "model_state_sha256": model.model_state_sha256,
        "evidence": PUBLIC_S1_FINETUNE_EVIDENCE_FILENAME,
        "evidence_sha256": _sha256(report_bytes),
        "evidence_content_sha256": content_sha256,
        "archive": PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME,
        "archive_sha256": _sha256(archive_bytes),
        "archive_size_bytes": len(archive_bytes),
    }


def intake_public_s1_finetune_release(
    source_root: Path, output_root: Path, *, policy_path: Path
) -> dict[str, Any]:
    """Validate and atomically project one completed public-S1-finetune release."""

    destination = Path(os.path.abspath(output_root))
    if destination.exists() or destination.is_symlink():
        raise PublicS1ReleaseIntakeError("public S1 intake destination already exists")
    try:
        parent = destination.parent.resolve(strict=True)
    except OSError as exc:
        raise PublicS1ReleaseIntakeError("public S1 intake parent is unavailable") from exc
    if parent != destination.parent:
        raise PublicS1ReleaseIntakeError("public S1 intake parent contains a symlink")
    policy = load_public_s1_finetune_intake_policy(policy_path)
    source = _source_root(source_root, policy=policy)
    archive_bytes = _archive_bytes(source.portable.payloads)
    if len(archive_bytes) > _MAXIMUM_ARCHIVE_BYTES:
        raise PublicS1ReleaseIntakeError("portable archive exceeds its intake ceiling")
    report = _projection(source, archive_bytes=archive_bytes, policy=policy)
    report_bytes = canonical_json_bytes(report)
    if len(report_bytes) > _MAXIMUM_INTAKE_JSON_BYTES:
        raise PublicS1ReleaseIntakeError("public S1 intake evidence exceeds its byte ceiling")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.intake-", dir=parent))
    temporary_stat = temporary.stat()
    temporary_identity = (temporary_stat.st_dev, temporary_stat.st_ino)
    renamed = False
    complete = False
    try:
        _write_exclusive(temporary / PUBLIC_S1_FINETUNE_ARCHIVE_FILENAME, archive_bytes)
        _write_exclusive(temporary / PUBLIC_S1_FINETUNE_EVIDENCE_FILENAME, report_bytes)
        _rename_no_replace(temporary, destination)
        renamed = True
        result = verify_public_s1_finetune_intake(destination, policy_path=policy_path)
        complete = True
        return result
    finally:
        if not complete and not renamed:
            _cleanup_unpublished(temporary, expected_identity=temporary_identity)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and project a completed public-S1-finetune release"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    intake = subparsers.add_parser("intake")
    intake.add_argument("--source", type=Path, required=True)
    intake.add_argument("--output", type=Path, required=True)
    intake.add_argument("--policy", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--policy", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "intake":
            result = intake_public_s1_finetune_release(
                arguments.source, arguments.output, policy_path=arguments.policy
            )
        else:
            result = verify_public_s1_finetune_intake(arguments.root, policy_path=arguments.policy)
    except (CanonicalJSONError, OSError, PublicS1ReleaseIntakeError) as exc:
        print(f"public S1 intake failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
