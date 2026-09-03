from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, cast

from bitsign_motion import s1_portable_runtime as portable
from bitsign_motion.s1_portable_runtime import S1PortableError, load_s1_portable_bundle
from bitsign_motion.s1_release_evidence import (
    EVIDENCE_FILES,
    MOTION_ABLATION_FILENAME,
    MOTION_ABLATION_SCHEMA,
    RELEASE_E2E_FILENAME,
    RELEASE_E2E_SCHEMA,
    RIGHTS_EVIDENCE_FILENAME,
    RIGHTS_EVIDENCE_SCHEMA,
    SELECTION_LEDGER_FILENAME,
    SELECTION_LEDGER_SCHEMA,
    S1ReleaseEvidenceError,
    load_motion_ablation_bytes,
    load_release_e2e_bytes,
    load_rights_evidence_bytes,
    load_selection_ledger_bytes,
)

SCHEMA = "umi-reference-model-release/2"
STATUS = "component_test_no_weight"
CONTENT_DOMAIN = b"umi-reference-model-release-v2\0"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_REVISION = re.compile(r"[0-9a-f]{40}")
RELEASE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
LABEL = re.compile(r"[a-z][a-z0-9-]{0,63}")
MAXIMUM_ARTIFACT_BYTES = 128 * 1024 * 1024
MODEL_FILENAME = "umi-s1-baseline-v0-portable.zip"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPANION_ARTIFACT_SOURCES = {
    "code-license": ("LICENSE", "LICENSE"),
    "notice": ("NOTICE", "NOTICE"),
    "model-license": ("CC-BY-SA-4.0.txt", "licenses/CC-BY-SA-4.0.txt"),
    "fleurs-attribution": (
        "FLEURS-ATTRIBUTION.txt",
        "licenses/FLEURS-ATTRIBUTION.txt",
    ),
    "fsboard-license": ("CC-BY-4.0.txt", "licenses/CC-BY-4.0.txt"),
    "fsboard-attribution": (
        "FSBOARD-ATTRIBUTION.txt",
        "licenses/FSBOARD-ATTRIBUTION.txt",
    ),
}
MODEL_ARCHIVE_FILES = portable._EXPECTED_FILES
MODEL_ARCHIVE_TIME = (1980, 1, 1, 0, 0, 0)
MAXIMUM_MODEL_ARCHIVE_CONTENT_BYTES = 80 * 1024 * 1024
TASK_MODEL_SHA256 = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8"
TASK_MODEL_SOURCE = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task?"
    "generation=1703178474695092"
)
CLAIM_BOUNDARY = (
    "This is an integration fixture and miner replacement target, not a usable ASL "
    "translator. UMI translation weights are inactive. Post-test-open validation used MPS "
    "over ARM64-materialized tensors; zero motion outscored real motion on the bound "
    "diagnostic, so useful motion grounding is not established. The release provides no "
    "Linux/AMD64 end-to-end quality, UMI activation, accessibility, production-quality, "
    "or reward claim. The extractor image is locally built and is not distributed."
)
EXPECTED_ARTIFACTS = {
    "model": MODEL_FILENAME,
    **EVIDENCE_FILES,
    **{
        label: release_filename
        for label, (release_filename, _source_path) in COMPANION_ARTIFACT_SOURCES.items()
    },
}
EVIDENCE_SCHEMAS = {
    "selection-ledger": SELECTION_LEDGER_SCHEMA,
    "motion-ablation-evidence": MOTION_ABLATION_SCHEMA,
    "rights-evidence": RIGHTS_EVIDENCE_SCHEMA,
    "release-e2e-evidence": RELEASE_E2E_SCHEMA,
}
LICENSE_DECLARATION = {
    "runtime_code": {
        "license_id": "Apache-2.0",
        "license_artifact": "LICENSE",
        "notice_artifact": "NOTICE",
    },
    "model_weights": {
        "license_id": "CC-BY-SA-4.0",
        "license_artifact": "CC-BY-SA-4.0.txt",
        "attribution_artifact": "FLEURS-ATTRIBUTION.txt",
    },
    "training_lineage": [
        {
            "source_id": "fleurs-asl-v1",
            "license_id": "CC-BY-SA-4.0",
            "license_artifact": "CC-BY-SA-4.0.txt",
            "attribution_artifact": "FLEURS-ATTRIBUTION.txt",
            "source_data_distributed": False,
        },
        {
            "source_id": "fsboard-v3",
            "license_id": "CC-BY-4.0",
            "license_artifact": "CC-BY-4.0.txt",
            "attribution_artifact": "FSBOARD-ATTRIBUTION.txt",
            "source_data_distributed": False,
        },
    ],
}
LOCAL_EXTRACTOR = {
    "binary_distributed": False,
    "build_context": "docker/mediapipe-holistic",
    "build_platform": "linux/amd64",
    "derived_bundle_required": True,
    "workflow": "source-build-validate-rebind-probe/1",
}
RELEASE_COMMIT_POLICY = {
    "source_revision_relation": "direct_parent",
    "allowed_changed_paths": ["release/SHA256SUMS", "release/release-manifest.json"],
}
PREPROCESSING_SOURCE_PATHS = {
    "amd64_container_host_source_sha256": "src/bitsign_motion/amd64_holistic_container.py",
    "amd64_container_requirements_sha256": ("docker/mediapipe-holistic/requirements.amd64.lock"),
    "amd64_container_worker_source_sha256": "docker/mediapipe-holistic/worker.amd64.py",
    "container_host_source_sha256": "src/bitsign_motion/holistic_container.py",
    "container_requirements_sha256": "docker/mediapipe-holistic/requirements.lock",
    "container_worker_source_sha256": "docker/mediapipe-holistic/worker.py",
    "landmark_mapping_source_sha256": "src/bitsign_motion/mediapipe_mapping.py",
    "motion_composition_source_sha256": "src/bitsign_motion/motion_artifact.py",
}


class ReleaseArtifactError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _content_sha256(record: dict[str, Any]) -> str:
    unsigned = dict(record)
    unsigned.pop("content_sha256", None)
    return hashlib.sha256(CONTENT_DOMAIN + _canonical(unsigned)).hexdigest()


def _read_regular(path: Path, *, maximum_bytes: int = MAXIMUM_ARTIFACT_BYTES) -> bytes:
    direct = Path(os.path.abspath(path))
    try:
        path_state = direct.lstat()
    except OSError as exc:
        raise ReleaseArtifactError(f"release artifact is unavailable: {path.name}") from exc
    if (
        stat.S_ISLNK(path_state.st_mode)
        or not stat.S_ISREG(path_state.st_mode)
        or path_state.st_nlink != 1
        or not 1 <= path_state.st_size <= maximum_bytes
    ):
        raise ReleaseArtifactError(f"release artifact violates its file contract: {path.name}")
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
            payload = b"".join(chunks)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ReleaseArtifactError(f"release artifact is unavailable: {path.name}") from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (path_state.st_dev, path_state.st_ino) != (before.st_dev, before.st_ino)
        or not 1 <= len(payload) <= maximum_bytes
        or len(payload) != before.st_size
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ReleaseArtifactError(f"release artifact changed while reading: {path.name}")
    return payload


def _hash_regular(path: Path) -> tuple[str, int]:
    payload = _read_regular(path)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _validate_companion_artifacts(payloads: dict[str, bytes]) -> None:
    for label, (release_filename, source_relative) in COMPANION_ARTIFACT_SOURCES.items():
        release_payload = payloads[release_filename]
        source_payload = _read_regular(REPOSITORY_ROOT / source_relative)
        if release_payload != source_payload:
            raise ReleaseArtifactError(f"{label} differs from its canonical repository source")


def _parse_artifacts(
    values: list[str], output_directory: Path
) -> tuple[list[dict[str, object]], dict[str, bytes]]:
    parsed: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or LABEL.fullmatch(label) is None or label in parsed:
            raise ReleaseArtifactError(f"invalid or duplicate artifact argument: {value}")
        parsed[label] = Path(os.path.abspath(raw_path))
    if set(parsed) != set(EXPECTED_ARTIFACTS):
        raise ReleaseArtifactError("release requires the complete fixed artifact set")

    output = output_directory.resolve(strict=True)
    records: list[dict[str, object]] = []
    payloads: dict[str, bytes] = {}
    for label, expected_name in EXPECTED_ARTIFACTS.items():
        path = parsed[label]
        if path.parent != output or path.name != expected_name:
            raise ReleaseArtifactError(f"{label} must use the fixed name in the output directory")
        payload = _read_regular(path)
        digest, size = hashlib.sha256(payload).hexdigest(), len(payload)
        payloads[path.name] = payload
        records.append(
            {"label": label, "filename": path.name, "size_bytes": size, "sha256": digest}
        )
    return sorted(records, key=lambda item: str(item["filename"])), payloads


def _validate_model_archive(
    archive_payload: bytes,
    expected_revision: str,
) -> dict[str, Any]:
    payloads: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != MODEL_ARCHIVE_FILES:
                raise ReleaseArtifactError("model archive has an unexpected file set")
            if archive.comment:
                raise ReleaseArtifactError("model archive has an unexpected archive comment")
            total = 0
            for info in infos:
                mode = info.external_attr >> 16
                maximum = portable._MAXIMUM_FILE_BYTES[info.filename]
                if (
                    info.create_system != 3
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.date_time != MODEL_ARCHIVE_TIME
                    or info.flag_bits != 0
                    or info.extra
                    or info.comment
                    or stat.S_IFMT(mode) != stat.S_IFREG
                    or stat.S_IMODE(mode) != 0o644
                    or not 1 <= info.file_size <= maximum
                    or info.compress_size != info.file_size
                ):
                    raise ReleaseArtifactError(
                        f"model archive member violates its contract: {info.filename}"
                    )
                total += info.file_size
                if total > MAXIMUM_MODEL_ARCHIVE_CONTENT_BYTES:
                    raise ReleaseArtifactError("model archive content exceeds its ceiling")
                payload = archive.read(info)
                if len(payload) != info.file_size:
                    raise ReleaseArtifactError(
                        f"model archive member changed while reading: {info.filename}"
                    )
                payloads[info.filename] = payload
    except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as exc:
        raise ReleaseArtifactError("model archive cannot be inspected") from exc
    with tempfile.TemporaryDirectory(prefix=".umi-release-model-") as temporary:
        bundle = Path(temporary)
        bundle.chmod(0o700)
        try:
            for name in MODEL_ARCHIVE_FILES:
                portable._write_exclusive(bundle / name, payloads[name])
            loaded = load_s1_portable_bundle(
                bundle,
                expected_inference_revision=expected_revision,
            )
        except S1PortableError as exc:
            raise ReleaseArtifactError("model archive failed strict runtime loading") from exc
        identity = loaded.identity
        del loaded
    return identity


def _validate_model_release_binding(
    identity: dict[str, Any],
    *,
    inference_revision: str,
    rights_decision_sha256: str,
) -> None:
    rights = identity.get("rights")
    if (
        identity.get("inference_revision") != inference_revision
        or not isinstance(rights, dict)
        or rights.get("release_rights_decision_sha256") != rights_decision_sha256
        or rights.get("runtime_code_license") != "Apache-2.0"
        or rights.get("intended_public_weight_license") != "CC-BY-SA-4.0"
        or rights.get("public_weight_eligible") is not True
        or rights.get("redistribution_blocked_pending_final_rights_review") is not False
    ):
        raise ReleaseArtifactError("model archive rights or release identity differs")


def _model_authority(identity: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    model = identity.get("model")
    tokenizer = identity.get("tokenizer")
    runtime = identity.get("runtime")
    postprocess = identity.get("text_postprocess")
    rights = identity.get("rights")
    if any(
        not isinstance(value, dict) for value in (model, tokenizer, runtime, postprocess, rights)
    ):
        raise ReleaseArtifactError("portable model authority is incomplete")
    return cast(
        tuple[dict[str, Any], ...],
        (model, tokenizer, runtime, postprocess, rights),
    )


def _validate_decoder_contract(runtime: dict[str, Any], postprocess: dict[str, Any]) -> None:
    decoding = runtime.get("decoding")
    expected_decoding = {
        "algorithm": "beam-search",
        "beam_width": 2,
        "default_maximum_decode_tokens": 24,
        "score": "cumulative-log-probability",
        "length_normalization": False,
        "suppressed_token_ids": [0, 1],
        "eos_token_id": 2,
        "no_repeat_ngram_size": 3,
        "tie_break": "lexicographically-smallest-token-sequence",
    }
    if decoding != expected_decoding or postprocess.get("maximum_words") != 8:
        raise ReleaseArtifactError("portable decoder differs from selected evidence")


def _validate_selection_model_binding(
    evidence: dict[str, Any], model_identity: dict[str, Any]
) -> None:
    model, tokenizer, runtime, postprocess, rights = _model_authority(model_identity)
    candidate = evidence["release_candidate"]
    source = evidence["source"]
    if (
        candidate["release_inference_revision"] != model_identity.get("inference_revision")
        or candidate["runtime_revision"] != runtime.get("revision")
        or candidate["selected_model_state_sha256"] != model.get("state_sha256")
        or candidate["tokenizer_model_sha256"] != tokenizer.get("model_sha256")
        or candidate["tokenizer_record_sha256"] != tokenizer.get("record_sha256")
        or source["source_id"] != rights.get("source_id")
        or source["source_version"] != rights.get("source_version")
        or source["license_id"] != rights.get("source_license_id")
        or source["public_data_eligible"] != rights.get("public_data_eligible")
    ):
        raise ReleaseArtifactError("selection evidence differs from portable model")
    _validate_decoder_contract(runtime, postprocess)


def _validate_motion_model_binding(
    evidence: dict[str, Any], model_identity: dict[str, Any]
) -> None:
    model, tokenizer, runtime, postprocess, _rights = _model_authority(model_identity)
    candidate = evidence["release_candidate"]
    if (
        candidate["release_inference_revision"] != model_identity.get("inference_revision")
        or candidate["selected_model_state_sha256"] != model.get("state_sha256")
        or candidate["tokenizer_model_sha256"] != tokenizer.get("model_sha256")
        or candidate["tokenizer_record_sha256"] != tokenizer.get("record_sha256")
        or candidate["decoder_profile"] != "beam2-max24-no-repeat3-eight-word-cap"
    ):
        raise ReleaseArtifactError("motion evidence differs from portable model")
    _validate_decoder_contract(runtime, postprocess)


def _validate_rights_model_binding(
    evidence: dict[str, Any],
    model_identity: dict[str, Any],
    *,
    rights_decision_sha256: str,
) -> None:
    _model, _tokenizer, _runtime, _postprocess, rights = _model_authority(model_identity)
    terms = evidence["release_terms"]
    sources = evidence["sources"]
    source = next(
        (
            item
            for item in sources
            if item["source_id"] == rights.get("source_id")
            and item["source_version"] == rights.get("source_version")
        ),
        None,
    )
    if (
        evidence["private_review"]["content_sha256"] != rights_decision_sha256
        or rights.get("release_rights_decision_sha256") != rights_decision_sha256
        or terms["runtime_code_license"] != rights.get("runtime_code_license")
        or terms["weight_license"] != rights.get("intended_public_weight_license")
        or terms["public_weight_redistribution_approved"] != rights.get("public_weight_eligible")
        or not isinstance(source, dict)
        or source["license_id"] != rights.get("source_license_id")
        or source["public_data_eligible"] != rights.get("public_data_eligible")
        or source["public_weight_eligible"] != rights.get("public_weight_eligible")
    ):
        raise ReleaseArtifactError("rights evidence differs from portable model")


def _validate_e2e_release_binding(
    evidence: dict[str, Any],
    model_identity: dict[str, Any],
    *,
    umi_git_revision: str,
) -> None:
    runtime = evidence["runtime"]
    rebound_identity = copy.deepcopy(model_identity)
    del rebound_identity["inference_revision"]
    rebound_identity["preprocessing"]["supported_oci_images"]["linux/amd64"] = runtime[
        "extractor_image_id"
    ]
    expected_derived = hashlib.sha256(
        portable._IDENTITY_DOMAIN + _canonical(rebound_identity)
    ).hexdigest()
    if (
        evidence["base_inference_revision"] != model_identity.get("inference_revision")
        or evidence["umi_git_revision"] != umi_git_revision
        or evidence["derived_inference_revision"] != expected_derived
    ):
        raise ReleaseArtifactError("release E2E evidence differs from release identity")


def _load_evidence_set(
    payloads: dict[str, bytes],
    *,
    expected_revision: str,
    model_identity: dict[str, Any],
    rights_decision_sha256: str,
    umi_git_revision: str,
) -> dict[str, dict[str, Any]]:
    try:
        selection = load_selection_ledger_bytes(
            payloads[SELECTION_LEDGER_FILENAME],
            expected_inference_revision=expected_revision,
        )
        motion = load_motion_ablation_bytes(
            payloads[MOTION_ABLATION_FILENAME],
            expected_inference_revision=expected_revision,
        )
        rights = load_rights_evidence_bytes(payloads[RIGHTS_EVIDENCE_FILENAME])
        e2e = load_release_e2e_bytes(payloads[RELEASE_E2E_FILENAME])
    except S1ReleaseEvidenceError as exc:
        raise ReleaseArtifactError("release evidence failed strict validation") from exc
    _validate_selection_model_binding(selection, model_identity)
    _validate_motion_model_binding(motion, model_identity)
    _validate_rights_model_binding(
        rights,
        model_identity,
        rights_decision_sha256=rights_decision_sha256,
    )
    _validate_e2e_release_binding(
        e2e,
        model_identity,
        umi_git_revision=umi_git_revision,
    )
    return {
        "selection-ledger": selection,
        "motion-ablation-evidence": motion,
        "rights-evidence": rights,
        "release-e2e-evidence": e2e,
    }


def _validate_artifact_records(
    artifacts: object,
    payloads: dict[str, bytes],
) -> list[str]:
    if not isinstance(artifacts, list) or len(artifacts) != len(EXPECTED_ARTIFACTS):
        raise ReleaseArtifactError("release artifact set is invalid")
    if any(not isinstance(item, dict) for item in artifacts):
        raise ReleaseArtifactError("release artifact record is invalid")
    records = cast(list[dict[str, Any]], artifacts)
    if records != sorted(records, key=lambda item: str(item.get("filename", ""))):
        raise ReleaseArtifactError("release artifact records are not sorted by filename")
    expected_checksum_lines: list[str] = []
    labels: set[str] = set()
    for item in records:
        if set(item) != {"label", "filename", "size_bytes", "sha256"}:
            raise ReleaseArtifactError("release artifact record is invalid")
        label = item["label"]
        filename = item["filename"]
        if (
            not isinstance(label, str)
            or label in labels
            or not isinstance(filename, str)
            or EXPECTED_ARTIFACTS.get(label) != filename
            or Path(filename).name != filename
        ):
            raise ReleaseArtifactError("release artifact name or label differs")
        labels.add(label)
        payload = payloads.get(filename)
        if payload is None:
            raise ReleaseArtifactError(f"release artifact is missing: {filename}")
        digest, size = hashlib.sha256(payload).hexdigest(), len(payload)
        if (
            not isinstance(item["sha256"], str)
            or SHA256.fullmatch(item["sha256"]) is None
            or type(item["size_bytes"]) is not int
            or not 1 <= item["size_bytes"] <= MAXIMUM_ARTIFACT_BYTES
            or digest != item["sha256"]
            or size != item["size_bytes"]
        ):
            raise ReleaseArtifactError(f"release artifact digest differs: {filename}")
        expected_checksum_lines.append(f"{digest}  {filename}\n")
    if set(labels) != set(EXPECTED_ARTIFACTS):
        raise ReleaseArtifactError("release artifact labels are incomplete")
    return expected_checksum_lines


def _validate_manifest_evidence(value: object) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict) or set(value) != set(EVIDENCE_SCHEMAS):
        raise ReleaseArtifactError("release evidence manifest is incomplete")
    records = cast(dict[str, object], value)
    validated: dict[str, dict[str, str]] = {}
    for label, schema in EVIDENCE_SCHEMAS.items():
        record = records[label]
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "content_sha256"}
            or record["schema"] != schema
            or not isinstance(record["content_sha256"], str)
            or SHA256.fullmatch(record["content_sha256"]) is None
        ):
            raise ReleaseArtifactError("release evidence manifest record differs")
        validated[label] = cast(dict[str, str], record)
    return validated


def _write_atomic(path: Path, payload: bytes, *, replace: bool) -> None:
    if (path.exists() or path.is_symlink()) and not replace:
        raise ReleaseArtifactError(f"release metadata already exists: {path.name}")
    temporary = path.with_name(f".{path.name}.release-{os.getpid()}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o644,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise ReleaseArtifactError("metadata write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def create_release(arguments: argparse.Namespace) -> dict[str, Any]:
    for value, pattern, label in (
        (arguments.release_id, RELEASE_ID, "release ID"),
        (arguments.inference_revision, SHA256, "inference revision"),
        (arguments.rights_decision_sha256, SHA256, "rights decision digest"),
        (arguments.source_git_revision, GIT_REVISION, "source Git revision"),
        (arguments.umi_git_revision, GIT_REVISION, "UMI Git revision"),
    ):
        if pattern.fullmatch(value) is None:
            raise ReleaseArtifactError(f"{label} is invalid")
    output = arguments.output_directory.resolve(strict=True)
    artifacts, payloads = _parse_artifacts(arguments.artifact, output)
    _validate_companion_artifacts(payloads)
    model_identity = _validate_model_archive(
        payloads[MODEL_FILENAME],
        arguments.inference_revision,
    )
    _validate_model_release_binding(
        model_identity,
        inference_revision=arguments.inference_revision,
        rights_decision_sha256=arguments.rights_decision_sha256,
    )
    evidence = _load_evidence_set(
        payloads,
        expected_revision=arguments.inference_revision,
        model_identity=model_identity,
        rights_decision_sha256=arguments.rights_decision_sha256,
        umi_git_revision=arguments.umi_git_revision,
    )
    _validate_artifact_records(artifacts, payloads)
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "release_id": arguments.release_id,
        "status": STATUS,
        "inference_revision": arguments.inference_revision,
        "rights_decision_sha256": arguments.rights_decision_sha256,
        "source_git_revision": arguments.source_git_revision,
        "umi_git_revision": arguments.umi_git_revision,
        "release_commit_policy": RELEASE_COMMIT_POLICY,
        "local_extractor": LOCAL_EXTRACTOR,
        "external_dependencies": {
            "mediapipe_holistic_task_model": {
                "distributed": False,
                "sha256": TASK_MODEL_SHA256,
                "source": TASK_MODEL_SOURCE,
            }
        },
        "licenses": LICENSE_DECLARATION,
        "evidence": {
            label: {
                "schema": report["schema"],
                "content_sha256": report["content_sha256"],
            }
            for label, report in evidence.items()
        },
        "artifacts": artifacts,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    record["content_sha256"] = _content_sha256(record)
    manifest_payload = _canonical(record) + b"\n"
    checksums = "".join(f"{item['sha256']}  {item['filename']}\n" for item in artifacts).encode()
    _write_atomic(output / "release-manifest.json", manifest_payload, replace=arguments.replace)
    _write_atomic(output / "SHA256SUMS", checksums, replace=arguments.replace)
    return record


def verify_release(path: Path, *, artifact_directory: Path | None = None) -> dict[str, Any]:
    manifest_path = Path(os.path.abspath(path))
    raw = _read_regular(manifest_path, maximum_bytes=1024 * 1024)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseArtifactError("release manifest contains duplicate members")
            result[key] = value
        return result

    try:
        record = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReleaseArtifactError(f"release manifest contains a non-finite number: {token}")
            ),
        )
        canonical = _canonical(record)
    except ReleaseArtifactError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ReleaseArtifactError("release manifest is invalid JSON") from exc
    if not isinstance(record, dict) or raw != canonical + b"\n":
        raise ReleaseArtifactError("release manifest is not canonical")
    expected_fields = {
        "schema",
        "release_id",
        "status",
        "inference_revision",
        "rights_decision_sha256",
        "source_git_revision",
        "umi_git_revision",
        "release_commit_policy",
        "local_extractor",
        "external_dependencies",
        "licenses",
        "evidence",
        "artifacts",
        "claim_boundary",
        "content_sha256",
    }
    if set(record) != expected_fields:
        raise ReleaseArtifactError("release manifest has an unexpected field set")
    if (
        record["schema"] != SCHEMA
        or record["status"] != STATUS
        or not isinstance(record["release_id"], str)
        or RELEASE_ID.fullmatch(record["release_id"]) is None
        or record["local_extractor"] != LOCAL_EXTRACTOR
        or record["release_commit_policy"] != RELEASE_COMMIT_POLICY
        or record["licenses"] != LICENSE_DECLARATION
        or record["external_dependencies"]
        != {
            "mediapipe_holistic_task_model": {
                "distributed": False,
                "sha256": TASK_MODEL_SHA256,
                "source": TASK_MODEL_SOURCE,
            }
        }
        or record["claim_boundary"] != CLAIM_BOUNDARY
        or record["content_sha256"] != _content_sha256(record)
        or not isinstance(record["inference_revision"], str)
        or SHA256.fullmatch(record["inference_revision"]) is None
        or not isinstance(record["rights_decision_sha256"], str)
        or SHA256.fullmatch(record["rights_decision_sha256"]) is None
        or not isinstance(record["source_git_revision"], str)
        or GIT_REVISION.fullmatch(record["source_git_revision"]) is None
        or not isinstance(record["umi_git_revision"], str)
        or GIT_REVISION.fullmatch(record["umi_git_revision"]) is None
    ):
        raise ReleaseArtifactError("release identity or content digest differs")
    manifest_evidence = _validate_manifest_evidence(record["evidence"])
    artifacts = record["artifacts"]
    artifact_root = (
        artifact_directory.resolve(strict=True)
        if artifact_directory is not None
        else manifest_path.parent
    )
    payloads = {
        filename: _read_regular(artifact_root / filename)
        for filename in EXPECTED_ARTIFACTS.values()
    }
    _validate_companion_artifacts(payloads)
    expected_checksum_lines = _validate_artifact_records(artifacts, payloads)
    try:
        checksum_payload = _read_regular(
            manifest_path.parent / "SHA256SUMS", maximum_bytes=64 * 1024
        ).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseArtifactError("SHA256SUMS is not UTF-8") from exc
    if checksum_payload != "".join(expected_checksum_lines):
        raise ReleaseArtifactError("SHA256SUMS differs from the manifest")
    model_identity = _validate_model_archive(payloads[MODEL_FILENAME], record["inference_revision"])
    _validate_model_release_binding(
        model_identity,
        inference_revision=record["inference_revision"],
        rights_decision_sha256=record["rights_decision_sha256"],
    )
    evidence = _load_evidence_set(
        payloads,
        expected_revision=record["inference_revision"],
        model_identity=model_identity,
        rights_decision_sha256=record["rights_decision_sha256"],
        umi_git_revision=record["umi_git_revision"],
    )
    for label, evidence_record in evidence.items():
        if evidence_record["content_sha256"] != manifest_evidence[label]["content_sha256"]:
            raise ReleaseArtifactError(f"{label} content digest differs from the manifest")
    return record


def _git(repository: Path, *arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", os.fspath(repository), *arguments],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseArtifactError("release Git history cannot be verified") from exc
    return completed.stdout


def _require_regular_git_blob(repository: Path, revision: str, relative: str) -> None:
    entry = _git(repository, "ls-tree", "-z", revision, "--", relative)
    prefix = b"100644 blob "
    suffix = b"\t" + relative.encode("utf-8") + b"\0"
    if (
        not entry.startswith(prefix)
        or not entry.endswith(suffix)
        or len(entry) != len(prefix) + 40 + len(suffix)
        or re.fullmatch(b"[0-9a-f]{40}", entry[len(prefix) : len(prefix) + 40]) is None
    ):
        raise ReleaseArtifactError(
            f"release history path is not one non-executable regular blob: {relative}"
        )


def _runtime_module_path(module: object) -> str:
    if module == "bitsign_motion":
        return "src/bitsign_motion/__init__.py"
    if (
        not isinstance(module, str)
        or re.fullmatch(r"bitsign_motion(?:\.[a-z][a-z0-9_]*)+", module) is None
    ):
        raise ReleaseArtifactError("model runtime names an invalid source module")
    return f"src/{module.replace('.', '/')}.py"


def _identity_source_closure(identity: dict[str, Any]) -> dict[str, str]:
    runtime = identity.get("runtime")
    preprocessing = identity.get("preprocessing")
    modules = runtime.get("modules") if isinstance(runtime, dict) else None
    sources = preprocessing.get("sources") if isinstance(preprocessing, dict) else None
    if not isinstance(modules, list) or not isinstance(sources, dict):
        raise ReleaseArtifactError("model identity has no reproducible source closure")
    closure: dict[str, str] = {}
    for record in modules:
        if not isinstance(record, dict) or set(record) != {"module", "source_sha256"}:
            raise ReleaseArtifactError("model runtime source record is invalid")
        path = _runtime_module_path(record["module"])
        digest = record["source_sha256"]
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise ReleaseArtifactError("model runtime source digest is invalid")
        if path in closure and closure[path] != digest:
            raise ReleaseArtifactError("model runtime source closure conflicts")
        closure[path] = digest
    for field, path in PREPROCESSING_SOURCE_PATHS.items():
        digest = sources.get(field)
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise ReleaseArtifactError("model preprocessing source digest is invalid")
        if path in closure and closure[path] != digest:
            raise ReleaseArtifactError("model preprocessing source closure conflicts")
        closure[path] = digest
    return closure


def _base_revision_from_source(payload: bytes) -> str:
    try:
        module = ast.parse(payload, filename="local_bundle_rebind.py")
    except (SyntaxError, ValueError) as exc:
        raise ReleaseArtifactError("local rebinder source is not valid Python") from exc
    values: list[str] = []
    for node in module.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "BASE_INFERENCE_REVISION"
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            values.append(node.value.value)
    if len(values) != 1 or SHA256.fullmatch(values[0]) is None:
        raise ReleaseArtifactError("local rebinder has no unique base inference revision")
    return values[0]


def _verify_source_revision_closure(
    identity: dict[str, Any],
    *,
    repository: Path,
    source_revision: str,
) -> None:
    for relative, expected in sorted(_identity_source_closure(identity).items()):
        payload = _git(repository, "show", f"{source_revision}:{relative}")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ReleaseArtifactError(
                f"source Git revision differs from model source closure: {relative}"
            )
    rebinder = _git(
        repository,
        "show",
        f"{source_revision}:src/bitsign_motion/local_bundle_rebind.py",
    )
    if _base_revision_from_source(rebinder) != identity["inference_revision"]:
        raise ReleaseArtifactError(
            "source Git revision's local rebinder pins a different base inference revision"
        )


def verify_release_commit(
    record: dict[str, Any],
    manifest_path: Path,
    repository: Path,
    release_revision: str = "HEAD",
    artifact_directory: Path | None = None,
) -> str:
    root = repository.resolve(strict=True)
    manifest = manifest_path.resolve(strict=True)
    verified_record = verify_release(
        manifest,
        artifact_directory=artifact_directory,
    )
    if record != verified_record:
        raise ReleaseArtifactError("supplied release record differs from the manifest")
    try:
        relative_manifest = manifest.relative_to(root).as_posix()
    except ValueError as exc:
        raise ReleaseArtifactError("release manifest is outside the repository") from exc
    if relative_manifest != "release/release-manifest.json":
        raise ReleaseArtifactError("release manifest is not at its fixed repository path")

    release_commit = (
        _git(root, "rev-parse", "--verify", f"{release_revision}^{{commit}}")
        .decode("ascii")
        .strip()
    )
    if GIT_REVISION.fullmatch(release_commit) is None:
        raise ReleaseArtifactError("release Git revision is invalid")
    parent_line = _git(root, "rev-list", "--parents", "-n", "1", release_commit)
    parents = parent_line.decode("ascii").strip().split()
    if len(parents) != 2 or parents[1] != record["source_git_revision"]:
        raise ReleaseArtifactError(
            "source Git revision is not the release commit's sole direct parent"
        )
    changed = {
        value
        for value in _git(
            root,
            "diff",
            "--name-only",
            "--no-renames",
            parents[1],
            release_commit,
        )
        .decode("utf-8")
        .splitlines()
        if value
    }
    expected = set(RELEASE_COMMIT_POLICY["allowed_changed_paths"])
    if changed != expected:
        raise ReleaseArtifactError("release commit is not metadata-only")
    for relative_path, local_path in (
        ("release/release-manifest.json", manifest),
        ("release/SHA256SUMS", manifest.parent / "SHA256SUMS"),
    ):
        _require_regular_git_blob(root, release_commit, relative_path)
        committed = _git(root, "show", f"{release_commit}:{relative_path}")
        if committed != local_path.read_bytes():
            raise ReleaseArtifactError(f"working {relative_path} differs from the release commit")
    artifact_root = (
        artifact_directory.resolve(strict=True)
        if artifact_directory is not None
        else manifest.parent
    )
    model_identity = _validate_model_archive(
        _read_regular(artifact_root / MODEL_FILENAME),
        str(record["inference_revision"]),
    )
    _validate_model_release_binding(
        model_identity,
        inference_revision=str(record["inference_revision"]),
        rights_decision_sha256=str(record["rights_decision_sha256"]),
    )
    _verify_source_revision_closure(
        model_identity,
        repository=root,
        source_revision=str(record["source_git_revision"]),
    )
    source_committed_filenames = (
        MODEL_FILENAME,
        *EVIDENCE_FILES.values(),
        *(item[0] for item in COMPANION_ARTIFACT_SOURCES.values()),
    )
    for filename in source_committed_filenames:
        artifact_path = artifact_root / filename
        _require_regular_git_blob(root, str(record["source_git_revision"]), f"release/{filename}")
        committed_artifact = _git(
            root,
            "show",
            f"{record['source_git_revision']}:release/{filename}",
        )
        if committed_artifact != _read_regular(artifact_path):
            raise ReleaseArtifactError(f"{filename} differs from the source-parent commit")
    for label, (release_filename, source_relative) in COMPANION_ARTIFACT_SOURCES.items():
        _require_regular_git_blob(root, str(record["source_git_revision"]), source_relative)
        committed_source = _git(
            root,
            "show",
            f"{record['source_git_revision']}:{source_relative}",
        )
        if committed_source != _read_regular(artifact_root / release_filename):
            raise ReleaseArtifactError(
                f"{label} differs from its canonical source at the source-parent commit"
            )
    e2e = load_release_e2e_bytes(_read_regular(artifact_root / RELEASE_E2E_FILENAME))
    tested_revision = str(e2e["tested_reference_model_git_revision"])
    source_parent_line = _git(
        root, "rev-list", "--parents", "-n", "1", str(record["source_git_revision"])
    )
    source_parents = source_parent_line.decode("ascii").strip().split()
    if len(source_parents) != 2 or source_parents[1] != tested_revision:
        raise ReleaseArtifactError(
            "E2E-tested revision is not the evidence commit's sole direct parent"
        )
    evidence_commit_changes = {
        value
        for value in _git(
            root,
            "diff",
            "--name-only",
            "--no-renames",
            tested_revision,
            str(record["source_git_revision"]),
        )
        .decode("utf-8")
        .splitlines()
        if value
    }
    if evidence_commit_changes != {f"release/{RELEASE_E2E_FILENAME}"}:
        raise ReleaseArtifactError("post-E2E source commit is not E2E-evidence-only")
    return release_commit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or verify reference-model release metadata"
    )
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--release-git-revision")
    parser.add_argument("--release-id")
    parser.add_argument("--inference-revision")
    parser.add_argument("--rights-decision-sha256")
    parser.add_argument("--source-git-revision")
    parser.add_argument("--umi-git-revision")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    try:
        if arguments.verify is not None:
            supplied_generation = (
                any(
                    value is not None
                    for value in (
                        arguments.release_id,
                        arguments.inference_revision,
                        arguments.rights_decision_sha256,
                        arguments.source_git_revision,
                        arguments.umi_git_revision,
                        arguments.output_directory,
                    )
                )
                or bool(arguments.artifact)
                or arguments.replace
            )
            if supplied_generation:
                parser.error("--verify cannot be combined with generation options")
            result = verify_release(
                arguments.verify, artifact_directory=arguments.artifact_directory
            )
            if arguments.release_git_revision is not None and arguments.repository is None:
                parser.error("--release-git-revision requires --repository")
            if arguments.repository is not None:
                verify_release_commit(
                    result,
                    arguments.verify,
                    arguments.repository,
                    arguments.release_git_revision or "HEAD",
                    artifact_directory=arguments.artifact_directory,
                )
        else:
            if any(
                value is not None
                for value in (
                    arguments.artifact_directory,
                    arguments.repository,
                    arguments.release_git_revision,
                )
            ):
                parser.error("repository verification options require --verify")
            required = (
                arguments.release_id,
                arguments.inference_revision,
                arguments.rights_decision_sha256,
                arguments.source_git_revision,
                arguments.umi_git_revision,
                arguments.output_directory,
            )
            if any(value is None for value in required):
                parser.error("generation requires every release identity option")
            result = create_release(arguments)
    except (OSError, ReleaseArtifactError) as exc:
        print(f"release metadata failed: {exc}", file=sys.stderr)
        return 2
    print(_canonical(result).decode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
