from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import sys
import zipfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from .canonical import canonical_json_bytes, canonical_json_sha256
from .holistic_container import (
    MAXIMUM_RAW_ARTIFACT_BYTES,
    MAXIMUM_RECEIPT_BYTES,
    RAW_FILENAME,
    RAW_SCHEMA,
    RECEIPT_FILENAME,
    RECEIPT_SCHEMA,
    SAFE_SAMPLE_ID,
    SHA256,
    TRACK_FIELDS,
    TRACK_SPECS,
    HolisticExtractionResult,
)
from .mediapipe_mapping import (
    DEFAULT_MEDIAPIPE_MAPPING_POLICY,
    MediaPipeMappingDiagnostics,
    MediaPipeMappingError,
    MediaPipeMappingPolicy,
    RawHolisticSequence,
    RawLandmarkTrack,
    map_holistic_to_canonical,
)
from .motion_artifact import (
    S1_MOTION_ARTIFACT_EVIDENCE_SCHEMA,
    S1_TARGET_FEATURE_PROFILE,
    ComposedMotion,
    MotionArtifactError,
    WrittenMotionArtifact,
    compose_motion,
    encode_motion_artifact,
)

HOLISTIC_MOTION_CONVERSION_SCHEMA: Final = "umi-holistic-motion-conversion/1"
HOLISTIC_MOTION_CONVERSION_STATUS: Final = "ex-203-candidate-only"
_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_RAW_METADATA_FIELDS: Final = {
    "schema",
    "sample_id",
    "status",
    "canonical_or_release_quality",
    "hand_slot_semantics",
    "track_prefixes",
}
_RECEIPT_FIELDS: Final = {
    "schema",
    "sample_id",
    "status",
    "claim_boundary",
    "container",
    "input_video",
    "model",
    "source_media",
    "clip",
    "derived_view",
    "selection",
    "extractor",
    "raw_artifact",
    "publication",
    "content_sha256",
}
_EXPECTED_TRACK_PREFIXES: Final = (
    "pose_image",
    "observed_hand_image_0",
    "observed_hand_image_1",
    "face_image",
    "pose_world",
    "observed_hand_world_0",
    "observed_hand_world_1",
)

Array = NDArray[Any]


class HolisticMotionIntegrationError(ValueError):
    """Raised when contained raw output cannot enter the canonical motion pipeline."""


@dataclass(frozen=True, slots=True)
class HolisticMotionConversionResult:
    sample_id: str
    raw_artifact_sha256: str
    receipt_sha256: str
    receipt_content_sha256: str
    frame_count: int
    mapping_diagnostics: MediaPipeMappingDiagnostics
    motion_artifact: WrittenMotionArtifact
    record: dict[str, object]


def _strict_json(raw: bytes, *, label: str, maximum_bytes: int) -> dict[str, Any]:
    if not raw or len(raw) > maximum_bytes:
        raise HolisticMotionIntegrationError(f"{label} violates its byte ceiling")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise HolisticMotionIntegrationError(f"{label} contains a duplicate object member")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HolisticMotionIntegrationError(f"{label} contains a non-finite number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HolisticMotionIntegrationError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise HolisticMotionIntegrationError(f"{label} is not a canonical JSON object")
    return value


def _require_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise HolisticMotionIntegrationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_descriptor(
    directory_descriptor: int,
    name: str,
    *,
    maximum_bytes: int,
) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise HolisticMotionIntegrationError("extraction file cannot be opened safely") from exc
    chunks: list[bytes] = []
    observed = 0
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_mode & 0o077
            or opened.st_size <= 0
            or opened.st_size > maximum_bytes
        ):
            raise HolisticMotionIntegrationError(
                "extraction files must be bounded owner-only regular files"
            )
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed))
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum_bytes:
                raise HolisticMotionIntegrationError("extraction file exceeded its byte ceiling")
            chunks.append(chunk)
        closed = os.fstat(descriptor)
        if (
            observed != opened.st_size
            or closed.st_dev != opened.st_dev
            or closed.st_ino != opened.st_ino
            or closed.st_size != opened.st_size
            or closed.st_mtime_ns != opened.st_mtime_ns
        ):
            raise HolisticMotionIntegrationError("extraction file changed while being read")
        return b"".join(chunks), opened
    finally:
        os.close(descriptor)


def _read_extraction_pair(
    *,
    raw_path: Path,
    receipt_path: Path,
    sample_id: str,
) -> tuple[bytes, bytes]:
    raw = Path(raw_path)
    receipt = Path(receipt_path)
    if raw.name != RAW_FILENAME or receipt.name != RECEIPT_FILENAME:
        raise HolisticMotionIntegrationError("extraction filenames do not match the contract")
    raw_parent = Path(os.path.abspath(raw.parent))
    receipt_parent = Path(os.path.abspath(receipt.parent))
    if raw_parent != receipt_parent or raw_parent.name != sample_id:
        raise HolisticMotionIntegrationError(
            "raw and receipt paths must share the sample-named directory"
        )
    try:
        resolved_parent = raw_parent.resolve(strict=True)
        initial = raw_parent.lstat()
    except OSError as exc:
        raise HolisticMotionIntegrationError("extraction directory cannot be resolved") from exc
    if (
        resolved_parent != raw_parent
        or raw_parent.is_symlink()
        or not stat.S_ISDIR(initial.st_mode)
        or initial.st_uid != os.geteuid()
        or initial.st_mode & 0o077
    ):
        raise HolisticMotionIntegrationError(
            "extraction directory must be a direct owner-only non-symlink path"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_descriptor = os.open(raw_parent, flags)
    except OSError as exc:
        raise HolisticMotionIntegrationError(
            "extraction directory cannot be opened safely"
        ) from exc
    try:
        opened = os.fstat(directory_descriptor)
        if opened.st_dev != initial.st_dev or opened.st_ino != initial.st_ino:
            raise HolisticMotionIntegrationError("extraction directory changed before reading")
        if sorted(os.listdir(directory_descriptor)) != [RAW_FILENAME, RECEIPT_FILENAME]:
            raise HolisticMotionIntegrationError(
                "extraction directory contains an unexpected entry"
            )
        raw_bytes, _ = _read_descriptor(
            directory_descriptor,
            RAW_FILENAME,
            maximum_bytes=MAXIMUM_RAW_ARTIFACT_BYTES,
        )
        receipt_bytes, _ = _read_descriptor(
            directory_descriptor,
            RECEIPT_FILENAME,
            maximum_bytes=MAXIMUM_RECEIPT_BYTES,
        )
        final = os.fstat(directory_descriptor)
        if final.st_dev != opened.st_dev or final.st_ino != opened.st_ino:
            raise HolisticMotionIntegrationError("extraction directory changed while reading")
    finally:
        os.close(directory_descriptor)
    return raw_bytes, receipt_bytes


def _receipt_inventory(
    receipt: dict[str, Any],
    *,
    frame_count: int,
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], dict[str, str]]:
    raw = receipt.get("raw_artifact")
    if not isinstance(raw, dict):
        raise HolisticMotionIntegrationError("receipt raw-artifact record is missing")
    inventory = raw.get("array_inventory")
    if not isinstance(inventory, list):
        raise HolisticMotionIntegrationError("receipt array inventory is missing")
    if [item.get("name") if isinstance(item, dict) else None for item in inventory] != sorted(
        item.get("name") if isinstance(item, dict) else "" for item in inventory
    ):
        raise HolisticMotionIntegrationError("receipt array inventory is not canonically sorted")

    expected: dict[str, tuple[str, tuple[int, ...] | None]] = {
        "timestamps_us": ("<i8", (frame_count,)),
        "requested_timestamps_us": ("<i8", (frame_count,)),
        "source_timestamps_us": ("<i8", (frame_count,)),
        "displayed_timestamps_us": ("<i8", (frame_count,)),
        "source_pts": ("<i8", (frame_count,)),
        "source_time_base": ("<i8", (2,)),
        "source_frame_indices": ("<i4", (frame_count,)),
        "clip_bounds_us": ("<i8", (2,)),
        "displayed_rgb_sha256": ("|u1", (frame_count, 32)),
        "metadata_json_utf8": ("|u1", None),
    }
    for prefix, points in TRACK_SPECS.items():
        for field, (dtype, shape_builder) in TRACK_FIELDS.items():
            expected[f"{prefix}_{field}"] = (
                dtype,
                tuple(shape_builder(frame_count, points)),
            )

    specifications: dict[str, tuple[str, tuple[int, ...]]] = {}
    digests: dict[str, str] = {}
    for entry in inventory:
        if not isinstance(entry, dict) or set(entry) != {
            "name",
            "dtype",
            "shape",
            "tensor_sha256",
        }:
            raise HolisticMotionIntegrationError("receipt array inventory entry is malformed")
        name = entry["name"]
        dtype = entry["dtype"]
        shape = entry["shape"]
        if not isinstance(name, str) or name in specifications or name not in expected:
            raise HolisticMotionIntegrationError("receipt array inventory name is invalid")
        expected_dtype, expected_shape = expected[name]
        if (
            dtype != expected_dtype
            or not isinstance(shape, list)
            or not all(type(dimension) is int and dimension >= 0 for dimension in shape)
        ):
            raise HolisticMotionIntegrationError("receipt array inventory type is invalid")
        observed_shape = tuple(shape)
        if expected_shape is None:
            if len(observed_shape) != 1 or not 1 <= observed_shape[0] <= 4096:
                raise HolisticMotionIntegrationError("raw metadata shape is invalid")
        elif observed_shape != expected_shape:
            raise HolisticMotionIntegrationError("receipt array inventory shape is invalid")
        specifications[name] = (dtype, observed_shape)
        digests[name] = _require_digest(entry["tensor_sha256"], "inventory tensor digest")
    if set(specifications) != set(expected):
        raise HolisticMotionIntegrationError("receipt array inventory is incomplete")
    return specifications, digests


def _decode_npy(
    payload: bytes,
    *,
    expected_dtype: str,
    expected_shape: tuple[int, ...],
) -> Array:
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        elif version == (3, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise HolisticMotionIntegrationError("raw NPY member uses an unsupported version")
    except (ValueError, EOFError) as exc:
        raise HolisticMotionIntegrationError("raw NPY member header is invalid") from exc
    dtype = np.dtype(dtype)
    if (
        fortran_order
        or dtype.hasobject
        or dtype.str != expected_dtype
        or tuple(shape) != expected_shape
    ):
        raise HolisticMotionIntegrationError("raw NPY member contract is invalid")
    element_count = int(np.prod(expected_shape, dtype=np.int64))
    expected_bytes = element_count * dtype.itemsize
    offset = stream.tell()
    if len(payload) - offset != expected_bytes:
        raise HolisticMotionIntegrationError("raw NPY member byte count is invalid")
    data = memoryview(payload)[offset:]
    if dtype == np.dtype(np.bool_) and any(value not in (0, 1) for value in data):
        raise HolisticMotionIntegrationError("raw boolean member has a noncanonical value")
    array = np.frombuffer(data, dtype=dtype, count=element_count).reshape(expected_shape)
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


def _decode_raw_npz(
    payload: bytes,
    *,
    specifications: dict[str, tuple[str, tuple[int, ...]]],
    inventory_digests: dict[str, str],
) -> dict[str, Array]:
    expected_members = {f"{name}.npy" for name in specifications}
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            if archive.comment or len(infos) != len(expected_members):
                raise HolisticMotionIntegrationError("raw NPZ archive member set is invalid")
            if [info.filename for info in infos] != sorted(expected_members):
                raise HolisticMotionIntegrationError("raw NPZ members are not canonically sorted")
            if len({info.filename for info in infos}) != len(infos):
                raise HolisticMotionIntegrationError("raw NPZ contains duplicate members")
            arrays: dict[str, Array] = {}
            for info in infos:
                if (
                    info.filename not in expected_members
                    or info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.flag_bits & 0x1
                    or info.date_time != _ZIP_TIMESTAMP
                    or info.create_system != 3
                    or stat.S_IMODE(info.external_attr >> 16) != 0o600
                ):
                    raise HolisticMotionIntegrationError("raw NPZ member metadata is invalid")
                name = info.filename.removesuffix(".npy")
                dtype, shape = specifications[name]
                expected_data_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
                if info.file_size > expected_data_bytes + 8192:
                    raise HolisticMotionIntegrationError("raw NPZ member exceeds its bound")
                member = archive.read(info)
                if len(member) != info.file_size:
                    raise HolisticMotionIntegrationError("raw NPZ member changed while decoding")
                array = _decode_npy(
                    member,
                    expected_dtype=dtype,
                    expected_shape=shape,
                )
                digest = hashlib.sha256(
                    memoryview(np.ascontiguousarray(array)).cast("B")
                ).hexdigest()
                if digest != inventory_digests[name]:
                    raise HolisticMotionIntegrationError(
                        "raw NPZ tensor does not match the receipt inventory"
                    )
                arrays[name] = array
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, HolisticMotionIntegrationError):
            raise
        raise HolisticMotionIntegrationError("raw NPZ cannot be decoded safely") from exc
    if set(arrays) != set(specifications):
        raise HolisticMotionIntegrationError("raw NPZ array set is incomplete")
    return arrays


def _track(arrays: dict[str, Array], prefix: str) -> RawLandmarkTrack:
    try:
        return RawLandmarkTrack(
            coordinates=arrays[f"{prefix}_coordinates"],
            coordinate_available=arrays[f"{prefix}_coordinate_available"],
            presence=arrays[f"{prefix}_presence"],
            presence_available=arrays[f"{prefix}_presence_available"],
            visibility=arrays[f"{prefix}_visibility"],
            visibility_available=arrays[f"{prefix}_visibility_available"],
        )
    except (KeyError, MediaPipeMappingError) as exc:
        raise HolisticMotionIntegrationError("raw landmark track is invalid") from exc


def _validate_receipt_and_arrays(
    *,
    receipt: dict[str, Any],
    arrays: dict[str, Array],
    sample_id: str,
    raw_sha256: str,
    raw_bytes: int,
    frame_count: int,
) -> RawHolisticSequence:
    if set(receipt) != _RECEIPT_FIELDS:
        raise HolisticMotionIntegrationError("worker receipt field set is invalid")
    claim = receipt.get("claim_boundary")
    raw_record = receipt.get("raw_artifact")
    selection = receipt.get("selection")
    publication = receipt.get("publication")
    source_media = receipt.get("source_media")
    clip = receipt.get("clip")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("sample_id") != sample_id
        or receipt.get("status") != HOLISTIC_MOTION_CONVERSION_STATUS
        or not isinstance(claim, dict)
        or claim.get("canonical_or_release_quality") is not False
        or claim.get("cross_runtime_equivalence_established") is not False
        or claim.get("source_is_umi_challenge_media") is not False
        or not isinstance(raw_record, dict)
        or raw_record.get("schema") != RAW_SCHEMA
        or raw_record.get("filename") != RAW_FILENAME
        or raw_record.get("sha256") != raw_sha256
        or raw_record.get("byte_count") != raw_bytes
        or raw_record.get("anatomical_side_assignment") != "deferred-to-mapping-layer"
        or not isinstance(selection, dict)
        or selection.get("frame_count") != frame_count
        or not isinstance(publication, dict)
        or publication.get("directory") != sample_id
        or publication.get("raw_filename") != RAW_FILENAME
        or publication.get("receipt_filename") != RECEIPT_FILENAME
        or not isinstance(source_media, dict)
        or not isinstance(clip, dict)
    ):
        raise HolisticMotionIntegrationError("worker receipt does not bind the raw extraction")

    metadata_bytes = arrays["metadata_json_utf8"].tobytes(order="C")
    metadata = _strict_json(metadata_bytes, label="raw NPZ metadata", maximum_bytes=4096)
    if (
        set(metadata) != _RAW_METADATA_FIELDS
        or metadata.get("schema") != RAW_SCHEMA
        or metadata.get("sample_id") != sample_id
        or metadata.get("status") != HOLISTIC_MOTION_CONVERSION_STATUS
        or metadata.get("canonical_or_release_quality") is not False
        or metadata.get("hand_slot_semantics")
        != "opaque-observation-slots-mapper-assigns-anatomical-side"
        or metadata.get("track_prefixes") != list(_EXPECTED_TRACK_PREFIXES)
    ):
        raise HolisticMotionIntegrationError("raw NPZ metadata binding is invalid")

    timestamps = arrays["timestamps_us"]
    requested = arrays["requested_timestamps_us"]
    source_timestamps = arrays["source_timestamps_us"]
    displayed_timestamps = arrays["displayed_timestamps_us"]
    source_pts = arrays["source_pts"]
    source_indices = arrays["source_frame_indices"]
    clip_bounds = arrays["clip_bounds_us"]
    source_time_base = arrays["source_time_base"]
    displayed_hashes = arrays["displayed_rgb_sha256"]
    clip_start_us = int(clip_bounds[0])
    clip_end_us = int(clip_bounds[1])
    if (
        not np.array_equal(timestamps, requested)
        or not np.array_equal(displayed_timestamps, source_timestamps)
        or np.any(timestamps[1:] <= timestamps[:-1])
        or np.any(source_timestamps[1:] < source_timestamps[:-1])
        or np.any(displayed_timestamps[1:] < displayed_timestamps[:-1])
        or np.any(source_pts[1:] < source_pts[:-1])
        or np.any(source_indices[1:] < source_indices[:-1])
        or clip_start_us < 0
        or not 2_000_000 <= clip_end_us - clip_start_us <= 15_000_000
        or int(timestamps[0]) < clip_start_us
        or int(timestamps[-1]) >= clip_end_us
        or int(source_timestamps[0]) < clip_start_us
        or int(source_timestamps[-1]) >= clip_end_us
        or int(source_time_base[0]) <= 0
        or int(source_time_base[1]) <= 0
    ):
        raise HolisticMotionIntegrationError("raw timing arrays violate the extraction policy")
    try:
        source_duration = Fraction(source_media.get("duration"))
        nominal_frame_rate = Fraction(source_media.get("nominal_frame_rate"))
        clip_end_overshoot = Fraction(source_media.get("clip_end_overshoot"))
        maximum_clip_end_overshoot = Fraction(source_media.get("maximum_clip_end_overshoot"))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise HolisticMotionIntegrationError("source duration tolerance is invalid") from exc
    expected_clip_end_overshoot = max(
        Fraction(0), Fraction(clip_end_us, 1_000_000) - source_duration
    )
    if (
        set(clip) != {"clip_start_us", "clip_end_us", "duration_us", "interval"}
        or clip.get("clip_start_us") != clip_start_us
        or clip.get("clip_end_us") != clip_end_us
        or clip.get("duration_us") != clip_end_us - clip_start_us
        or clip.get("interval") != "half-open-[start,end)"
        or selection.get("clip_bounds_us") != [clip_start_us, clip_end_us]
        or type(source_media.get("duration_us")) is not int
        or Fraction(clip_start_us, 1_000_000) >= source_duration
        or nominal_frame_rate <= 0
        or source_media.get("clip_end_tolerance_policy")
        != "source-metadata-rounding-at-most-one-nominal-frame-v1"
        or clip_end_overshoot != expected_clip_end_overshoot
        or maximum_clip_end_overshoot != Fraction(1, 1) / nominal_frame_rate
        or clip_end_overshoot > maximum_clip_end_overshoot
    ):
        raise HolisticMotionIntegrationError("raw clip bounds disagree with the receipt")
    try:
        receipt_time_base = Fraction(source_media.get("source_time_base"))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise HolisticMotionIntegrationError("source time base is invalid") from exc
    if receipt_time_base != Fraction(
        int(source_time_base[0]),
        int(source_time_base[1]),
    ):
        raise HolisticMotionIntegrationError("raw source time base disagrees with the receipt")
    receipt_hashes = [bytes(row).hex() for row in displayed_hashes]
    expected_selection = {
        "requested_timestamps_us": requested.tolist(),
        "source_timestamps_us": source_timestamps.tolist(),
        "source_frame_indices": source_indices.tolist(),
        "source_pts": source_pts.tolist(),
        "displayed_rgb_sha256": receipt_hashes,
    }
    if any(selection.get(name) != value for name, value in expected_selection.items()):
        raise HolisticMotionIntegrationError("raw timing arrays disagree with the receipt")

    try:
        return RawHolisticSequence(
            pose_image=_track(arrays, "pose_image"),
            pose_world=_track(arrays, "pose_world"),
            observed_hand_images=(
                _track(arrays, "observed_hand_image_0"),
                _track(arrays, "observed_hand_image_1"),
            ),
            observed_hand_worlds=(
                _track(arrays, "observed_hand_world_0"),
                _track(arrays, "observed_hand_world_1"),
            ),
            face_image=_track(arrays, "face_image"),
            timestamps_us=timestamps,
            clip_start_us=clip_start_us,
            clip_end_us=clip_end_us,
        )
    except MediaPipeMappingError as exc:
        raise HolisticMotionIntegrationError("raw Holistic sequence is invalid") from exc


def _prepare_output(path: Path, *, sample_id: str) -> Path:
    output = Path(path)
    if output.name != f"{sample_id}.npz":
        raise HolisticMotionIntegrationError("motion output filename must equal sample_id.npz")
    try:
        output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        parent = Path(os.path.abspath(output.parent))
        resolved = parent.resolve(strict=True)
        metadata = parent.lstat()
    except OSError as exc:
        raise HolisticMotionIntegrationError("motion output directory cannot be prepared") from exc
    if (
        resolved != parent
        or parent.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise HolisticMotionIntegrationError(
            "motion output directory must be a direct owner-only path"
        )
    return output


def _motion_evidence(
    *,
    path: Path,
    sample_id: str,
    composed: ComposedMotion,
    payload: bytes,
) -> WrittenMotionArtifact:
    artifact_sha256 = hashlib.sha256(payload).hexdigest()
    evidence: dict[str, object] = {
        "schema": S1_MOTION_ARTIFACT_EVIDENCE_SCHEMA,
        "sample_id": sample_id,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(payload),
        "feature_profile": S1_TARGET_FEATURE_PROFILE,
        "feature_policy_sha256": composed.feature_policy_sha256,
        "valid_frame_count": composed.valid_frame_count,
        "float32_tensor_sha256": composed.float32_tensor_sha256,
        "float16_projection_sha256": composed.float16_projection_sha256,
    }
    evidence["content_sha256"] = canonical_json_sha256(
        evidence,
        domain=b"umi-s1-motion-artifact-evidence-v1\0",
    )
    return WrittenMotionArtifact(
        path=path,
        artifact_sha256=artifact_sha256,
        byte_count=len(payload),
        evidence=evidence,
    )


def _same_inode(first: os.stat_result, second: os.stat_result) -> bool:
    return first.st_dev == second.st_dev and first.st_ino == second.st_ino


def _remove_created_output(
    directory_descriptor: int,
    filename: str,
    created: os.stat_result,
) -> None:
    try:
        current = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if _same_inode(current, created):
            os.unlink(filename, dir_fd=directory_descriptor)
    except OSError:
        pass


def _read_existing_output(
    directory_descriptor: int,
    filename: str,
    expected_payload: bytes,
) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(filename, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise HolisticMotionIntegrationError(
            "existing motion artifact cannot be opened safely"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_mode & 0o077
            or opened.st_nlink != 1
            or opened.st_size != len(expected_payload)
        ):
            raise HolisticMotionIntegrationError(
                "existing motion artifact is not a single-link owner-only regular artifact"
            )
        offset = 0
        while offset < len(expected_payload):
            observed = os.read(descriptor, min(1024 * 1024, len(expected_payload) - offset))
            if not observed or observed != expected_payload[offset : offset + len(observed)]:
                raise HolisticMotionIntegrationError(
                    "existing motion artifact differs from deterministic recomposition"
                )
            offset += len(observed)
        if os.read(descriptor, 1):
            raise HolisticMotionIntegrationError(
                "existing motion artifact differs from deterministic recomposition"
            )
        final = os.fstat(descriptor)
        current = os.stat(
            filename,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if (
            not _same_inode(opened, final)
            or not _same_inode(opened, current)
            or final.st_size != opened.st_size
            or final.st_mtime_ns != opened.st_mtime_ns
        ):
            raise HolisticMotionIntegrationError(
                "existing motion artifact changed during verification"
            )
    finally:
        os.close(descriptor)


def _publish_motion_artifact_idempotently(
    path: Path,
    *,
    sample_id: str,
    composed: ComposedMotion,
) -> WrittenMotionArtifact:
    """Create once, or accept only exact deterministic prior publication."""

    try:
        payload = encode_motion_artifact(sample_id, composed)
    except MotionArtifactError as exc:
        raise HolisticMotionIntegrationError("motion artifact encoding failed") from exc
    parent = path.parent
    initial = parent.lstat()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_descriptor = os.open(parent, directory_flags)
    except OSError as exc:
        raise HolisticMotionIntegrationError(
            "motion output directory cannot be opened safely"
        ) from exc
    try:
        opened_directory = os.fstat(directory_descriptor)
        if not _same_inode(initial, opened_directory):
            raise HolisticMotionIntegrationError(
                "motion output directory changed before publication"
            )
        create_flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(
                path.name,
                create_flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        except FileExistsError:
            _read_existing_output(
                directory_descriptor,
                path.name,
                payload,
            )
        except OSError as exc:
            raise HolisticMotionIntegrationError(
                "motion artifact cannot be created safely"
            ) from exc
        else:
            created = os.fstat(descriptor)
            closed = False
            try:
                if (
                    not stat.S_ISREG(created.st_mode)
                    or created.st_uid != os.geteuid()
                    or created.st_mode & 0o077
                    or created.st_nlink != 1
                ):
                    raise HolisticMotionIntegrationError(
                        "new motion artifact is not an owner-only regular file"
                    )
                offset = 0
                while offset < len(payload):
                    count = os.write(descriptor, payload[offset:])
                    if count <= 0:  # pragma: no cover - os.write raises normally
                        raise OSError("motion artifact write made no progress")
                    offset += count
                os.fsync(descriptor)
                final = os.fstat(descriptor)
                current = os.stat(
                    path.name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                current_parent = parent.lstat()
                if (
                    not _same_inode(created, final)
                    or not _same_inode(created, current)
                    or not _same_inode(opened_directory, current_parent)
                    or final.st_size != len(payload)
                    or final.st_nlink != 1
                ):
                    raise HolisticMotionIntegrationError(
                        "motion artifact changed during publication"
                    )
                os.close(descriptor)
                closed = True
                os.fsync(directory_descriptor)
            except BaseException:
                if not closed:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
                _remove_created_output(
                    directory_descriptor,
                    path.name,
                    created,
                )
                raise
        final_directory = os.fstat(directory_descriptor)
        current_parent = parent.lstat()
        if not _same_inode(opened_directory, final_directory) or not _same_inode(
            opened_directory, current_parent
        ):
            raise HolisticMotionIntegrationError(
                "motion output directory changed during publication"
            )
    except OSError as exc:
        raise HolisticMotionIntegrationError("motion artifact publication failed") from exc
    finally:
        os.close(directory_descriptor)
    return _motion_evidence(
        path=path,
        sample_id=sample_id,
        composed=composed,
        payload=payload,
    )


def convert_holistic_raw_to_motion(
    *,
    raw_path: Path,
    receipt_path: Path,
    output_path: Path,
    sample_id: str,
    expected_raw_sha256: str,
    expected_receipt_sha256: str,
    expected_receipt_content_sha256: str,
    expected_frame_count: int,
    mapping_policy: MediaPipeMappingPolicy = DEFAULT_MEDIAPIPE_MAPPING_POLICY,
) -> HolisticMotionConversionResult:
    """Verify, map, compose, and publish one container extraction create-once."""

    if not isinstance(sample_id, str) or SAFE_SAMPLE_ID.fullmatch(sample_id) is None:
        raise HolisticMotionIntegrationError("sample_id must be a safe lowercase identifier")
    raw_digest_expected = _require_digest(expected_raw_sha256, "expected raw digest")
    receipt_digest_expected = _require_digest(expected_receipt_sha256, "expected receipt digest")
    content_digest_expected = _require_digest(
        expected_receipt_content_sha256,
        "expected receipt content digest",
    )
    if type(expected_frame_count) is not int or not 16 <= expected_frame_count <= 120:
        raise HolisticMotionIntegrationError("expected frame count must be from 16 through 120")
    if not isinstance(mapping_policy, MediaPipeMappingPolicy):
        raise TypeError("mapping_policy must be a MediaPipeMappingPolicy")

    raw_bytes, receipt_bytes = _read_extraction_pair(
        raw_path=raw_path,
        receipt_path=receipt_path,
        sample_id=sample_id,
    )
    raw_digest = hashlib.sha256(raw_bytes).hexdigest()
    receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
    if raw_digest != raw_digest_expected or receipt_digest != receipt_digest_expected:
        raise HolisticMotionIntegrationError("extraction file digest does not match its authority")
    receipt = _strict_json(
        receipt_bytes,
        label="worker receipt",
        maximum_bytes=MAXIMUM_RECEIPT_BYTES,
    )
    content_digest = _require_digest(receipt.get("content_sha256"), "worker receipt content digest")
    receipt_without_digest = dict(receipt)
    del receipt_without_digest["content_sha256"]
    calculated_content = hashlib.sha256(
        b"umi-raw-holistic-extraction-receipt-v3\0" + canonical_json_bytes(receipt_without_digest)
    ).hexdigest()
    if content_digest != content_digest_expected or content_digest != calculated_content:
        raise HolisticMotionIntegrationError("worker receipt content digest does not reproduce")

    specifications, inventory_digests = _receipt_inventory(
        receipt,
        frame_count=expected_frame_count,
    )
    arrays = _decode_raw_npz(
        raw_bytes,
        specifications=specifications,
        inventory_digests=inventory_digests,
    )
    raw_sequence = _validate_receipt_and_arrays(
        receipt=receipt,
        arrays=arrays,
        sample_id=sample_id,
        raw_sha256=raw_digest,
        raw_bytes=len(raw_bytes),
        frame_count=expected_frame_count,
    )
    try:
        mapping = map_holistic_to_canonical(raw_sequence, mapping_policy)
        composed = compose_motion(mapping.sequence)
    except (MediaPipeMappingError, MotionArtifactError) as exc:
        raise HolisticMotionIntegrationError("Holistic mapping or composition failed") from exc
    output = _prepare_output(output_path, sample_id=sample_id)
    written = _publish_motion_artifact_idempotently(
        output,
        sample_id=sample_id,
        composed=composed,
    )

    mapping_summary = {
        "profile": mapping.diagnostics.profile,
        "status": mapping.diagnostics.status,
        "policy": mapping.diagnostics.policy,
        "invalid_torso_frame_count": mapping.diagnostics.invalid_torso_frame_count,
        "ambiguous_hand_frame_count": mapping.diagnostics.ambiguous_hand_frame_count,
        "rejected_hand_fit_count": mapping.diagnostics.rejected_hand_fit_count,
        "rejected_face_fit_count": mapping.diagnostics.rejected_face_fit_count,
        "missing_point_count": mapping.diagnostics.missing_point_count,
        "frames": [asdict(frame) for frame in mapping.diagnostics.frames],
    }
    record: dict[str, object] = {
        "schema": HOLISTIC_MOTION_CONVERSION_SCHEMA,
        "sample_id": sample_id,
        "status": HOLISTIC_MOTION_CONVERSION_STATUS,
        "claim_boundary": {
            "canonical_or_release_quality": False,
            "cross_runtime_equivalence_established": False,
            "ex203_complete": False,
        },
        "raw_extraction": {
            "schema": RAW_SCHEMA,
            "raw_artifact_sha256": raw_digest,
            "receipt_sha256": receipt_digest,
            "receipt_content_sha256": content_digest,
            "frame_count": expected_frame_count,
        },
        "mapping": mapping_summary,
        "motion_artifact": written.evidence,
    }
    record["content_sha256"] = canonical_json_sha256(
        record,
        domain=b"umi-holistic-motion-conversion-v1\0",
    )
    return HolisticMotionConversionResult(
        sample_id=sample_id,
        raw_artifact_sha256=raw_digest,
        receipt_sha256=receipt_digest,
        receipt_content_sha256=content_digest,
        frame_count=expected_frame_count,
        mapping_diagnostics=mapping.diagnostics,
        motion_artifact=written,
        record=record,
    )


def convert_holistic_extraction_to_motion(
    extraction: HolisticExtractionResult,
    *,
    output_path: Path,
    mapping_policy: MediaPipeMappingPolicy = DEFAULT_MEDIAPIPE_MAPPING_POLICY,
) -> HolisticMotionConversionResult:
    if not isinstance(extraction, HolisticExtractionResult):
        raise TypeError("extraction must be a HolisticExtractionResult")
    if extraction.status != HOLISTIC_MOTION_CONVERSION_STATUS:
        raise HolisticMotionIntegrationError("extraction status is not an EX-203 candidate")
    return convert_holistic_raw_to_motion(
        raw_path=extraction.raw_artifact_path,
        receipt_path=extraction.receipt_path,
        output_path=output_path,
        sample_id=extraction.sample_id,
        expected_raw_sha256=extraction.raw_artifact_sha256,
        expected_receipt_sha256=extraction.receipt_sha256,
        expected_receipt_content_sha256=extraction.receipt_content_sha256,
        expected_frame_count=extraction.frame_count,
        mapping_policy=mapping_policy,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bitsign_motion.holistic_motion",
        description="Verify and convert one contained Holistic result into an S1 motion artifact.",
    )
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--raw-npz", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-receipt-content-sha256", required=True)
    parser.add_argument("--expected-frame-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = convert_holistic_raw_to_motion(
            raw_path=arguments.raw_npz,
            receipt_path=arguments.receipt,
            output_path=arguments.output,
            sample_id=arguments.sample_id,
            expected_raw_sha256=arguments.expected_raw_sha256,
            expected_receipt_sha256=arguments.expected_receipt_sha256,
            expected_receipt_content_sha256=(arguments.expected_receipt_content_sha256),
            expected_frame_count=arguments.expected_frame_count,
        )
    except (HolisticMotionIntegrationError, OSError) as exc:
        print(f"Holistic motion conversion failed: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(canonical_json_bytes(result.record) + b"\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main
    raise SystemExit(main())
