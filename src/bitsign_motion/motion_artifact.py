from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .canonical import canonical_json_bytes, canonical_json_sha256
from .portable_model import MOTION_FEATURE_DIM

S1_MOTION_ARTIFACT_SCHEMA: Final = "umi-s1-motion-artifact/1"
S1_MOTION_ARTIFACT_EVIDENCE_SCHEMA: Final = "umi-s1-motion-artifact-evidence/1"
S1_TARGET_FEATURE_PROFILE: Final = "normalized-monocular-3d/1"

MOTION_FRAME_COUNT: Final = 120
MOTION_POINT_COUNT: Final = 107
MOTION_POINT_WIDTH: Final = 11
MOTION_GLOBAL_WIDTH: Final = 7

POINT_FIELDS: Final = (
    "x",
    "y",
    "z",
    "delta_x",
    "delta_y",
    "delta_z",
    "presence",
    "visibility",
    "missing",
    "presence_available",
    "visibility_available",
)
GLOBAL_FIELDS: Final = (
    "normalized_time",
    "delta_time",
    "motion_boundary",
    "pose_present",
    "left_hand_present",
    "right_hand_present",
    "face_present",
)
TRACK_SLICES: Final = (
    slice(0, 33),
    slice(33, 54),
    slice(54, 75),
    slice(75, 107),
)

_SAFE_SAMPLE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_POSITIVE_ZERO32 = np.float32(0.0)
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]
BoolArray = NDArray[np.bool_]
Int64Array = NDArray[np.int64]


class MotionArtifactError(ValueError):
    """Raised when canonical motion features or their persisted artifact are invalid."""


@dataclass(frozen=True, slots=True)
class MotionFeaturePolicy:
    coordinate_minimum: float = -8.0
    coordinate_maximum: float = 8.0
    delta_minimum: float = -4.0
    delta_maximum: float = 4.0

    def __post_init__(self) -> None:
        values = (
            self.coordinate_minimum,
            self.coordinate_maximum,
            self.delta_minimum,
            self.delta_maximum,
        )
        if any(type(value) is not float or not np.isfinite(value) for value in values):
            raise MotionArtifactError("motion clamp bounds must be finite floats")
        if self.coordinate_minimum >= self.coordinate_maximum:
            raise MotionArtifactError("coordinate clamp bounds are not ordered")
        if self.delta_minimum >= self.delta_maximum:
            raise MotionArtifactError("delta clamp bounds are not ordered")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "normalized-monocular-3d-feature-policy/1",
            "feature_profile": S1_TARGET_FEATURE_PROFILE,
            "coordinate_clamp": [self.coordinate_minimum, self.coordinate_maximum],
            "delta_clamp": [self.delta_minimum, self.delta_maximum],
            "score_clamp": [0.0, 1.0],
            "point_fields": list(POINT_FIELDS),
            "global_fields": list(GLOBAL_FIELDS),
            "rounding": "binary32-compose-then-binary16-round-to-nearest-ties-to-even",
            "padding": "positive-zero-with-frame-mask-zero",
        }


DEFAULT_MOTION_FEATURE_POLICY: Final = MotionFeaturePolicy()


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalPointSequence:
    coordinates: Float32Array
    coordinate_present: BoolArray
    presence: Float32Array
    presence_available: BoolArray
    visibility: Float32Array
    visibility_available: BoolArray
    timestamps_us: Int64Array
    clip_start_us: int
    clip_end_us: int
    track_present: BoolArray
    motion_boundary: BoolArray

    @property
    def frame_count(self) -> int:
        return int(self.coordinates.shape[0])

    def __repr__(self) -> str:
        return (
            "CanonicalPointSequence("
            f"frame_count={self.frame_count}, point_count={MOTION_POINT_COUNT}, "
            f"clip_start_us={self.clip_start_us}, clip_end_us={self.clip_end_us})"
        )


@dataclass(frozen=True, slots=True)
class ComposedMotion:
    motion: Float32Array
    frame_mask: Int32Array
    valid_frame_count: int
    feature_policy_sha256: str
    float32_tensor_sha256: str
    float16_projection_sha256: str


@dataclass(frozen=True, slots=True)
class WrittenMotionArtifact:
    path: Path
    artifact_sha256: str
    byte_count: int
    evidence: dict[str, object]


def _is_positive_zero(array: np.ndarray) -> bool:
    return bool(np.all(array == 0) and not np.signbit(array).any())


def _validate_point_sequence(sequence: CanonicalPointSequence) -> None:
    if not isinstance(sequence, CanonicalPointSequence):
        raise TypeError("sequence must be a CanonicalPointSequence")
    frame_count = sequence.frame_count
    if not 1 <= frame_count <= MOTION_FRAME_COUNT:
        raise MotionArtifactError("motion sequence must contain 1 through 120 frames")
    expected_points = (frame_count, MOTION_POINT_COUNT)
    if (
        not isinstance(sequence.coordinates, np.ndarray)
        or sequence.coordinates.dtype != np.float32
        or sequence.coordinates.shape != (*expected_points, 3)
        or not np.isfinite(sequence.coordinates).all()
    ):
        raise MotionArtifactError("coordinates must be finite float32 [frames,107,3]")
    for name, value in (
        ("coordinate_present", sequence.coordinate_present),
        ("presence_available", sequence.presence_available),
        ("visibility_available", sequence.visibility_available),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.bool_
            or value.shape != expected_points
        ):
            raise MotionArtifactError(f"{name} must be bool [frames,107]")
    for name, value in (("presence", sequence.presence), ("visibility", sequence.visibility)):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float32
            or value.shape != expected_points
            or not np.isfinite(value).all()
        ):
            raise MotionArtifactError(f"{name} must be finite float32 [frames,107]")
    if not _is_positive_zero(sequence.coordinates[~sequence.coordinate_present]):
        raise MotionArtifactError("missing coordinates must be explicit positive zero")
    if not _is_positive_zero(sequence.presence[~sequence.presence_available]):
        raise MotionArtifactError("unavailable presence values must be explicit positive zero")
    if not _is_positive_zero(sequence.visibility[~sequence.visibility_available]):
        raise MotionArtifactError("unavailable visibility values must be explicit positive zero")
    if (
        not isinstance(sequence.timestamps_us, np.ndarray)
        or sequence.timestamps_us.dtype != np.int64
        or sequence.timestamps_us.shape != (frame_count,)
    ):
        raise MotionArtifactError("timestamps_us must be int64 [frames]")
    if frame_count > 1 and np.any(sequence.timestamps_us[1:] < sequence.timestamps_us[:-1]):
        raise MotionArtifactError("motion timestamps must be nondecreasing")
    if type(sequence.clip_start_us) is not int or type(sequence.clip_end_us) is not int:
        raise MotionArtifactError("clip bounds must be integer microseconds")
    if sequence.clip_end_us <= sequence.clip_start_us:
        raise MotionArtifactError("clip end must be later than clip start")
    if (
        int(sequence.timestamps_us[0]) < sequence.clip_start_us
        or int(sequence.timestamps_us[-1]) > sequence.clip_end_us
    ):
        raise MotionArtifactError("motion timestamps escape the displayed clip interval")
    if (
        not isinstance(sequence.track_present, np.ndarray)
        or sequence.track_present.dtype != np.bool_
        or sequence.track_present.shape != (frame_count, 4)
    ):
        raise MotionArtifactError("track_present must be bool [frames,4]")
    if (
        not isinstance(sequence.motion_boundary, np.ndarray)
        or sequence.motion_boundary.dtype != np.bool_
        or sequence.motion_boundary.shape != (frame_count,)
    ):
        raise MotionArtifactError("motion_boundary must be bool [frames]")
    for track_index, point_slice in enumerate(TRACK_SLICES):
        observed = sequence.coordinate_present[:, point_slice].any(axis=1)
        if not np.array_equal(observed, sequence.track_present[:, track_index]):
            raise MotionArtifactError("track-present flags disagree with point coordinates")


def _tensor_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def compose_motion(
    sequence: CanonicalPointSequence,
    policy: MotionFeaturePolicy = DEFAULT_MOTION_FEATURE_POLICY,
) -> ComposedMotion:
    """Compose the fixed S1 input tensor from points already in the body basis."""

    if not isinstance(policy, MotionFeaturePolicy):
        raise TypeError("policy must be a MotionFeaturePolicy")
    _validate_point_sequence(sequence)
    frame_count = sequence.frame_count
    point_features = np.zeros(
        (frame_count, MOTION_POINT_COUNT, MOTION_POINT_WIDTH),
        dtype=np.float32,
    )
    point_features[:, :, 0:3] = np.clip(
        sequence.coordinates,
        np.float32(policy.coordinate_minimum),
        np.float32(policy.coordinate_maximum),
    )
    point_features[:, :, 0:3][~sequence.coordinate_present] = _POSITIVE_ZERO32
    if frame_count > 1:
        valid_difference = sequence.coordinate_present[1:] & sequence.coordinate_present[:-1]
        difference = np.asarray(
            sequence.coordinates[1:] - sequence.coordinates[:-1],
            dtype=np.float32,
        )
        difference = np.clip(
            difference,
            np.float32(policy.delta_minimum),
            np.float32(policy.delta_maximum),
        )
        difference[~valid_difference] = _POSITIVE_ZERO32
        point_features[1:, :, 3:6] = difference
    point_features[:, :, 6] = np.where(
        sequence.presence_available,
        np.clip(sequence.presence, np.float32(0.0), np.float32(1.0)),
        _POSITIVE_ZERO32,
    )
    point_features[:, :, 7] = np.where(
        sequence.visibility_available,
        np.clip(sequence.visibility, np.float32(0.0), np.float32(1.0)),
        _POSITIVE_ZERO32,
    )
    point_features[:, :, 8] = (~sequence.coordinate_present).astype(np.float32)
    point_features[:, :, 9] = sequence.presence_available.astype(np.float32)
    point_features[:, :, 10] = sequence.visibility_available.astype(np.float32)

    globals_ = np.zeros((frame_count, MOTION_GLOBAL_WIDTH), dtype=np.float32)
    duration = sequence.clip_end_us - sequence.clip_start_us
    normalized_time = np.asarray(
        [
            np.float32((int(timestamp) - sequence.clip_start_us) / duration)
            for timestamp in sequence.timestamps_us
        ],
        dtype=np.float32,
    )
    globals_[:, 0] = normalized_time
    if frame_count > 1:
        globals_[1:, 1] = np.asarray(normalized_time[1:] - normalized_time[:-1], dtype=np.float32)
    globals_[:, 2] = sequence.motion_boundary.astype(np.float32)
    globals_[:, 3:7] = sequence.track_present.astype(np.float32)

    motion = np.zeros((MOTION_FRAME_COUNT, MOTION_FEATURE_DIM), dtype=np.float32)
    motion[:frame_count, : MOTION_POINT_COUNT * MOTION_POINT_WIDTH] = point_features.reshape(
        frame_count, -1
    )
    motion[:frame_count, -MOTION_GLOBAL_WIDTH:] = globals_
    frame_mask = np.zeros(MOTION_FRAME_COUNT, dtype=np.int32)
    frame_mask[:frame_count] = 1
    if not np.isfinite(motion).all() or not _is_positive_zero(motion[frame_count:]):
        raise MotionArtifactError("composed motion tensor is nonfinite or has invalid padding")
    float16_projection = np.asarray(motion, dtype=np.float16)
    if not np.isfinite(float16_projection).all():
        raise MotionArtifactError("motion tensor cannot be projected safely to binary16")
    motion.setflags(write=False)
    frame_mask.setflags(write=False)
    return ComposedMotion(
        motion=motion,
        frame_mask=frame_mask,
        valid_frame_count=frame_count,
        feature_policy_sha256=canonical_json_sha256(policy.as_dict()),
        float32_tensor_sha256=_tensor_sha256(motion),
        float16_projection_sha256=_tensor_sha256(float16_projection),
    )


def _npy_bytes(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, np.asarray(array), allow_pickle=False)
    return output.getvalue()


def encode_motion_artifact(sample_id: str, composed: ComposedMotion) -> bytes:
    if not isinstance(sample_id, str) or _SAFE_SAMPLE_ID.fullmatch(sample_id) is None:
        raise MotionArtifactError("sample_id must be a lowercase filesystem-safe identifier")
    if not isinstance(composed, ComposedMotion):
        raise TypeError("composed must be a ComposedMotion")
    metadata = {
        "schema": S1_MOTION_ARTIFACT_SCHEMA,
        "sample_id": sample_id,
        "feature_profile": S1_TARGET_FEATURE_PROFILE,
        "motion_shape": [MOTION_FRAME_COUNT, MOTION_FEATURE_DIM],
        "motion_dtype": "float32",
        "frame_mask_shape": [MOTION_FRAME_COUNT],
        "frame_mask_dtype": "int32",
    }
    members = {
        "motion.npy": _npy_bytes(composed.motion),
        "frame_mask.npy": _npy_bytes(composed.frame_mask),
        "metadata.npy": _npy_bytes(np.asarray(canonical_json_bytes(metadata).decode("utf-8"))),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in ("motion.npy", "frame_mask.npy", "metadata.npy"):
            info = zipfile.ZipInfo(name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, members[name])
    return output.getvalue()


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:  # pragma: no cover - os.write raises for normal failures
            raise OSError("motion artifact write made no progress")
        offset += written


def _unlink_owned_file(path: Path, metadata: os.stat_result) -> None:
    try:
        observed = path.lstat()
    except OSError:
        return
    if observed.st_dev == metadata.st_dev and observed.st_ino == metadata.st_ino:
        try:
            path.unlink()
        except OSError:
            pass


def write_motion_artifact(
    path: Path,
    *,
    sample_id: str,
    composed: ComposedMotion,
) -> WrittenMotionArtifact:
    payload = encode_motion_artifact(sample_id, composed)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    closed = False
    try:
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("motion artifact output must be a regular file")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        final_metadata = os.fstat(descriptor)
        if final_metadata.st_size != len(payload):
            raise OSError("motion artifact size changed during publication")
        os.close(descriptor)
        closed = True
    except BaseException:
        if not closed:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _unlink_owned_file(path, metadata)
        raise
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


def inspect_motion_artifact_metadata(encoded: bytes) -> dict[str, object]:
    """Read only the small canonical metadata member from an in-memory artifact."""

    try:
        with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
            metadata = archive["metadata"]
            if metadata.shape != () or metadata.dtype.kind != "U":
                raise MotionArtifactError("motion artifact metadata array is invalid")
            value = json.loads(metadata.item())
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise MotionArtifactError("motion artifact metadata cannot be decoded") from exc
    if not isinstance(value, dict):
        raise MotionArtifactError("motion artifact metadata must be an object")
    return value
