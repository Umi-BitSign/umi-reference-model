from __future__ import annotations

import hashlib
import io
import json
import stat
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from bitsign_motion.canonical import canonical_json_bytes, canonical_json_sha256
from bitsign_motion.holistic_container import (
    RAW_FILENAME,
    RAW_SCHEMA,
    RECEIPT_FILENAME,
    RECEIPT_SCHEMA,
    TRACK_FIELDS,
    TRACK_SPECS,
    HolisticExtractionResult,
)
from bitsign_motion.holistic_motion import (
    HOLISTIC_MOTION_CONVERSION_SCHEMA,
    HolisticMotionIntegrationError,
    convert_holistic_extraction_to_motion,
    main,
)
from bitsign_motion.motion_artifact import inspect_motion_artifact_metadata

FRAME_COUNT = 16
SAMPLE_ID = "fixture-one"
CLIP_START_US = 1_000_000
CLIP_END_US = 3_000_000


def _arrays(*, metadata_schema: str = RAW_SCHEMA) -> dict[str, np.ndarray]:
    timestamps = np.arange(FRAME_COUNT, dtype=np.int64) * 125_000 + CLIP_START_US + 62_500
    arrays: dict[str, np.ndarray] = {
        "timestamps_us": timestamps,
        "requested_timestamps_us": np.array(timestamps, copy=True),
        "source_timestamps_us": np.array(timestamps, copy=True),
        "displayed_timestamps_us": np.array(timestamps, copy=True),
        "source_pts": np.array(timestamps, copy=True),
        "source_time_base": np.asarray([1, 1_000_000], dtype=np.int64),
        "source_frame_indices": np.arange(FRAME_COUNT, dtype=np.int32),
        "clip_bounds_us": np.asarray([CLIP_START_US, CLIP_END_US], dtype=np.int64),
        "displayed_rgb_sha256": np.zeros((FRAME_COUNT, 32), dtype=np.uint8),
    }
    for prefix, points in TRACK_SPECS.items():
        for field, (_, shape_builder) in TRACK_FIELDS.items():
            shape = shape_builder(FRAME_COUNT, points)
            dtype = np.float32 if field in {"coordinates", "presence", "visibility"} else np.bool_
            arrays[f"{prefix}_{field}"] = np.zeros(shape, dtype=dtype)

    pose_world = arrays["pose_world_coordinates"]
    pose_world_available = arrays["pose_world_coordinate_available"]
    pose_world[:, 11] = np.asarray([-1.0, 0.0, 0.0], dtype=np.float32)
    pose_world[:, 12] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    pose_world_available[:, [11, 12]] = True
    metadata = {
        "schema": metadata_schema,
        "sample_id": SAMPLE_ID,
        "status": "ex-203-candidate-only",
        "canonical_or_release_quality": False,
        "hand_slot_semantics": "opaque-observation-slots-mapper-assigns-anatomical-side",
        "track_prefixes": [
            "pose_image",
            "observed_hand_image_0",
            "observed_hand_image_1",
            "face_image",
            "pose_world",
            "observed_hand_world_0",
            "observed_hand_world_1",
        ],
    }
    arrays["metadata_json_utf8"] = np.frombuffer(
        canonical_json_bytes(metadata), dtype=np.uint8
    ).copy()
    return arrays


def _npy(array: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.lib.format.write_array(output, array, allow_pickle=False)
    return output.getvalue()


def _raw_payload(arrays: dict[str, np.ndarray]) -> tuple[bytes, list[dict[str, object]]]:
    inventory = []
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            array = arrays[name]
            inventory.append(
                {
                    "name": name,
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "tensor_sha256": hashlib.sha256(
                        memoryview(np.ascontiguousarray(array)).cast("B")
                    ).hexdigest(),
                }
            )
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            info.create_system = 3
            archive.writestr(info, _npy(array))
    return output.getvalue(), inventory


def _receipt(
    *,
    raw_payload: bytes,
    inventory: list[dict[str, object]],
    frame_count: int = FRAME_COUNT,
) -> tuple[bytes, str]:
    timestamps = (
        np.arange(FRAME_COUNT, dtype=np.int64) * 125_000 + CLIP_START_US + 62_500
    ).tolist()
    source_timestamps = list(timestamps)
    source_pts = list(timestamps)
    source_indices = np.arange(FRAME_COUNT, dtype=np.int32).tolist()
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "sample_id": SAMPLE_ID,
        "status": "ex-203-candidate-only",
        "claim_boundary": {
            "canonical_or_release_quality": False,
            "cross_runtime_equivalence_established": False,
            "source_is_umi_challenge_media": False,
            "purpose": "bootstrap-timed-source-isolated-raw-holistic-research",
        },
        "container": {"image_id": "sha256:" + "a" * 64},
        "input_video": {"sha256": "b" * 64},
        "model": {"sha256": "c" * 64},
        "source_media": {
            "duration": "4/1",
            "duration_us": 4_000_000,
            "nominal_frame_rate": "30/1",
            "clip_end_tolerance_policy": "source-metadata-rounding-at-most-one-nominal-frame-v1",
            "clip_end_overshoot": "0/1",
            "maximum_clip_end_overshoot": "1/30",
            "source_time_base": "1/1000000",
        },
        "clip": {
            "clip_start_us": CLIP_START_US,
            "clip_end_us": CLIP_END_US,
            "duration_us": CLIP_END_US - CLIP_START_US,
            "interval": "half-open-[start,end)",
        },
        "derived_view": {"dimensions": [640, 480]},
        "selection": {
            "frame_count": frame_count,
            "clip_bounds_us": [CLIP_START_US, CLIP_END_US],
            "requested_timestamps_us": timestamps,
            "source_timestamps_us": source_timestamps,
            "source_frame_indices": source_indices,
            "source_pts": source_pts,
            "displayed_rgb_sha256": ["00" * 32] * FRAME_COUNT,
        },
        "extractor": {"mediapipe_version": "1.0.1"},
        "raw_artifact": {
            "schema": RAW_SCHEMA,
            "filename": RAW_FILENAME,
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
            "byte_count": len(raw_payload),
            "array_inventory": inventory,
            "hand_observation_slots": [{"slot": 0}, {"slot": 1}],
            "anatomical_side_assignment": "deferred-to-mapping-layer",
        },
        "publication": {
            "directory": SAMPLE_ID,
            "raw_filename": RAW_FILENAME,
            "receipt_filename": RECEIPT_FILENAME,
        },
    }
    content_sha256 = hashlib.sha256(
        b"umi-raw-holistic-extraction-receipt-v3\0" + canonical_json_bytes(receipt)
    ).hexdigest()
    receipt["content_sha256"] = content_sha256
    return canonical_json_bytes(receipt), content_sha256


def _write_private(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)
    path.chmod(0o600)


def _extraction(
    tmp_path: Path,
    *,
    metadata_schema: str = RAW_SCHEMA,
    receipt_frame_count: int = FRAME_COUNT,
) -> HolisticExtractionResult:
    arrays = _arrays(metadata_schema=metadata_schema)
    raw_payload, inventory = _raw_payload(arrays)
    receipt_payload, receipt_content_sha256 = _receipt(
        raw_payload=raw_payload,
        inventory=inventory,
        frame_count=receipt_frame_count,
    )
    sample = tmp_path / "raw" / SAMPLE_ID
    sample.mkdir(mode=0o700, parents=True)
    sample.chmod(0o700)
    raw = sample / RAW_FILENAME
    receipt = sample / RECEIPT_FILENAME
    _write_private(raw, raw_payload)
    _write_private(receipt, receipt_payload)
    return HolisticExtractionResult(
        sample_id=SAMPLE_ID,
        image_id="sha256:" + "a" * 64,
        raw_artifact_path=raw,
        receipt_path=receipt,
        raw_artifact_sha256=hashlib.sha256(raw_payload).hexdigest(),
        receipt_sha256=hashlib.sha256(receipt_payload).hexdigest(),
        receipt_content_sha256=receipt_content_sha256,
        frame_count=FRAME_COUNT,
        status="ex-203-candidate-only",
    )


def _output(tmp_path: Path) -> Path:
    output_root = tmp_path / "motion"
    output_root.mkdir(mode=0o700)
    output_root.chmod(0o700)
    return output_root / f"{SAMPLE_ID}.npz"


def test_verified_raw_tracks_map_compose_and_publish_idempotently(tmp_path: Path) -> None:
    extraction = _extraction(tmp_path)
    output = _output(tmp_path)
    result = convert_holistic_extraction_to_motion(extraction, output_path=output)

    assert result.frame_count == FRAME_COUNT
    assert result.record["schema"] == HOLISTIC_MOTION_CONVERSION_SCHEMA
    assert result.record["status"] == "ex-203-candidate-only"
    assert result.record["claim_boundary"] == {
        "canonical_or_release_quality": False,
        "cross_runtime_equivalence_established": False,
        "ex203_complete": False,
    }
    record_without_digest = dict(result.record)
    content_sha256 = record_without_digest.pop("content_sha256")
    assert content_sha256 == canonical_json_sha256(
        record_without_digest,
        domain=b"umi-holistic-motion-conversion-v1\0",
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    metadata = inspect_motion_artifact_metadata(output.read_bytes())
    assert metadata["sample_id"] == SAMPLE_ID
    assert metadata["motion_shape"] == [120, 1184]
    with np.load(output, allow_pickle=False) as archive:
        assert archive["motion"].shape == (120, 1184)
        assert archive["frame_mask"].tolist() == [1] * FRAME_COUNT + [0] * (120 - FRAME_COUNT)
    first_bytes = output.read_bytes()
    repeated = convert_holistic_extraction_to_motion(extraction, output_path=output)
    assert output.read_bytes() == first_bytes
    assert repeated.motion_artifact == result.motion_artifact
    assert repeated.record == result.record


def test_different_existing_motion_artifact_is_never_replaced(tmp_path: Path) -> None:
    extraction = _extraction(tmp_path)
    output = _output(tmp_path)
    convert_holistic_extraction_to_motion(extraction, output_path=output)
    substituted = bytearray(output.read_bytes())
    substituted[len(substituted) // 2] ^= 1
    output.write_bytes(substituted)
    output.chmod(0o600)

    with pytest.raises(HolisticMotionIntegrationError, match="differs"):
        convert_holistic_extraction_to_motion(extraction, output_path=output)
    assert output.read_bytes() == substituted


def test_existing_symlink_or_hardlink_motion_path_is_rejected(tmp_path: Path) -> None:
    extraction = _extraction(tmp_path)
    output = _output(tmp_path)
    convert_holistic_extraction_to_motion(extraction, output_path=output)
    original = output.read_bytes()

    hardlink = output.parent / "hardlink.npz"
    hardlink.hardlink_to(output)
    with pytest.raises(HolisticMotionIntegrationError, match="regular artifact"):
        convert_holistic_extraction_to_motion(extraction, output_path=output)
    assert output.read_bytes() == original

    hardlink.unlink()
    output.unlink()
    target = output.parent / "target.npz"
    target.write_bytes(original)
    target.chmod(0o600)
    output.symlink_to(target.name)
    with pytest.raises(HolisticMotionIntegrationError, match="opened safely"):
        convert_holistic_extraction_to_motion(extraction, output_path=output)
    assert output.is_symlink()
    assert target.read_bytes() == original


def test_cli_publishes_and_prints_canonical_conversion_record(
    tmp_path: Path, capsysbinary: pytest.CaptureFixture[bytes]
) -> None:
    extraction = _extraction(tmp_path)
    output = _output(tmp_path)
    exit_code = main(
        [
            "--sample-id",
            SAMPLE_ID,
            "--raw-npz",
            str(extraction.raw_artifact_path),
            "--receipt",
            str(extraction.receipt_path),
            "--expected-raw-sha256",
            extraction.raw_artifact_sha256,
            "--expected-receipt-sha256",
            extraction.receipt_sha256,
            "--expected-receipt-content-sha256",
            extraction.receipt_content_sha256,
            "--expected-frame-count",
            str(extraction.frame_count),
            "--output",
            str(output),
        ]
    )
    captured = capsysbinary.readouterr()
    assert exit_code == 0
    assert captured.err == b""
    assert captured.out.endswith(b"\n") and captured.out.count(b"\n") == 1
    parsed = json.loads(captured.out)
    assert canonical_json_bytes(parsed) + b"\n" == captured.out
    assert (
        parsed["motion_artifact"]["artifact_sha256"]
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )


def test_external_raw_hash_mismatch_fails_before_publication(tmp_path: Path) -> None:
    extraction = replace(_extraction(tmp_path), raw_artifact_sha256="0" * 64)
    output = _output(tmp_path)
    with pytest.raises(HolisticMotionIntegrationError, match="digest"):
        convert_holistic_extraction_to_motion(extraction, output_path=output)
    assert not output.exists()


def test_rehashed_wrong_raw_schema_still_fails_semantic_validation(tmp_path: Path) -> None:
    extraction = _extraction(tmp_path, metadata_schema="attacker-schema/1")
    output = _output(tmp_path)
    with pytest.raises(HolisticMotionIntegrationError, match="metadata binding"):
        convert_holistic_extraction_to_motion(extraction, output_path=output)
    assert not output.exists()


def test_frame_count_substitution_fails_against_array_inventory(tmp_path: Path) -> None:
    extraction = _extraction(tmp_path, receipt_frame_count=17)
    output = _output(tmp_path)
    with pytest.raises(HolisticMotionIntegrationError, match="bind"):
        convert_holistic_extraction_to_motion(extraction, output_path=output)
    assert not output.exists()


def test_receipt_path_from_another_sample_directory_is_rejected(tmp_path: Path) -> None:
    extraction = _extraction(tmp_path)
    other = tmp_path / "raw" / "other-sample"
    other.mkdir(mode=0o700)
    other.chmod(0o700)
    substituted = other / RECEIPT_FILENAME
    _write_private(substituted, extraction.receipt_path.read_bytes())
    extraction = replace(extraction, receipt_path=substituted)
    output = _output(tmp_path)

    with pytest.raises(HolisticMotionIntegrationError, match="share"):
        convert_holistic_extraction_to_motion(extraction, output_path=output)
    assert not output.exists()
