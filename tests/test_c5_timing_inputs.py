"""Synthetic source/visibility/cache checks; no video, model or grader execution."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_timing_native_input as inputs
import c5_timing_visibility as visibility
import prepare_c5_timing_features as producer


def hand_row(count=1):
    return {"hand_landmarks": np.zeros((count, 21, 3)).tolist()}


def test_unsampled_missing_frame_breaks_interval_without_dropping_sampled_frames():
    rows = {i: hand_row() for i in range(5)}
    rows[1] = None
    observed, valid = visibility.visibility(rows, 5, np.array([0, 2, 4], dtype=np.int64))
    assert observed.tolist() == [True, False, True, True, True]
    assert valid.tolist() == [False, True]


@pytest.mark.parametrize("row", [None, {"hand_landmarks": None}, hand_row(0), hand_row(3)])
def test_missing_or_ambiguous_multi_hand_frames_are_not_visible(row):
    assert visibility.visible(row) is False


def test_visibility_uses_xy_tips_without_clipping_or_palm_and_depth_inputs():
    row = hand_row(2)
    a = np.asarray(row["hand_landmarks"])
    a[:, :, 2] = np.nan
    a[:, visibility.TIPS, :2] = 10
    a[:, 0, :2] = np.nan
    assert visibility.visible({"hand_landmarks": a}) is True
    a[0, 8, 0] = np.nan
    assert visibility.visible({"hand_landmarks": a}) is False


@pytest.mark.parametrize(
    "indices",
    [
        np.array([0, 4], dtype=np.int64),
        np.array([0, 2, 4], dtype=np.int32),
        np.array([1, 3], dtype=np.int64),
    ],
)
def test_incomplete_or_retimed_sampling_grid_is_rejected(indices):
    with pytest.raises(ValueError, match="complete original stride2"):
        visibility.interval_mask(np.ones(5, dtype=np.bool_), indices)


def test_incomplete_detector_timeline_and_malformed_hands_are_rejected():
    with pytest.raises(ValueError, match="every original detector frame"):
        visibility.visibility({0: hand_row()}, 3, np.array([0, 2], dtype=np.int64))
    with pytest.raises(ValueError, match="21-point"):
        visibility.visible({"hand_landmarks": np.zeros((1, 5, 3))})


@pytest.fixture
def native(monkeypatch, tmp_path):
    monkeypatch.setattr(inputs, "ROOT", tmp_path)
    clip = {"path": "data/video.mp4", "sha256": "synthetic-video"}
    sidecar = {"path": "data/landmarks.npz", "sha256": "synthetic-landmarks"}
    case = {"kind": "word", "word": {"record": {"clip": clip, "landmarks": sidecar}}}
    monkeypatch.setattr(inputs.mixed, "training_row", lambda c: {"role": "training"})
    header = {"source_fps": 25, "source_frames": 5, "source_shape": [480, 640], "sampled_frames": 3}
    pose = np.arange(42, dtype=np.float32).reshape(3, 14)
    pose[0, 0] = -0.0
    verified = {
        "pixels": np.full((3, 3, 112, 112, 3), 200, dtype=np.uint8),
        "pose_features": pose,
        "frame_indices": np.array([0, 2, 4], dtype=np.int64),
        "header": header,
    }
    monkeypatch.setattr(inputs.mixed, "read_input", lambda c: copy.deepcopy(verified))
    monkeypatch.setattr(inputs.mixed, "expected_header", lambda c: copy.deepcopy(header))
    monkeypatch.setattr(inputs, "media_artifact", lambda p: clip if p.suffix == ".mp4" else sidecar)
    rows = {i: hand_row() for i in range(5)}
    rows[1] = None
    metadata = {
        "source_sha256": clip["sha256"],
        "frames": 5,
        "fps": 25,
        "height": 480,
        "width": 640,
    }
    monkeypatch.setattr(inputs, "unpack_landmarks", lambda p: (rows, metadata))
    state = {"mismatch": False}

    class Reader:
        def __init__(self, *args):
            self.reference = False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Wrapper:
        def __init__(self, *args):
            self.reference = True

    def crops(reader, sampled):
        delta = int(reader.reference and state["mismatch"])
        return [
            np.broadcast_to(
                np.array([10 + i + delta, 20 + i, 30 + i], dtype=np.uint8), (224, 224, 3)
            ).copy()
            for i in range(len(sampled))
        ]

    class Hands:
        def __init__(self, output_size):
            assert output_size == (224, 224)

        def extract_hand_frames(self, reader, sampled):
            return crops(reader, sampled), crops(reader, sampled)

    class Face(Hands):
        def extract_face_frames(self, reader, sampled):
            return crops(reader, sampled)

    monkeypatch.setattr(inputs, "SampledRgbReader", Reader)
    monkeypatch.setattr(inputs, "NativeSampledReader", Wrapper)
    monkeypatch.setattr(inputs, "native_helpers", lambda: (Hands, Face, Reader))
    return SimpleNamespace(case=case, pose=pose, state=state, metadata=metadata)


def test_native_training224_is_fresh_full_size_rgb_with_exact_pose_and_visibility(native):
    values = inputs.read_training224(native.case)
    assert values["pixels"].shape == (3, 3, 224, 224, 3)
    assert values["pixels"][0, 0, 0, 0].tolist() == [30, 20, 10]
    assert values["pose_features"].tobytes() == native.pose.tobytes()
    assert values["valid_intervals"].tolist() == [False, True]
    assert inputs.input_receipt(values)["native_side"] == 224


def test_native_crop_mismatch_and_source_clock_changes_are_rejected(native):
    native.state["mismatch"] = True
    with pytest.raises(ValueError, match="crop pixels differ"):
        inputs.read_training224(native.case)
    native.state["mismatch"] = False
    native.metadata["fps"] = 24
    with pytest.raises(ValueError, match="original source timeline"):
        inputs.read_training224(native.case)


def test_changed_gap_mask_cannot_pass_input_receipt(native):
    values = inputs.read_training224(native.case)
    values["valid_intervals"][:] = True
    with pytest.raises(ValueError, match="bridged"):
        inputs.input_receipt(values)


def test_nontraining_role_fails_before_any_source_read(monkeypatch):
    def refuse(case):
        raise ValueError("reserved role")

    monkeypatch.setattr(inputs.mixed, "training_row", refuse)
    monkeypatch.setattr(inputs.mixed, "read_input", lambda c: pytest.fail("read reserved data"))
    with pytest.raises(ValueError, match="reserved role"):
        inputs.read_training224({"kind": "word"})


def test_native_feature_reference_keeps_region_order_and_fixed_error_limit():
    class Embedder:
        def extract_embeddings_from_frames(self, frames):
            return np.broadcast_to(
                frames[:, 0, 0, 0, None].astype(np.float32), (len(frames), 384)
            ).copy()

    pixels = np.zeros((3, 3, 224, 224, 3), dtype=np.uint8)
    pixels[:, 0] = 1
    pixels[:, 1] = 2
    pixels[:, 2] = 3
    indices = np.array([0, 2], dtype=np.int64)
    reference = producer.native_reference(Embedder(), Embedder(), pixels, indices)
    assert [
        reference[k][0, 0] for k in ("face_features", "left_hand_features", "right_hand_features")
    ] == [1, 2, 3]
    streams = {k: np.full((3, 384), v[0, 0], dtype=np.float32) for k, v in reference.items()}
    assert producer.compare_features(streams, reference, indices)["sample_indices"] == [0, 2]
    reference["right_hand_features"][-1, 0] += 0.01
    with pytest.raises(ValueError, match="feature comparison failed"):
        producer.compare_features(streams, reference, indices)


def test_storage_guard_retains_physical_floor_and_remaining_output_reserve(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "ROOT", tmp_path)
    p = producer.POLICY
    required = p["physical_free_floor_bytes"] + p["headroom_bytes"] + p["output_budget_bytes"]
    monkeypatch.setattr(producer.shutil, "disk_usage", lambda r: SimpleNamespace(free=required))
    producer.guard(tmp_path / "output")
    monkeypatch.setattr(producer.shutil, "disk_usage", lambda r: SimpleNamespace(free=required - 1))
    with pytest.raises(ValueError, match="physical reserve"):
        producer.guard(tmp_path / "output")


def test_release_blocks_actual_previous_worker_and_allows_unrelated_pid_reuse(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(producer, "PREDECESSOR", tmp_path)
    monkeypatch.setattr(producer, "artifact", lambda p: {"path": str(p), "sha256": "synthetic"})
    import json

    for name, value in {
        "worker.json": {"pid": 123, "uid": os.getuid(), "ps_started_local": "synthetic start"},
        "exit.json": {"pid": 123, "exit_code": 0},
        "complete.json": {"complete": True, "post_hoc_diagnostic_only": True},
    }.items():
        (tmp_path / name).write_text(json.dumps(value))
    monkeypatch.setattr(
        producer.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=f"{os.getuid()} synthetic start"),
    )
    with pytest.raises(ValueError, match="still present"):
        producer.release()
    monkeypatch.setattr(
        producer.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=f"{os.getuid()} different start"),
    )
    assert set(producer.release()) == {"worker.json", "exit.json", "complete.json"}
