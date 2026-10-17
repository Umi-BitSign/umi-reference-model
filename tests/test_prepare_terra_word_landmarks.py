from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_terra_word_landmarks as landmarks


def case():
    record = {
        "clip": {"path": "data/training/word.mp4", "sha256": "video"},
        "normalized_probe": {"width": 640, "height": 480, "decoded_frames": 3},
    }
    metadata = {
        "source_sha256": "video",
        "frames": 3,
        "fps": 25,
        "width": 640,
        "height": 480,
        "mediapipe_version": "0.10.14",
        "delegate": "cpu",
    }
    rows = {
        i: {"hand_landmarks": [[[0.1, 0.2, 0.3]]], "face_landmarks": None, "pose_landmarks": None}
        for i in range(3)
    }
    return record, metadata, rows


def test_complete_native_timeline_and_missing_detection_behavior_are_preserved(monkeypatch):
    record, metadata, rows = case()
    rows[1] = None
    monkeypatch.setattr(landmarks, "unpack_landmarks", lambda path: (rows, metadata))
    assert landmarks.checked_landmarks(Path("unused"), record) == metadata


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sha256", "changed"),
        ("frames", 2),
        ("fps", 30),
        ("width", 320),
        ("height", 240),
        ("mediapipe_version", "other"),
        ("delegate", "gpu"),
    ],
)
def test_changed_source_timeline_geometry_detector_or_delegate_cannot_pass(
    monkeypatch, field, value
):
    record, metadata, rows = case()
    metadata[field] = value
    monkeypatch.setattr(landmarks, "unpack_landmarks", lambda path: (rows, metadata))
    with pytest.raises(ValueError):
        landmarks.checked_landmarks(Path("unused"), record)


@pytest.mark.parametrize("kind", ["missing_frame", "reordered_frame_key", "active_nan"])
def test_missing_frame_indices_and_nonfinite_active_coordinates_are_rejected(monkeypatch, kind):
    record, metadata, rows = case()
    if kind == "missing_frame":
        rows.pop(1)
    elif kind == "reordered_frame_key":
        rows[4] = rows.pop(1)
    else:
        rows[1]["hand_landmarks"][0][0][0] = np.nan
    monkeypatch.setattr(landmarks, "unpack_landmarks", lambda path: (rows, metadata))
    with pytest.raises(ValueError):
        landmarks.checked_landmarks(Path("unused"), record)


def test_existing_container_contract_has_no_network_and_has_bounded_cpu_memory(tmp_path):
    clip = landmarks.ROOT / "data/asl-stem-wiki/word-pilot-v1/clips-25fps/training/word.mp4"
    command = landmarks.container_command(
        docker_socket=landmarks.DOCKER_SOCKET,
        clip=clip,
        output_name="word.npz",
        landmark_dir=tmp_path,
    )
    before = copy.deepcopy(command)
    command[command.index(landmarks.IMAGE)] = "sha256:" + "1" * 64
    assert before != command
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--cpus") + 1] == "2"
    assert command[command.index("--memory") + 1] == "3g"
    assert f"{landmarks.ROOT}:/workspace:ro" in command
    assert landmarks.POLICY["service_checks_performed"] is False
    assert landmarks.POLICY["model_training_ready"] is False
