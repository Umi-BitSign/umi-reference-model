from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from extract_mediapipe_landmarks import unpack_landmarks, write_landmarks


def _points(count: int, offset: float = 0.0) -> list[list[float]]:
    return [[offset + index, offset + index + 0.25, offset - index] for index in range(count)]


def test_landmark_sidecar_round_trip_without_pickle(tmp_path: Path) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    sidecar = tmp_path / "clip.npz"
    rows = {
        0: {
            "face_landmarks": [_points(478)],
            "hand_landmarks": [_points(21, 1), _points(21, 2)],
            "pose_landmarks": [_points(33, 3)],
        },
        1: {
            "face_landmarks": None,
            "hand_landmarks": None,
            "pose_landmarks": None,
        },
        2: None,
    }
    metadata = write_landmarks(
        sidecar,
        rows,
        source=source,
        width=640,
        height=480,
        fps=25.0,
        mediapipe_version="test",
    )
    restored, restored_metadata = unpack_landmarks(sidecar)
    assert restored == rows
    assert restored_metadata == metadata
    assert restored_metadata["source_sha256"]


def test_landmark_sidecar_rejects_noncontiguous_frames() -> None:
    from extract_mediapipe_landmarks import packed_landmarks

    try:
        packed_landmarks({1: None})
    except ValueError as error:
        assert "contiguous" in str(error)
    else:
        raise AssertionError("noncontiguous landmark frames were accepted")
