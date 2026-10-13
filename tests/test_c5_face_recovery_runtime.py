from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_face_recovery_runtime import RecoveryLandmarks, recover_one_frame, install_face_recovery
from c5_stride2_runtime import install_post_mediapipe_stride


def row(face=None, pose=True):
    points = np.full((33, 3), .5).tolist()
    return {"face_landmarks": face, "hand_landmarks": None, "pose_landmarks": [points] if pose else None}


class Cropper:
    output_size = (2, 2)

    def extract_face_frames(self, frames, landmarks):
        # Deliberately retain a previous legacy crop across a missing detection.
        previous = np.full((2, 2, 3), 7, dtype=np.uint8)
        return [previous.copy() for _ in frames]

    def cues_on_grey_background(self, frame, points):
        return frame.copy()

    def resize_frame(self, frame, size):
        return frame.copy()

    def extract_hand_frames(self, frames, landmarks):
        return frames, landmarks


def runtime():
    return SimpleNamespace(_face_extractor=Cropper(), _hand_extractor=Cropper(),
                           _process_pose_landmarks=lambda landmarks: landmarks)


def test_overlay_keeps_legacy_entries_imputation_and_nonface_paths_unchanged():
    model = runtime(); install_post_mediapipe_stride(model)
    original_hand = model._hand_extractor; original_pose = model._process_pose_landmarks
    install_face_recovery(model)
    legacy = {i: row() for i in range(5)}; frozen = deepcopy(legacy)
    points = np.full((478, 3), .5)
    recovered = RecoveryLandmarks(legacy, {0: points})
    frames = np.tile(np.asarray([11, 22, 33], dtype=np.uint8), (5, 2, 2, 1))
    crops = model._face_extractor.extract_face_frames(frames, recovered)
    assert recovered == legacy == frozen
    assert np.array_equal(crops[0][..., ::-1], frames[0])  # Exactly one RGB/BGR round trip.
    assert np.all(crops[1] == 7) and np.all(crops[2] == 7)  # No recovery carry-forward.
    assert model._hand_extractor is original_hand and model._process_pose_landmarks is original_pose
    next_request = RecoveryLandmarks(legacy, {})
    assert all(np.all(crop == 7) for crop in model._face_extractor.extract_face_frames(frames, next_request))


def test_recovery_side_information_rejects_existing_faces_odd_frames_and_bad_points():
    legacy = {0: row(), 1: row(), 2: row(face=[[[.5, .5, .5]]])}
    for index, values in [(1, np.zeros((478, 3))), (2, np.zeros((478, 3))), (0, np.zeros((2, 3))), (0, np.full((478, 3), np.nan))]:
        with pytest.raises(ValueError, match="replacement policy"):
            RecoveryLandmarks(legacy, {index: values})


def test_roi_only_runs_for_sampled_missing_faces_with_current_pose():
    frame = np.zeros((100, 200, 3), dtype=np.uint8); calls = []
    def detector(region):
        calls.append(region.shape)
        return [np.full((478, 3), .5)]
    for index, landmarks in [(1, row()), (0, None), (0, row(pose=False)), (0, row(face=[[[0., 0., 0.]]]))]:
        assert recover_one_frame(frame, landmarks, index, detector) is None
    assert calls == []
    points = recover_one_frame(frame, row(), 0, detector)
    assert calls == [(64, 64, 3)] and points.shape == (478, 3)
    assert np.allclose(points[:, 0], .5) and np.allclose(points[:, 1], .5)
    assert np.allclose(points[:, 2], .16)


def test_install_requires_stride_and_rejects_duplicate_install():
    model = runtime()
    with pytest.raises(ValueError, match="stride-2"): install_face_recovery(model)
    install_post_mediapipe_stride(model); install_face_recovery(model)
    with pytest.raises(RuntimeError, match="already"): install_face_recovery(model)
