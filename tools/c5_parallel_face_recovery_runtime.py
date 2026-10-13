"""Overlap independent legacy landmark graphs while retaining frame order."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
from pathlib import Path
import time

import numpy as np

import c5_face_recovery_runtime as legacy


def detect_legacy_frame(frame, image, face, hand, pose, executor):
    face_future = executor.submit(face.detect, image)
    hand_future = executor.submit(hand.detect, image)
    try:
        body = pose.process(frame)
    finally:
        # Join both graphs even if pose raises. No graph may process another
        # frame or close while a previous call is still in progress.
        wait((face_future, hand_future))
    return face_future.result(), hand_future.result(), body


def detect_video_landmarks(frames, face_model: Path, hand_model: Path, *, recovery: bool):
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_model), delegate=python.BaseOptions.Delegate.CPU),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6,
    )
    rows = {}; recovered = {}; base_seconds = 0.; recovery_seconds = 0.; attempted = 0
    with ExitStack() as stack:
        face = vision.FaceLandmarker.create_from_options(options); stack.callback(face.close)
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_model), delegate=python.BaseOptions.Delegate.CPU),
            num_hands=6, min_hand_detection_confidence=.05,
        )); stack.callback(hand.close)
        pose = mp.solutions.pose.Pose(min_detection_confidence=.1); stack.callback(pose.close)
        roi_face = vision.FaceLandmarker.create_from_options(options) if recovery else None
        if roi_face is not None: stack.callback(roi_face.close)
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="c5-offline-landmarks")
        stack.callback(executor.shutdown, wait=True)

        def detect(region):
            result = roi_face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=region))
            return [np.asarray([[p.x, p.y, p.z] for p in item], dtype=np.float64) for item in result.face_landmarks]

        for index, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            started = time.perf_counter()
            faces, hands, body = detect_legacy_frame(frame, image, face, hand, pose, executor)
            row = {
                "face_landmarks": [[[p.x, p.y, p.z] for p in item] for item in faces.face_landmarks] or None,
                "hand_landmarks": [[[p.x, p.y, p.z] for p in item] for item in hands.hand_landmarks] or None,
                "pose_landmarks": [[[p.x, p.y, p.z] for p in body.pose_landmarks.landmark]] if body.pose_landmarks else None,
            }
            rows[index] = row; base_seconds += time.perf_counter() - started
            if recovery and index % 2 == 0 and not row["face_landmarks"] and row["pose_landmarks"]:
                attempted += 1; started = time.perf_counter()
                chosen = legacy.recover_one_frame(frame, row, index, detect)
                recovery_seconds += time.perf_counter() - started
                if chosen is not None: recovered[index] = chosen
    return legacy.RecoveryLandmarks(rows, recovered), {
        "baseline_landmarks_seconds": base_seconds, "face_recovery_seconds": recovery_seconds,
        "recovery_attempts": attempted, "recovered_frames": len(recovered), "mediapipe_version": mp.__version__,
        "legacy_landmarks_modified": False, "separate_recovery_detector": recovery,
        "parallel_legacy_graphs": True, "parallel_source_sha256": legacy.sha256(Path(__file__)),
    }


def main() -> None:
    # Reuse the unchanged input validation, decoder and safe NPZ output path.
    # This substitution is private to this CLI process, never the live miner.
    legacy.detect_video_landmarks = detect_video_landmarks
    legacy.main()


if __name__ == "__main__": main()
