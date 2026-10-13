"""Extract C5-compatible MediaPipe landmarks in a Linux CPU container."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

SCHEMA = "umi-studio-mediapipe-landmarks/1"
MAX_FACES = 6
MAX_HANDS = 6
FACE_POINTS = 478
HAND_POINTS = 21
POSE_POINTS = 33


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def packed_landmarks(landmarks: dict[int, object]) -> dict[str, np.ndarray]:
    """Pack the author pipeline's ragged dictionaries without using pickle."""
    if not landmarks or set(landmarks) != set(range(len(landmarks))):
        raise ValueError("landmark frames must be nonempty and contiguous")
    frames = len(landmarks)
    face = np.full((frames, MAX_FACES, FACE_POINTS, 3), np.nan, dtype=np.float64)
    hand = np.full((frames, MAX_HANDS, HAND_POINTS, 3), np.nan, dtype=np.float64)
    pose = np.full((frames, POSE_POINTS, 3), np.nan, dtype=np.float64)
    face_counts = np.zeros(frames, dtype=np.uint8)
    hand_counts = np.zeros(frames, dtype=np.uint8)
    pose_present = np.zeros(frames, dtype=np.bool_)
    frame_valid = np.zeros(frames, dtype=np.bool_)
    for index, row in landmarks.items():
        if row is None:
            continue
        if not isinstance(row, dict):
            raise ValueError("landmark frame must be a dictionary or None")
        frame_valid[index] = True
        faces = row.get("face_landmarks") or []
        hands = row.get("hand_landmarks") or []
        poses = row.get("pose_landmarks") or []
        if len(faces) > MAX_FACES or len(hands) > MAX_HANDS or len(poses) > 1:
            raise ValueError("landmark detector exceeded configured object count")
        face_counts[index] = len(faces)
        hand_counts[index] = len(hands)
        if faces:
            values = np.asarray(faces, dtype=np.float64)
            if values.shape != (len(faces), FACE_POINTS, 3):
                raise ValueError("face landmark topology changed")
            face[index, : len(faces)] = values
        if hands:
            values = np.asarray(hands, dtype=np.float64)
            if values.shape != (len(hands), HAND_POINTS, 3):
                raise ValueError("hand landmark topology changed")
            hand[index, : len(hands)] = values
        if poses:
            values = np.asarray(poses, dtype=np.float64)
            if values.shape != (1, POSE_POINTS, 3):
                raise ValueError("pose landmark topology changed")
            pose[index] = values[0]
            pose_present[index] = True
    return {
        "face_landmarks": face,
        "face_counts": face_counts,
        "hand_landmarks": hand,
        "hand_counts": hand_counts,
        "pose_landmarks": pose,
        "pose_present": pose_present,
        "frame_valid": frame_valid,
    }


def unpack_landmarks(path: Path) -> tuple[dict[int, object], dict]:
    """Reconstruct dictionaries accepted by the unmodified C5 crop helpers."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {name: archive[name] for name in archive.files if name != "metadata"}
    required = {
        "face_landmarks",
        "face_counts",
        "hand_landmarks",
        "hand_counts",
        "pose_landmarks",
        "pose_present",
        "frame_valid",
    }
    frames = metadata.get("frames")
    if (
        metadata.get("schema") != SCHEMA
        or type(frames) is not int
        or frames < 1
        or set(arrays) != required
        or arrays["face_landmarks"].shape
        != (frames, MAX_FACES, FACE_POINTS, 3)
        or arrays["face_counts"].shape != (frames,)
        or arrays["hand_landmarks"].shape
        != (frames, MAX_HANDS, HAND_POINTS, 3)
        or arrays["hand_counts"].shape != (frames,)
        or arrays["pose_landmarks"].shape != (frames, POSE_POINTS, 3)
        or arrays["pose_present"].shape != (frames,)
        or arrays["frame_valid"].shape != (frames,)
    ):
        raise ValueError("landmark sidecar topology differs")
    output = {}
    for index in range(frames):
        if not bool(arrays["frame_valid"][index]):
            output[index] = None
            continue
        face_count = int(arrays["face_counts"][index])
        hand_count = int(arrays["hand_counts"][index])
        if not 0 <= face_count <= MAX_FACES or not 0 <= hand_count <= MAX_HANDS:
            raise ValueError("landmark sidecar object count differs")
        output[index] = {
            "face_landmarks": (
                arrays["face_landmarks"][index, :face_count].tolist()
                if face_count
                else None
            ),
            "hand_landmarks": (
                arrays["hand_landmarks"][index, :hand_count].tolist()
                if hand_count
                else None
            ),
            "pose_landmarks": (
                [arrays["pose_landmarks"][index].tolist()]
                if bool(arrays["pose_present"][index])
                else None
            ),
        }
    return output, metadata


def write_landmarks(
    path: Path,
    landmarks: dict[int, object],
    *,
    source: Path,
    width: int,
    height: int,
    fps: float,
    mediapipe_version: str,
) -> dict:
    arrays = packed_landmarks(landmarks)
    metadata = {
        "schema": SCHEMA,
        "source_sha256": sha256(source),
        "frames": len(landmarks),
        "width": width,
        "height": height,
        "fps": fps,
        "mediapipe_version": mediapipe_version,
        "delegate": "cpu",
        "face_model": "face_landmarker_v2_with_blendshapes.task",
        "hand_model": "hand_landmarker.task",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.npz")
    if path.exists() or temporary.exists():
        raise ValueError("landmark sidecar output already exists")
    try:
        with temporary.open("xb") as output:
            np.savez_compressed(
                output,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                **arrays,
            )
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.input.is_file() or not args.face_model.is_file() or not args.hand_model.is_file():
        parser.error("input video and landmark models must exist")

    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(args.face_model),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=MAX_FACES,
        )
    )
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(args.hand_model),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=0.05,
        )
    )
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.1)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError("could not open input video")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    landmarks = {}
    try:
        while True:
            available, bgr = capture.read()
            if not available:
                break
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            face_result = face.detect(image)
            hand_result = hand.detect(image)
            pose_result = pose.process(rgb)
            landmarks[len(landmarks)] = {
                "face_landmarks": [
                    [[point.x, point.y, point.z] for point in item]
                    for item in face_result.face_landmarks
                ]
                or None,
                "hand_landmarks": [
                    [[point.x, point.y, point.z] for point in item]
                    for item in hand_result.hand_landmarks
                ]
                or None,
                "pose_landmarks": (
                    [
                        [
                            [point.x, point.y, point.z]
                            for point in pose_result.pose_landmarks.landmark
                        ]
                    ]
                    if pose_result.pose_landmarks
                    else None
                ),
            }
    finally:
        capture.release()
        face.close()
        hand.close()
        pose.close()
    if not landmarks or width < 1 or height < 1 or fps <= 0:
        raise RuntimeError("decoded video metadata is invalid")
    metadata = write_landmarks(
        args.output,
        landmarks,
        source=args.input,
        width=width,
        height=height,
        fps=fps,
        mediapipe_version=mp.__version__,
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
