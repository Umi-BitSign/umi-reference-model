"""Isolated pose-guided face recovery and single-pass runtime crop overlay.

Legacy landmark entries and crop imputation remain unchanged. Recovered faces
are request-local side information, applied only after legacy face cropping.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np

from probe_c5_face_detection import head_roi, remap_face, sha256


class RecoveryLandmarks(dict):
    def __init__(self, legacy: dict, recovered_faces: dict[int, np.ndarray]) -> None:
        if not legacy or set(legacy) != set(range(len(legacy))):
            raise ValueError("legacy landmark frames must be contiguous")
        super().__init__(legacy)
        self.recovered_faces = {}
        for index, points in recovered_faces.items():
            row = legacy.get(index)
            values = np.asarray(points, dtype=np.float64)
            if (
                type(index) is not int or index % 2 or row is None
                or row.get("face_landmarks") or not row.get("pose_landmarks")
                or values.shape != (478, 3) or not np.isfinite(values).all()
            ):
                raise ValueError("recovered face violates the frozen replacement policy")
            self.recovered_faces[index] = values.copy()


def recover_one_frame(frame_rgb: np.ndarray, row: dict | None, index: int, detect) -> np.ndarray | None:
    """Detect only a sampled, missing face; detect returns ROI-normalized faces."""
    if index % 2 or row is None or row.get("face_landmarks") or not row.get("pose_landmarks"):
        return None
    pose = np.asarray(row["pose_landmarks"][0], dtype=np.float64)
    height, width = frame_rgb.shape[:2]
    box = head_roi(pose, width, height)
    x0, y0, x1, y1 = box
    region = np.ascontiguousarray(frame_rgb[y0:y1, x0:x1])
    faces = [remap_face(points, box, width, height) for points in detect(region)]
    if not faces:
        return None
    chosen = min(faces, key=lambda points: np.linalg.norm(pose[0] - points[0]))
    if chosen.shape != (478, 3) or not np.isfinite(chosen).all():
        raise ValueError("ROI detector returned invalid face coordinates")
    return chosen


class RecoveryFaceExtractor:
    """Wrap an already-strided extractor without changing its imputation state."""

    def __init__(self, strided_base) -> None:
        if getattr(strided_base, "stride", None) != 2:
            raise ValueError("face overlay requires the confirmed stride-2 extractor")
        self.base = strided_base
        self.cropper = strided_base.base

    def extract_face_frames(self, frames: np.ndarray, landmarks: RecoveryLandmarks):
        if not isinstance(landmarks, RecoveryLandmarks) or len(frames) != len(landmarks):
            raise ValueError("face overlay requires matching request-local recovery landmarks")
        # Complete legacy cropping first. A recovered crop must never become the
        # previous-face value used for a later frame whose face was not recovered.
        crops = list(self.base.extract_face_frames(frames, landmarks))
        if len(crops) != (len(frames) + 1) // 2:
            raise ValueError("legacy face crop count differs from stride-2 policy")
        for source_index, points in landmarks.recovered_faces.items():
            # Runtime frames are RGB; legacy crop helpers return BGR, which the
            # runtime converts back to RGB once immediately before DINO.
            bgr = np.ascontiguousarray(frames[source_index][..., ::-1])
            crop = self.cropper.cues_on_grey_background(bgr, points.tolist())
            crops[source_index // 2] = self.cropper.resize_frame(crop, self.cropper.output_size)
        return crops


def install_face_recovery(runtime) -> None:
    if getattr(runtime, "_c5_temporal_stride", None) != 2:
        raise ValueError("install confirmed stride-2 policy before face recovery")
    if getattr(runtime, "_c5_face_recovery", False):
        raise RuntimeError("face recovery is already installed")
    runtime._face_extractor = RecoveryFaceExtractor(runtime._face_extractor)
    runtime._c5_face_recovery = True


def detect_video_landmarks(frames, face_model: Path, hand_model: Path, *, recovery: bool) -> tuple[RecoveryLandmarks, dict]:
    """Run full-rate legacy graphs and a separate face-only ROI graph on CPU."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(face_model), delegate=python.BaseOptions.Delegate.CPU),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6,
    )
    legacy = {}; recovered = {}; base_seconds = 0.; recovery_seconds = 0.; attempted = 0
    with ExitStack() as stack:
        face = vision.FaceLandmarker.create_from_options(options); stack.callback(face.close)
        hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(hand_model), delegate=python.BaseOptions.Delegate.CPU),
            num_hands=6, min_hand_detection_confidence=.05,
        )); stack.callback(hand.close)
        pose = mp.solutions.pose.Pose(min_detection_confidence=.1); stack.callback(pose.close)
        roi_face = vision.FaceLandmarker.create_from_options(options) if recovery else None
        if roi_face is not None: stack.callback(roi_face.close)

        def detect(region):
            result = roi_face.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=region))
            return [np.asarray([[p.x, p.y, p.z] for p in item], dtype=np.float64) for item in result.face_landmarks]

        for index, frame in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            started = time.perf_counter()
            faces = face.detect(image); hands = hand.detect(image); body = pose.process(frame)
            row = {
                "face_landmarks": [[[p.x, p.y, p.z] for p in item] for item in faces.face_landmarks] or None,
                "hand_landmarks": [[[p.x, p.y, p.z] for p in item] for item in hands.hand_landmarks] or None,
                "pose_landmarks": [[[p.x, p.y, p.z] for p in body.pose_landmarks.landmark]] if body.pose_landmarks else None,
            }
            legacy[index] = row; base_seconds += time.perf_counter() - started
            if recovery and index % 2 == 0 and not row["face_landmarks"] and row["pose_landmarks"]:
                attempted += 1; started = time.perf_counter()
                chosen = recover_one_frame(frame, row, index, detect)
                recovery_seconds += time.perf_counter() - started
                if chosen is not None: recovered[index] = chosen
    return RecoveryLandmarks(legacy, recovered), {
        "baseline_landmarks_seconds": base_seconds, "face_recovery_seconds": recovery_seconds,
        "recovery_attempts": attempted, "recovered_frames": len(recovered),
        "mediapipe_version": mp.__version__, "legacy_landmarks_modified": False,
        "separate_recovery_detector": recovery,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--recovery", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() or not all(path.is_file() for path in (args.input, args.face_model, args.hand_model)):
        parser.error("landmark runtime needs existing inputs and a new output directory")
    import cv2
    from extract_mediapipe_landmarks import write_landmarks

    capture = cv2.VideoCapture(str(args.input))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps != 25. or min(width, height) < 1:
        capture.release(); raise ValueError("landmark runtime requires a normalized native-aspect 25fps clip")

    def frames():
        while True:
            valid, bgr = capture.read()
            if not valid: break
            yield np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    try:
        landmarks, timings = detect_video_landmarks(frames(), args.face_model, args.hand_model, recovery=args.recovery)
    finally:
        capture.release()
    args.output_dir.mkdir()
    metadata = write_landmarks(args.output_dir / "legacy.npz", landmarks, source=args.input,
                               width=width, height=height, fps=fps, mediapipe_version=timings["mediapipe_version"])
    metadata.update(timings)
    metadata.update({"runtime_source_sha256": sha256(Path(__file__)), "face_model_sha256": sha256(args.face_model),
                     "hand_model_sha256": sha256(args.hand_model), "recovery_enabled": args.recovery})
    mask = np.zeros(len(landmarks), dtype=np.bool_)
    points = np.full((len(landmarks), 478, 3), np.nan, dtype=np.float64)
    for index, face in landmarks.recovered_faces.items(): mask[index] = True; points[index] = face
    with (args.output_dir / "recovery.npz").open("xb") as output:
        np.savez_compressed(output, recovery_mask=mask, recovered_faces=points,
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
