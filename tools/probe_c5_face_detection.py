"""Compare full-frame and pose-guided head-ROI face detection on three frames."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def head_roi(pose: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    if pose.shape != (33, 3) or not np.isfinite(pose).all() or min(width, height) < 1:
        raise ValueError("head ROI requires valid normalized pose coordinates")
    points = pose[:11, :2] * np.array([width, height])
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    side = min(min(width, height), max(64, int(np.ceil(float((high-low).max()) * 2.5))))
    x = min(width-side, max(0, int(center[0]-side/2)))
    y = min(height-side, max(0, int(center[1]-side/2)))
    return x, y, x + side, y + side


def remap_face(points: np.ndarray, roi: tuple, width: int, height: int) -> np.ndarray:
    x0, y0, x1, y1 = roi
    mapped = np.asarray(points, dtype=np.float64).copy()
    mapped[:, 0] = (mapped[:, 0] * (x1-x0) + x0) / width
    mapped[:, 1] = (mapped[:, 1] * (y1-y0) + y0) / height
    mapped[:, 2] *= (x1-x0) / width
    return mapped


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.input.is_file() or not args.face_model.is_file():
        parser.error("face probe needs existing inputs and a new output")
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.face_model), delegate=python.BaseOptions.Delegate.CPU),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6,
    ))
    pose_detector = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=.1)
    capture = cv2.VideoCapture(str(args.input))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 3:
        raise ValueError("face probe needs at least three video frames")
    rows = []
    try:
        for index in sorted({int((total-1)*fraction) for fraction in (.1, .5, .9)}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            available, bgr = capture.read()
            if not available:
                raise RuntimeError("face probe frame unavailable")
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            height, width = rgb.shape[:2]
            full = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            pose = pose_detector.process(rgb)
            roi = None
            recovered = []
            if pose.pose_landmarks:
                points = np.array([[point.x, point.y, point.z] for point in pose.pose_landmarks.landmark])
                roi = head_roi(points, width, height)
                x0, y0, x1, y1 = roi
                crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
                localized = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
                recovered = [remap_face(np.array([[p.x,p.y,p.z] for p in face]), roi, width, height).tolist() for face in localized.face_landmarks]
            rows.append({"frame_index":index, "width":width, "height":height, "full_frame_faces":len(full.face_landmarks),
                         "pose_present":pose.pose_landmarks is not None, "head_roi":roi, "head_roi_faces":len(recovered),
                         "head_roi_landmarks":recovered})
    finally:
        capture.release()
        detector.close()
        pose_detector.close()
    report = {"schema":"umi-studio-c5-face-roi-probe/1", "clip_sha256":sha256(args.input),
              "face_model_sha256":sha256(args.face_model), "probe_source_sha256":sha256(Path(__file__)),
              "mediapipe_version":mp.__version__, "device":"cpu", "rows":rows,
              "full_frame_detections":sum(r['full_frame_faces']>0 for r in rows),
              "head_roi_detections":sum(r['head_roi_faces']>0 for r in rows), "translation_evaluated":False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"frames":len(rows),"full_frame_detections":report['full_frame_detections'],"head_roi_detections":report['head_roi_detections']}))


if __name__ == "__main__":
    main()
