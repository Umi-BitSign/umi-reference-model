"""Pose-guided face recovery with unchanged cached non-face model inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from probe_c5_face_detection import head_roi, remap_face

ROOT = Path(__file__).resolve().parents[1]
RECIPE = {
    "id": "c5-pose-guided-face-fallback-v1",
    "temporal_stride": 2,
    "visual_crop_size": 224,
    "replace_only_when": "full_frame_face_detection_missing_and_head_roi_face_recovered",
    "head_roi": "pose_head_points_0_to_10_scale2.5_min64_square",
    "keep_nonrecovered_face_features_exact": True,
    "keep_hand_and_pose_features_exact": True,
    "trainable_parameters": 0,
    "cpu_threads": 2,
    "generation_max_length": 2048,
    "generation_num_beams": 5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def overlay_faces(control: dict, replacement: np.ndarray, mask: np.ndarray) -> dict:
    original = control["face_features"]
    if replacement.shape != original.shape or replacement.dtype != np.float32 or mask.shape != (len(original),) or mask.dtype != np.bool_:
        raise ValueError("face replacement shapes or dtypes differ")
    if not np.isfinite(replacement).all():
        raise ValueError("face replacement is not finite")
    candidate = {name: values.copy() for name, values in control.items()}
    candidate["face_features"][mask] = replacement[mask]
    if any(not np.array_equal(candidate[name], control[name]) for name in control if name != "face_features"):
        raise RuntimeError("face recovery changed a non-face stream")
    if not np.array_equal(candidate["face_features"][~mask], original[~mask]):
        raise RuntimeError("face recovery changed an unrecovered frame")
    return candidate


def extract(plan_path: Path, output_dir: Path) -> None:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    plan = json.loads(plan_path.read_text())
    if plan.get("recipe") != RECIPE or plan["recovery_source_sha256"] != sha256(Path(__file__)) or plan["head_roi_source_sha256"] != sha256(ROOT / "tools/probe_c5_face_detection.py"):
        raise ValueError("frozen face-recovery code changed")
    asset = ROOT / plan["face_landmarker"]
    if sha256(asset) != plan["face_landmarker_sha256"]:
        raise ValueError("face-recovery task asset changed")
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(asset), delegate=python.BaseOptions.Delegate.CPU),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6,
    ))
    pose_detector = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=.1)
    try:
        for row in plan["examples"]:
            clip = ROOT / row["clip"]
            target = output_dir / f"{row['example_id']}.npz"
            if target.exists() or sha256(clip) != row["clip_sha256"]:
                raise ValueError("face-recovery clip changed or output exists")
            frames = row["source_frames"]
            count = (frames + 1) // 2
            points = np.full((count, 478, 3), np.nan, dtype=np.float64)
            mask = np.zeros(count, dtype=np.bool_)
            full_counts = np.zeros(count, dtype=np.uint8)
            capture = cv2.VideoCapture(str(clip))
            index = 0
            try:
                while True:
                    available, bgr = capture.read()
                    if not available:
                        break
                    current = index
                    index += 1
                    if current % 2:
                        continue
                    if current >= frames:
                        raise ValueError("face-recovery video exceeds its frozen cache")
                    position = current // 2
                    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    height, width = rgb.shape[:2]
                    full = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                    full_counts[position] = len(full.face_landmarks)
                    if full.face_landmarks:
                        continue
                    pose = pose_detector.process(rgb)
                    if pose.pose_landmarks is None:
                        continue
                    pose_points = np.array([[p.x,p.y,p.z] for p in pose.pose_landmarks.landmark])
                    box = head_roi(pose_points, width, height)
                    x0,y0,x1,y1 = box
                    region = np.ascontiguousarray(rgb[y0:y1,x0:x1])
                    local = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=region))
                    if not local.face_landmarks:
                        continue
                    faces = [remap_face(np.array([[p.x,p.y,p.z] for p in face]), box, width, height) for face in local.face_landmarks]
                    chosen = min(faces, key=lambda face: np.linalg.norm(pose_points[0]-face[0]))
                    points[position] = chosen
                    mask[position] = True
                if index != frames:
                    raise ValueError("face-recovery decoded frame count differs")
            finally:
                capture.release()
            metadata = {
                "schema":"umi-studio-c5-face-recovery-landmarks/1", "plan_sha256":sha256(plan_path),
                "clip_sha256":row["clip_sha256"], "example_id":row["example_id"],
                "source_frames":frames, "temporal_stride":2, "mediapipe_version":mp.__version__,
                "face_model_sha256":plan["face_landmarker_sha256"], "recovery_source_sha256":sha256(Path(__file__)),
            }
            with target.open("xb") as output:
                np.savez_compressed(output, metadata=np.asarray(json.dumps(metadata,sort_keys=True)),
                                    recovery_mask=mask, recovered_face_landmarks=points, full_frame_face_counts=full_counts)
            print(f"{row['domain']} {row['example_id']} recovered={int(mask.sum())}/{count}",flush=True)
    finally:
        detector.close()
        pose_detector.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args = parser.parse_args()
    extract(args.plan,args.output_dir)


if __name__ == "__main__":
    main()
