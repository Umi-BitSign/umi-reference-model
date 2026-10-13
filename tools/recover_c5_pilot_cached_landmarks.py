"""Add isolated ROI detections to already-verified normalized legacy landmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np

from c5_face_recovery_runtime import recover_one_frame
from extract_mediapipe_landmarks import unpack_landmarks
from probe_c5_face_detection import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--face-model", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): parser.error("cached face recovery needs a new output directory")
    legacy, metadata = unpack_landmarks(args.legacy)
    if metadata["source_sha256"] != sha256(args.input) or metadata["fps"] != 25. or metadata["mediapipe_version"] != "0.10.14":
        raise ValueError("cached landmark provenance does not match this 25fps clip")
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(args.face_model), delegate=python.BaseOptions.Delegate.CPU),
        output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=6))
    capture = cv2.VideoCapture(str(args.input)); mask = np.zeros(len(legacy), dtype=np.bool_)
    faces = np.full((len(legacy), 478, 3), np.nan, dtype=np.float64); index = 0; seconds = 0.
    def detect(region):
        result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=region))
        return [np.asarray([[p.x, p.y, p.z] for p in face]) for face in result.face_landmarks]
    try:
        while True:
            valid, bgr = capture.read()
            if not valid: break
            if index >= len(legacy): raise ValueError("cached landmarks have fewer frames than the clip")
            if index % 2 == 0:
                rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                started = time.perf_counter(); chosen = recover_one_frame(rgb, legacy[index], index, detect)
                seconds += time.perf_counter() - started
                if chosen is not None: mask[index] = True; faces[index] = chosen
            index += 1
    finally: capture.release(); detector.close()
    if index != len(legacy): raise ValueError("cached landmark frame count differs from the clip")
    args.output_dir.mkdir()
    shutil.copy2(args.legacy, args.output_dir / "legacy.npz")
    extra = {**metadata, "recovery_enabled": True, "recovered_frames": int(mask.sum()),
        "cached_legacy_sha256": sha256(args.legacy), "face_recovery_seconds": seconds,
        "recovery_source_sha256": sha256(Path(__file__)), "face_model_sha256": sha256(args.face_model),
        "legacy_landmarks_modified": False, "separate_recovery_detector": True}
    with (args.output_dir / "recovery.npz").open("xb") as stream:
        np.savez_compressed(stream, metadata=np.asarray(json.dumps(extra, sort_keys=True)), recovery_mask=mask, recovered_faces=faces)
    print(json.dumps(extra, sort_keys=True))


if __name__ == "__main__": main()
