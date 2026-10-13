"""Extract MediaPipe Holistic landmarks from pinned FSboard validation clips.

This is an offline diagnostic. It never reads the FSboard test split or calls
the live miner for inference. Output retains missing parts as NaN so different
fingerspelling preprocessing variants can be compared without re-decoding video.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from benchmark_fleurs_dev import miner_revision, sha256
from fetch_fsboard_validation import validate_selection

PARTS = (
    ("face_landmarks", 468),
    ("left_hand_landmarks", 21),
    ("pose_landmarks", 33),
    ("right_hand_landmarks", 21),
)
LANDMARK_COUNT = sum(count for _, count in PARTS)


def assemble(result: object, *, face_points: int = 468) -> np.ndarray:
    """Keep face/left-hand/pose/right-hand blocks with an explicit face width."""
    if face_points not in (468, 478):
        raise ValueError("face point count must be 468 or 478")
    parts = (("face_landmarks", face_points), *PARTS[1:])
    points = np.full((face_points + LANDMARK_COUNT - 468, 3), np.nan, dtype=np.float32)
    offset = 0
    for field, count in parts:
        landmarks = getattr(result, field)
        if (
            field == "face_landmarks" and face_points == 478
            and landmarks is not None and 0 < len(landmarks) < 478
        ):
            raise ValueError("MediaPipe result has fewer than 478 detected face points")
        if landmarks is not None and len(landmarks) >= count:
            points[offset : offset + count] = [
                (landmark.x, landmark.y, landmark.z) for landmark in landmarks[:count]
            ]
        offset += count
    return points


def extract(
    video: Path, landmarker: object, *, mirror_input: bool = False, face_points: int = 468
) -> tuple[np.ndarray, float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open FSboard clip: {video.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"invalid FSboard frame rate: {video.name}")
    frames = []
    last_timestamp = -1
    try:
        while len(frames) < 768:
            ok, bgr = capture.read()
            if not ok:
                break
            if mirror_input:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = max(last_timestamp + 1, round(1000 * len(frames) / fps))
            result = landmarker.detect_for_video(image, timestamp)
            frames.append(assemble(result, face_points=face_points))
            last_timestamp = timestamp
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"FSboard clip has no decoded frames: {video.name}")
    return np.stack(frames), fps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--clip-receipts", type=Path, required=True)
    parser.add_argument("--holistic-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--mirror-input", action="store_true")
    parser.add_argument("--face-points", type=int, choices=(468, 478), default=468)
    parser.add_argument("--only-left-dominant-from", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    rows = validate_selection(json.loads(args.selection.read_text()))
    selection_hash = sha256(args.selection)
    model_hash = sha256(args.holistic_model)
    if args.only_left_dominant_from is not None:
        prior = {
            record["clip_filename"]: record
            for record in map(json.loads, args.only_left_dominant_from.read_text().splitlines())
        }
        if any(
            record["selection_sha256"] != selection_hash or record.get("mirror_input", False)
            for record in prior.values()
        ):
            raise ValueError("left-hand selector requires original validation landmarks")
        rows = [
            row
            for row in rows
            if prior[row["clip_filename"]]["left_hand_finite_scalars"]
            > prior[row["clip_filename"]]["right_hand_finite_scalars"]
        ]
        if not rows:
            raise ValueError("no left-hand-dominant clips found")
    clip_receipts = {
        row["clip_filename"]: row
        for row in map(json.loads, args.clip_receipts.read_text().splitlines())
    }
    previous = {}
    if args.receipts.exists():
        for line in args.receipts.read_text().splitlines():
            record = json.loads(line)
            name = record["clip_filename"]
            if (
                name in previous
                or record["selection_sha256"] != selection_hash
                or record["holistic_model_sha256"] != model_hash
                or record["clip_sha256"] != clip_receipts[name]["sha256"]
                or record.get("mirror_input", False) != args.mirror_input
                or record.get("face_points", 468) != args.face_points
                or record.get("schema") != (
                    "umi-studio-fsboard-landmarks-full/1" if args.face_points == 478
                    else "umi-studio-fsboard-landmarks/1"
                )
            ):
                raise ValueError("landmark receipt mismatch")
            output = args.output_dir / f"{Path(name).stem}.npz"
            if not output.is_file() or sha256(output) != record["sha256"]:
                raise ValueError(f"landmark artifact missing or changed: {name}")
            previous[name] = record
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    revision = miner_revision(args.miner_health_url, None)
    options = mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.holistic_model.resolve())),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
    )
    processed = 0
    for row in rows:
        name = row["clip_filename"]
        if name in previous or processed >= args.limit:
            continue
        clip_receipt = clip_receipts.get(name)
        clip = args.clips_dir / name
        if (
            clip_receipt is None
            or clip_receipt["selection_sha256"] != selection_hash
            or not clip.is_file()
            or sha256(clip) != clip_receipt["sha256"]
        ):
            raise ValueError(f"FSboard clip receipt mismatch: {name}")
        revision = miner_revision(args.miner_health_url, revision)
        # VIDEO mode timestamps and tracking state belong to one recording.
        with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
            landmarks, fps = extract(
                clip, landmarker, mirror_input=args.mirror_input, face_points=args.face_points
            )
        revision = miner_revision(args.miner_health_url, revision)
        output = args.output_dir / f"{Path(name).stem}.npz"
        if output.exists():
            raise ValueError(f"unreceipted landmark artifact needs inspection: {output}")
        with tempfile.NamedTemporaryFile(
            suffix=".npz", prefix="fsboard-landmarks-", dir=args.output_dir, delete=False
        ) as temp:
            temp_path = Path(temp.name)
        try:
            np.savez_compressed(temp_path, landmarks=landmarks)
            os.replace(temp_path, output)
        finally:
            temp_path.unlink(missing_ok=True)
        left = int(np.isfinite(landmarks[:, args.face_points:args.face_points + 21]).sum())
        right_start = args.face_points + 54
        right = int(np.isfinite(landmarks[:, right_start:right_start + 21]).sum())
        receipt = {
            "schema": (
                "umi-studio-fsboard-landmarks-full/1" if args.face_points == 478
                else "umi-studio-fsboard-landmarks/1"
            ),
            "clip_filename": name,
            "clip_sha256": clip_receipt["sha256"],
            "selection_sha256": selection_hash,
            "holistic_model_sha256": model_hash,
            "mirror_input": args.mirror_input,
            "face_points": args.face_points,
            "sha256": sha256(output),
            "shape": list(landmarks.shape),
            "fps": fps,
            "left_hand_finite_scalars": left,
            "right_hand_finite_scalars": right,
            "fully_missing_frames": int(np.count_nonzero(np.isnan(landmarks).all(axis=(1, 2)))),
            "miner_model_revision": revision,
        }
        with args.receipts.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(receipt, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        processed += 1
        print(f"{len(previous) + processed}: extracted {name}", flush=True)


if __name__ == "__main__":
    main()
