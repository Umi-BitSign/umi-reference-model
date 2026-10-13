"""Measure frozen-baseline resolution sensitivity with pinned FSboard landmarks.

The landmark cache is held fixed across variants. This isolates source-pixel and
visual-backbone resolution from MediaPipe detection variance and avoids creating
new landmarks. Results are public-validation diagnostics, not official UMI scores.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from benchmark_fleurs_dev import distance, miner_revision, sha256
from fetch_fsboard_validation import validate_selection
from resolution_policy import (
    make_short_side_reader,
    set_visual_crop_size,
    validate_source_short_side,
    validate_visual_crop_size,
)
from sequential_video_reader import make_sequential_reader

FACE_POINTS = 468
LEFT_HAND_START = FACE_POINTS
POSE_START = LEFT_HAND_START + 21
RIGHT_HAND_START = POSE_START + 33
TOTAL_POINTS = RIGHT_HAND_START + 21


def complete_part(frame: np.ndarray, start: int, count: int) -> list[list[float]] | None:
    part = frame[start : start + count]
    finite = np.isfinite(part)
    if not finite.any():
        return None
    if not finite.all():
        raise ValueError("cached landmark part is partially finite")
    return part.tolist()


def baseline_landmark_dict(landmarks: np.ndarray) -> dict[int, dict]:
    if (
        landmarks.ndim != 3
        or landmarks.shape[1:] != (TOTAL_POINTS, 3)
        or landmarks.dtype != np.float32
    ):
        raise ValueError("cached FSboard landmarks have an unexpected tensor schema")
    converted = {}
    for index, frame in enumerate(landmarks):
        face = complete_part(frame, 0, FACE_POINTS)
        left = complete_part(frame, LEFT_HAND_START, 21)
        pose = complete_part(frame, POSE_START, 33)
        right = complete_part(frame, RIGHT_HAND_START, 21)
        hands = [hand for hand in (left, right) if hand is not None]
        converted[index] = {
            "face_landmarks": [face] if face is not None else None,
            "hand_landmarks": hands or None,
            "pose_landmarks": [pose] if pose is not None else None,
        }
    return converted


def main() -> None:
    from evaluate_fsboard_coreml import comparable_units

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clip-receipts", type=Path, required=True)
    parser.add_argument("--landmark-receipts", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--landmarks-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--source-short-side", type=int)
    parser.add_argument("--visual-crop-size", type=int, default=224)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1 or not args.baseline.is_dir():
        parser.error("limit must be positive and baseline must exist")
    try:
        validate_source_short_side(args.source_short_side)
        validate_visual_crop_size(args.visual_crop_size)
    except ValueError as error:
        parser.error(str(error))

    rows = validate_selection(json.loads(args.selection.read_text()))
    selection_hash = sha256(args.selection)
    clip_receipts = {
        row["clip_filename"]: row
        for row in map(json.loads, args.clip_receipts.read_text().splitlines())
    }
    landmark_lines = [
        json.loads(line) for line in args.landmark_receipts.read_text().splitlines()
    ]
    landmark_receipts = {row["clip_filename"]: row for row in landmark_lines}
    if (
        len(clip_receipts) != len(args.clip_receipts.read_text().splitlines())
        or len(landmark_receipts) != len(landmark_lines)
    ):
        raise ValueError("resolution ablation receipts contain duplicates")

    completed = set()
    if args.results.exists():
        for line in args.results.read_text().splitlines():
            record = json.loads(line)
            name = record["clip_filename"]
            if (
                record.get("schema") != "umi-studio-fsboard-resolution-ablation/1"
                or name in completed
                or record.get("selection_sha256") != selection_hash
                or record.get("source_short_side") != args.source_short_side
                or record.get("visual_crop_size") != args.visual_crop_size
                or record.get("landmark_receipts_sha256")
                != sha256(args.landmark_receipts)
            ):
                raise ValueError("existing resolution results use another frozen policy")
            completed.add(name)

    baseline = args.baseline.resolve()
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(baseline / "run/hf"),
        "TORCH_HOME": str(baseline / "run/torch"),
        "MPLCONFIGDIR": str(baseline / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(baseline / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(baseline))
    from runtime import SHuBERTInferenceRuntime

    revision = miner_revision(args.miner_health_url, None)
    model = SHuBERTInferenceRuntime(baseline, device="cpu")
    model._video_reader = make_sequential_reader(model._video_reader)
    if args.source_short_side is not None:
        model._video_reader = make_short_side_reader(
            model._video_reader, args.source_short_side
        )
    visual_policy = set_visual_crop_size(model, args.visual_crop_size)
    revision = miner_revision(args.miner_health_url, revision)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    for row in rows:
        name = row["clip_filename"]
        if name in completed or count >= args.limit:
            continue
        clip = args.clips_dir / name
        landmark_path = args.landmarks_dir / f"{Path(name).stem}.npz"
        clip_receipt = clip_receipts.get(name)
        landmark_receipt = landmark_receipts.get(name)
        if (
            clip_receipt is None
            or landmark_receipt is None
            or clip_receipt.get("selection_sha256") != selection_hash
            or landmark_receipt.get("selection_sha256") != selection_hash
            or landmark_receipt.get("schema") != "umi-studio-fsboard-landmarks/1"
            or landmark_receipt.get("face_points", 468) != 468
            or not clip.is_file()
            or not landmark_path.is_file()
            or sha256(clip) != clip_receipt.get("sha256")
            or sha256(landmark_path) != landmark_receipt.get("sha256")
            or landmark_receipt.get("clip_sha256") != clip_receipt.get("sha256")
        ):
            raise ValueError(f"resolution input receipt mismatch: {name}")
        with np.load(landmark_path, allow_pickle=False) as payload:
            if set(payload.files) != {"landmarks"}:
                raise ValueError(f"resolution landmark fields differ: {name}")
            cached = baseline_landmark_dict(payload["landmarks"])

        def pinned_landmarks(frames, *_args):
            if len(frames) != len(cached):
                raise ValueError("video frame count differs from the pinned landmark cache")
            return cached

        model._video_holistic = pinned_landmarks
        revision = miner_revision(args.miner_health_url, revision)
        hypothesis = model.translate_path(clip)
        revision = miner_revision(args.miner_health_url, revision)
        reference_units = comparable_units(row["reference"])
        hypothesis_units = comparable_units(hypothesis)
        edits = distance(reference_units, hypothesis_units)
        record = {
            "schema": "umi-studio-fsboard-resolution-ablation/1",
            "clip_filename": name,
            "selection_sha256": selection_hash,
            "clip_receipts_sha256": sha256(args.clip_receipts),
            "landmark_receipts_sha256": sha256(args.landmark_receipts),
            "clip_sha256": clip_receipt["sha256"],
            "landmarks_sha256": landmark_receipt["sha256"],
            "landmark_policy": "pinned_public_fsboard_holistic_cache",
            "source_short_side": args.source_short_side,
            "source_resize_policy": (
                "aspect_preserving_downscale_only"
                if args.source_short_side is not None
                else "native"
            ),
            **visual_policy,
            "baseline_identity_sha256": sha256(baseline / "MODEL_IDENTITY.json"),
            "baseline_runtime_sha256": sha256(baseline / "runtime.py"),
            "reference": row["reference"],
            "hypothesis": hypothesis,
            "single_reference_cer": edits / max(1, len(reference_units)),
            "single_reference_character_edits": edits,
            "reference_character_count": len(reference_units),
            "miner_model_revision": revision,
            "diagnostics": model.last_diagnostics,
        }
        with args.results.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        count += 1
        print(f"{len(completed) + count}: {name} CER={record['single_reference_cer']:.3f}")


if __name__ == "__main__":
    main()
