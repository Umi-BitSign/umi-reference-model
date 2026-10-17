"""Read receipted complete word clips as native112px C5 RGB and original pose."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import sha256
from extract_mediapipe_landmarks import unpack_landmarks
from prepare_roi_appearance import DEMO, SampledRgbReader, native_helpers
from prepare_roi_short_corpus import NativeSampledReader
from prepare_terra_word_landmarks import checked, checked_landmarks
from probe_c5_small_input import pack_rgb
from terra_training_recovery import artifact
from training_media_artifact import media_artifact

ROOT = Path(__file__).resolve().parents[1]
RESERVED = frozenset(("P10", "P13", "P14", "P21", "P22", "P31", "P4", "P6"))
DIAGNOSTIC = frozenset(("P0", "P19"))


def require_role(record, purpose):
    row = record["original"]["row"]
    if (
        purpose not in ("training", "development")
        or row["role"] != purpose
        or row["training_allowed"] is not (purpose == "training")
        or (row["participant"] in RESERVED) != (purpose == "development")
        or record["ROI_lineage_training_allowed"]
        is not (purpose == "training" and row["participant"] not in DIAGNOSTIC)
        or (purpose == "training" and row["participant"] in DIAGNOSTIC)
    ):
        raise ValueError("word native input cannot cross development or retained diagnostic roles")


def bind_records(plan_path, audit_path):
    """Bind every receipt only after the complete324-clip extraction was audited."""
    plan = checked(plan_path)
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("schema") != "umi-studio-word-complete-landmark-audit/1"
        or audit.get("complete_landmark_audit") is not True
        or audit.get("plan") != artifact(plan_path)
        or audit.get("all_native_source_timeline_and_coordinate_checks_passed") is not True
        or len(audit["records"]) != len(plan["records"])
        or audit.get("by_role") != {"training": 228, "development": 96}
        or audit.get("frames") != 35245
    ):
        raise ValueError("word native inputs require the complete bound landmark audit")
    records = []
    for record, audited in zip(plan["records"], audit["records"], strict=True):
        receipt = ROOT / audited["receipt"]["path"]
        saved = json.loads(receipt.read_text())
        target = ROOT / audited["landmarks"]["path"]
        if (
            artifact(receipt) != audited["receipt"]
            or media_artifact(target) != audited["landmarks"]
            or saved.get("plan_sha256") != sha256(plan_path)
            or saved.get("record") != record
            or saved.get("landmarks") != audited["landmarks"]
            or saved.get("metadata") != checked_landmarks(target, record)
        ):
            raise ValueError("word native input audit lost its original source or role binding")
        records.append(
            {
                **record,
                "landmarks": audited["landmarks"],
                "landmark_receipt": audited["receipt"],
                "landmark_plan": artifact(plan_path),
            }
        )
    return records


def pose_processor():
    source = DEMO / "body_features.py"
    spec = importlib.util.spec_from_file_location("word_native_body_features", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PoseProcessor()


def read_input(record, *, purpose):
    """Read one full stride2 timeline; references never enter crop or pose helpers.

    A frozen consuming plan must pin this reader, every native helper and the
    bound records from bind_records. This reader performs no model evaluation,
    training, caching, acquisition or scoring.
    """
    require_role(record, purpose)  # Reject held-out inputs before any file read.
    source = {
        k: v
        for k, v in record.items()
        if k not in ("landmarks", "landmark_receipt", "landmark_plan")
    }
    video = ROOT / record["clip"]["path"]
    landmark = ROOT / record["landmarks"]["path"]
    receipt = ROOT / record["landmark_receipt"]["path"]
    saved = json.loads(receipt.read_text())
    if (
        artifact(receipt) != record["landmark_receipt"]
        or media_artifact(video) != record["clip"]
        or media_artifact(landmark) != record["landmarks"]
        or saved.get("plan_sha256") != record["landmark_plan"]["sha256"]
        or saved.get("record") != source
        or saved.get("landmarks") != record["landmarks"]
    ):
        raise ValueError("word native input video, landmark or source receipt bytes changed")
    metadata = checked_landmarks(landmark, source)
    if saved.get("metadata") != metadata:
        raise ValueError("word native input metadata differs from its receipt")
    landmarks, _ = unpack_landmarks(landmark)
    indices = np.arange(0, metadata["frames"], 2, dtype=np.int64)
    sampled = {i: landmarks[int(index)] for i, index in enumerate(indices)}
    header = {
        "source_fps": 25,
        "source_frames": metadata["frames"],
        "source_shape": [metadata["height"], metadata["width"]],
        "sampled_frames": len(indices),
        "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }
    hands, face, native_reader = native_helpers()
    with SampledRgbReader(video, header) as reader:
        left, right = hands(output_size=(112, 112)).extract_hand_frames(reader, sampled)
    with SampledRgbReader(video, header) as reader:
        facial = face(output_size=(112, 112)).extract_face_frames(reader, sampled)
    pixels = np.stack([pack_rgb(v, len(indices), 112) for v in (facial, left, right)], axis=1)
    del facial, left, right
    reader = native_reader(video)
    try:
        left, right = hands(output_size=(112, 112)).extract_hand_frames(
            NativeSampledReader(reader, header), sampled
        )
        facial = face(output_size=(112, 112)).extract_face_frames(
            NativeSampledReader(reader, header), sampled
        )
    finally:
        del reader
    reference = np.stack([pack_rgb(v, len(indices), 112) for v in (facial, left, right)], axis=1)
    if not np.array_equal(pixels, reference):
        raise ValueError("word native input streamed/reference crop pixels differ")
    # Preserve the native distinction between a missing frame and a frame with
    # no pose, including its previous-pose and -9999 sentinel behavior.
    pose = pose_processor().process_landmarks_sequence(sampled).astype(np.float32)
    if pose.shape != (len(indices), 14) or not np.isfinite(pose).all():
        raise ValueError("word native pose must preserve complete finite14D coverage")
    return {"pixels": pixels, "pose_features": pose, "frame_indices": indices, "header": header}
