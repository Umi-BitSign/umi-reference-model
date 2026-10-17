"""Authenticate complete word preflight receipts and reproduce inputs before learning."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from audit_terra_word_landmarks import terminal
from benchmark_fleurs_dev import sha256
from prepare_terra_word_native import checked, eligible_records
from terra_training_recovery import artifact
from terra_word_native_input import read_input, require_role

ROOT = Path(__file__).resolve().parents[1]


def expected_header(record):
    probe = record["normalized_probe"]
    indices = np.arange(0, probe["decoded_frames"], 2, dtype=np.int64)
    return {
        "source_fps": 25,
        "source_frames": probe["decoded_frames"],
        "source_shape": [probe["height"], probe["width"]],
        "sampled_frames": len(indices),
        "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def require_receipt(record, saved):
    header = expected_header(record)
    frames = header["sampled_frames"]
    expected = {
        "pixels": ([frames, 3, 112, 112, 3], "uint8"),
        "pose_features": ([frames, 14], "float32"),
        "frame_indices": ([frames], "int64"),
    }
    arrays = saved.get("arrays", {})
    if (
        saved.get("record") != record
        or saved.get("header") != header
        or saved.get("streamed_pixels_equal_native_reference") is not True
        or set(arrays) != set(expected)
        or any(
            arrays[name].get("shape") != shape
            or arrays[name].get("dtype") != dtype
            or re.fullmatch(r"[0-9a-f]{64}", arrays[name].get("sha256", "")) is None
            for name, (shape, dtype) in expected.items()
        )
        or arrays["frame_indices"]["sha256"] != header["frame_indices_sha256"]
    ):
        raise ValueError("word preflight receipt changed native shapes, full timeline or identity")


def checked_result(plan_path, result_path):
    original = result_path.absolute()
    result_path = result_path.resolve()
    if original != result_path or not result_path.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("word preflight result must be a canonical owned receipt")
    plan = checked(plan_path)
    records, excluded = eligible_records(plan["records"])
    authority = terminal(result_path.parent)
    result = json.loads(result_path.read_text())
    frames = Counter()
    for record in records:
        frames[record["original"]["row"]["role"]] += expected_header(record)["sampled_frames"]
    if (
        result.get("complete") is not True
        or result.get("plan_sha256") != sha256(plan_path)
        or result.get("counts") != plan["eligible_counts"]
        or result.get("sampled_frames_by_role") != dict(frames)
        or result.get("excluded_original_training_identities")
        != [record["original"]["row"]["example_id"] for record in excluded]
        or result.get("all_streamed_pixels_equal_native_reference") is not True
        or result.get("model_training_ready") is not False
        or result.get("optimizer_updates") != 0
        or result.get("new_Terra_requests") != 0
        or result.get("translation_quality_evaluated") is not False
        or result.get("promotion_authorized") is not False
        or len(result.get("receipts", [])) != len(records)
    ):
        raise ValueError(
            "word preflight is incomplete or differs from the complete fixed inventory"
        )
    bound = []
    expected_receipts = {
        result_path.with_name(f"input-{index:04d}.json") for index in range(len(records))
    }
    if set(result_path.parent.glob("input-*.json")) != expected_receipts:
        raise ValueError("word preflight per-clip receipt inventory changed")
    for index, (record, descriptor) in enumerate(zip(records, result["receipts"], strict=True)):
        receipt = result_path.with_name(f"input-{index:04d}.json")
        if artifact(receipt) != descriptor:
            raise ValueError("word preflight changed an actual per-clip receipt")
        saved = json.loads(receipt.read_text())
        require_receipt(record, saved)
        bound.append({"record": record, "receipt": descriptor, "verified_input": saved})
    return {
        "plan": artifact(plan_path),
        "result": artifact(result_path),
        "authority": authority,
        "records": bound,
        "counts": plan["eligible_counts"],
        "model_training_qualification_still_required": True,
        "verification_source_sha256": sha256(Path(__file__)),
    }


def load_verified_input(bound, *, purpose):
    record = bound["record"]
    require_role(record, purpose)
    receipt = ROOT / bound["receipt"]["path"]
    if artifact(receipt) != bound["receipt"]:
        raise ValueError("word learning input lost its actual preflight receipt")
    saved = json.loads(receipt.read_text())
    if saved != bound["verified_input"]:
        raise ValueError("word learning input changed its frozen preflight binding")
    require_receipt(record, saved)
    values = read_input(record, purpose=purpose)
    arrays = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in values.items()
        if name != "header"
    }
    if arrays != saved["arrays"] or values["header"] != saved["header"]:
        raise ValueError("word learning pixels, pose or timeline differ from the actual preflight")
    return values
