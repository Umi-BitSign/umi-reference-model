"""Native224 confirmation inputs with the unchanged full-rate timing masks.

Uses the existing complete broad sidecar audit and native crop reader. No
training eligibility, model loading, prediction or confirmation consumption.
"""

from __future__ import annotations

import hashlib

import c5_broad_native_input as native
import numpy as np
from c5_timing_feature_data import ROOT, bound, read, require, sha
from c5_timing_visibility import visibility
from extract_mediapipe_landmarks import unpack_landmarks


def confirmation_row(record):
    native.require_role(record, "confirmation")
    return record["example"]


def source_visibility(record):
    confirmation_row(record)  # Reject other roles before reading any source.
    audit = read(bound(record["broad_audit"]))
    require(
        audit["complete"] is True
        and audit["training_allowed"] is False
        and len(audit["records"]) == 48
        and {k: v for k, v in record.items() if k != "broad_audit"} in audit["records"]
        and audit["plan"] == record["landmark_plan"]
        and bool(audit["source_sha256"])
        and all(sha(ROOT / name) == digest for name, digest in audit["source_sha256"].items()),
        "timing confirmation lost its audited reserved-inventory membership",
    )
    bound(record["clip"])
    landmarks, metadata = unpack_landmarks(bound(record["landmarks"]))
    example, probe = record["example"], record["normalized_probe"]
    require(
        example["video"] == record["clip"]
        and metadata["source_sha256"] == record["clip"]["sha256"]
        and metadata["fps"] == example["fps"] == 25
        and metadata["frames"] == example["frames"] == probe["decoded_frames"]
        and [metadata["height"], metadata["width"]] == [probe["height"], probe["width"]],
        "timing confirmation landmark source or clock changed",
    )
    indices = np.arange(0, metadata["frames"], 2, dtype=np.int64)
    header = {
        "source_fps": 25,
        "source_frames": metadata["frames"],
        "source_shape": [metadata["height"], metadata["width"]],
        "sampled_frames": len(indices),
        "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }
    observed, intervals = visibility(landmarks, metadata["frames"], indices)
    return header, indices, observed, intervals


def read_confirmation224(record):
    header, indices, observed, intervals = source_visibility(record)
    values = native.read_input(record, purpose="confirmation", side=224)
    native.array_receipt(values, 224)
    require(
        values["header"] == header and values["frame_indices"].tobytes() == indices.tobytes(),
        "timing confirmation changed the original complete native224 grid",
    )
    result = {
        **values,
        "source_hand_tip_visible": observed,
        "valid_intervals": intervals,
    }
    from c5_timing_native_input import input_receipt

    input_receipt(result)
    return result
