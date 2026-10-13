"""Load audited FSboard training landmarks without importing validation data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from tfrecord import example_pb2

from audit_fsboard_train_arrow import (
    context_bytes,
    context_frames,
    hand_fraction,
    metadata_clip_name,
    validate_coordinate_streams,
)
from benchmark_fleurs_dev import sha256

AUDIT_SCHEMA = "umi-studio-fsboard-train-arrow-audit/1"
ROW_SCHEMA = "umi-studio-fsboard-train-shard-row/1"
PARTS = (("face", 0, 468), ("left_hand", 468, 21), ("pose", 489, 33), ("right_hand", 522, 21))


def canonical_landmarks(example: example_pb2.SequenceExample) -> np.ndarray:
    """Map publisher face478/pose25-or-33 to canonical face468/pose33."""
    frames = context_frames(example)
    validate_coordinate_streams(example, frames)
    landmarks = np.full((frames, 543, 3), np.nan, dtype=np.float32)
    for part, offset, capacity in PARTS:
        for axis_index, axis in enumerate("xyz"):
            values = example.feature_lists.feature_list[f"{axis}_{part}"].feature
            for frame_index, frame in enumerate(values):
                coordinates = frame.float_list.value
                if coordinates:
                    count = min(len(coordinates), capacity)
                    landmarks[frame_index, offset : offset + count, axis_index] = coordinates[:count]
    if np.isinf(landmarks).any():
        raise ValueError("FSboard arrow landmarks contain infinite coordinates")
    return landmarks


class AuditedTrainingShard:
    def __init__(
        self, shard: Path, index: Path, audit: Path, *, allow_unverified: bool = False
    ) -> None:
        document = json.loads(audit.read_text())
        if (
            document.get("schema") != AUDIT_SCHEMA
            or document.get("source_split") != "FSboard daun_v3 train"
            or document.get("test_used") is not False
            or document.get("shard_name") != shard.name
            or document.get("shard_sha256") != sha256(shard)
            or document.get("training_index_sha256") != sha256(index)
        ):
            raise ValueError("FSboard training shard differs from frozen audit")
        if document.get("coordinate_integrity") != "verified_clean" and not allow_unverified:
            raise ValueError("FSboard coordinate integrity is unverified; training is prohibited")
        rows = [json.loads(line) for line in index.read_text().splitlines()]
        if (
            len(rows) != document.get("shard_safe_examples")
            or len({row["shard_row"] for row in rows}) != len(rows)
            or len({row["clip_filename"] for row in rows}) != len(rows)
            or any(row.get("schema") != ROW_SCHEMA for row in rows)
        ):
            raise ValueError("FSboard training index differs from frozen audit")
        self._serialized = feather.read_table(
            shard, columns=["serialized"], memory_map=True
        )["serialized"]
        if len(self._serialized) != document["shard_examples"] or any(
            not 0 <= row["shard_row"] < len(self._serialized) for row in rows
        ):
            raise ValueError("FSboard training index exceeds source shard")
        self.rows = tuple(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def example(self, position: int) -> tuple[np.ndarray, str, dict]:
        row = self.rows[position]
        example = example_pb2.SequenceExample.FromString(
            self._serialized[row["shard_row"]].as_py()
        )
        if (
            metadata_clip_name(context_bytes(example, "video_name")) != row["clip_filename"]
            or context_bytes(example, "prompt") != row["reference"]
            or context_bytes(example, "signer_id") != row["signer_id"]
            or context_frames(example) != row["frames"]
        ):
            raise ValueError("FSboard training row differs from frozen index")
        landmarks = canonical_landmarks(example)
        left = hand_fraction(example, len(landmarks), "left_hand")
        right = hand_fraction(example, len(landmarks), "right_hand")
        if (
            abs(left - row["left_hand_detection_fraction"]) > 1e-9
            or abs(right - row["right_hand_detection_fraction"]) > 1e-9
        ):
            raise ValueError("FSboard training hand coverage differs from frozen index")
        return landmarks, row["reference"], row
