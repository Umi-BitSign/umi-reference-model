"""Read all 478 publisher face points for an independent FSboard experiment."""

from __future__ import annotations

import numpy as np
from tfrecord import example_pb2

from audit_fsboard_train_arrow import context_frames, validate_coordinate_streams
from fsboard_arrow_training_data import AuditedTrainingShard

FULL_PARTS = (
    ("face", 0, 478),
    ("left_hand", 478, 21),
    ("pose", 499, 33),
    ("right_hand", 532, 21),
)


def full_landmarks(example: example_pb2.SequenceExample) -> np.ndarray:
    """Return publisher face478/hand21/pose33-padded/hand21 landmarks."""
    frames = context_frames(example)
    validate_coordinate_streams(example, frames)
    landmarks = np.full((frames, 553, 3), np.nan, dtype=np.float32)
    for part, offset, capacity in FULL_PARTS:
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


class AuditedFullTrainingShard(AuditedTrainingShard):
    """Keep the existing audit checks, then preserve the publisher's iris points."""

    def example_full(self, position: int) -> tuple[np.ndarray, str, dict]:
        _, reference, row = self.example(position)
        example = example_pb2.SequenceExample.FromString(
            self._serialized[row["shard_row"]].as_py()
        )
        return full_landmarks(example), reference, row
