from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tfrecord import example_pb2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from fsboard_full_landmarks import AuditedFullTrainingShard, full_landmarks
from fsboard_paper_like_features import paper_like_features


def test_full_landmarks_keep_iris_and_pad_short_pose() -> None:
    example = example_pb2.SequenceExample()
    example.context.feature["num_frames"].int64_list.value.append(1)
    for part, width, value in (
        ("face", 478, 1.0),
        ("left_hand", 21, 2.0),
        ("pose", 25, 3.0),
        ("right_hand", 21, 4.0),
    ):
        for axis in "xyz":
            example.feature_lists.feature_list[f"{axis}_{part}"].feature.add().float_list.value.extend(
                [value] * width
            )
    landmarks = full_landmarks(example)
    assert landmarks.shape == (1, 553, 3)
    np.testing.assert_array_equal(landmarks[0, 468], [1, 1, 1])
    np.testing.assert_array_equal(landmarks[0, 473], [1, 1, 1])
    np.testing.assert_array_equal(landmarks[0, 478], [2, 2, 2])
    np.testing.assert_array_equal(landmarks[0, 499], [3, 3, 3])
    assert np.isnan(landmarks[0, 524:532]).all()
    np.testing.assert_array_equal(landmarks[0, 532], [4, 4, 4])
    assert paper_like_features(landmarks, frame_stride=1).shape == (1, 255)


def test_audited_first_training_row_exposes_full_face() -> None:
    ledger = ROOT / "runs/fsboard-train-shard00-v3"
    loader = AuditedFullTrainingShard(
        ROOT / "data/fsboard/source/daun_v3-train.arrow-00000-of-00100",
        ledger / "training-index.jsonl",
        ledger / "audit.json",
        allow_unverified=True,
    )
    landmarks, reference, row = loader.example_full(0)
    assert landmarks.shape == (159, 553, 3)
    assert np.isfinite(landmarks[:, 468:478]).any()
    assert reference == "+82-823-308-41-57"
    assert row["signer_id"] == "51ee"
