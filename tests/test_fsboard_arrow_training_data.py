from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from tfrecord import example_pb2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from fsboard_arrow_training_data import AuditedTrainingShard, canonical_landmarks


def test_canonical_landmarks_drops_iris_points_and_pads_pose() -> None:
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
    result = canonical_landmarks(example)
    assert result.shape == (1, 543, 3)
    np.testing.assert_array_equal(result[0, 467], [1, 1, 1])
    np.testing.assert_array_equal(result[0, 468], [2, 2, 2])
    np.testing.assert_array_equal(result[0, 489], [3, 3, 3])
    assert np.isnan(result[0, 514:522]).all()
    np.testing.assert_array_equal(result[0, 522], [4, 4, 4])


def test_first_frozen_training_row_loads_with_its_reference() -> None:
    ledger = ROOT / "runs/fsboard-train-shard00-v3"
    shard = ROOT / "data/fsboard/source/daun_v3-train.arrow-00000-of-00100"
    with pytest.raises(ValueError, match="coordinate integrity"):
        AuditedTrainingShard(shard, ledger / "training-index.jsonl", ledger / "audit.json")
    loader = AuditedTrainingShard(
        shard,
        ledger / "training-index.jsonl",
        ledger / "audit.json",
        allow_unverified=True,
    )
    assert len(loader) == 736
    landmarks, reference, row = loader.example(0)
    assert landmarks.shape == (159, 543, 3)
    assert reference == "+82-823-308-41-57"
    assert row["signer_id"] == "51ee"
