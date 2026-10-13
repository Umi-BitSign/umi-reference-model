"""The generic frozen replay trainer binds every scheduled example exactly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_c5_frozen_replay import materialize_schedule


def test_materializes_exact_schedule_order():
    replay = [
        {"id": "a", "source": "how2sign_raw", "signer": 2},
        {"id": "b", "source": "fleurs", "signer": 3},
    ]
    frozen = [
        {
            "position": 0,
            "source": "fleurs",
            "example_id": "b",
            "signer": 3,
            "stored_temporal_stride": 1,
        },
        {
            "position": 1,
            "source": "how2sign_raw",
            "example_id": "a",
            "signer": 2,
            "stored_temporal_stride": 1,
        },
    ]
    assert [row["id"] for row in materialize_schedule(replay, [], frozen)] == ["b", "a"]


def test_rejects_repeated_schedule_example():
    replay = [{"id": "a", "source": "how2sign_raw", "signer": 2}]
    row = {
        "position": 0,
        "source": "how2sign_raw",
        "example_id": "a",
        "signer": 2,
        "stored_temporal_stride": 1,
    }
    repeated = [row, {**row, "position": 1}]
    with pytest.raises(ValueError, match="repeats"):
        materialize_schedule(replay, [], repeated)
