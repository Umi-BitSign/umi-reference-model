"""The scaled STEM trainer binds and serializes its exact replay schedule."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from train_c5_stem_scale import frozen_schedule_rows


def test_frozen_schedule_rows_preserve_source_stride_and_position():
    schedule = [
        {
            "id": "stem-1",
            "source": "asl_stem_supplemental",
            "signer": "P1",
            "stored_temporal_stride": 2,
        },
        {"id": "h-1", "source": "how2sign_raw", "signer": 8},
    ]
    assert frozen_schedule_rows(schedule) == [
        {
            "position": 0,
            "source": "asl_stem_supplemental",
            "example_id": "stem-1",
            "signer": "P1",
            "stored_temporal_stride": 2,
        },
        {
            "position": 1,
            "source": "how2sign_raw",
            "example_id": "h-1",
            "signer": 8,
            "stored_temporal_stride": 1,
        },
    ]
