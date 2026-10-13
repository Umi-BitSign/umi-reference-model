"""Corrective tuning decisions retain temporal-sampling provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from decide_visual_lora_corrective import check_temporal_stride


def test_accepts_consistent_stride_two_counts():
    check_temporal_stride(
        [
            {
                "temporal_stride": 2,
                "source_frame_count": 5,
                "evaluated_frame_count": 3,
            },
            {
                "temporal_stride": 2,
                "source_frame_count": 6,
                "evaluated_frame_count": 3,
            },
        ],
        2,
    )


def test_rejects_mismatched_stride_or_frame_count():
    with pytest.raises(ValueError, match="another temporal stride"):
        check_temporal_stride(
            [
                {
                    "temporal_stride": 1,
                    "source_frame_count": 5,
                    "evaluated_frame_count": 5,
                }
            ],
            2,
        )
    with pytest.raises(ValueError, match="another temporal stride"):
        check_temporal_stride(
            [
                {
                    "temporal_stride": 2,
                    "source_frame_count": 5,
                    "evaluated_frame_count": 2,
                }
            ],
            2,
        )
