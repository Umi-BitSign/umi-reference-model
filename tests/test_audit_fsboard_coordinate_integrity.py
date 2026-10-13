from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_fsboard_coordinate_integrity import axis_shifted, sample_indices


def test_stride_one_coordinate_views_are_detected() -> None:
    interleaved = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    assert axis_shifted(interleaved[:6], interleaved[1:7], interleaved[2:8])
    assert not axis_shifted([0.1, 0.4, 0.7, 1.0, 1.3, 1.6],
                            [0.2, 0.5, 0.8, 1.1, 1.4, 1.7],
                            [0.3, 0.6, 0.9, 1.2, 1.5, 1.8])


def test_sample_indices_are_bounded_and_unique() -> None:
    assert sample_indices(3, 32) == [0, 1, 2]
    assert len(sample_indices(739, 32)) == 32
