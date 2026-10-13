from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from train_fsboard_byt5_smoke import eligible_positions


def test_eligible_positions_are_deterministic_and_exclude_low_hand_coverage() -> None:
    rows = (
        {"clip_filename": "a.mp4", "left_hand_detection_fraction": 0.5, "right_hand_detection_fraction": 0.0},
        {"clip_filename": "b.mp4", "left_hand_detection_fraction": 0.2, "right_hand_detection_fraction": 0.4},
        {"clip_filename": "c.mp4", "left_hand_detection_fraction": 0.0, "right_hand_detection_fraction": 0.9},
    )
    first = eligible_positions(rows, 17)
    assert first == eligible_positions(rows, 17)
    assert set(first) == {0, 2}
    assert eligible_positions(rows, 18) == eligible_positions(rows, 18)
