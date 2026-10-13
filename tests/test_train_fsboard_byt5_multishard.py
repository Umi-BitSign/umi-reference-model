from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from train_fsboard_byt5_multishard import schedule


def row(name: str, *, frames: int = 100, hand: float = 0.8) -> dict:
    return {
        "clip_filename": name,
        "frames": frames,
        "left_hand_detection_fraction": hand,
        "right_hand_detection_fraction": 0.0,
    }


def test_multishard_schedule_is_deterministic_and_bounded() -> None:
    loaders = [
        SimpleNamespace(rows=(row("a"), row("b", frames=513))),
        SimpleNamespace(rows=(row("c"), row("d", hand=0.4))),
    ]
    first = schedule(loaders, 17)
    assert first == schedule(loaders, 17)
    assert set(first) == {(0, 0), (1, 0)}


def test_multishard_schedule_rejects_duplicate_clips() -> None:
    loaders = [SimpleNamespace(rows=(row("a"),)), SimpleNamespace(rows=(row("a"),))]
    with pytest.raises(ValueError, match="duplicate"):
        schedule(loaders, 17)
