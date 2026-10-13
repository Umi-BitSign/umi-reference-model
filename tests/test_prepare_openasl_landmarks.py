from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prepare_openasl_landmarks import bbox_crop_geometry, clip_id


def test_clip_id_is_ordered_and_path_safe() -> None:
    assert clip_id(7, "aB_12-cD345") == "openasl-0007-aB_12-cD345"


@pytest.mark.parametrize("position,youtube_id", [(-1, "abcdefghijk"), (0, "short")])
def test_clip_id_rejects_invalid_identity(position: int, youtube_id: str) -> None:
    with pytest.raises(ValueError):
        clip_id(position, youtube_id)


def test_bbox_crop_geometry_preserves_rectangle_and_pads_outside_frame() -> None:
    geometry = bbox_crop_geometry([0.25, -0.25, 0.75, 1.25], 1280, 720)
    assert geometry == {
        "crop_x": 320,
        "crop_y": 0,
        "crop_width": 640,
        "crop_height": 1080,
        "pad_left": 0,
        "pad_top": 180,
        "pad_right": 0,
        "pad_bottom": 180,
    }


def test_bbox_crop_geometry_rounds_output_to_even_dimensions() -> None:
    geometry = bbox_crop_geometry([0.1, 0.1, 0.5, 0.5], 101, 101)
    assert geometry["crop_width"] % 2 == 0
    assert geometry["crop_height"] % 2 == 0
