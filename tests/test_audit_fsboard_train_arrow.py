from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tfrecord import example_pb2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from audit_fsboard_train_arrow import (
    assert_clean_coordinate_streams,
    context_frames,
    hand_fraction,
    metadata_clip_name,
    validate_coordinate_streams,
)


def test_arrow_video_name_maps_to_metadata_milliseconds() -> None:
    assert metadata_clip_name("51ee_379_20220813_203034_619.011_625.644.mp4") == (
        "51ee_379_20220813_203034_619011_625644.mp4"
    )
    assert metadata_clip_name("x_0.001_1.020.mp4") == "x_1_1020.mp4"
    assert metadata_clip_name("x_752.45_757.5.mp4") == "x_752450_757500.mp4"
    with pytest.raises(ValueError, match="unexpected"):
        metadata_clip_name("x_1000_1020.mp4")


def test_hand_fraction_checks_frame_count_and_width() -> None:
    example = example_pb2.SequenceExample()
    example.context.feature["num_frames"].int64_list.value.append(2)
    left = example.feature_lists.feature_list["x_left_hand"].feature
    left.add().float_list.value.extend([0.5] * 21)
    left.add()
    assert context_frames(example) == 2
    assert hand_fraction(example, 2, "left_hand") == 0.5
    with pytest.raises(ValueError, match="different frame count"):
        hand_fraction(example, 1, "left_hand")
    left[1].float_list.value.extend([0.5] * 20)
    with pytest.raises(ValueError, match="unexpected width"):
        hand_fraction(example, 2, "left_hand")


def test_long_publisher_clip_is_auditable_but_implausible_length_is_rejected() -> None:
    example = example_pb2.SequenceExample()
    example.context.feature["num_frames"].int64_list.value.append(788)
    assert context_frames(example) == 788
    example.context.feature["num_frames"].int64_list.value[0] = 4097
    with pytest.raises(ValueError, match="frame count"):
        context_frames(example)


def test_coordinate_streams_require_matching_axis_widths() -> None:
    example = example_pb2.SequenceExample()
    for part, width in (("face", 478), ("left_hand", 21), ("right_hand", 21), ("pose", 25)):
        for axis in "xyz":
            example.feature_lists.feature_list[f"{axis}_{part}"].feature.add().float_list.value.extend(
                [0.5] * width
            )
    validate_coordinate_streams(example, 1)
    example.feature_lists.feature_list["z_pose"].feature[0].float_list.value.append(0.5)
    with pytest.raises(ValueError, match="unexpected width"):
        validate_coordinate_streams(example, 1)


def test_shifted_coordinate_streams_are_rejected() -> None:
    example = example_pb2.SequenceExample()
    values = [float(i) for i in range(24)]
    for part in ("face", "left_hand", "right_hand", "pose"):
        for axis, offset in zip("xyz", range(3), strict=True):
            example.feature_lists.feature_list[f"{axis}_{part}"].feature.add().float_list.value.extend(
                values[offset:offset + 21]
            )
    with pytest.raises(ValueError, match="shifted interleaved views"):
        assert_clean_coordinate_streams(example)


def test_shifted_later_frame_is_rejected() -> None:
    example = example_pb2.SequenceExample()
    for part in ("face", "left_hand", "right_hand", "pose"):
        for axis, offset in zip("xyz", range(3), strict=True):
            frames = example.feature_lists.feature_list[f"{axis}_{part}"].feature
            frames.add().float_list.value.extend([float(i + 100 * offset) for i in range(21)])
            frames.add().float_list.value.extend([float(i) for i in range(offset, offset + 21)])
    with pytest.raises(ValueError, match="shifted interleaved views"):
        assert_clean_coordinate_streams(example)
