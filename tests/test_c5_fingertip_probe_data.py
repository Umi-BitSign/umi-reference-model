from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_fingertip_probe_data as probe


def fixture(frames=17):
    tips = np.arange(frames * 30, dtype=np.float32).reshape(frames, 10, 3) / 10000
    hands = np.zeros((frames, 2, 21, 3), dtype=np.float32)
    hands[:, :, [4, 8, 12, 16, 20]] = tips.reshape(frames, 2, 5, 3)
    record = {
        "frames": frames,
        "fps": 25,
        "kind": "word",
        "source": "asl_stem_word",
        "signer": "P1",
        "source_group": "article",
        "example_id": "fixture",
    }
    arrays = {
        "fingertips": tips,
        "full_hands": hands.reshape(frames, 42, 3),
        "hand_continuity": probe.tip_continuity(tips.reshape(frames, 2, 5, 3)),
        "source_frame_indices": np.arange(frames),
        "canonical_seconds": np.arange(frames) / 25,
    }
    return record, arrays


@pytest.mark.parametrize("width", [1, 5, 13])
def test_every_original_center_is_retained_with_explicit_edge_padding(width):
    record, arrays = fixture()
    sequence = probe.sequence_view(record, arrays)
    for center in range(record["frames"]):
        sample = probe.context(sequence, center, width)
        assert sample["input_values"].shape == (width, 2, 5, 2)
        assert sample["center_frame"] == center
        assert sample["center_seconds"] == center / 25
        np.testing.assert_array_equal(sample["input_values"][width // 2], sequence.tips[center])
        padded = sample["context_frame_indices"] == -1
        assert not sample["input_valid"][padded].any()
        assert (sample["input_values"][padded] == 0).all()
        assert np.isnan(sample["context_seconds"][padded]).all()


def test_target_values_and_missingness_never_change_inputs_or_context_selection():
    record, arrays = fixture()
    before = probe.context(probe.sequence_view(record, arrays), 8, 13)
    changed = copy.deepcopy(arrays)
    changed["full_hands"].reshape(17, 2, 21, 3)[:, :, probe.TARGET_JOINTS] = np.nan
    after = probe.context(probe.sequence_view(record, changed), 8, 13)
    for key in ("input_values", "input_valid", "context_frame_indices", "context_seconds"):
        np.testing.assert_array_equal(before[key], after[key])
    assert before["target_valid"].all()
    assert not after["target_valid"].any()


def test_missing_hand_remains_masked_without_dropping_other_hand_or_timeline():
    record, arrays = fixture()
    arrays["fingertips"][:, 5:] = np.nan
    arrays["hand_continuity"] = probe.tip_continuity(arrays["fingertips"].reshape(17, 2, 5, 3))
    sequence = probe.sequence_view(record, arrays)
    sample = probe.context(sequence, 8, 13)
    assert len(sequence.frame_indices) == 17
    assert sample["input_valid"][:, 0].all()
    assert not sample["input_valid"][:, 1].any()
    assert np.isfinite(sample["input_values"]).all()


def test_context_does_not_bridge_a_missing_frame_or_a_track_reset():
    record, arrays = fixture()
    arrays["fingertips"][7, :5] = np.nan
    arrays["fingertips"][10:, :5] += 0.8
    arrays["hand_continuity"] = probe.tip_continuity(arrays["fingertips"].reshape(17, 2, 5, 3))
    sample = probe.context(probe.sequence_view(record, arrays), 9, 13)
    frames = sample["context_frame_indices"]
    assert not sample["input_valid"][(frames < 8) | (frames >= 10), 0].any()
    assert sample["input_valid"][(frames >= 8) & (frames <= 9), 0].all()
    assert sample["input_valid"][:, 1].all()


def test_xy_does_not_receive_native_depth_and_xyz_preserves_original_depth():
    record, arrays = fixture()
    before = probe.context(probe.sequence_view(record, arrays), 8, 5)
    arrays["fingertips"][..., 2] += 100
    after = probe.context(probe.sequence_view(record, arrays), 8, 5)
    np.testing.assert_array_equal(before["input_values"], after["input_values"])
    native = probe.context(probe.sequence_view(record, arrays, "native_xyz"), 8, 5)
    np.testing.assert_array_equal(native["input_values"][..., :2], before["input_values"])
    assert (native["input_values"][..., 2] >= 100).all()


def test_claimed_continuity_and_compressed_timeline_are_rejected():
    record, arrays = fixture()
    arrays["hand_continuity"][0] = True
    with pytest.raises(ValueError, match="fingertip-only"):
        probe.sequence_view(record, arrays)
    record, arrays = fixture()
    arrays["source_frame_indices"][5:] += 1
    with pytest.raises(ValueError, match="original frame"):
        probe.sequence_view(record, arrays)


@pytest.mark.parametrize("width", [0, 2, 7, True])
def test_only_predeclared_context_widths_are_accepted(width):
    record, arrays = fixture()
    with pytest.raises(ValueError, match="width"):
        probe.context(probe.sequence_view(record, arrays), 0, width)


def test_packet_rejects_unbound_manifest_before_opening_features(tmp_path):
    directory = tmp_path / probe.PACKET
    directory.mkdir(parents=True)
    (directory / "complete.json").write_text('{"complete":true}')
    with pytest.raises(ValueError, match="binding changed"):
        probe.Packet(tmp_path)
