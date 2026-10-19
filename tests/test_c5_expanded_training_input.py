"""Expanded source admission retains full signer and temporal boundaries."""

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_expanded_training_input import expected_header, read_training224, training_row


def record():
    identity = {"source": "asl_stem_original", "example_id": "a", "source_group": "group"}
    raw = {
        **identity,
        "role": "training",
        "training_allowed": True,
        "signer": "P2",
        "reference": "A full caption.",
        "video": "data/a.mp4",
        "video_sha256": "abc",
        "video_fps": 25,
    }
    grid = np.arange(0, 7, 2, dtype=np.int64)
    clock = {
        "sampled_frames": 4,
        "frame_indices_sha256": hashlib.sha256(grid.tobytes()).hexdigest(),
    }
    header = {**raw, **clock, "source_fps": 25, "source_frames": 7, "source_shape": [20, 30]}
    teacher = {**raw, "stored_temporal_stride": 2, "frames": 4}
    binding = {
        **identity,
        "role": "training",
        "training_allowed": True,
        "original_index": 2,
        "source_frames": 7,
        "sampled_frames": 4,
        "canonical_video": {"path": "data/a.mp4", "sha256": "abc"},
    }
    return {
        "pair": {
            "raw": raw,
            "header": header,
            "alignment": {**raw, **clock},
            "teacher": teacher,
            "native_teacher_stride": 2,
            "feature_pretraining_allowed": True,
        },
        "candidate": {**binding, "stored_stride": 2},
        "annotation": copy.deepcopy(binding),
    }


def test_keeps_complete_odd_source_clock_and_caption():
    actual = record()
    assert training_row(actual)["reference"] == "A full caption."
    assert expected_header(actual)["source_frames"] == 7
    assert expected_header(actual)["sampled_frames"] == 4


@pytest.mark.parametrize(
    "signer", ["P0", "P19", "P10", "P13", "P14", "P21", "P22", "P31", "P4", "P6"]
)
def test_all_protected_signers_fail_before_source_reads(signer):
    actual = record()
    actual["pair"]["raw"]["signer"] = actual["pair"]["teacher"]["signer"] = signer
    with pytest.raises(ValueError, match="reserved or retained diagnostic"):
        read_training224(actual)


@pytest.mark.parametrize("part", ["raw", "header", "alignment", "teacher"])
def test_nested_development_role_fails_before_source_reads(part):
    actual = record()
    actual["pair"][part]["role"] = "development"
    with pytest.raises(ValueError, match="affirmative training roles"):
        read_training224(actual)


@pytest.mark.parametrize(
    "change", ["group", "stride", "source_count", "sample_count", "grid", "fps"]
)
def test_source_join_or_clock_change_fails_before_source_reads(change):
    actual = record()
    if change == "group":
        actual["annotation"]["source_group"] = "other"
    elif change == "stride":
        actual["candidate"]["stored_stride"] = 1
    elif change == "source_count":
        actual["annotation"]["source_frames"] = 8
    elif change == "sample_count":
        actual["pair"]["teacher"]["frames"] = 3
    elif change == "grid":
        actual["pair"]["header"]["frame_indices_sha256"] = "changed"
    else:
        actual["pair"]["raw"]["video_fps"] = 30
    with pytest.raises(ValueError, match="expanded input"):
        read_training224(actual)
