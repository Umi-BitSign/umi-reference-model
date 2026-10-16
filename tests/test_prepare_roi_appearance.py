from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_roi_appearance import SampledRgbReader, rgb_crops, select


class Capture:
    def __init__(self, count=5, *, actual=None):
        self.count, self.actual, self.position = count, actual if actual is not None else count, 0
        self.closed = False

    def isOpened(self):
        return True

    def get(self, name):
        return 25 if name == cv2.CAP_PROP_FPS else self.count

    def read(self):
        if self.position >= self.actual:
            return False, None
        value = np.full((2, 3, 3), [self.position, 17, 241], dtype=np.uint8)
        self.position += 1
        return True, value

    def release(self):
        self.closed = True


def header(count=5):
    indices = np.arange(0, count, 2, dtype=np.int64)
    return {
        "source_frames": count,
        "source_fps": 25,
        "source_shape": [2, 3],
        "sampled_frames": len(indices),
        "frame_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
    }


def test_streaming_preserves_exact_source_indices_and_RGB_color():
    capture = Capture()
    with SampledRgbReader("unused", header(), capture_factory=lambda _: capture) as reader:
        assert len(reader) == 3
        values = [reader[i][0, 0].tolist() for i in range(3)]
        assert values == [[241, 17, 0], [241, 17, 2], [241, 17, 4]]
    assert capture.closed


def test_decoder_checks_the_unsampled_final_source_frame_too():
    capture = Capture(4, actual=3)
    with pytest.raises(ValueError, match="complete bound source"):
        with SampledRgbReader("unused", header(4), capture_factory=lambda _: capture) as reader:
            for i in range(len(reader)):
                reader[i]
    assert capture.closed


def test_extra_decoded_source_frames_are_not_accepted():
    capture = Capture(4, actual=5)
    with pytest.raises(ValueError, match="more source frames"):
        with SampledRgbReader("unused", header(4), capture_factory=lambda _: capture) as reader:
            for i in range(len(reader)):
                reader[i]


def test_seeks_repeats_and_changed_timeline_hash_are_rejected():
    capture = Capture()
    with pytest.raises(ValueError, match="seek"):
        with SampledRgbReader("unused", header(), capture_factory=lambda _: capture) as reader:
            reader[1]
    with pytest.raises(ValueError, match="timeline"):
        SampledRgbReader(
            "unused",
            {**header(), "frame_indices_sha256": "wrong"},
            capture_factory=lambda _: Capture(),
        )


def test_native_BGR_crop_becomes_independent_small_RGB_input():
    source = np.full((64, 64, 3), [3, 17, 241], dtype=np.uint8)
    converted = rgb_crops([source], 1)
    np.testing.assert_array_equal(converted[0, 0, 0], [241, 17, 3])
    assert converted.flags.c_contiguous
    assert not np.shares_memory(converted, source)


@pytest.mark.parametrize(
    "value", [np.zeros((32, 64, 3), dtype=np.uint8), np.zeros((64, 64, 3), dtype=np.float32), None]
)
def test_no_wrong_surface_or_missing_crop_is_silently_adopted(value):
    with pytest.raises(ValueError):
        rgb_crops([value], 1)


def rows():
    return [
        {
            "source": "asl_stem_original",
            "example_id": str(i),
            "signer": f"P{i}",
            "role": "training",
            "training_allowed": True,
            "original_pair": {"header": {"source_frames": 100}},
        }
        for i in range(10)
    ]


def test_selection_is_deterministic_source_distinct_and_never_uses_reserved_people():
    data = {"records": rows(), "reserved_STEM_participants": []}
    selected = select(data)
    assert selected == select({**data, "records": list(reversed(data["records"]))})
    assert len({r["signer"] for r in selected}) == 8
    forbidden = selected[0]["signer"]
    with pytest.raises(ValueError, match="reserved"):
        select({**data, "reserved_STEM_participants": [forbidden]})


def test_development_permission_cannot_be_overridden_by_old_nested_data():
    inputs = rows()
    inputs[0]["role"] = "development"
    inputs[0]["training_allowed"] = False
    assert inputs[0] not in select({"records": inputs, "reserved_STEM_participants": []})
    inputs[0]["training_allowed"] = True
    with pytest.raises(ValueError, match="role"):
        select({"records": inputs, "reserved_STEM_participants": []})
