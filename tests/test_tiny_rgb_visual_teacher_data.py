from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tiny_rgb_visual_teacher_data as data
from tiny_rgb_visual_teacher_data import bind_pair


def fixtures(stride=2):
    raw = {
        "source": "how2sign_new",
        "example_id": "fixture",
        "reference": "hello world",
        "source_group": "how2sign:fixture",
        "video_fps": 25,
        "video_sha256": "a" * 64,
        "role": "training",
        "training_allowed": True,
    }
    header = {
        "source": raw["source"],
        "example_id": raw["example_id"],
        "role": "training",
        "source_fps": 25,
        "source_frames": 3,
        "sampled_frames": 2,
        "video_sha256": raw["video_sha256"],
        "frame_indices_sha256": hashlib.sha256(
            np.array([0, 2], dtype=np.int64).tobytes()
        ).hexdigest(),
    }
    teacher = {
        "source": raw["source"],
        "id": raw["example_id"],
        "reference": raw["reference"],
        "role": "training",
        "training_allowed": True,
        "cache_sha256": "b" * 64,
        "stored_temporal_stride": stride,
        "frames": 3 if stride == 1 else 2,
    }
    receipt = {
        "sentence_name": raw["example_id"],
        "reference": raw["reference"],
        "split": "training",
        "clip_fps": 25,
        "source_frames": 3,
        "temporal_stride": stride,
        "effective_fps": 12.5,
        "resampled_clip_sha256": raw["video_sha256"],
        "encoder_cache_sha256": "b" * 64,
        "encoder_shape": [teacher["frames"], 1536],
    }
    return raw, header, teacher, receipt


@pytest.mark.parametrize("stride", [1, 2])
def test_native_receipts_bind_actual_source_hash_and_exact_even_frame_timeline(stride):
    pair = bind_pair(*fixtures(stride), "c" * 64)
    assert pair["alignment"]["sampled_frames"] == 2
    assert pair["alignment"]["video_sha256"] == "a" * 64
    assert pair["native_teacher_stride"] == stride


@pytest.mark.parametrize(
    "index,field,value",
    [
        (0, "training_allowed", False),
        (1, "role", "open_development"),
        (2, "id", "other"),
        (3, "split", "test"),
        (3, "reference", "wrong caption"),
        (3, "resampled_clip_sha256", "d" * 64),
        (3, "source_frames", 4),
        (1, "frame_indices_sha256", "d" * 64),
        (2, "frames", 3),
        (3, "encoder_cache_sha256", "d" * 64),
        (3, "temporal_stride", 1),
    ],
)
def test_equal_length_is_not_enough_to_authorize_teacher_pair(index, field, value):
    rows = list(copy.deepcopy(fixtures()))
    rows[index][field] = value
    with pytest.raises(ValueError):
        bind_pair(*rows, "c" * 64)


def test_canonical_input_uses_receipted_derivative_without_mutating_original(tmp_path, monkeypatch):
    import cv2
    from benchmark_fleurs_dev import sha256

    raw, header, _, receipt = fixtures()
    raw["source"] = "how2sign_precut_expansion"
    raw["video"] = "data/source.mp4"
    raw["video_fps"] = 24
    header["source_fps"] = 24
    header["frame_indices_sha256"] = hashlib.sha256(
        np.array([0, 1], dtype=np.int64).tobytes()
    ).hexdigest()
    original = tmp_path / raw["video"]
    original.parent.mkdir()
    original.write_bytes(b"original24")
    derivative = tmp_path / "data/how2sign/precut-expansion-clips-25fps/fixture.mp4"
    derivative.parent.mkdir(parents=True)
    derivative.write_bytes(b"derivative25")
    raw["video_sha256"] = sha256(original)
    receipt["source_clip_sha256"] = raw["video_sha256"]
    receipt["resampled_clip_sha256"] = sha256(derivative)
    receipt["source_frames"] = 4
    before = copy.deepcopy(raw)

    class Capture:
        def isOpened(self):
            return True

        def get(self, property):
            return {
                cv2.CAP_PROP_FPS: 25,
                cv2.CAP_PROP_FRAME_COUNT: 4,
                cv2.CAP_PROP_FRAME_HEIGHT: 32,
                cv2.CAP_PROP_FRAME_WIDTH: 48,
            }[property]

        def release(self):
            pass

    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: Capture())
    canonical, timeline = data.canonical_input(raw, header, receipt)
    assert canonical["video_sha256"] == sha256(derivative)
    assert canonical["video_fps"] == 25
    assert timeline["source_frames"] == 4
    assert timeline["frame_indices_sha256"] != header["frame_indices_sha256"]
    assert timeline["sampled_frames"] == header["sampled_frames"]
    assert raw == before and original.read_bytes() == b"original24"
    receipt["source_clip_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="no receipted"):
        data.canonical_input(raw, header, receipt)
