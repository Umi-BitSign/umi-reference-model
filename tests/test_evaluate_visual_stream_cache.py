"""Visual-stream evaluation rejects malformed or mismatched caches."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_fleurs_cache import (
    cached_visual_streams,
    evaluation_identifiers,
    temporal_subsample,
)


class Payload:
    def __init__(self, values):
        self.values = values
        self.files = list(values)

    def __getitem__(self, key):
        return self.values[key]


def _case():
    values = {
        "encoder_hidden_state": np.ones((2, 1536), dtype=np.float32),
        "face_features": np.ones((2, 384), dtype=np.float32),
        "left_hand_features": np.ones((2, 384), dtype=np.float32),
        "right_hand_features": np.ones((2, 384), dtype=np.float32),
        "pose_features": np.ones((2, 14), dtype=np.float32),
    }
    row = {
        "encoder_shape": [2, 1536],
        "stream_feature_dtype": "float32",
        "stream_feature_shapes": {
            name: list(value.shape)
            for name, value in values.items()
            if name != "encoder_hidden_state"
        },
    }
    return Payload(values), row


def test_accepts_frame_aligned_float32_streams():
    payload, row = _case()
    streams = cached_visual_streams(payload, row)
    assert set(streams) == set(row["stream_feature_shapes"])
    assert all(value.dtype == np.float32 for value in streams.values())


def test_rejects_missing_wrong_dtype_or_nonfinite_stream():
    payload, row = _case()
    payload.files.remove("pose_features")
    with pytest.raises(ValueError, match="lacks"):
        cached_visual_streams(payload, row)
    payload, row = _case()
    payload.values["pose_features"] = payload.values["pose_features"].astype(np.float64)
    with pytest.raises(ValueError, match="dtype"):
        cached_visual_streams(payload, row)
    payload, row = _case()
    payload.values["face_features"][0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        cached_visual_streams(payload, row)


def test_temporal_subsample_keeps_aligned_even_frames():
    streams = {
        "face_features": np.arange(15, dtype=np.float32).reshape(5, 3),
        "left_hand_features": np.arange(10, dtype=np.float32).reshape(5, 2),
    }
    result = temporal_subsample(streams, 2)
    assert result["face_features"].shape == (3, 3)
    assert np.array_equal(result["face_features"], streams["face_features"][[0, 2, 4]])
    assert np.array_equal(result["left_hand_features"], streams["left_hand_features"][[0, 2, 4]])


def test_temporal_subsample_rejects_nonpositive_stride():
    with pytest.raises(ValueError, match="positive"):
        temporal_subsample({"face_features": np.ones((2, 3), dtype=np.float32)}, 0)


def test_openasl_training_identifiers_use_clip_ids():
    rows = [
        {
            "schema": "umi-studio-openasl-stream-cache/1",
            "split": "training",
            "clip_id": "openasl-0000-abcdefghijk",
        }
    ]
    assert evaluation_identifiers(rows, "openasl_training") == [
        "openasl-0000-abcdefghijk"
    ]


def test_expansion_holdout_identifiers_use_training_sentence_names():
    rows = [
        {
            "schema": "umi-studio-how2sign-precut-expansion-stream-cache/1",
            "split": "training",
            "sentence_name": "clip-1",
        }
    ]
    assert evaluation_identifiers(rows, "how2sign_expansion_holdout") == ["clip-1"]
