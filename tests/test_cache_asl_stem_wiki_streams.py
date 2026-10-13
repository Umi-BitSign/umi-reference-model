"""ASL STEM C5 stream caching uses the confirmed post-MediaPipe stride."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cache_asl_stem_wiki_streams import temporal_subsample_landmarks


def test_temporal_subsample_reindexes_even_frames():
    landmarks = {index: {"value": index} for index in range(5)}
    assert temporal_subsample_landmarks(landmarks, 2) == {
        0: {"value": 0},
        1: {"value": 2},
        2: {"value": 4},
    }


def test_temporal_subsample_rejects_noncontiguous_input():
    with pytest.raises(ValueError, match="invalid"):
        temporal_subsample_landmarks({0: {}, 2: {}}, 2)
