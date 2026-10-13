import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from probe_visual_retrieval import nearest_neighbor, normalized_mean, source_video_id


def test_source_video_id() -> None:
    assert source_video_id("a_b_12-3-rgb_front") == "a_b"
    with pytest.raises(ValueError):
        source_video_id("a_b")


def test_normalized_mean() -> None:
    state = np.zeros((2, 1536), dtype=np.float32)
    state[:, 0] = 1
    vector = normalized_mean(state, 2)
    assert vector[0] == 1
    assert np.linalg.norm(vector) == 1
    with pytest.raises(ValueError):
        normalized_mean(state.astype(np.float64), 2)


def test_nearest_neighbor_excludes_source_video() -> None:
    vectors = np.array([[1, 0], [1, 0], [0.5, 0.5]], dtype=np.float32)
    index, score = nearest_neighbor(0, vectors, ["same", "same", "other"])
    assert index == 2
    assert score == 0.5
