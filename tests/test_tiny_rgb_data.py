from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tiny_rgb_data


def test_streaming_resize_is_aspect_preserving_temporally_complete_and_pre_network(
    tmp_path, monkeypatch
):
    import cv2

    video_dir = tmp_path / "data"
    video_dir.mkdir()
    path = video_dir / "training.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (112, 64))
    assert writer.isOpened()
    for index in range(20):
        writer.write(np.full((64, 112, 3), index * 5, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(tiny_rgb_data, "ROOT", tmp_path)
    pixels, indices, metadata = tiny_rgb_data.decode_rgb_video("data/training.mp4", 32)
    assert pixels.shape == (10, 32, 56, 3)
    assert indices.tolist() == list(range(0, 20, 2))
    assert metadata["source_fps"] == 25
    assert metadata["source_frames"] == 20
    assert tiny_rgb_data.pixel_tensor(pixels).shape == (10, 3, 32, 56)
    with pytest.raises(ValueError, match="do not truncate"):
        tiny_rgb_data.decode_rgb_video("data/training.mp4", 32, max_frames=4)


def test_raw_decoder_rejects_paths_outside_private_data_and_embedding_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(tiny_rgb_data, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="private data"):
        tiny_rgb_data.decode_rgb_video("outside.mp4", 128)
    with pytest.raises(ValueError, match="not learned embeddings"):
        tiny_rgb_data.pixel_tensor(np.zeros((20, 768), dtype=np.float32))
