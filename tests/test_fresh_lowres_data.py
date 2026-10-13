from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fresh_lowres_data import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    KINETICS400_MEAN,
    _resized_dimensions,
    augment_video,
    load_inventory,
    sample_frame_indices,
)
from fresh_lowres_model import MobileNetFrameEncoder


def test_sampling_reduces_25_fps_and_caps_across_the_whole_clip() -> None:
    assert sample_frame_indices(10, 25, 12.5, 20) == [0, 2, 4, 6, 8]
    capped = sample_frame_indices(100, 25, 12.5, 5)
    assert capped == [0, 24, 48, 74, 98]
    with pytest.raises(ValueError, match="sampling"):
        sample_frame_indices(100, 25, 30, 10)


def test_resize_dimensions_preserve_aspect_ratio() -> None:
    assert _resized_dimensions(360, 640, 192) == (341, 192)
    assert _resized_dimensions(640, 360, 192) == (192, 341)


def test_inventory_rejects_model_cache_fields(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    row = {
        "schema": "umi-studio-fresh-lowres-training-example/1",
        "source": "fleurs",
        "example_id": "example",
        "reference": "text",
        "video": "video.mp4",
        "video_sha256": "a" * 64,
        "encoder_cache_sha256": "b" * 64,
    }
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="invalid"):
        load_inventory(inventory, tmp_path)


def test_mobilenet_frame_encoder_chunks_frames() -> None:
    class _Features(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.batch_sizes = []

        def forward(self, frames: torch.Tensor) -> torch.Tensor:
            self.batch_sizes.append(len(frames))
            return torch.ones(len(frames), 576, 2, 2)

    features = _Features()
    encoder = MobileNetFrameEncoder(features, frame_batch_size=2)
    encoded = encoder(torch.rand(1, 5, 3, 8, 8))
    assert encoded.shape == (1, 5, 576)
    assert features.batch_sizes == [2, 2, 1]
    assert IMAGENET_MEAN.shape == IMAGENET_STD.shape == (1, 3, 1, 1)
    assert KINETICS400_MEAN.shape == (1, 3, 1, 1)


def test_mild_augmentation_is_seeded_clip_consistent_and_bounded() -> None:
    frame = torch.linspace(0, 1, 3 * 32 * 48).reshape(3, 32, 48)
    video = torch.stack((frame, frame))
    first = augment_video(video, profile="mild_v1", seed=41)
    repeated = augment_video(video, profile="mild_v1", seed=41)
    changed = augment_video(video, profile="mild_v1", seed=42)
    assert first.shape == video.shape
    assert torch.equal(first, repeated)
    assert not torch.equal(first, changed)
    assert torch.equal(first[0], first[1])
    assert 0 <= first.min() <= first.max() <= 1


def test_augmentation_rejects_missing_seed_and_unknown_profile() -> None:
    video = torch.zeros(1, 3, 16, 16)
    with pytest.raises(ValueError, match="seed"):
        augment_video(video, profile="mild_v1")
    with pytest.raises(ValueError, match="unknown"):
        augment_video(video, profile="heavy")
