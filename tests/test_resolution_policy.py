from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from resolution_policy import (
    aspect_preserving_downscale,
    make_short_side_reader,
    set_visual_crop_size,
    validate_visual_crop_size,
)


class Batch:
    def __init__(self, value: np.ndarray):
        self.value = value

    def asnumpy(self) -> np.ndarray:
        return self.value


class Reader:
    def __init__(self, frame: np.ndarray):
        self.frame = frame

    def _read(self, index: int) -> np.ndarray:
        return self.frame.copy()

    def get_batch(self, indices) -> Batch:
        return Batch(np.stack([self._read(index) for index in indices]))


@pytest.fixture(autouse=True)
def fake_cv2(monkeypatch):
    def resize(frame: np.ndarray, size: tuple[int, int], interpolation: int) -> np.ndarray:
        assert interpolation == 3
        return np.zeros((size[1], size[0], frame.shape[2]), dtype=frame.dtype)

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        types.SimpleNamespace(INTER_AREA=3, resize=resize),
    )


def test_aspect_preserving_downscale_and_no_upscale() -> None:
    landscape = np.zeros((360, 640, 3), dtype=np.uint8)
    assert aspect_preserving_downscale(landscape, 256).shape == (256, 455, 3)
    portrait = np.zeros((640, 360, 3), dtype=np.uint8)
    assert aspect_preserving_downscale(portrait, 256).shape == (455, 256, 3)
    small = np.zeros((120, 160, 3), dtype=np.uint8)
    resized = aspect_preserving_downscale(small, 256)
    assert resized.shape == small.shape
    assert resized.flags.c_contiguous


def test_short_side_reader_resizes_single_and_batch_reads() -> None:
    wrapped = make_short_side_reader(Reader, 128)
    reader = wrapped(np.zeros((360, 640, 3), dtype=np.uint8))
    assert reader._read(0).shape == (128, 228, 3)
    assert reader.get_batch(range(2)).asnumpy().shape == (2, 128, 228, 3)


def test_crop_size_must_follow_dino_patch_grid() -> None:
    assert validate_visual_crop_size(112) == 112
    assert validate_visual_crop_size(168) == 168
    assert validate_visual_crop_size(224) == 224
    with pytest.raises(ValueError, match="multiple"):
        validate_visual_crop_size(160)
    with pytest.raises(ValueError, match="between"):
        validate_visual_crop_size(42)


def test_visual_crop_policy_updates_both_streams(monkeypatch) -> None:
    class Resize:
        def __init__(self, size):
            self.size = size

    class Compose:
        def __init__(self, stages):
            self.transforms = stages

    transforms = types.SimpleNamespace(Resize=Resize, Compose=Compose)
    monkeypatch.setitem(sys.modules, "torchvision", types.SimpleNamespace(transforms=transforms))

    def embedder():
        return types.SimpleNamespace(
            model=types.SimpleNamespace(patch_size=14),
            transform=Compose([Resize((224, 224)), object()]),
        )

    model = types.SimpleNamespace(
        _hand_model=embedder(),
        _face_model=embedder(),
        _hand_extractor=types.SimpleNamespace(output_size=(224, 224)),
        _face_extractor=types.SimpleNamespace(output_size=(224, 224)),
    )
    policy = set_visual_crop_size(model, 168)
    assert policy == {
        "visual_crop_size": 168,
        "dino_patch_size": 14,
        "patch_tokens_per_stream_frame": 144,
    }
    assert model._hand_extractor.output_size == (168, 168)
    assert model._face_extractor.output_size == (168, 168)
    assert model._hand_model.transform.transforms[0].size == (168, 168)
    assert model._face_model.transform.transforms[0].size == (168, 168)
