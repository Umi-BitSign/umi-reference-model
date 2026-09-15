from __future__ import annotations

import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime():
    spec = importlib.util.spec_from_file_location(
        "crop_color_runtime", ROOT / "candidates/community-baseline-v0.2/runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crop_conversion_preserves_pixels_and_ownership(runtime):
    source = np.array([[[3, 17, 241], [0, 99, 255]]], dtype=np.uint8)
    before = source.copy()
    output = runtime._rgb_crop_frames([source])[0]
    np.testing.assert_array_equal(output, [[[241, 17, 3], [255, 99, 0]]])
    np.testing.assert_array_equal(source, before)
    assert output.dtype == np.uint8
    assert output.flags.c_contiguous
    assert not np.shares_memory(source, output)


def test_black_fallback_remains_black(runtime):
    source = np.zeros((224, 224, 3), dtype=np.uint8)
    np.testing.assert_array_equal(runtime._rgb_crop_frames([source])[0], source)
    assert runtime._rgb_crop_frames([]) == []


@pytest.mark.parametrize(
    "value",
    [
        None,
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2, 4), dtype=np.uint8),
        np.zeros((0, 2, 3), dtype=np.uint8),
        np.zeros((2, 2, 3), dtype=np.float32),
    ],
)
def test_invalid_crop_is_not_silently_dropped(runtime, value):
    with pytest.raises(ValueError, match="uint8 BGR crop"):
        runtime._rgb_crop_frames([value])


def test_translation_sends_rgb_to_all_three_dino_streams(runtime, tmp_path):
    # No checkpoint or contributed model is loaded. Exercise the actual
    # translate_path wiring with different known BGR values for each crop.
    crops = [
        np.full((2, 2, 3), pixel, dtype=np.uint8)
        for pixel in ([1, 2, 3], [4, 5, 6], [7, 8, 9])
    ]
    source = np.full((1, 4, 4, 3), [200, 100, 50], dtype=np.uint8)

    class Reader:
        def __len__(self):
            return 1

        def get_batch(self, indices):
            assert list(indices) == [0]
            return SimpleNamespace(asnumpy=lambda: source)

    received = []

    def embed(frames):
        received.append(frames[0].copy())
        return np.zeros((1, 384), dtype=np.float32)

    model = runtime.SHuBERTInferenceRuntime.__new__(runtime.SHuBERTInferenceRuntime)
    model._thread_state = threading.local()
    model._model_execution_slots = threading.BoundedSemaphore(1)
    model._model_execution_concurrency = 1
    model._video_reader = lambda _: Reader()
    model._video_holistic = lambda *_: {0: {}}
    model.models = tmp_path
    model._hand_extractor = SimpleNamespace(extract_hand_frames=lambda *_: ([crops[0]], [crops[1]]))
    model._face_extractor = SimpleNamespace(extract_face_frames=lambda *_: [crops[2]])
    model._process_pose_landmarks = lambda _: np.zeros((1, 14), dtype=np.float32)
    model._hand_model = SimpleNamespace(extract_embeddings_from_frames=embed)
    model._face_model = SimpleNamespace(extract_embeddings_from_frames=embed)
    model._translator = SimpleNamespace(generate=lambda **_: torch.tensor([[1]]))
    model._tokenizer = SimpleNamespace(
        pad_token_id=0, eos_token_id=1, decode=lambda *_args, **_kw: "fixture"
    )
    model.device = torch.device("cpu")
    model.generation_max_length = 2048
    model.generation_num_beams = 5
    model.dino_batch_size = 128
    model.component_devices = {}
    assert model.translate_path(tmp_path / "fixture.mp4") == "fixture"
    assert len(received) == 3
    for actual, original in zip(received, crops, strict=True):
        np.testing.assert_array_equal(actual, original[..., ::-1])
    # Landmark input remains RGB and is not modified by the crop conversion.
    np.testing.assert_array_equal(source[0, 0, 0], [200, 100, 50])
