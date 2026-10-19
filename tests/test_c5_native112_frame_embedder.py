from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_native112_frame_embedder import Native112FrameEmbedder
from c5_patch_projection import LinearPatchProjection
from c5_small_input_vision import input_tensor


class TinyBackbone(nn.Module):
    """Exercise real patch projection with no pretrained transformer or assets."""

    def __init__(self, fault=None):
        super().__init__()
        self.patch_embed = nn.Module()
        self.patch_embed.proj = LinearPatchProjection(nn.Conv2d(3, 384, 14, stride=14))
        self.pos_embed = nn.Parameter(torch.zeros(1, 257, 384))
        self.mask_token = nn.Parameter(torch.zeros(1, 384))
        self.seen = []
        self.fault = fault
        self.requires_grad_(False).eval()

    def forward(self, pixels):
        self.seen.append(pixels.cpu().clone())
        if self.fault == "copy":
            pixels = pixels.clone()
        elif self.fault == "resize":
            pixels = torch.nn.functional.interpolate(pixels, size=(224, 224))
        elif self.fault == "skip":
            return pixels.new_zeros(len(pixels), 384)
        values = self.patch_embed.proj(pixels).mean((2, 3))
        if self.fault == "twice":
            values += self.patch_embed.proj(pixels).mean((2, 3))
        elif self.fault == "nonfinite":
            values[0, 0] = float("nan")
        elif self.fault == "drop":
            values = values[:-1]
        elif self.fault == "dtype":
            values = values.double()
        elif self.fault == "raise":
            raise RuntimeError("synthetic model failure")
        return values


def crops(count=5):
    rng = np.random.default_rng(719)
    return [rng.integers(0, 256, (112, 112, 3), dtype=np.uint8) for _ in range(count)]


def test_all_rgb_pixels_match_training_normalization_and_batches_preserve_every_frame():
    model = TinyBackbone()
    frames = crops()
    before_frames = [frame.copy() for frame in frames]
    before_weights = {key: value.clone() for key, value in model.state_dict().items()}
    embedder = Native112FrameEmbedder(model, batch_context=nullcontext, batch_size=2)
    actual = embedder.extract_embeddings_from_frames(frames)
    received = torch.cat(model.seen)
    expected = input_tensor(np.repeat(np.stack(frames)[:, None], 3, axis=1))[:, 0]
    assert torch.equal(received, expected)
    assert [len(part) for part in model.seen] == [2, 2, 1]
    with torch.inference_mode():
        reference = model.patch_embed.proj(expected).mean((2, 3)).numpy()
    np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-6)
    assert actual.shape == (5, 384) and actual.dtype == np.float32
    assert all(np.array_equal(a, b) for a, b in zip(frames, before_frames, strict=True))
    assert all(torch.equal(value, before_weights[key]) for key, value in model.state_dict().items())
    assert not model.patch_embed.proj._forward_pre_hooks


def test_execution_context_covers_forward_and_returns_host_values_before_leaving():
    model = TinyBackbone()
    state = {"active": False, "entries": 0}

    @contextmanager
    def execution():
        assert not state["active"]
        state["active"] = True
        state["entries"] += 1
        try:
            yield
        finally:
            state["active"] = False

    def entered(module, args):
        assert state["active"] and not torch.is_grad_enabled()

    handle = model.register_forward_pre_hook(entered)
    try:
        value = Native112FrameEmbedder(
            model, batch_context=execution
        ).extract_embeddings_from_frames(np.stack(crops()))
    finally:
        handle.remove()
    assert state == {"active": False, "entries": 3}
    assert value.shape == (5, 384) and np.isfinite(value).all()


@pytest.mark.parametrize("fault", ["copy", "resize", "skip", "twice", "nonfinite", "drop", "dtype"])
def test_hidden_resize_changed_patch_inputs_and_incomplete_features_are_rejected(fault):
    model = TinyBackbone(fault)
    embedder = Native112FrameEmbedder(model, batch_context=nullcontext)
    with pytest.raises(ValueError, match="native112 backbone"):
        embedder.extract_embeddings_from_frames(crops(2))
    assert not model.patch_embed.proj._forward_pre_hooks


def test_forward_failure_releases_context_and_removes_input_guard():
    model = TinyBackbone("raise")
    active = []

    @contextmanager
    def execution():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    with pytest.raises(RuntimeError, match="synthetic model failure"):
        Native112FrameEmbedder(model, batch_context=execution).extract_embeddings_from_frames(
            crops(1)
        )
    assert not active and not model.patch_embed.proj._forward_pre_hooks


@pytest.mark.parametrize(
    "fault", ["empty", "wrong_size", "float", "late_bad_frame", "scalar", "none", "string"]
)
def test_invalid_crop_sequence_is_rejected_before_any_model_execution(fault):
    model = TinyBackbone()
    frames = crops()
    if fault == "empty":
        frames = []
    elif fault == "wrong_size":
        frames[0] = np.zeros((224, 224, 3), np.uint8)
    elif fault == "float":
        frames[0] = frames[0].astype(np.float32)
    elif fault == "late_bad_frame":
        frames[-1] = np.zeros((112, 112, 4), np.uint8)
    elif fault == "scalar":
        frames = np.array(5)
    elif fault == "none":
        frames = None
    else:
        frames = "not decoded RGB crops"
    with pytest.raises(ValueError, match="complete uint8 RGB112px crops"):
        Native112FrameEmbedder(model, batch_context=nullcontext).extract_embeddings_from_frames(
            frames
        )
    assert not model.seen


@pytest.mark.parametrize("fault", ["training", "trainable", "double", "convolution", "position"])
def test_wrong_or_training_backbone_is_rejected_without_changing_it(fault):
    model = TinyBackbone()
    if fault == "training":
        model.train()
    elif fault == "trainable":
        model.mask_token.requires_grad_(True)
    elif fault == "double":
        model.double()
    elif fault == "convolution":
        model.patch_embed.proj = nn.Conv2d(3, 384, 14, stride=14).requires_grad_(False).eval()
    else:
        model.pos_embed = nn.Parameter(torch.zeros(1, 65, 384), requires_grad=False)
    flags = [(id(p), p.requires_grad, p.dtype) for p in model.parameters()]
    mode = model.training
    with pytest.raises(ValueError, match="frozen FP32 DINO with linear patches"):
        Native112FrameEmbedder(model, batch_context=nullcontext)
    assert [(id(p), p.requires_grad, p.dtype) for p in model.parameters()] == flags
    assert model.training == mode and not model.seen


def test_backbone_changed_to_training_after_construction_is_rejected_before_execution():
    model = TinyBackbone()
    embedder = Native112FrameEmbedder(model, batch_context=nullcontext)
    model.train()
    with pytest.raises(ValueError, match="frozen FP32 DINO with linear patches"):
        embedder.extract_embeddings_from_frames(crops(1))
    assert not model.seen


@pytest.mark.parametrize("size", [True, 0, -1, 1.5])
def test_invalid_batch_size_is_rejected(size):
    with pytest.raises(ValueError, match="positive batch"):
        Native112FrameEmbedder(TinyBackbone(), batch_context=nullcontext, batch_size=size)
