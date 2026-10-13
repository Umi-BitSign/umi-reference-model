from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fresh_lowres_model import (
    FreshLowResolutionTranslator,
    S3DVideoEncoder,
    initialization_manifest,
    resize_video_short_side,
)


class _TinyFrameEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convolution = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch, time, channels, height, width = frames.shape
        hidden = self.convolution(frames.reshape(batch * time, channels, height, width)).mean(
            dim=(-2, -1)
        )
        return hidden.reshape(batch, time, 8)


def _tiny_model() -> FreshLowResolutionTranslator:
    config = T5Config(
        vocab_size=384,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=16,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    return FreshLowResolutionTranslator(
        T5ForConditionalGeneration(config),
        _TinyFrameEncoder(),
        8,
        temporal_width=16,
        temporal_layers=1,
        temporal_heads=2,
        temporal_feedforward_width=32,
        max_frames=8,
        dropout=0.0,
    )


def test_resize_video_short_side_preserves_aspect_ratio() -> None:
    landscape = torch.rand(3, 3, 40, 80)
    portrait = torch.rand(3, 3, 80, 40)
    assert resize_video_short_side(landscape, 64).shape == (3, 3, 64, 128)
    assert resize_video_short_side(portrait, 64).shape == (3, 3, 128, 64)
    with pytest.raises(ValueError, match="at least 64"):
        resize_video_short_side(landscape, 32)


def test_fresh_model_propagates_gradients_from_text_to_raw_frames() -> None:
    model = _tiny_model()
    frames = torch.rand(2, 4, 3, 16, 24)
    frame_mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    labels = torch.tensor([[100, 101, 1], [102, 103, 1]])
    output = model(frames, frame_mask, labels)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.frame_encoder.convolution.weight.grad is not None
    assert model.temporal_encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.language_model.shared.weight.grad is not None


def test_fresh_model_generates_without_sign_specific_state() -> None:
    model = _tiny_model().eval()
    tokens = model.generate(
        torch.rand(1, 3, 3, 16, 24),
        torch.ones(1, 3, dtype=torch.bool),
        max_new_tokens=3,
        num_beams=1,
    )
    assert tokens.shape[0] == 1
    assert 1 < tokens.shape[1] <= 4


def test_initialization_manifest_distinguishes_clean_pretraining_and_scratch() -> None:
    clean = initialization_manifest(
        visual_initialization="imagenet1k_v1",
        language_initialization="google_byt5_small",
        source_short_side=192,
        temporal_sample_fps=12.5,
    )
    scratch = initialization_manifest(
        visual_initialization="random",
        language_initialization="random",
        source_short_side=192,
        temporal_sample_fps=12.5,
    )
    assert clean["sign_specific_checkpoint"] is None
    assert not clean["sign_specific_encoder_cache"]
    assert scratch["visual_initialization"] == "random"
    assert scratch["language_initialization"] == "random"
    assert clean["language_encoder_policy"] == "discarded"


def test_initialization_manifest_accepts_subword_t5() -> None:
    manifest = initialization_manifest(
        visual_initialization="imagenet1k_v1",
        language_initialization="google_t5_small",
        source_short_side=192,
        temporal_sample_fps=12.5,
        use_language_encoder=True,
    )
    assert manifest["language_initialization"] == "google_t5_small"
    assert manifest["language_encoder_policy"] == "generic_t5_small"


def test_generic_language_encoder_can_contextualize_visual_states() -> None:
    config = T5Config(
        vocab_size=384,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        d_kv=16,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = FreshLowResolutionTranslator(
        T5ForConditionalGeneration(config),
        _TinyFrameEncoder(),
        8,
        temporal_width=16,
        temporal_layers=1,
        temporal_heads=2,
        temporal_feedforward_width=32,
        max_frames=8,
        dropout=0.0,
        use_language_encoder=True,
    )
    memory = model.encode(torch.rand(1, 3, 3, 16, 24), torch.ones(1, 3, dtype=torch.bool))
    assert memory.shape == (1, 3, 32)
    assert torch.isfinite(memory).all()


def test_model_rejects_empty_or_malformed_frame_masks() -> None:
    model = _tiny_model()
    frames = torch.rand(1, 3, 3, 16, 24)
    with pytest.raises(ValueError, match="boolean"):
        model.encode(frames, torch.ones(1, 3))
    with pytest.raises(ValueError, match="empty"):
        model.encode(frames, torch.zeros(1, 3, dtype=torch.bool))


def test_s3d_encoder_retains_a_downsampled_temporal_sequence() -> None:
    class _Features(torch.nn.Module):
        def forward(self, video: torch.Tensor) -> torch.Tensor:
            assert video.shape == (2, 3, 8, 16, 24)
            return torch.ones(2, 1024, 2, 2, 3)

    encoded = S3DVideoEncoder(_Features())(torch.rand(2, 8, 3, 16, 24))
    assert encoded.shape == (2, 2, 1024)
