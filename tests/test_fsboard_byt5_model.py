from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from fsboard_byt5_model import FSboardByT5, normalized_features


def test_normalized_features_downsamples_without_truncating_or_losing_missingness() -> None:
    points = np.full((5, 543, 3), np.nan, dtype=np.float32)
    points[0, 468, :] = [0.1, 0.2, 0.3]
    points[2, 468, :] = [0.2, 0.3, 0.4]
    features = normalized_features(points, frame_stride=2, max_frames=2)
    assert features.shape == (2, 1629)
    assert np.isfinite(features).all()
    assert np.count_nonzero(features[:, :468 * 3]) == 0
    with pytest.raises(ValueError, match="frame limit"):
        normalized_features(points, frame_stride=1, max_frames=1)


def test_tiny_byt5_projection_and_decoder_receive_gradients() -> None:
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
    model = FSboardByT5(T5ForConditionalGeneration(config))
    features = torch.randn(1, 3, 1629)
    mask = torch.ones(1, 3, dtype=torch.long)
    labels = torch.tensor([[100, 101, 1]], dtype=torch.long)
    loss = model(features, mask, labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert model.language_model.shared.weight.grad is not None


def test_tiny_byt5_generates_from_projected_landmarks() -> None:
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
    model = FSboardByT5(T5ForConditionalGeneration(config)).eval()
    with torch.inference_mode():
        tokens = model.language_model.generate(
            inputs_embeds=model.projection(torch.randn(1, 3, 1629)),
            attention_mask=torch.ones(1, 3, dtype=torch.long),
            max_new_tokens=4,
            num_beams=1,
        )
    assert tokens.shape[0] == 1
    assert 1 < tokens.shape[1] <= 5


def test_tiny_byt5_checkpoint_loads_with_weights_only(tmp_path: Path) -> None:
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
    original = FSboardByT5(T5ForConditionalGeneration(config))
    checkpoint = tmp_path / "model.pt"
    torch.save(original.state_dict(), checkpoint)
    restored = FSboardByT5(T5ForConditionalGeneration(config))
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    torch.testing.assert_close(restored.projection.weight, original.projection.weight)
    torch.testing.assert_close(
        restored.language_model.shared.weight, original.language_model.shared.weight
    )
