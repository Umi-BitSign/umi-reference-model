from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import T5Config, T5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_language_model import TinyRgbLanguageTranslator
from tiny_rgb_model import TinyRgbConfig


def language():
    # A tiny randomly initialized stand-in tests preservation, not pretrained quality.
    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=32,
            d_model=32,
            d_ff=64,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            d_kv=8,
            dropout_rate=0.0,
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
        )
    )


def visual():
    return TinyRgbConfig(width=32, heads=4, encoder_layers=1, decoder_layers=1, dropout=0.0)


def test_supplied_language_weights_are_preserved_and_scratch_decoder_is_absent():
    text_model = language()
    before = {
        name: tensor.clone()
        for name, tensor in text_model.state_dict().items()
        if not name.startswith("encoder.")
    }
    model = TinyRgbLanguageTranslator(text_model, visual())
    after = model.language_model.state_dict()
    assert all(torch.equal(tensor, after[name]) for name, tensor in before.items())
    assert not hasattr(model.vision, "decoder") and not hasattr(model.vision, "embedding")
    assert not list(model.language_model.encoder.parameters())
    assert all(p.requires_grad for p in model.vision.parameters())


def test_loss_and_gradients_condition_pretrained_language_interface_on_raw_pixels():
    model = TinyRgbLanguageTranslator(language(), visual())
    video = torch.randn(1, 2, 3, 32, 56)
    mask = torch.ones(1, 2, dtype=torch.bool)
    labels = torch.tensor([[4, 5, 1]])
    output = model(video, mask, labels)
    assert torch.isfinite(output.loss) and output.logits.shape == (1, 3, 32)
    output.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.vision.stem[0].weight.grad.abs().sum() > 0
    model.eval()
    assert not torch.allclose(
        model(video, mask, labels).logits, model(-video, mask, labels).logits, atol=1e-6
    )


def test_parameter_counts_separate_scratch_vision_from_language_and_generation_runs():
    model = TinyRgbLanguageTranslator(language(), visual()).eval()
    counts = model.parameter_counts()
    assert (
        counts["total"]
        == counts["scratch_visual_encoder"]
        + counts["scratch_visual_language_bridge"]
        + counts["pretrained_language_decoder_and_shared_embeddings"]
    )
    assert counts["removed_unused_text_encoder"] > 0
    generated = model.generate(
        torch.randn(1, 2, 3, 32, 56),
        torch.ones(1, 2, dtype=torch.bool),
        num_beams=2,
        max_new_tokens=4,
    )
    assert generated.shape[0] == 1
