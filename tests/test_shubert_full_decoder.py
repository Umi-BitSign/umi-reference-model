"""A full-decoder adapter must leave the video encoder frozen."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from shubert_full_decoder import full_decoder_state, load_full_decoder, train_full_decoder


class TinyTranslator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(3, 4)
        self.decoder_emb = nn.Embedding(8, 4)
        self.decoder = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.decoder_emb.weight


def test_full_decoder_excludes_encoder_and_round_trips():
    model = TinyTranslator()
    selected = train_full_decoder(model)
    assert selected
    assert all(parameter.requires_grad for parameter in selected.values())
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    state = full_decoder_state(model)
    original = model.decoder[0].weight.detach().clone()
    with torch.no_grad():
        model.decoder[0].weight.add_(1)
    load_full_decoder(model, state)
    assert torch.equal(model.decoder[0].weight, original)


def test_full_decoder_rejects_incomplete_or_invalid_checkpoint():
    model = TinyTranslator()
    train_full_decoder(model)
    state = full_decoder_state(model)
    with pytest.raises(ValueError, match="missing"):
        load_full_decoder(model, dict(list(state.items())[1:]))
    changed = {
        **state,
        "decoder.0.weight": torch.full_like(state["decoder.0.weight"], float("nan")),
    }
    with pytest.raises(ValueError, match="non-finite"):
        load_full_decoder(model, changed)
    changed = {**state, "decoder.0.weight": state["decoder.0.weight"][:1]}
    with pytest.raises(ValueError, match="shape"):
        load_full_decoder(model, changed)


def test_full_decoder_rejects_shared_visual_weight():
    model = TinyTranslator()
    model.encoder.weight = model.decoder[0].weight
    with pytest.raises(ValueError, match="shared"):
        train_full_decoder(model)
