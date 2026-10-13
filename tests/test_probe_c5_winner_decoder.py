from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from probe_c5_winner_decoder import decoder_gradient_guard, trainable_decoder, trainable_full


def model():
    result = torch.nn.Module()
    for name in ("encoder", "decoder", "decoder_emb", "lm_head"):
        setattr(result, name, torch.nn.Linear(4, 4))
    return result


def test_full_decoder_is_trainable_without_changing_any_winning_weights():
    candidate = model()
    before = {name: value.clone() for name, value in candidate.state_dict().items()}
    selected = trainable_decoder(candidate)
    assert len(selected) == 6
    assert not any(parameter.requires_grad for parameter in candidate.encoder.parameters())
    assert all(torch.equal(value, before[name]) for name, value in candidate.state_dict().items())
    for parameter in selected.values():
        parameter.grad = torch.ones_like(parameter)
    assert decoder_gradient_guard(candidate, selected, torch)["nonzero_gradient_tensors"] == 6


def test_full_translator_scope_includes_encoder_and_keeps_values_unchanged():
    candidate = model()
    before = {name: value.clone() for name, value in candidate.state_dict().items()}
    selected = trainable_full(candidate)
    assert len(selected) == 8 and any(name.startswith("encoder.") for name in selected)
    assert all(torch.equal(value, before[name]) for name, value in candidate.state_dict().items())


@pytest.mark.parametrize("bad", ("missing", "nan", "frozen"))
def test_decoder_preflight_rejects_bad_gradients_or_encoder_updates(bad):
    candidate = model()
    selected = trainable_decoder(candidate)
    for parameter in selected.values():
        parameter.grad = torch.ones_like(parameter)
    if bad == "missing":
        next(iter(selected.values())).grad = None
    elif bad == "nan":
        next(iter(selected.values())).grad.fill_(float("nan"))
    else:
        candidate.encoder.weight.grad = torch.ones_like(candidate.encoder.weight)
    with pytest.raises(RuntimeError, match="winner"):
        decoder_gradient_guard(candidate, selected, torch)
