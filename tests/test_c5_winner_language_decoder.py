from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_winner_language_decoder import WinnerLanguageDecoder


@pytest.mark.parametrize("tied", [False, True])
def test_decoder_only_forward_gradient_and_generation_match_full_t5_given_same_memory(tied):
    config = T5Config(
        vocab_size=32,
        d_model=32,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=1,
        num_heads=4,
        d_kv=8,
        dropout_rate=0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        tie_word_embeddings=tied,
    )
    original = T5ForConditionalGeneration(config).eval()
    decoder = WinnerLanguageDecoder(config).eval()
    decoder.shared.load_state_dict(original.shared.state_dict())
    decoder.decoder.load_state_dict(original.decoder.state_dict())
    decoder.lm_head.load_state_dict(original.lm_head.state_dict())
    assert not list(decoder.encoder.parameters())
    assert decoder.shared.weight is decoder.decoder.embed_tokens.weight
    memory = torch.randn(1, 3, 32, requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    labels = torch.tensor([[4, 7, 1]])
    before = original(
        encoder_outputs=BaseModelOutput(last_hidden_state=memory),
        attention_mask=mask,
        labels=labels,
        use_cache=False,
    )
    after = decoder(
        encoder_outputs=BaseModelOutput(last_hidden_state=memory),
        attention_mask=mask,
        labels=labels,
        use_cache=False,
    )
    assert torch.equal(before.logits, after.logits)
    assert torch.equal(before.loss, after.loss)
    first = torch.autograd.grad(before.loss, memory)[0]
    second = torch.autograd.grad(after.loss, memory)[0]
    assert torch.equal(first, second)
    for beams in (1, 2):
        options = dict(
            attention_mask=mask,
            num_beams=beams,
            max_new_tokens=6,
            do_sample=False,
        )
        # Generation expands the supplied ModelOutput in place for beam search.
        first_tokens = original.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory.detach().clone()), **options
        )
        second_tokens = decoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory.detach().clone()), **options
        )
        assert torch.equal(first_tokens, second_tokens)


def test_frozen_language_stays_in_eval_but_passes_gradients_to_visual_states():
    config = T5Config(
        vocab_size=32,
        d_model=32,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        d_kv=8,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    decoder = WinnerLanguageDecoder(config)
    decoder.requires_grad_(False)
    decoder.frozen_winner_weights = True
    decoder.train()
    assert not decoder.training and all(not module.training for module in decoder.modules())
    memory = torch.randn(1, 2, 32, requires_grad=True)
    result = decoder(
        encoder_outputs=BaseModelOutput(last_hidden_state=memory),
        attention_mask=torch.ones(1, 2, dtype=torch.bool),
        labels=torch.tensor([[4, 1]]),
        use_cache=False,
    )
    result.loss.backward()
    assert memory.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in decoder.parameters())
    with pytest.raises(RuntimeError, match="discarded text encoder"):
        decoder(input_ids=torch.tensor([[4, 1]]), labels=torch.tensor([[4, 1]]))
