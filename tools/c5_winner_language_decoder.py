"""Retain the winner's exact language weights without constructing its visual encoder."""

from __future__ import annotations

import copy
from typing import ClassVar

import torch
from fresh_lowres_model import DiscardedTextEncoder
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Stack


class WinnerLanguageDecoder(T5ForConditionalGeneration):
    """Standard T5 decode from supplied states, with no text/SHuBERT encoder allocation."""

    _tied_weights_keys: ClassVar[list[str]] = ["decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config):
        T5PreTrainedModel.__init__(self, config)
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = DiscardedTextEncoder()
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.frozen_winner_weights = False
        self.post_init()
        self.model_parallel = False
        self.device_map = None

    def _tie_weights(self):
        # The encoder is absent. Decoder input embeddings still alias shared exactly.
        self.decoder.embed_tokens = self.shared

    def train(self, mode=True):
        return super().train(False if self.frozen_winner_weights else mode)


def load_winner_language():
    """Only decoder tensors are read from the verified safetensors reference.

    Freezing parameters preserves language weights while gradients can still reach
    new visual states. This alone does not establish CPU/Metal parity or ASL quality.
    """
    import json

    from c5_winner_reference import ROOT, load_reference
    from safetensors import safe_open

    reference = load_reference()
    root = ROOT / reference["bundle_root"] / "model/models/checkpoint-11625"
    config = T5Config.from_dict(json.loads(root.joinpath("config.json").read_text()))
    if (
        config.d_model,
        config.num_decoder_layers,
        config.vocab_size,
        config.tie_word_embeddings,
    ) != (1536, 6, 384, False):
        raise ValueError("winner language topology differs from the reviewed reference")
    language = WinnerLanguageDecoder(config)
    with safe_open(root / "model.safetensors", framework="pt", device="cpu") as weights:
        embedding = weights.get_tensor("decoder_emb.weight")
        if not torch.equal(embedding, weights.get_tensor("decoder.embed_tokens.weight")):
            raise ValueError("winner decoder input embedding aliases differ")
        decoder = {
            name: weights.get_tensor(f"decoder.{name}") for name in language.decoder.state_dict()
        }
        language.decoder.load_state_dict(decoder, strict=True)
        with torch.no_grad():
            language.shared.weight.copy_(embedding)
            language.lm_head.weight.copy_(weights.get_tensor("lm_head.weight"))
    if sum(parameter.numel() for parameter in language.parameters()) != 167539584:
        raise ValueError("winner language extraction parameter count changed")
    language.requires_grad_(False)
    language.frozen_winner_weights = True
    return language.eval(), reference
