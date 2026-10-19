"""Frozen public C5 with a trainable local adapter on its complete semantic memory.

Original native features, semantic encoder, decoder and generation options stay
explicit. This module supplies an interface, not data or training admission.
"""

from __future__ import annotations

import torch
from c5_monotone_feature_timing import validate
from c5_temporal_memory_adapter import WIDTH, TemporalMemoryAdapter
from c5_timing_semantic_model import FrozenFeatureClip
from tiny_rgb_feature_winner_student import RawFeatureWinnerStudent
from torch import nn
from transformers.modeling_outputs import BaseModelOutput


def require(value, message):
    if not value:
        raise ValueError(message)


class TemporalMemoryStudent(RawFeatureWinnerStudent):
    def __init__(self, semantic_encoder, language, *, adapter=None):
        nn.Module.__init__(self)
        self.semantic_encoder = semantic_encoder.requires_grad_(False).eval()
        self.language = language.requires_grad_(False).eval()
        if hasattr(language, "frozen_winner_weights"):
            language.frozen_winner_weights = True
        self.memory_adapter = adapter if adapter is not None else TemporalMemoryAdapter()
        require(
            isinstance(self.memory_adapter, TemporalMemoryAdapter),
            "unknown temporal memory adapter",
        )
        self.require_fixed()

    def train(self, mode=True):
        nn.Module.train(self, False)
        self.training = mode
        self.memory_adapter.train(mode)
        return self

    def trainable_parameters(self):
        return {
            "memory_adapter." + name: parameter
            for name, parameter in sorted(self.memory_adapter.named_parameters())
        }

    def require_fixed(self):
        require(
            all(
                not p.requires_grad and p.device.type == "cpu" and p.dtype == torch.float32
                for module in (self.semantic_encoder, self.language)
                for p in module.parameters()
            )
            and not self.semantic_encoder.training
            and not self.language.training,
            "memory student original C5 encoder and language must remain frozen CPU FP32 eval",
        )
        rank = self.memory_adapter.rank
        selected = self.trainable_parameters()
        expected = {
            "memory_adapter.input_projection.weight": (rank, WIDTH),
            "memory_adapter.output_projection.weight": (WIDTH, rank),
            "memory_adapter.temporal_weights": (rank, 3),
        }
        require(
            set(selected) == set(expected)
            and {name for name, p in self.named_parameters() if p.requires_grad} == set(expected)
            and all(
                p.requires_grad
                and p.device.type == "cpu"
                and p.dtype == torch.float32
                and tuple(p.shape) == expected[name]
                and torch.isfinite(p).all()
                for name, p in selected.items()
            ),
            "memory student changed its declared trainable adapter scope",
        )

    def encode_details(self, clip, frame_mask):
        self.require_fixed()
        require(
            isinstance(clip, FrozenFeatureClip),
            "memory student requires original frozen feature clips",
        )
        frames = validate(clip.streams, clip.valid_intervals)
        require(
            not any(value.requires_grad for value in clip.streams.values()),
            "memory student cannot learn upstream input features",
        )
        require(
            isinstance(frame_mask, torch.Tensor)
            and frame_mask.device.type == "cpu"
            and frame_mask.dtype == torch.bool
            and frame_mask.shape == (1, frames)
            and frame_mask.all(),
            "memory student cannot mask or drop original time steps",
        )
        features = {name: value[None] for name, value in clip.streams.items()}
        with torch.no_grad():
            base = self.semantic_memory(features, frame_mask)
        adapted = self.memory_adapter(base, valid_intervals=clip.valid_intervals)
        return {**adapted, "base_memory": base, "frame_mask": frame_mask, "features": features}

    def encode(self, clip, frame_mask):
        details = self.encode_details(clip, frame_mask)
        return details["memory"], details["frame_mask"]

    def forward(self, clip, frame_mask, labels):
        details = self.encode_details(clip, frame_mask)
        output = self.language(
            encoder_outputs=BaseModelOutput(last_hidden_state=details["memory"]),
            attention_mask=details["frame_mask"].long(),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return {**details, "language": output}


def require_model(model, model_plan, configuration):
    from probe_roi_semantic_learning import fixed_hashes

    model.require_fixed()
    require(
        model.memory_adapter.configuration() == configuration,
        "memory student changed its exact adapter configuration",
    )
    require(
        fixed_hashes(model) == model_plan["initial_reference_hashes"],
        "memory student loaded different original C5 values",
    )


def load_student(model_plan, configuration):
    require(
        isinstance(configuration, dict)
        and all(k in configuration for k in ("rank", "maximum_relative_residual", "seed")),
        "memory student requires a complete configuration",
    )
    adapter = TemporalMemoryAdapter(
        rank=configuration["rank"],
        maximum_relative_residual=configuration["maximum_relative_residual"],
        seed=configuration["seed"],
    )
    require(adapter.configuration() == configuration, "memory student configuration changed")
    from evaluate_c5_words import models

    encoder, language, tokenizer = models(
        {
            "reference": model_plan["reference"],
            "initial_reference_hashes": model_plan["initial_reference_hashes"],
        }
    )
    model = TemporalMemoryStudent(encoder, language, adapter=adapter).eval()
    require_model(model, model_plan, configuration)
    return model, tokenizer
