"""Learn timing of frozen native streams through unchanged C5 sequence/language.

This supplies an interface, not data admission, an optimizer recipe or a quality
claim. Actual C5 identity and authentic semantic-risk gradients remain required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from c5_monotone_feature_timing import WIDTHS, MonotoneFeatureTiming
from tiny_rgb_feature_winner_student import RawFeatureWinnerStudent
from torch import nn
from transformers.modeling_outputs import BaseModelOutput


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class FrozenFeatureClip:
    streams: dict[str, torch.Tensor]
    valid_intervals: torch.Tensor

    def to(self, device):
        require(torch.device(device) == torch.device("cpu"), "timing feature clips stay on CPU")
        return self


def clip_input(values):
    require(
        set(values["streams"]) == set(WIDTHS), "timing clip requires every frozen native stream"
    )
    streams = {}
    for name in WIDTHS:
        value = values["streams"][name]
        require(
            isinstance(value, np.ndarray) and value.dtype == np.float32,
            "timing clip cannot silently cast or replace feature values",
        )
        streams[name] = torch.from_numpy(value.copy())
    intervals = values["valid_intervals"]
    require(
        isinstance(intervals, np.ndarray) and intervals.dtype == np.bool_,
        "timing clip requires explicit boolean interval visibility",
    )
    return FrozenFeatureClip(streams, torch.from_numpy(intervals.copy()))


class TimingSemanticStudent(RawFeatureWinnerStudent):
    """Use the existing exact C5 memory and generation paths, with no visual model."""

    def __init__(self, semantic_encoder, language, *, timing=None):
        # Avoid the parent's randomly initialized RGB/feature-head constructor.
        nn.Module.__init__(self)
        self.semantic_encoder = semantic_encoder.requires_grad_(False).eval()
        self.language = language.requires_grad_(False).eval()
        if hasattr(language, "frozen_winner_weights"):
            language.frozen_winner_weights = True
        self.timing = timing if timing is not None else MonotoneFeatureTiming()
        require(isinstance(self.timing, MonotoneFeatureTiming), "unknown timing feature adapter")
        self.require_fixed()

    def require_fixed(self):
        require(
            all(
                not p.requires_grad and p.device.type == "cpu" and p.dtype == torch.float32
                for module in (self.semantic_encoder, self.language)
                for p in module.parameters()
            ),
            "timing sequence/language values must stay frozen CPU FP32",
        )
        require(
            not self.semantic_encoder.training and not self.language.training,
            "timing C5 backbones must retain evaluation mode",
        )

    def timed_features(self, clip, frame_mask):
        self.require_fixed()
        require(isinstance(clip, FrozenFeatureClip), "timing model requires a frozen feature clip")
        frames = len(clip.streams["pose_features"])
        require(
            isinstance(frame_mask, torch.Tensor)
            and frame_mask.device.type == "cpu"
            and frame_mask.dtype == torch.bool
            and frame_mask.shape == (1, frames)
            and frame_mask.all(),
            "timing model cannot mask or drop original input frames",
        )
        timing = self.timing(clip.streams, valid_intervals=clip.valid_intervals)
        features = {name: value[None] for name, value in timing["streams"].items()}
        return features, frame_mask, timing

    def predict_features(self, clip, frame_mask):
        features, mask, _ = self.timed_features(clip, frame_mask)
        return features, mask

    def forward(self, clip, frame_mask, labels):
        features, mask, timing = self.timed_features(clip, frame_mask)
        memory = self.semantic_memory(features, mask)
        output = self.language(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask.long(),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return {
            "language": output,
            "memory": memory,
            "frame_mask": mask,
            "features": features,
            "timing": timing,
        }


def require_model(model, model_plan):
    from probe_roi_semantic_learning import fixed_hashes

    selected = model.trainable_parameters()
    require(
        set(selected) == {"timing.rate_weights"}
        and selected["timing.rate_weights"].shape == (4,)
        and sum(p.numel() for p in selected.values()) == 4,
        "timing model changed its four-parameter training scope",
    )
    model.require_fixed()
    require(
        fixed_hashes(model) == model_plan["initial_reference_hashes"],
        "timing model loaded different C5 sequence/language values",
    )


def load_student(model_plan):
    """Caller must admit complete data and hold the existing exclusive model lease."""
    from evaluate_c5_words import models

    encoder, language, tokenizer = models(
        {
            "reference": model_plan["reference"],
            "initial_reference_hashes": model_plan["initial_reference_hashes"],
        }
    )
    model = TimingSemanticStudent(encoder, language).eval()
    require_model(model, model_plan)
    return model, tokenizer


def gradient_report(model, *, require_signal=True):
    model.require_fixed()
    selected = model.trainable_parameters()
    require(set(selected) == {"timing.rate_weights"}, "timing optimizer scope changed")
    require(
        all(
            p.grad is None
            for module in (model.semantic_encoder, model.language)
            for p in module.parameters()
        ),
        "frozen C5 backbone received gradients",
    )
    value = selected["timing.rate_weights"].grad
    require(
        value is not None
        and value.shape == (4,)
        and value.dtype == torch.float32
        and value.device.type == "cpu"
        and torch.isfinite(value).all(),
        "timing model lacks a complete finite CPU gradient vector",
    )
    require(not require_signal or value.norm() > 0, "timing case has no learning signal")
    return {
        "parameters": 4,
        "gradient_values": value.detach().tolist(),
        "gradient_norm": float(value.norm()),
        "frozen_backbone_gradients_absent": True,
    }
