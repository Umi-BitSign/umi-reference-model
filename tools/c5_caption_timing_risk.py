"""Caption supervision plus token-normalized graded risk through frozen C5.

The complete candidate tokens determine normalization. One differentiable
encoder call receives the exact sum of frozen-decoder memory derivatives.
This module admits no data, runs no grader and performs no optimizer update.
"""

from __future__ import annotations

import math

import torch
from c5_timing_semantic_model import FrozenFeatureClip
from terra_quality_training import minimum_risk, sequence_log_probability
from transformers.modeling_outputs import BaseModelOutput


def require(value, message):
    if not value:
        raise ValueError(message)


def backward_caption_timing_risk(
    model, clip, mask, labels, quality, *, alpha=1.0, caption_weight=0.1, divisor=1
):
    require(
        isinstance(clip, FrozenFeatureClip)
        and type(divisor) is int
        and divisor >= 1
        and len(labels) == len(quality)
        and bool(quality)
        and all(type(q) is int and 0 <= q <= 10 for q in quality)
        and quality[0] == 10,
        "caption timing requires complete integer grades and the verified exact anchor",
    )
    require(
        type(alpha) in (int, float)
        and math.isfinite(alpha)
        and alpha > 0
        and type(caption_weight) in (int, float)
        and math.isfinite(caption_weight)
        and caption_weight >= 0,
        "caption timing requires a finite positive risk scale and nonnegative caption weight",
    )
    counts = []
    for value in labels:
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.long
            and value.ndim == 2
            and value.shape[0] == 1
            and value.shape[1] > 0
            and (value != -100).any(),
            "caption timing requires complete nonempty CPU candidate labels",
        )
        counts.append(int((value != -100).sum()))
    model.eval()
    memory, actual_mask = model.encode(clip, mask)
    require(
        memory.device.type == "cpu"
        and memory.dtype == torch.float32
        and torch.isfinite(memory).all(),
        "caption timing memory must retain finite original CPU FP32 values",
    )

    def language(value, selected_memory):
        return model.language(
            encoder_outputs=BaseModelOutput(last_hidden_state=selected_memory),
            attention_mask=actual_mask.long(),
            labels=value,
            use_cache=False,
            return_dict=True,
        )

    with torch.no_grad():
        logs = torch.stack(
            [
                sequence_log_probability(language(value, memory.detach()).logits, value)
                for value in labels
            ]
        )
        normalized = logs / logs.new_tensor(counts)
        risk = minimum_risk(normalized, quality, alpha=alpha)
        coefficients = risk["coefficients"].detach()
    memory_gradient = torch.zeros_like(memory)
    caption_loss = None
    for index, value in enumerate(labels):
        leaf = memory.detach().requires_grad_(True)
        output = language(value, leaf)
        objective = (
            coefficients[index] * sequence_log_probability(output.logits, value) / counts[index]
        )
        if index == 0:
            caption_loss = float(output.loss.detach())
            objective = objective + caption_weight * output.loss
        (gradient,) = torch.autograd.grad(objective / divisor, leaf)
        require(
            torch.isfinite(gradient).all(), "caption timing received a nonfinite memory gradient"
        )
        memory_gradient.add_(gradient)
        del output, objective, leaf, gradient
    if memory.requires_grad:
        memory.backward(memory_gradient)
    else:
        (model.timing.rate_weights.sum() * 0).backward()
    return {
        "expected_terra_error": float(risk["loss"]),
        "caption_loss": caption_loss,
        "loss": float(risk["loss"]) + caption_weight * caption_loss,
        "caption_weight": caption_weight,
        "risk_alpha": alpha,
        "sequence_score_normalization": "mean_valid_token_log_probability",
        "candidate_token_counts": counts,
        "candidate_sequence_log_probabilities": logs.tolist(),
        "candidate_normalized_log_probabilities": normalized.tolist(),
        "candidate_probabilities": risk["probabilities"].tolist(),
        "terra_gradient_coefficient_l1": float(coefficients.abs().sum()),
        "memory_gradient_norm": float(memory_gradient.norm()),
        "timing_memory_has_gradient_path": memory.requires_grad,
        "differentiable_encoder_calls": 1,
    }
