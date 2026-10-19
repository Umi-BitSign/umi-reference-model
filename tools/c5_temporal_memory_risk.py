"""Caption and graded-risk derivatives for the local temporal memory adapter.

The original encoder runs once outside autograd. Decoder derivatives accumulate
on detached memory leaves, then enter the adapter graph with its penalty once.
No optimizer, grader, dataset admission or checkpoint loading occurs here.
"""

from __future__ import annotations

import math

import torch
from c5_temporal_memory_model import TemporalMemoryStudent
from c5_timing_semantic_model import FrozenFeatureClip
from terra_quality_training import minimum_risk, sequence_log_probability
from transformers.modeling_outputs import BaseModelOutput


def require(value, message):
    if not value:
        raise ValueError(message)


def gradient_report(model, *, require_signal=True):
    model.require_fixed()
    require(
        all(
            p.grad is None
            for module in (model.semantic_encoder, model.language)
            for p in module.parameters()
        ),
        "temporal memory frozen backbone received gradients",
    )
    selected = model.trainable_parameters()
    values = {}
    for name, parameter in selected.items():
        gradient = parameter.grad
        require(
            gradient is not None
            and gradient.shape == parameter.shape
            and gradient.dtype == torch.float32
            and gradient.device.type == "cpu"
            and torch.isfinite(gradient).all(),
            "temporal memory lacks a complete finite adapter gradient",
        )
        values[name] = gradient.detach().tolist()
    norm = float(torch.cat([p.grad.flatten() for p in selected.values()]).norm())
    require(not require_signal or norm > 0, "temporal memory lacks a learning signal")
    return {
        "parameters": sum(p.numel() for p in selected.values()),
        "gradient_values": values,
        "gradient_norm": norm,
        "frozen_backbone_gradients_absent": True,
    }


def backward_memory_risk(
    model,
    clip,
    mask,
    labels,
    quality,
    *,
    alpha=1.0,
    caption_weight=0.1,
    residual_weight=0.01,
    divisor=1,
):
    require(
        isinstance(model, TemporalMemoryStudent)
        and isinstance(clip, FrozenFeatureClip)
        and type(divisor) is int
        and divisor >= 1
        and len(labels) == len(quality)
        and bool(quality)
        and all(type(q) is int and 0 <= q <= 10 for q in quality)
        and quality[0] == 10,
        "temporal memory requires complete integer grades and the exact human anchor",
    )
    require(
        all(
            type(value) in (int, float) and math.isfinite(value) and value >= 0
            for value in (alpha, caption_weight, residual_weight)
        )
        and alpha > 0,
        "temporal memory requires a positive risk scale and finite nonnegative loss weights",
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
            "temporal memory requires complete nonempty CPU candidate labels",
        )
        counts.append(int((value != -100).sum()))
    model.eval()
    details = model.encode_details(clip, mask)
    memory = details["memory"]
    penalty = details["normalized_residual_penalty"]
    require(
        memory.requires_grad and penalty.requires_grad,
        "temporal memory training requires the complete adapter graph",
    )

    def language(value, selected_memory):
        return model.language(
            encoder_outputs=BaseModelOutput(last_hidden_state=selected_memory),
            attention_mask=details["frame_mask"].long(),
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
        require(torch.isfinite(gradient).all(), "temporal memory decoder gradient is nonfinite")
        memory_gradient.add_(gradient)
        del leaf, output, objective, gradient
    torch.autograd.backward(
        (memory, penalty),
        (memory_gradient, penalty.new_tensor(residual_weight / divisor)),
    )
    return {
        "loss": float(risk["loss"])
        + caption_weight * caption_loss
        + residual_weight * float(penalty.detach()),
        "expected_terra_error": float(risk["loss"]),
        "caption_loss": caption_loss,
        "normalized_residual_penalty": float(penalty.detach()),
        "risk_alpha": alpha,
        "caption_weight": caption_weight,
        "residual_weight": residual_weight,
        "sequence_score_normalization": "mean_valid_token_log_probability",
        "candidate_token_counts": counts,
        "candidate_sequence_log_probabilities": logs.tolist(),
        "candidate_normalized_log_probabilities": normalized.tolist(),
        "candidate_probabilities": risk["probabilities"].tolist(),
        "terra_gradient_coefficient_l1": float(coefficients.abs().sum()),
        "memory_gradient_norm": float(memory_gradient.norm()),
        "original_encoder_calls": 1,
        "differentiable_encoder_calls": 0,
        "adapter_backward_calls": 1,
    }
