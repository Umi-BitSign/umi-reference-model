"""Exact graded candidate-risk gradient with one differentiable encoder call.

The frozen decoder is differentiated separately for each candidate against a
detached memory leaf. Its summed memory gradient is then propagated once through
the original encoder and timing adapter. This changes neither candidate
probabilities nor the minimum-risk objective. No judge or optimizer runs here.
"""

from __future__ import annotations

import torch
from c5_timing_semantic_model import FrozenFeatureClip
from terra_quality_training import minimum_risk, sequence_log_probability
from transformers.modeling_outputs import BaseModelOutput


def require(value, message):
    if not value:
        raise ValueError(message)


def backward_timing_risk(model, clip, mask, labels, quality, *, alpha=0.005, divisor=1):
    require(
        isinstance(clip, FrozenFeatureClip)
        and type(divisor) is int
        and divisor >= 1
        and len(labels) == len(quality)
        and bool(quality)
        and quality[0] == 10,
        "timing risk requires actual complete grades and a verified exact anchor",
    )
    model.eval()
    memory, actual_mask = model.encode(clip, mask)
    require(
        memory.device.type == "cpu"
        and memory.dtype == torch.float32
        and torch.isfinite(memory).all(),
        "timing risk memory must retain finite original CPU FP32 values",
    )
    for value in labels:
        require(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.long
            and value.ndim == 2
            and value.shape[0] == 1,
            "timing risk requires complete CPU candidate labels",
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
        risk = minimum_risk(logs, quality, alpha=alpha)
        coefficients = risk["coefficients"].detach()
    memory_gradient = torch.zeros_like(memory)
    caption_loss = None
    for index, value in enumerate(labels):
        leaf = memory.detach().requires_grad_(True)
        output = language(value, leaf)
        if index == 0:
            caption_loss = float(output.loss.detach())
        objective = coefficients[index] * sequence_log_probability(output.logits, value) / divisor
        (gradient,) = torch.autograd.grad(objective, leaf)
        require(torch.isfinite(gradient).all(), "timing risk received a nonfinite memory gradient")
        memory_gradient.add_(gradient)
        del output, objective, leaf, gradient
    if memory.requires_grad:
        memory.backward(memory_gradient)
    else:
        # Complete no-support clips remain valid exposures with a measured zero
        # timing derivative. Frozen language/encoder still run on every frame.
        (model.timing.rate_weights.sum() * 0).backward()
    return {
        "expected_terra_error": float(risk["loss"]),
        "caption_loss": caption_loss,
        "loss": float(risk["loss"]),
        "terra_gradient_coefficient_l1": float(coefficients.abs().sum()),
        "candidate_probabilities": risk["probabilities"].tolist(),
        "candidate_sequence_log_probabilities": logs.tolist(),
        "memory_gradient_norm": float(memory_gradient.norm()),
        "timing_memory_has_gradient_path": memory.requires_grad,
        "differentiable_encoder_calls": 1,
        "caption_weight": 0,
    }
