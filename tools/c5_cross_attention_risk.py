"""Accumulate exact graded-risk, caption and frozen-original retention gradients.

Ungraded supplemental captions contribute only the declared auxiliary losses.
No dataset admission, optimizer, label inference, model loader or API call.
"""

from __future__ import annotations

import math

import torch
from c5_cross_attention_alignment import CrossAttentionAlignment
from c5_retention import teacher_kl
from terra_quality_training import minimum_risk, sequence_log_probability


def backward_alignment_risk(
    model,
    memory,
    mask,
    labels,
    quality,
    *,
    alpha=1.0,
    risk_weight=1.0,
    caption_weight=0.1,
    retention_weight=0.01,
    retention_temperature=1.0,
    divisor=1,
):
    if not isinstance(model, CrossAttentionAlignment) or not labels:
        raise ValueError("alignment objective requires its declared model and human caption")
    if (
        any(
            type(w) not in (int, float) or not math.isfinite(w) or w < 0
            for w in (alpha, risk_weight, caption_weight, retention_weight, retention_temperature)
        )
        or alpha == 0
        or retention_temperature == 0
        or type(divisor) is not int
        or divisor < 1
    ):
        raise ValueError("alignment objective has invalid scales or divisor")
    if quality is None:
        if len(labels) != 1:
            raise ValueError("ungraded supplemental input may supply only its human caption")
    elif (
        len(quality) != len(labels)
        or not quality
        or quality[0] != 10
        or any(type(q) is not int or not 0 <= q <= 10 for q in quality)
    ):
        raise ValueError("alignment graded input requires complete real integer rewards and anchor")
    if not all(module.enabled for module in model.projections().values()):
        raise ValueError("alignment objective cannot fit with adapters disabled")
    counts = [int((value != -100).sum()) for value in labels]
    if any(count < 1 for count in counts):
        raise ValueError("alignment cannot fit an empty target")
    teacher = None
    with torch.no_grad():
        if quality is not None:
            logs = torch.stack(
                [
                    sequence_log_probability(model(memory, mask, value).logits, value)
                    for value in labels
                ]
            )
            normalized = logs / logs.new_tensor(counts)
            risk = minimum_risk(normalized, quality, alpha=alpha)
            coefficients = risk["coefficients"].detach()
        else:
            logs, normalized, risk = None, None, None
            coefficients = memory.new_zeros(1)
        if retention_weight:
            with model.original_decoder():
                teacher = model(memory, mask, labels[0]).logits.detach()
    caption, retention = None, 0.0
    for index, value in enumerate(labels):
        output = model(memory, mask, value)
        objective = (
            risk_weight
            * coefficients[index]
            * sequence_log_probability(output.logits, value)
            / counts[index]
        )
        if index == 0:
            caption = float(output.loss.detach())
            objective = objective + caption_weight * output.loss
            if teacher is not None:
                kl = teacher_kl(output.logits, teacher, value, retention_temperature)
                retention = float(kl.detach())
                objective = objective + retention_weight * kl
        (objective / divisor).backward()
        del output, objective
    model.require_scope()
    expected_error = float(risk["loss"]) if risk is not None else None
    return {
        "loss": (risk_weight * expected_error if expected_error is not None else 0)
        + caption_weight * caption
        + retention_weight * retention,
        "expected_terra_error": expected_error,
        "real_graded_candidates_used": quality is not None,
        "caption_loss": caption,
        "original_decoder_KL": retention,
        "risk_alpha": alpha,
        "risk_weight": risk_weight,
        "caption_weight": caption_weight,
        "retention_weight": retention_weight,
        "retention_temperature": retention_temperature,
        "sequence_score_normalization": "mean_valid_token_log_probability",
        "candidate_token_counts": counts,
        "candidate_sequence_log_probabilities": logs.tolist() if logs is not None else None,
        "candidate_normalized_log_probabilities": normalized.tolist()
        if normalized is not None
        else None,
        "candidate_probabilities": risk["probabilities"].tolist() if risk is not None else None,
        "terra_gradient_coefficient_l1": float(coefficients.abs().sum()),
        "original_encoder_calls": 0,
    }
