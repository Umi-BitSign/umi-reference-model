"""Training-scope helpers for the fresh low-resolution model."""

from __future__ import annotations

import random
from itertools import combinations
from math import ceil

import torch
import torch.nn.functional as F
from fresh_lowres_model import FreshLowResolutionTranslator
from torch import nn


def ordered_examples(rows: list[dict], seed: int) -> list[dict]:
    if not rows:
        raise ValueError("training inventory is empty")
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    return ordered


def repeat_source_examples(
    rows: list[dict], source_repeats: dict[str, int], seed: int
) -> list[dict]:
    """Deterministically interleave extra exposures for selected sources."""
    if not rows or any(repeat < 1 for repeat in source_repeats.values()):
        raise ValueError("source repeats need rows and positive multipliers")
    expanded = [
        row
        for row in rows
        for _ in range(source_repeats.get(row["source"], 1))
    ]
    if len(expanded) == len(rows):
        return list(rows)
    random.Random(seed).shuffle(expanded)
    return expanded


def training_schedule(
    rows: list[dict], epochs: int
) -> list[tuple[int, int, dict]]:
    """Repeat a frozen row order and identify each epoch and local example."""
    if not rows or epochs < 1:
        raise ValueError("training schedule needs rows and at least one epoch")
    return [
        (epoch, local_example, row)
        for epoch in range(1, epochs + 1)
        for local_example, row in enumerate(rows, start=1)
    ]


def accumulation_group(
    example_update: int, total_examples: int, accumulation_steps: int
) -> tuple[int, int, bool]:
    """Return optimizer step, group size, and boundary for one example update."""
    if (
        example_update < 1
        or total_examples < example_update
        or accumulation_steps < 1
    ):
        raise ValueError("invalid gradient-accumulation position")
    optimizer_step = ceil(example_update / accumulation_steps)
    group_start = (optimizer_step - 1) * accumulation_steps + 1
    group_size = min(accumulation_steps, total_examples - group_start + 1)
    boundary = example_update == group_start + group_size - 1
    return optimizer_step, group_size, boundary


def allowed_cumulative_state_keys(
    stage_scopes: dict[str, set[str] | frozenset[str]],
) -> set[frozenset[str]]:
    """Return exact checkpoint key sets allowed across clean training stages."""
    scopes = [frozenset(scope) for scope in stage_scopes.values() if scope]
    if not scopes:
        raise ValueError("training stages have no parameter scopes")
    allowed: set[frozenset[str]] = set()
    for count in range(1, len(scopes) + 1):
        for selected in combinations(scopes, count):
            allowed.add(frozenset().union(*selected))
    return allowed


def ctc_alignment_loss(
    memory: torch.Tensor,
    memory_mask: torch.Tensor,
    labels: torch.Tensor,
    token_embeddings: torch.Tensor,
    *,
    pad_token_id: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Align temporal video states to subword targets with a frozen token basis."""
    if (
        memory.ndim != 3
        or memory_mask.shape != memory.shape[:2]
        or labels.ndim != 2
        or labels.shape[0] != memory.shape[0]
        or token_embeddings.ndim != 2
        or token_embeddings.shape[1] != memory.shape[2]
    ):
        raise ValueError("CTC alignment tensors have incompatible shapes")
    targets = []
    target_lengths = []
    for row in labels:
        target = row[
            (row != pad_token_id) & (row != eos_token_id) & (row != -100)
        ]
        if target.numel() < 1:
            raise ValueError("CTC alignment target is empty")
        targets.append(target)
        target_lengths.append(target.numel())
    input_lengths = memory_mask.sum(dim=1, dtype=torch.long)
    for target, input_length in zip(targets, input_lengths.tolist(), strict=True):
        repeated = int((target[1:] == target[:-1]).sum().item())
        if target.numel() + repeated > input_length:
            raise ValueError("CTC alignment target exceeds its temporal sequence")
    logits = ctc_token_logits(memory, token_embeddings)
    log_probabilities = logits.log_softmax(dim=-1).transpose(0, 1).to("cpu")
    return F.ctc_loss(
        log_probabilities,
        torch.cat(targets).to("cpu"),
        input_lengths.to("cpu"),
        torch.tensor(target_lengths, dtype=torch.long),
        blank=pad_token_id,
        reduction="mean",
        zero_infinity=False,
    )


def ctc_alignment_is_feasible(
    memory_mask: torch.Tensor,
    labels: torch.Tensor,
    *,
    pad_token_id: int,
    eos_token_id: int,
) -> bool:
    """Return whether every target can be aligned to its temporal sequence."""
    if (
        memory_mask.ndim != 2
        or labels.ndim != 2
        or memory_mask.shape[0] != labels.shape[0]
    ):
        raise ValueError("CTC feasibility tensors have incompatible shapes")
    input_lengths = memory_mask.sum(dim=1, dtype=torch.long).tolist()
    for row, input_length in zip(labels, input_lengths, strict=True):
        target = row[
            (row != pad_token_id) & (row != eos_token_id) & (row != -100)
        ]
        if target.numel() < 1:
            return False
        repeated = int((target[1:] == target[:-1]).sum().item())
        if target.numel() + repeated > input_length:
            return False
    return True


def ctc_token_logits(memory: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
    """Score video states against a normalized, tied language-token basis."""
    if (
        memory.ndim != 3
        or token_embeddings.ndim != 2
        or token_embeddings.shape[1] != memory.shape[2]
    ):
        raise ValueError("CTC token tensors have incompatible shapes")
    return F.linear(
        F.normalize(memory, dim=-1),
        F.normalize(token_embeddings, dim=-1),
    ) / 0.1


def semantic_alignment_loss(
    memory: torch.Tensor,
    memory_mask: torch.Tensor,
    text_targets: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Match one pooled video to its own text target ahead of fixed negatives."""
    if (
        memory.ndim != 3
        or memory.shape[0] != 1
        or memory_mask.shape != memory.shape[:2]
        or text_targets.ndim != 2
        or text_targets.shape[0] < 2
        or text_targets.shape[1] != memory.shape[2]
        or temperature <= 0
    ):
        raise ValueError("semantic alignment tensors or temperature are invalid")
    weights = memory_mask.to(memory.dtype).unsqueeze(-1)
    pooled_video = (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    logits = F.linear(
        F.normalize(pooled_video, dim=-1),
        F.normalize(text_targets.to(memory.device), dim=-1),
    ) / temperature
    return F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=memory.device))


def configure_alignment_stage(model: FreshLowResolutionTranslator) -> dict[str, nn.Parameter]:
    """Train new visual-temporal layers and decoder cross-attention only."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.frame_projection,
        model.temporal_encoder,
        model.language_projection,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.temporal_positions.requires_grad_(True)
    for block in model.language_model.decoder.block:
        attention = block.layer[1].EncDecAttention
        for projection in (attention.q, attention.k, attention.v, attention.o):
            for parameter in projection.parameters():
                parameter.requires_grad_(True)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if not trainable or any(name.startswith("frame_encoder.") for name in trainable):
        raise ValueError("alignment-stage trainable scope is invalid")
    return trainable


def configure_scratch_stage(model: FreshLowResolutionTranslator) -> dict[str, nn.Parameter]:
    """Train every retained parameter for the strict random-initialization control."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    trainable = dict(model.named_parameters())
    if not trainable:
        raise ValueError("scratch model has no trainable parameters")
    return trainable


def configure_decoder_stage(model: FreshLowResolutionTranslator) -> dict[str, nn.Parameter]:
    """Train the new visual-temporal layers and the retained language decoder."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for module in (
        model.frame_projection,
        model.temporal_encoder,
        model.language_projection,
        model.language_model.decoder,
        model.language_model.lm_head,
    ):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    model.temporal_positions.requires_grad_(True)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if (
        not trainable
        or any(name.startswith("frame_encoder.") for name in trainable)
        or not any(name.startswith("language_model.shared.") for name in trainable)
        or not any(name.startswith("language_model.decoder.") for name in trainable)
    ):
        raise ValueError("decoder-stage trainable scope is invalid")
    return trainable


def configure_encoder_alignment_stage(
    model: FreshLowResolutionTranslator,
) -> dict[str, nn.Parameter]:
    """Adapt the generic T5 encoder and cross-attention while preserving its decoder."""
    if not model.use_language_encoder:
        raise ValueError("encoder alignment requires the generic language encoder")
    trainable = configure_alignment_stage(model)
    for parameter in model.language_model.encoder.parameters():
        parameter.requires_grad_(True)
    for parameter in model.language_model.shared.parameters():
        parameter.requires_grad_(False)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if (
        not any(name.startswith("language_model.encoder.") for name in trainable)
        or any(name.startswith("frame_encoder.") for name in trainable)
        or any(name.startswith("language_model.shared.") for name in trainable)
    ):
        raise ValueError("encoder-alignment trainable scope is invalid")
    return trainable


def configure_visual_alignment_stage(
    model: FreshLowResolutionTranslator, trainable_blocks: int = 4
) -> dict[str, nn.Parameter]:
    """Adapt a suffix of MobileNet plus new alignment and cross-attention layers."""
    features = getattr(model.frame_encoder, "features", None)
    if not isinstance(features, nn.Sequential) or not 1 <= trainable_blocks <= len(features):
        raise ValueError("visual alignment requires a valid MobileNet block suffix")
    configure_alignment_stage(model)
    for block in features[-trainable_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    trainable = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    visual_names = [name for name in trainable if name.startswith("frame_encoder.features.")]
    if not visual_names or any(
        name.startswith("language_model.encoder.")
        or name.startswith("language_model.decoder.block.0.layer.0")
        for name in trainable
    ):
        raise ValueError("visual-alignment trainable scope is invalid")
    return trainable
