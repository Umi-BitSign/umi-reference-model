"""Training-only feature supervision for raw RGB; not a qualified training recipe."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

STREAM_WIDTHS = {
    "face_features": 384,
    "left_hand_features": 384,
    "right_hand_features": 384,
    "pose_features": 14,
}


def verify_training_alignment(student: dict, teacher: dict) -> None:
    """Reject label/identity/timeline substitution before any teacher target is loaded.

    Bindings must first come from the existing protected, duration-vetted inventory.
    Matching these metadata alone does not replace source-byte or cache-byte audits.
    """
    for row in (student, teacher):
        if row.get("role") != "training" or row.get("training_allowed") is not True:
            raise ValueError("visual teaching accepts approved training identities only")
        if type(row.get("sampled_frames")) is not int or row["sampled_frames"] < 1:
            raise ValueError("visual teaching requires a complete positive frame timeline")
        for name in ("video_sha256", "frame_indices_sha256", "duration_integrity_audit_sha256"):
            value = row.get(name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError("visual teaching needs audited source/timeline hashes")
    fields = (
        "source",
        "example_id",
        "source_group",
        "video_sha256",
        "sampled_frames",
        "frame_indices_sha256",
        "duration_integrity_audit_sha256",
    )
    if any(not student.get(field) or student[field] != teacher.get(field) for field in fields):
        raise ValueError("visual teacher and raw RGB identity/timeline differ; never interpolate")


class SpatialTeacherHeads(nn.Module):
    """Supervise the four learned spatial queries as face/left/right/body features.

    These auxiliary heads can be discarded after pretraining. They never load a
    teacher checkpoint or reinitialize the student's visual/language weights.
    """

    def __init__(self, width: int):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("visual teacher head width must be a positive integer")
        self.width = width
        self.heads = nn.ModuleDict(
            {name: nn.Linear(width, dimension) for name, dimension in STREAM_WIDTHS.items()}
        )

    def forward(self, hidden, token_mask):
        if (
            hidden.ndim != 3
            or hidden.shape[-1] != self.width
            or hidden.shape[1] < 4
            or hidden.shape[1] % 4
            or token_mask.shape != hidden.shape[:2]
            or token_mask.dtype != torch.bool
        ):
            raise ValueError("visual teacher requires four spatial tokens per complete frame")
        batch, tokens, width = hidden.shape
        frame_tokens = hidden.reshape(batch, tokens // 4, 4, width)
        masks = token_mask.reshape(batch, tokens // 4, 4)
        if not torch.equal(masks, masks[..., :1].expand_as(masks)):
            raise ValueError("visual teacher cannot mask only part of a real frame")
        return {
            name: self.heads[name](frame_tokens[:, :, query])
            for query, name in enumerate(STREAM_WIDTHS)
        }, masks[:, :, 0]


def visual_teacher_loss(predictions, targets, frame_mask, *, pose_weight=0.1):
    """Masked feature loss with detached targets; absent DINO features are not labels.

    DINO streams use cosine distance; normalized pose uses SmoothL1. Padding is
    excluded before finite checks, and no teacher or language gradient is created.
    This is a proposed auxiliary objective, not evidence of better translation.
    """
    if (
        set(predictions) != set(STREAM_WIDTHS)
        or set(targets) != set(STREAM_WIDTHS)
        or frame_mask.ndim != 2
        or frame_mask.dtype != torch.bool
        or not frame_mask.any()
        or not math.isfinite(pose_weight)
        or pose_weight < 0
    ):
        raise ValueError("invalid visual teacher streams, frame mask or pose coefficient")
    components, usable = {}, {}
    for name, width in STREAM_WIDTHS.items():
        predicted, target = predictions[name], targets[name]
        shape = (*frame_mask.shape, width)
        if predicted.shape != shape or target.shape != shape:
            raise ValueError(f"visual teacher frame/feature shape mismatch: {name}")
        predicted, target = predicted[frame_mask], target.detach()[frame_mask]
        if not torch.isfinite(predicted).all() or not torch.isfinite(target).all():
            raise ValueError(f"nonfinite valid visual teacher features: {name}")
        if name == "pose_features":
            components[name] = F.smooth_l1_loss(predicted, target, beta=0.1)
            usable[name] = len(target)
        else:
            present = target.square().sum(-1) > 1e-12
            usable[name] = int(present.sum())
            components[name] = (
                (1 - F.cosine_similarity(predicted[present], target[present], dim=-1)).mean()
                if present.any()
                else predicted.sum() * 0
            )
    if not any(usable[name] for name in STREAM_WIDTHS if name != "pose_features"):
        raise ValueError("visual teacher clip has no usable hand/face feature signal")
    semantic = sum(components[name] for name in STREAM_WIDTHS if name != "pose_features") / 3
    return {
        "loss": semantic + pose_weight * components["pose_features"],
        "components": components,
        "usable_frames": usable,
    }
