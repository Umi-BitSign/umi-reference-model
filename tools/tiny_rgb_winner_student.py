"""Fresh small-pixel vision and trainable bridge with the winner's frozen language."""

from __future__ import annotations

import torch
from tiny_rgb_model import RmsNorm
from tiny_rgb_visual_teacher import visual_teacher_loss
from torch import nn
from torch.nn import functional as F
from train_tiny_rgb_visual_teacher import build_student
from transformers.modeling_outputs import BaseModelOutput


def winner_length_buckets(video, labels):
    """Extend only this byte-decoder family's target buckets, without dropping tokens."""
    from tiny_rgb_full_training_tools import masked_length_buckets

    padded, mask, _ = masked_length_buckets(video, None, torch)
    if labels is None:
        return padded, mask, None
    length = labels.shape[1]
    size = next(
        (
            size
            for size in (8, 16, 24, 32, 48, 64, 80, 96, 128, 192, 256, 384, 512)
            if size >= length
        ),
        None,
    )
    if length < 1 or size is None:
        raise ValueError("winner student cannot truncate a byte reference")
    target = labels.new_full((labels.shape[0], size), -100)
    target[:, :length] = labels
    return padded, mask, target


class RawRgbWinnerStudent(nn.Module):
    def __init__(self, language):
        super().__init__()
        fresh = build_student()
        self.vision = fresh["vision"]
        self.teacher_heads = fresh["heads"]
        self.language = language
        self.language.requires_grad_(False)
        if hasattr(self.language, "frozen_winner_weights"):
            self.language.frozen_winner_weights = True
        incoming = self.vision.config.width * self.vision.config.spatial_queries
        self.bridge = nn.Linear(incoming, language.config.d_model)
        if incoming == language.config.d_model:
            nn.init.eye_(self.bridge.weight)
        else:
            nn.init.normal_(self.bridge.weight, std=0.02)
        nn.init.zeros_(self.bridge.bias)
        self.memory_norm = RmsNorm(language.config.d_model)
        self.language.eval()

    def train(self, mode=True):
        super().train(mode)
        self.language.eval()
        return self

    def encode(self, video, frame_mask, *, features=False):
        hidden, token_mask = self.vision(video, frame_mask)
        queries = self.vision.config.spatial_queries
        if hidden.shape[1] != video.shape[1] * queries:
            raise ValueError("winner student cannot truncate or invent visual frames")
        memory = self.memory_norm(self.bridge(hidden.reshape(video.shape[0], video.shape[1], -1)))
        if features:
            predictions, mask = self.teacher_heads(hidden, token_mask)
            if not torch.equal(mask, frame_mask):
                raise ValueError("winner student feature and caption masks differ")
            return memory, mask, predictions
        return memory, frame_mask

    def forward(self, video, frame_mask, labels):
        memory, mask, features = self.encode(video, frame_mask, features=True)
        output = self.language(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask.long(),
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
        return {"language": output, "memory": memory, "frame_mask": mask, "features": features}

    @torch.no_grad()
    def generate(self, video, frame_mask, **kwargs):
        memory, mask = self.encode(video, frame_mask)
        return self.language.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=memory),
            attention_mask=mask.long(),
            **kwargs,
        )

    def trainable_parameters(self):
        return {name: value for name, value in self.named_parameters() if value.requires_grad}


def student_objective(
    output, latent_teacher, feature_teacher, *, latent_coefficient=0.5, feature_coefficient=0.1
):
    """Human CE plus fixed teacher representations; no teacher or language update."""
    mask = output["frame_mask"]
    memory = output["memory"]
    if latent_teacher.shape != memory.shape:
        raise ValueError("winner teacher and raw student latent frame/width shapes differ")
    target = latent_teacher.detach()[mask]
    predicted = memory[mask]
    if not len(target) or not torch.isfinite(target).all() or not torch.isfinite(predicted).all():
        raise ValueError("winner latent teacher signal is empty or nonfinite")
    latent = (1 - F.cosine_similarity(predicted, target, dim=-1)).mean()
    latent = latent + 0.1 * F.smooth_l1_loss(predicted, target, beta=0.1)
    feature = visual_teacher_loss(output["features"], feature_teacher, mask, pose_weight=0.1)[
        "loss"
    ]
    caption = output["language"].loss
    return {
        "loss": caption + latent_coefficient * latent + feature_coefficient * feature,
        "caption_loss": caption,
        "latent_loss": latent,
        "feature_loss": feature,
    }


def load_frozen_winner_encoder(reference):
    """Use the exact published forward path, never the old cached encoder states."""
    import os
    import sys

    from c5_winner_mps_safe import replace_fairseq_exact_gelu, replace_native_layer_norms
    from c5_winner_reference import ROOT
    from tiny_rgb_mps_safe import replace_native_gelu

    root = ROOT / reference["bundle_root"] / "model"
    sys.dont_write_bytecode = True
    os.environ["SHUBERT_DINOV2_SOURCE"] = str(root / "vendor/dinov2-source")
    sys.path.insert(0, str(root / "vendor"))
    sys.path.insert(0, str(root / "vendor/demo"))
    from inference import SignLanguageByT5ForConditionalGeneration

    teacher = SignLanguageByT5ForConditionalGeneration.from_pretrained(
        root / "models/checkpoint-11625", local_files_only=True, use_safetensors=True
    ).eval()
    if (
        replace_fairseq_exact_gelu(teacher),
        replace_native_gelu(teacher),
        replace_native_layer_norms(teacher),
    ) != (12, 1, 30):
        raise ValueError("winner semantic teacher numerical topology differs from qualified source")
    teacher.requires_grad_(False)
    return teacher.encoder.eval()
