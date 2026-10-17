"""Load original public C5 values for the separately qualified native word risk proof."""

from __future__ import annotations

import torch
from c5_small_input_semantic import Native112SemanticStudent, clip_input
from evaluate_c5_small_input import appearance_model
from probe_c5_small_input import legacy_models
from probe_roi_semantic_learning import fixed_hashes
from probe_tiny_rgb_capacity import state_digest
from terra_word_verified_inputs import load_verified_input

PARAMETERS = 43260672
TENSORS = 350


def require_model(model, plan):
    selected = model.trainable_parameters()
    if (
        len(selected) != TENSORS
        or sum(parameter.numel() for parameter in selected.values()) != PARAMETERS
        or any(not name.startswith("vision.") for name in selected)
        or fixed_hashes(model) != plan["initial_reference_hashes"]
        or state_digest(model.vision.state_dict()) != plan["pretrained_state_sha256"]
    ):
        raise ValueError(
            "word semantic proof changed original C5 values or the qualified visual scope"
        )


def load_student(plan):
    from c5_terra_tail import load_model
    from c5_winner_reference import ROOT
    from prepare_roi_appearance import native_helpers
    from tiny_rgb_winner_student import load_frozen_winner_encoder
    from transformers import ByT5Tokenizer

    native_helpers()
    original, reference = load_model()
    if reference != plan["reference"]:
        raise ValueError("word semantic proof loaded another original C5 reference")
    encoder = load_frozen_winner_encoder(reference)
    hands, face = legacy_models(plan)
    vision = appearance_model(plan, hands.model, face.model)
    model = Native112SemanticStudent(vision, encoder, original.language, chunk_frames=2).eval()
    require_model(model, plan)
    tokenizer = ByT5Tokenizer.from_pretrained(
        ROOT / reference["bundle_root"] / "model/models/byt5_base", local_files_only=True
    )
    return model, tokenizer


def load_clip(case):
    values = load_verified_input(case, purpose="training")
    clip = clip_input(values)
    return clip, torch.ones(1, len(clip.pixels), dtype=torch.bool)
