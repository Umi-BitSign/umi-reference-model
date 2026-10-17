"""Load a complete word-final research initializer for later mixed risk qualification."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from c5_small_input_semantic import clip_input
from c5_word_final import checked_final
from c5_word_semantic_model import PARAMETERS, TENSORS
from c5_word_semantic_model import load_student as load_original_student
from prepare_terra_mixed_native import load_verified_input
from probe_roi_semantic_learning import fixed_hashes
from probe_tiny_rgb_capacity import state_digest
from terra_training_recovery import artifact
from train_c5_word_semantic import mask_digest

ROOT = Path(__file__).resolve().parents[1]


def word_initialization(training_path, result_path):
    training_path, result_path = training_path.resolve(), result_path.resolve()
    fit, result, evidence = checked_final(training_path, result_path)
    run = json.loads(result_path.with_name("run.json").read_text())
    return {
        "schema": "umi-studio-complete-word-final-mixed-initializer/1",
        "word_training_plan": artifact(training_path),
        "word_training_result": artifact(result_path),
        "word_training_evidence": evidence,
        "original_model_plan": fit["model_plan"],
        "initial_visual": result["visual"],
        "initial_visual_state_sha256": result["visual_final_sha256"],
        "original_mask_sha256": run["frozen_mask_sha256"],
        "scope": "research initializer provenance only; no stage-advance or promotion decision",
        "fresh_mixed_full_risk_qualification_required": True,
        "optimizer_state_resumed": False,
        "promotion_authorized": False,
    }


def checked_initialization(binding):
    expected = word_initialization(
        ROOT / binding["word_training_plan"]["path"],
        ROOT / binding["word_training_result"]["path"],
    )
    if binding != expected:
        raise ValueError("mixed initializer changed complete word training or actual final values")
    return binding


def _require_structure(model, binding):
    original = binding["original_model_plan"]
    selected = model.trainable_parameters()
    if (
        model.training
        or model.vision.training
        or model.vision.side != 112
        or len(selected) != TENSORS
        or set(selected) != set(original["expected_visual_gradient_names"])
        or sum(parameter.numel() for parameter in selected.values()) != PARAMETERS
        or any(not name.startswith("vision.") for name in selected)
        or any(
            component.training
            or any(parameter.requires_grad for parameter in component.parameters())
            for component in (model.semantic_encoder, model.language)
        )
        or any(getattr(model.vision, name).mask_token.requires_grad for name in ("hands", "face"))
        or mask_digest(model) != binding["original_mask_sha256"]
        or fixed_hashes(model) != original["initial_reference_hashes"]
    ):
        raise ValueError("mixed model changed original C5 backbones, masks or350-tensor scope")
    return selected


def require_model(model, plan):
    """Check proof values inside the caller's duty block when the model is on Metal."""
    binding = plan["initialization"]
    _require_structure(model, binding)
    if state_digest(model.vision.state_dict()) != binding["initial_visual_state_sha256"]:
        raise ValueError(
            "mixed model does not reproduce its complete word-final visual initializer"
        )


def install_visual(model, binding):
    from safetensors.torch import load_file

    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("mixed initializer must first reproduce original public C5 on CPU")
    selected = _require_structure(model, binding)
    if (
        state_digest(model.vision.state_dict())
        != binding["original_model_plan"]["pretrained_state_sha256"]
    ):
        raise ValueError("mixed initializer must first reproduce original public C5 on CPU")
    target = ROOT / binding["initial_visual"]["path"]
    if artifact(target) != binding["initial_visual"]:
        raise ValueError("mixed initializer's actual fixed-final visual file changed")
    incoming = load_file(str(target), device="cpu")
    if set(incoming) != set(selected):
        raise ValueError("mixed initializer cannot load partial, extra or frozen tensor names")
    proposed = dict(model.vision.state_dict())
    for name, value in incoming.items():
        key = name.removeprefix("vision.")
        if (
            key not in proposed
            or value.shape != proposed[key].shape
            or value.dtype != proposed[key].dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(
                "mixed initializer contains changed shapes, dtypes or nonfinite values"
            )
        proposed[key] = value
    # Validate the complete trained state, including both original masks, before
    # mutating the newly loaded model. Parameter identity and requires_grad stay.
    if state_digest(proposed) != binding["initial_visual_state_sha256"]:
        raise ValueError("mixed initializer's proposed complete visual-state digest differs")
    model.vision.load_state_dict(proposed, strict=True)
    require_model(model, {"initialization": binding})


def load_student(plan):
    binding = checked_initialization(plan["initialization"])
    model, tokenizer = load_original_student(binding["original_model_plan"])
    install_visual(model, binding)
    return model, tokenizer


def load_clip(bound):
    values = load_verified_input(bound)
    clip = clip_input(values)
    return clip, torch.ones(1, len(clip.pixels), dtype=torch.bool)
