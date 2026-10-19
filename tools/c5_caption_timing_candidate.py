"""Load exactly one fully audited final timing adapter into original frozen C5."""

from __future__ import annotations

import torch
from c5_timing_feature_data import bound, require


def install_timing(model, binding):
    """Only load_candidate admits an actual final audit; exposed for interface tests."""
    model.require_fixed()
    require(
        binding["schema"] == "umi-studio-complete-caption-timing-final-audit/1"
        and binding["complete"] is True
        and binding["optimizer_updates_replayed"] == 134
        and binding["complete_training_exposures"] == 1064
        and binding["full_epochs"] == 2
        and model.timing.maximum_log_rate == binding["maximum_log_rate"]
        and set(model.trainable_parameters()) == {"timing.rate_weights"}
        and torch.count_nonzero(model.timing.rate_weights).item() == 0,
        "timing candidate requires the audited complete final and fresh original initializer",
    )
    saved = torch.load(bound(binding["adapter"]), map_location="cpu", weights_only=True)
    require(
        set(saved) == {"timing", "optimizer", "epochs", "updates", "exposures", "plan"}
        and set(saved["timing"]) == {"rate_weights"}
        and saved["epochs"] == 2
        and saved["updates"] == 134
        and saved["exposures"] == 1064
        and saved["plan"] == binding["plan"],
        "timing candidate cannot install partial, inherited or expanded adapter state",
    )
    weights = saved["timing"]["rate_weights"]
    require(
        isinstance(weights, torch.Tensor)
        and weights.dtype == torch.float32
        and weights.device.type == "cpu"
        and weights.shape == (4,)
        and torch.isfinite(weights).all()
        and torch.count_nonzero(weights).item() > 0
        and (weights.abs() <= 0.5).all()
        and weights.tolist() == binding["weights"],
        "timing candidate changed its exact four learned FP32 values",
    )
    model.timing.load_state_dict(saved["timing"], strict=True)
    model.eval()
    require_candidate(model, binding)


def require_candidate(model, binding):
    model.require_fixed()
    selected = model.trainable_parameters()
    require(
        set(selected) == {"timing.rate_weights"}
        and selected["timing.rate_weights"].shape == (4,)
        and selected["timing.rate_weights"].dtype == torch.float32
        and selected["timing.rate_weights"].device.type == "cpu"
        and selected["timing.rate_weights"].detach().tolist() == binding["weights"]
        and model.timing.maximum_log_rate == binding["maximum_log_rate"],
        "timing candidate weights or rate bound changed during evaluation",
    )


def load_candidate(final_audit_path):
    """Caller must hold the shared C5 lease before model loading."""
    from c5_caption_timing_final import checked_final
    from c5_timing_semantic_model import load_student, require_model

    binding = checked_final(final_audit_path)
    model, tokenizer = load_student(binding["model_plan"])
    install_timing(model, binding)
    require_model(model, binding["model_plan"])
    return model, tokenizer, binding
