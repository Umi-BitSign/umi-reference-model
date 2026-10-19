"""Install only the complete audited final local temporal-memory adapter."""

from __future__ import annotations

import torch
from c5_timing_feature_data import bound, require
from train_c5_temporal_memory import state_hashes


def install_adapter(model, binding):
    model.require_fixed()
    require(
        binding["schema"] == "umi-studio-complete-temporal-memory-final-audit/1"
        and binding["complete"] is True
        and binding["optimizer_updates_replayed"] == 268
        and binding["complete_training_exposures"] == 2128
        and binding["full_epochs"] == 4
        and model.memory_adapter.configuration() == binding["adapter_configuration"]
        and torch.count_nonzero(model.memory_adapter.output_projection.weight).item() == 0,
        "memory candidate requires its audited complete final and fresh original initializer",
    )
    saved = torch.load(bound(binding["adapter"]), map_location="cpu", weights_only=True)
    require(
        set(saved)
        == {
            "memory_adapter",
            "configuration",
            "optimizer",
            "epochs",
            "updates",
            "exposures",
            "plan",
        }
        and saved["configuration"] == binding["adapter_configuration"]
        and saved["epochs"] == 4
        and saved["updates"] == 268
        and saved["exposures"] == 2128
        and saved["plan"] == binding["plan"],
        "memory candidate cannot install a partial or different adapter state",
    )
    expected = model.memory_adapter.state_dict()
    require(
        set(saved["memory_adapter"]) == set(expected)
        and all(
            isinstance(value, torch.Tensor)
            and value.device.type == "cpu"
            and value.dtype == torch.float32
            and value.shape == expected[name].shape
            and torch.isfinite(value).all()
            for name, value in saved["memory_adapter"].items()
        )
        and torch.count_nonzero(saved["memory_adapter"]["output_projection.weight"]).item() > 0,
        "memory candidate changed its finite declared adapter values",
    )
    model.memory_adapter.load_state_dict(saved["memory_adapter"], strict=True)
    model.eval()
    require_candidate(model, binding)


def require_candidate(model, binding):
    model.require_fixed()
    require(
        model.memory_adapter.configuration() == binding["adapter_configuration"]
        and state_hashes(model.memory_adapter) == binding["state_hashes"],
        "memory candidate weights or configuration changed during evaluation",
    )


def load_candidate(final_audit_path):
    """Caller must hold the shared C5 lease before loading the public model."""
    from c5_temporal_memory_final import checked_final
    from c5_temporal_memory_model import load_student, require_model

    binding = checked_final(final_audit_path)
    model, tokenizer = load_student(binding["model_plan"], binding["adapter_configuration"])
    install_adapter(model, binding)
    require_model(model, binding["model_plan"], binding["adapter_configuration"])
    return model, tokenizer, binding
