from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_tiny_rgb_feature_winner import CPU_LIMITS, CpuBudget, install_weights
from tiny_rgb_feature_winner_protocol import RECIPE as QUALIFIED
from train_tiny_rgb_feature_winner_full import RECIPE, optimizer_payload


def test_full_recipe_matches_real_qualified_path_and_preserves_full_horizon():
    for key in (
        "capacity",
        "short_side",
        "target_fps",
        "max_frames",
        "max_text_tokens",
        "latent_coefficient",
        "feature_coefficient",
        "retained_native_semantic_encoder",
        "initial_diagnostic_weights_reused",
    ):
        assert RECIPE[key] == QUALIFIED[key]
    assert RECIPE["epochs"] == 4 and RECIPE["checks_after_updates"] == [2048, 8192, 16384, 30092]
    assert RECIPE["evaluation_device"] == "isolated_cpu_subprocess"
    assert RECIPE["optimizer_state_saved_at_each_check"]


class Student(torch.nn.Linear):
    def trainable_parameters(self):
        return dict(self.named_parameters())


def test_checkpoint_restore_supports_later_updates_without_touching_parameter_identity():
    student = Student(3, 2)
    values = {name: torch.zeros_like(value) for name, value in student.named_parameters()}
    identities = [id(value) for value in student.parameters()]
    metadata = {"plan_sha256": "plan", "fixed_winner_checkpoint": "winner", "update": "8192"}
    install_weights(
        student, values, metadata, parent_sha="plan", reference_sha="winner", update=8192
    )
    assert identities == [id(value) for value in student.parameters()]
    assert all(not value.any() for value in student.parameters())
    with pytest.raises(ValueError, match="checkpoint"):
        install_weights(
            student, values, metadata, parent_sha="plan", reference_sha="winner", update=2048
        )


def test_optimizer_payload_retains_moments_steps_rng_and_live_parameter_references():
    student = Student(3, 2)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4, foreach=False)
    student(torch.ones(1, 3)).square().sum().backward()
    optimizer.step()
    parameters = student.trainable_parameters()
    before = [id(value) for value in optimizer.param_groups[0]["params"]]
    rng = {"cpu": torch.get_rng_state(), "mps": torch.tensor([1, 2], dtype=torch.uint8)}
    values, metadata = optimizer_payload(optimizer, parameters, rng)
    assert before == [id(value) for value in optimizer.param_groups[0]["params"]]
    assert json.loads(metadata["parameter_order"]) == list(parameters)
    assert len(values) == len(parameters) * 3 + 2
    for name, parameter in parameters.items():
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(values[f"{name}.{field}"], optimizer.state[parameter][field])
    assert torch.equal(values["rng.cpu"], rng["cpu"]) and torch.equal(values["rng.mps"], rng["mps"])


def test_new_full_backbone_cpu_budget_never_silently_changes_generation():
    guard = CpuBudget(clock=lambda: 0, memory=lambda: 0)
    tokens = torch.ones(5, 64, dtype=torch.long)
    assert not guard(tokens, None).any()
    assert CPU_LIMITS["maximum_process_peak_bytes"] == 24 * 1024**3
    guard = CpuBudget(clock=lambda: 0, memory=lambda: CPU_LIMITS["maximum_process_peak_bytes"] + 1)
    with pytest.raises(RuntimeError, match="resource policy"):
        guard(tokens, None)
