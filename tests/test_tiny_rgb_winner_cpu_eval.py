from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_tiny_rgb_winner_cpu as evaluator
from evaluate_tiny_rgb_winner_cpu import RECIPE, GenerationBudget, install_trainable_weights


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(3, 2)
        self.language = torch.nn.Linear(3, 2).requires_grad_(False)

    def trainable_parameters(self):
        return {
            name: parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }


def fixture():
    student = Student()
    weights = {
        name: torch.zeros_like(value) for name, value in student.trainable_parameters().items()
    }
    metadata = {"plan_sha256": "parent", "fixed_winner_checkpoint": "winner", "update": "2048"}
    return student, weights, metadata


def test_checkpoint_install_preserves_fixed_language_and_requires_exact_visual_keys():
    student, weights, metadata = fixture()
    before = {name: value.clone() for name, value in student.language.state_dict().items()}
    install_trainable_weights(student, weights, metadata, "parent", "winner")
    assert all(
        torch.equal(value, torch.zeros_like(value))
        for value in student.trainable_parameters().values()
    )
    assert all(
        torch.equal(before[name], value) for name, value in student.language.state_dict().items()
    )
    weights["language.weight"] = student.language.weight
    with pytest.raises(ValueError, match="topology"):
        install_trainable_weights(student, weights, metadata, "parent", "winner")


@pytest.mark.parametrize(
    "kind", ["wrong_plan", "wrong_winner", "wrong_update", "shape", "dtype", "nonfinite"]
)
def test_checkpoint_install_rejects_changed_provenance_or_weight_shapes(kind):
    student, weights, metadata = fixture()
    if kind.startswith("wrong_"):
        key = {
            "wrong_plan": "plan_sha256",
            "wrong_winner": "fixed_winner_checkpoint",
            "wrong_update": "update",
        }[kind]
        metadata[key] = "changed"
    elif kind == "shape":
        weights["visual.weight"] = torch.zeros(4, 3)
    elif kind == "dtype":
        weights["visual.weight"] = weights["visual.weight"].double()
    else:
        weights["visual.weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="topology"):
        install_trainable_weights(student, weights, metadata, "parent", "winner")


def test_successful_generation_guard_never_stops_or_mutates_tokens():
    guard = GenerationBudget(clock=lambda: 0, memory=lambda: 0)
    tokens = torch.ones(5, 64, dtype=torch.long)
    before = tokens.clone()
    for _ in range(128):
        assert not guard(tokens, None).any()
    assert torch.equal(tokens, before)
    assert RECIPE["generation"] == "unchanged_winner_five_beam_2048"


@pytest.mark.parametrize("resource", ["memory", "time"])
def test_resource_failure_raises_instead_of_scoring_truncated_hypothesis(resource):
    clock_value = [0]
    guard = GenerationBudget(
        clock=lambda: clock_value[0],
        memory=lambda: RECIPE["maximum_process_peak_bytes"] + (resource == "memory"),
    )
    if resource == "time":
        clock_value[0] = RECIPE["maximum_seconds_per_clip"] + 1
    with pytest.raises(RuntimeError, match="resource budget"):
        guard(torch.ones(5, 2, dtype=torch.long), None)


def test_one_owned_job_freezes_before_evaluation_without_intermediate_observer(
    monkeypatch, tmp_path
):
    calls = []
    monkeypatch.setattr(evaluator, "freeze", lambda path: calls.append(("freeze", path)))
    monkeypatch.setattr(evaluator, "run", lambda path, output: calls.append(("run", path, output)))
    for name in (
        "PYTHONDONTWRITEBYTECODE",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    ):
        monkeypatch.setenv(name, "test_original")
    plan, output = tmp_path / "plan.json", tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv", ["cpu-evaluation", "--freeze", str(plan), "--output-dir", str(output)]
    )
    evaluator.main()
    assert calls == [("freeze", plan), ("run", plan, output)]
