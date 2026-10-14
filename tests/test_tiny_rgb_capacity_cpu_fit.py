from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import tiny_rgb_capacity_protocol as numerical
import tiny_rgb_full_run as full
import train_tiny_rgb_capacity_cpu as fit
from benchmark_fleurs_dev import sha256


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    root = tmp_path
    plan_path = root / "runs/fit/plan.json"
    plan_path.parent.mkdir(parents=True)
    numerical_path = root / "numerical.json"
    audit_path, budget_path, video_path = (root / name for name in ("audit", "budget", "video"))
    for path in (numerical_path, audit_path, budget_path, video_path):
        path.write_text("fixture\n")
    examples = [{"video": "video", "video_sha256": sha256(video_path)}]
    numerical_plan = {"dependency_sha256": {}, "language_path": "language", "cases": []}
    monkeypatch.setattr(fit, "ROOT", root)
    monkeypatch.setattr(fit, "NUMERICAL_PLAN", numerical_path)
    monkeypatch.setattr(numerical, "BUDGET", budget_path)
    monkeypatch.setattr(numerical, "DURATION_AUDIT", audit_path)
    monkeypatch.setattr(fit, "sources", lambda: {"fixture": "source"})
    monkeypatch.setattr(full, "runtime_binding", lambda: {"fixture": "runtime"})
    monkeypatch.setattr(
        fit,
        "inputs",
        lambda: (numerical_plan, {"valid_training_clips": 7523}, examples),
    )
    plan = {
        "schema": fit.SCHEMA,
        "seed": fit.SEED,
        "recipe": copy.deepcopy(fit.RECIPE),
        "source_sha256": fit.sources(),
        "runtime_binding": full.runtime_binding(),
        "numerical_plan_sha256": sha256(numerical_path),
        "dependency_sha256": {
            path.name: sha256(path)
            for path in (numerical_path, audit_path, budget_path, video_path)
        },
        "examples": examples,
        "preflight_cases": [],
        "eligible_training_clips": 7523,
        "language_path": "language",
        "resource_policy": "budget",
        "metal_training_qualified": False,
        "gpu_used": False,
        "miner_modified": False,
    }
    snapshot = plan_path.with_name("source-snapshot")
    snapshot.mkdir()

    def save(value):
        plan_path.write_text(json.dumps(value))
        snapshot.joinpath("manifest.json").write_text(
            json.dumps({"plan_sha256": sha256(plan_path)})
        )

    save(plan)
    return plan_path, plan, save


def test_cpu_recipe_is_matched_and_never_claims_gpu_or_held_out_success(frozen):
    path, plan, _ = frozen
    assert fit.checked_plan(path) == plan
    assert fit.RECIPE["capacities"] == ["small", "medium", "large"]
    assert fit.RECIPE["short_side"] == 192
    assert fit.RECIPE["execution_device"] == "cpu"
    assert not fit.RECIPE["initial_weights_reused"]


@pytest.mark.parametrize("key", ["gpu_used", "metal_training_qualified", "miner_modified"])
def test_cpu_plan_rejects_authority_expansion(frozen, key):
    path, plan, save = frozen
    plan[key] = True
    save(plan)
    with pytest.raises(ValueError, match="immutable"):
        fit.checked_plan(path)


@pytest.mark.parametrize("field", ["seed", "recipe", "language_path", "dependency_sha256"])
def test_cpu_plan_rejects_changed_controls_even_with_resaved_snapshot(frozen, field):
    path, plan, save = frozen
    if field == "seed":
        plan[field] += 1
    elif field == "recipe":
        plan[field]["short_side"] = 256
    elif field == "language_path":
        plan[field] = "other-language"
    else:
        del plan[field]["audit"]
    save(plan)
    with pytest.raises(ValueError, match="changed"):
        fit.checked_plan(path)


def test_cpu_plan_rejects_edited_snapshot_and_dependency(frozen):
    path, plan, save = frozen
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="archived"):
        fit.checked_plan(path)
    save(plan)
    path.parents[2].joinpath("video").write_text("changed\n")
    with pytest.raises(ValueError, match="dependency changed"):
        fit.checked_plan(path)
