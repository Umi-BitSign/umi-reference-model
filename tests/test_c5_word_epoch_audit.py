from __future__ import annotations

import copy
import json
import random
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_word_epoch_audit as audit
from terra_training_recovery import artifact
from train_tiny_rgb_feature_winner_full import optimizer_payload


@pytest.fixture
def state(tmp_path):
    """350 one-value CPU parameters; no actual C5 weights or Metal RNG evidence."""
    parameters = {
        f"vision.fixture{index}": torch.nn.Parameter(torch.ones(1)) for index in range(350)
    }
    optimizer = torch.optim.AdamW(parameters.values(), lr=1e-5, weight_decay=0.01, foreach=False)
    sum(value.sum() for value in parameters.values()).backward()
    optimizer.step()
    values, metadata = optimizer_payload(
        optimizer,
        parameters,
        {
            "cpu": torch.get_rng_state(),
            "mps": torch.arange(16, dtype=torch.uint8),
        },
    )
    path = tmp_path / "optimizer.safetensors"
    save_file(values, str(path), metadata=metadata)
    return SimpleNamespace(
        parameters=parameters,
        values=values,
        metadata=metadata,
        path=path,
        shapes={name: [1] for name in parameters},
    )


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "missing",
        "extra",
        "order",
        "duplicate_order",
        "groups",
        "lr",
        "betas",
        "foreach",
        "step",
        "step_shape",
        "moment_shape",
        "moment_dtype",
        "nonfinite",
        "negative_second",
        "missing_rng",
        "rng_dtype",
        "rng_shape",
        "cpu_rng",
    ],
)
def test_saved_Adam_inventory_checks_all350_steps_moments_hyperparameters_and_RNG_without_mutation(
    state, mutation
):
    env = state
    values, metadata = dict(env.values), dict(env.metadata)
    name = list(env.parameters)[-1]
    if mutation == "missing":
        values.pop(f"{name}.exp_avg")
    elif mutation == "extra":
        values["extra"] = torch.zeros(1)
    elif mutation in ("order", "duplicate_order"):
        order = json.loads(metadata["parameter_order"])
        order[-1] = "other" if mutation == "order" else order[0]
        metadata["parameter_order"] = json.dumps(order)
    elif mutation in ("groups", "lr", "betas", "foreach"):
        groups = json.loads(metadata["groups_without_parameters"])
        if mutation == "groups":
            groups.append(copy.deepcopy(groups[0]))
        else:
            groups[0][mutation] = {"lr": 0.001, "betas": [0.8, 0.9], "foreach": True}[mutation]
        metadata["groups_without_parameters"] = json.dumps(groups)
    elif mutation in ("step", "step_shape"):
        values[f"{name}.step"] = torch.tensor(2.0) if mutation == "step" else torch.ones(1)
    elif mutation in ("moment_shape", "moment_dtype", "nonfinite"):
        values[f"{name}.exp_avg"] = {
            "moment_shape": torch.ones(2),
            "moment_dtype": torch.ones(1, dtype=torch.float64),
            "nonfinite": torch.full((1,), float("nan")),
        }[mutation]
    elif mutation == "negative_second":
        values[f"{name}.exp_avg_sq"] = -torch.ones(1)
    elif mutation == "missing_rng":
        values.pop("rng.mps")
    elif mutation == "rng_dtype":
        values["rng.mps"] = values["rng.mps"].to(torch.int32)
    elif mutation == "rng_shape":
        values["rng.mps"] = values["rng.mps"].reshape(2, 8)
    elif mutation == "cpu_rng":
        values["rng.cpu"] = torch.ones(1, dtype=torch.uint8)
    changed = state.path.with_name("changed.safetensors")
    save_file(values, str(changed), metadata=metadata)
    rng_before = torch.get_rng_state().clone()
    parameter_before = {name: value.detach().clone() for name, value in env.parameters.items()}
    if mutation is None:
        result = audit.optimizer_inventory(changed, env.shapes, 1, audit.RECIPE)
        assert result["parameter_tensors"] == 350 and result["all_Adam_steps"] == 1
        assert result["RNG"]["cpu"]["bytes"] == rng_before.numel()
    else:
        with pytest.raises((ValueError, RuntimeError)):
            audit.optimizer_inventory(changed, env.shapes, 1, audit.RECIPE)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(torch.equal(value, parameter_before[name]) for name, value in env.parameters.items())


@pytest.fixture
def epoch(state, tmp_path, monkeypatch):
    """Synthetic214-clip prefix and actual serialized tiny350-tensor Adam transaction."""
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "artifact", descriptor)
    monkeypatch.setattr(audit, "PARAMETERS", 350)
    path = tmp_path / "runs/training/plan.json"
    path.parent.mkdir(parents=True)
    output = tmp_path / "checkpoints/training"
    output.mkdir(parents=True)
    scope = path.with_name("scope.json")
    scope.write_text(
        json.dumps(
            {
                "parameters": [
                    {"parameter": name.removeprefix("vision."), "shape": [1]}
                    for name in state.shapes
                ]
            }
        )
    )
    batches = []
    for number in range(4):
        indices = list(range(214))
        random.Random(audit.RECIPE["seed"] + number).shuffle(indices)
        batches.extend(
            {"epoch": number, "training_record_indices": indices[offset : offset + 4]}
            for offset in range(0, 214, 4)
        )
    plan = {
        "recipe": copy.deepcopy(audit.RECIPE),
        "source_sha256": {},
        "records": [None] * 214,
        "planned_training_examples": 856,
        "batches": batches,
        "vision_initialization_sha256": "initial",
        "model_plan": {
            "initial_reference_hashes": {"sequence": "fixed", "language": "fixed"},
            "visual_scope": descriptor(scope),
            "expected_visual_gradient_names": list(state.shapes),
            "reference": {"translator_checkpoint_sha256": "original"},
        },
    }
    path.write_text(json.dumps(plan))
    snapshot = path.with_name("source-snapshot")
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(
        json.dumps({"plan_sha256": audit.sha256(path), "source_sha256": {}})
    )
    run = output / "run.json"
    run.write_text(
        json.dumps(
            {
                "plan_sha256": audit.sha256(path),
                "recipe": plan["recipe"],
                "initial_visual_sha256": "initial",
                "fixed_backbones": plan["model_plan"]["initial_reference_hashes"],
            }
        )
    )
    worker = output / "worker.json"
    worker.write_text(json.dumps({"pid": 123}))
    observer = path.with_name("observer.py")
    observer.write_text("synthetic observer fixture")
    steps = []
    examples = 0
    for number, batch in enumerate(batches[:54]):
        count = len(batch["training_record_indices"])
        examples += count
        steps.append(
            {
                **batch,
                "batch": number,
                "training_examples": examples,
                "optimizer_updates": 1,
                "contrasting_clips": count if number == 0 else 0,
                "Terra_qualities": [[10, 3] if number == 0 else [10, 10] for _ in range(count)],
            }
        )
    prefix = path.with_name("completed-steps.jsonl")
    prefix.write_text("".join(json.dumps(step) + "\n" for step in steps))
    visual = output / "visual-after-000214.safetensors"
    provenance = {"plan_sha256": audit.sha256(path), "update": "214"}
    save_file(
        {name: value.detach() for name, value in state.parameters.items()},
        str(visual),
        metadata={**provenance, "optimizer_updates": "1", "fixed_winner_checkpoint": "original"},
    )
    optimizer = visual.with_name(visual.stem + "-optimizer.safetensors")
    save_file(state.values, str(optimizer), metadata={**provenance, **state.metadata})
    commit_path = visual.with_suffix(".json")
    commit = {
        "complete": True,
        "visual": descriptor(visual),
        "optimizer": descriptor(optimizer),
        "training_examples": 214,
        "optimizer_updates": 1,
        "committed_batch_RNG": True,
    }
    commit_path.write_text(json.dumps(commit))
    observation = {
        "schema": "umi-studio-word-training-live-observation/1",
        "same_original_worker_confirmed_live": True,
        "previous_committed_prefix_and_evidence_unchanged": True,
        "new_Terra_requests_during_observation": 0,
        "optimizer_updates_during_observation": 0,
        "original_worker": descriptor(worker),
        "original_run": descriptor(run),
        "observer_source": descriptor(observer),
        "completed_steps_prefix": descriptor(prefix),
        "actual_raw_training_evidence": [],
        "complete_batch_records": 54,
        "epoch_checkpoint_commits": [descriptor(commit_path)],
        "training_examples": 214,
        "optimizer_updates": 1,
    }
    observed = path.with_name("observation.json")
    observed.write_text(json.dumps(observation))
    return SimpleNamespace(
        plan_path=path,
        observation_path=observed,
        commit_path=commit_path,
        plan=plan,
        observation=observation,
        commit=commit,
        visual=visual,
        optimizer=optimizer,
        steps=steps,
        prefix=prefix,
    )


def overwrite(path, record):
    path.write_text(json.dumps(record))


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "commit_hash",
        "optimizer_hash",
        "partial_commit",
        "rng_claim",
        "update_count",
        "plan",
        "weight",
        "weight_shape",
        "weight_dtype",
        "missing_weight",
        "optimizer_step",
        "metadata",
        "scope",
        "order",
        "prefix",
        "not_observed",
    ],
)
def test_epoch_transaction_rejoins_observed_prefix_weights_Adam_RNG_and_original_scope(
    epoch, mutation
):
    env = epoch
    if mutation in ("partial_commit", "rng_claim", "update_count", "optimizer_hash"):
        key, value = {
            "partial_commit": ("complete", False),
            "rng_claim": ("committed_batch_RNG", False),
            "update_count": ("optimizer_updates", 2),
            "optimizer_hash": ("optimizer", env.commit["visual"]),
        }[mutation]
        env.commit[key] = value
        overwrite(env.commit_path, env.commit)
        env.observation["epoch_checkpoint_commits"] = [audit.artifact(env.commit_path)]
        overwrite(env.observation_path, env.observation)
    elif mutation == "commit_hash":
        overwrite(env.commit_path, {})
    elif mutation == "not_observed":
        env.observation["epoch_checkpoint_commits"] = []
        overwrite(env.observation_path, env.observation)
    elif mutation == "scope":
        scope = env.plan_path.with_name("scope.json")
        overwrite(scope, {})
    elif mutation in ("plan", "order"):
        if mutation == "plan":
            env.plan["recipe"]["learning_rate"] = 0.001
        else:
            env.plan["batches"][0]["training_record_indices"].reverse()
        overwrite(env.plan_path, env.plan)
    elif mutation == "prefix":
        env.prefix.write_text(env.prefix.read_text() + "{")
    elif mutation in (
        "weight",
        "weight_shape",
        "weight_dtype",
        "missing_weight",
        "optimizer_step",
        "metadata",
    ):
        target = env.optimizer if mutation == "optimizer_step" else env.visual
        with safe_open(str(target), framework="pt", device="cpu") as saved:
            metadata = saved.metadata()
        values = load_file(str(target))
        name = list(values)[-1]
        if mutation == "weight":
            values[name] = torch.full_like(values[name], float("nan"))
        elif mutation == "weight_shape":
            values[name] = torch.zeros(2)
        elif mutation == "weight_dtype":
            values[name] = values[name].to(torch.float64)
        elif mutation == "missing_weight":
            values.pop(name)
        elif mutation == "optimizer_step":
            name = next(name for name in values if name.endswith(".step"))
            values[name] = torch.tensor(2.0)
        elif mutation == "metadata":
            metadata["fixed_winner_checkpoint"] = "different"
        target.unlink()
        save_file(values, str(target), metadata=metadata)
        env.commit["optimizer" if mutation == "optimizer_step" else "visual"] = audit.artifact(
            target
        )
        overwrite(env.commit_path, env.commit)
        env.observation["epoch_checkpoint_commits"] = [audit.artifact(env.commit_path)]
        overwrite(env.observation_path, env.observation)
    if mutation is None:
        result = audit.audit_epoch(env.plan_path, env.observation_path, env.commit_path)
        assert result["training_examples"] == 214 and result["tail_clips"] == 2
        assert result["all_Adam_steps"] == 1 and result["parameter_tensors"] == 350
        assert result["fixed_final_evaluation_eligible"] is False
        assert result["optimizer_or_RNG_restored"] is False
        assert result["MPS_RNG_device_restoration_tested"] is False
    else:
        with pytest.raises(ValueError):
            audit.audit_epoch(env.plan_path, env.observation_path, env.commit_path)
