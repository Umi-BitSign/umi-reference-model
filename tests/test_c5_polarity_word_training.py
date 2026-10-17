from __future__ import annotations

import ast
import copy
import inspect
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_c5_polarity_word as audit
import audit_c5_word_lineage as old_audit
import c5_polarity_word_final as final
import c5_word_final as old_final
import terra_polarity_judge as judge
import test_c5_word_training as fixtures
import train_c5_polarity_words as training
from terra_training_recovery import artifact


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_full_fresh_fit_reuses_original_numeric_loop_and_adam_constructor():
    bound = training.bound_fit()
    assert bound.__code__ is fixtures.trainer.fit.__code__
    assert {
        key
        for key in bound.__globals__
        if bound.__globals__[key] is not fixtures.trainer.fit.__globals__[key]
    } == {"checked", "RecoveryPacer", "score_batch"}
    for name in (
        "load_student",
        "load_clip",
        "synchronize_visual",
        "update_visual",
        "save_checkpoint",
        "RECIPE",
        "STORAGE",
        "verify_coverage",
    ):
        assert bound.__globals__[name] is fixtures.trainer.fit.__globals__[name]


def test_independent_auditor_keeps_complete_sampler_and_350_tensor_adam_inventory():
    assert ast.dump(ast.parse(inspect.getsource(audit.optimizer_inventory))) == ast.dump(
        ast.parse(inspect.getsource(old_audit.optimizer_inventory))
    )

    def loops(module):
        tree = ast.parse(inspect.getsource(module.main))
        return [ast.dump(node) for node in tree.body[0].body if isinstance(node, ast.For)]

    assert loops(audit) == loops(old_audit)


def test_pending_proof_blocks_freeze_before_any_plan_or_optimizer(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "ROOT", tmp_path)

    def pending():
        raise ValueError("qualification incomplete")

    monkeypatch.setattr(training, "qualified", pending)
    monkeypatch.setattr(torch.optim, "AdamW", lambda *a, **k: pytest.fail("optimizer constructed"))
    target = tmp_path / "runs/fresh/plan.json"
    with pytest.raises(ValueError, match="qualification incomplete"):
        training.freeze(target)
    assert not target.parent.exists()


def test_training_gate_uses_the_fresh_proof_paths_and_actual_new_validator(tmp_path, monkeypatch):
    fixtures.proof_gate.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(training, "PROOF_PLAN", fixtures.trainer.PROOF_PLAN)
    monkeypatch.setattr(training, "PROOF", fixtures.trainer.PROOF)
    plan, proof, calls = {"new_protocol": True}, {"actual_new_proof": True}, []
    monkeypatch.setattr(training.proof, "checked", lambda path: plan)

    def validate(*args):
        calls.append(args)
        return proof

    monkeypatch.setattr(training.proof, "checked_qualification", validate)
    assert training.qualified() == (plan, proof)
    assert calls == [(training.PROOF, training.PROOF_PLAN, plan)]


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """Tiny real CPU Adam files; all public-C5/model qualification is stubbed explicitly."""
    env = fixtures.tiny_training.__wrapped__(tmp_path, monkeypatch)
    descriptor = partial(artifact, root=tmp_path)
    for module in (training, final, old_final):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "artifact", descriptor)
    plan = env.plan
    plan.update(
        schema=training.SCHEMA,
        recipe=copy.deepcopy(fixtures.word.RECIPE),
        unique_training_clips=len(plan["records"]),
        maximum_optimizer_updates=len(plan["batches"]),
    )
    proof = tmp_path / "runs/qualified/plan.json"
    put(proof, {"synthetic_qualification_only": True})
    plan["qualified_plan"] = descriptor(proof)
    scope = tmp_path / "runs/scope.json"
    put(
        scope,
        {
            "parameters": [
                {"parameter": name.removeprefix("vision."), "shape": list(value.shape)}
                for name, value in env.template.trainable_parameters().items()
            ]
        },
    )
    plan["model_plan"].update(visual_scope=descriptor(scope))
    plan["model_plan"]["reference"]["bundle_root"] = "artifacts/fixture"
    put(env.path, plan)
    monkeypatch.setattr(training, "checked", lambda path: plan)
    monkeypatch.setattr(old_final, "checked_training", lambda path: plan)
    resources = {
        "current_allowed_gpu_duty_fraction": 0.5,
        "current_cpu_threads": 2,
        "memory_policy": "framework_default_no_artificial_six_GiB_cap",
        "mps_active_seconds": 1.0,
        "mps_idle_seconds": 1.0,
        "maximum_allocated_bytes": 1,
        "maximum_driver_allocated_bytes": 1,
    }
    monkeypatch.setattr(fixtures.FakeDuty, "receipt", lambda self: resources.copy())
    from transformers import ByT5Tokenizer

    monkeypatch.setattr(ByT5Tokenizer, "from_pretrained", lambda *a, **k: fixtures.Tokenizer())
    return SimpleNamespace(
        env=env, root=tmp_path, descriptor=descriptor, proof=proof, monkeypatch=monkeypatch
    )


def run_fresh(case, *, fail=False):
    env = case.env
    case.monkeypatch.setattr(fixtures, "terra_payload", judge.payload)
    case.monkeypatch.setattr(training, "Pacer", lambda: env.pacer)
    env.pacer.fail_call = 1 if fail else None
    put(env.output / "worker.json", {"pid": 123})
    training.fit(env.path, env.output, env.plan)
    put(env.output / "exit.json", {"pid": 123, "exit_code": 0})


def test_complete_fresh_run_rejoins_new_raw_rewards_and_real_checkpoint_files(fresh):
    run_fresh(fresh)
    env = fresh.env
    plan, result, evidence = final.checked_final(env.path, env.output / "result.json")
    assert plan == env.plan and result["training_examples"] == 24
    assert result["optimizer_updates"] == 8 and len(result["checkpoints"]) == 4
    start = json.loads((env.output / "run.json").read_text())
    assert start["training_examples"] == start["optimizer_updates"] == 0
    assert len([item for item in evidence if item["path"].endswith("/response.json")]) == 8
    assert len([item for item in evidence if item["path"].endswith("/rewards.json")]) == 8
    assert len([item for item in evidence if item["path"].endswith("-optimizer.safetensors")]) == 4
    with pytest.raises(ValueError, match="authentic model, rubric"):
        old_final.checked_final(env.path, env.output / "result.json")


def test_same_reward_tiny_training_matches_original_weights_adam_and_rng_exactly(fresh):
    env = fresh.env
    legacy = env.output
    fixtures.trainer.fit(env.path, legacy, env.plan)
    env.output = fresh.root / "checkpoints/new"
    env.output.mkdir()
    run_fresh(fresh)
    for suffix in (".safetensors", "-optimizer.safetensors"):
        with safe_open(str(legacy / ("visual-after-000024" + suffix)), framework="pt") as before:
            with safe_open(
                str(env.output / ("visual-after-000024" + suffix)), framework="pt"
            ) as after:
                assert set(before.keys()) == set(after.keys())
                assert all(
                    torch.equal(before.get_tensor(name), after.get_tensor(name))
                    for name in before.keys()
                )


def test_one_proof_cannot_fork_after_calibration_failure(fresh):
    env = fresh.env
    with pytest.raises(ValueError, match="original calibration"):
        run_fresh(fresh, fail=True)
    stopped = json.loads((env.output / "stopped.json").read_text())
    assert stopped["training_examples"] == stopped["optimizer_updates"] == 0
    assert not (env.output / "result.json").exists()
    loaded = len(env.models)
    other = fresh.root / "checkpoints/alias"
    put(other / "worker.json", {"pid": 124})
    with pytest.raises(FileExistsError):
        training.fit(env.path, other, env.plan)
    assert len(env.models) == loaded


@pytest.mark.parametrize("mutation", ["resume", "partial", "nested_stop", "claim"])
def test_fresh_final_rejects_inheritance_partial_hidden_stop_or_claim_drift(fresh, mutation):
    run_fresh(fresh)
    env = fresh.env
    if mutation == "resume":
        value = json.loads(env.path.read_text())
        value["resume"] = {}
        put(env.path, value)
    elif mutation == "partial":
        value = json.loads((env.output / "result.json").read_text())
        value["training_examples"] -= 1
        put(env.output / "result.json", value)
    elif mutation == "nested_stop":
        put(env.output / "batch-0000/stopped.json", {})
    else:
        claim = env.path.with_name("execution-claim.json")
        value = json.loads(claim.read_text())
        value["inherited_exposures"] = 1
        put(claim, value)
    with pytest.raises(ValueError):
        final.checked_final(env.path, env.output / "result.json")
