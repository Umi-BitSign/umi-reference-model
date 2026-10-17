from __future__ import annotations

import ast
import copy
import inspect
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_c5_word_lineage
import c5_experiment_sources
import c5_word_epoch_audit
import c5_word_final
import c5_word_resume as recovery
import c5_word_resumed_final
import roi_semantic_resume
import test_c5_word_epoch_audit as epoch_fixtures
import test_c5_word_training as fixtures
import train_c5_word_resume as controller
from terra_training_recovery import artifact

UTC = timezone.utc  # noqa: UP017 - pinned model runtime is Python3.10


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def resources():
    return {
        "current_allowed_gpu_duty_fraction": 0.5,
        "current_cpu_threads": 2,
        "memory_policy": "framework_default_no_artificial_six_GiB_cap",
        "mps_active_seconds": 1.0,
        "mps_idle_seconds": 1.0,
        "maximum_allocated_bytes": 1024,
        "maximum_driver_allocated_bytes": 2048,
    }


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    """Real tiny CPU Adam/RNG files; a synthetic independent-audit binding, never C5 evidence."""
    env = fixtures.tiny_training.__wrapped__(tmp_path, monkeypatch)
    descriptor = partial(artifact, root=tmp_path)
    for module in (recovery, controller, c5_word_final, c5_word_resumed_final, roi_semantic_resume):
        monkeypatch.setattr(module, "ROOT", tmp_path)
        monkeypatch.setattr(module, "artifact", descriptor)
    monkeypatch.setattr(c5_experiment_sources, "ROOT", tmp_path)
    monkeypatch.setattr(c5_word_epoch_audit, "TENSORS", 2)
    monkeypatch.setattr(recovery, "TENSORS", 2)
    monkeypatch.setattr(
        recovery,
        "PARAMETERS",
        sum(value.numel() for value in env.template.trainable_parameters().values()),
    )
    monkeypatch.setattr(recovery, "storage", lambda *a, **k: {})
    monkeypatch.setattr(fixtures.FakeDuty, "receipt", lambda self: resources())
    env.plan.update(
        recipe=copy.deepcopy(fixtures.word.RECIPE),
        source_sha256={},
        maximum_optimizer_updates=len(env.plan["batches"]),
    )
    env.plan["model_plan"].update(
        assets={},
        reference={"translator_checkpoint_sha256": "original", "bundle_root": "artifacts/fixture"},
    )
    scope = env.path.with_name("scope.json")
    put(
        scope,
        {
            "parameters": [
                {"parameter": name.removeprefix("vision."), "shape": list(value.shape)}
                for name, value in env.template.trainable_parameters().items()
            ]
        },
    )
    env.plan["model_plan"]["visual_scope"] = descriptor(scope)
    put(env.path, env.plan)
    put(
        env.path.with_name("source-snapshot") / "manifest.json",
        {"plan_sha256": recovery.sha256(env.path), "source_sha256": {}},
    )
    env.pacer.fail_call = 3
    with pytest.raises(ValueError, match="calibration"):
        fixtures.trainer.fit(env.path, env.output, env.plan)
    put(env.output / "worker.json", {"pid": 111})
    put(
        env.output / "exit.json",
        {"pid": 111, "exit_code": 1, "observed_at": "2026-10-17T20:00:00+00:00"},
    )
    put(
        env.output / "failure.json",
        {"error": "word judge failed complete integer rewards or original calibration"},
    )
    steps = [json.loads(line) for line in (env.output / "steps.jsonl").read_text().splitlines()]
    assert len(steps) == 2 and steps[-1]["training_examples"] == 6
    source = env.path.with_name("synthetic-auditor.py")
    source.write_text("synthetic audit binding only")
    epochs = [descriptor(path) for path in sorted(env.output.glob("visual-after-*.json"))]
    observed = env.path.with_name("observation.json")
    put(
        observed,
        {
            "epoch_checkpoint_commits": epochs,
            "original_worker": descriptor(env.output / "worker.json"),
            "original_run": descriptor(env.output / "run.json"),
        },
    )
    snapshot = env.path.with_name("audited-steps.jsonl")
    snapshot.write_bytes((env.output / "steps.jsonl").read_bytes())
    checkpoint = env.output / "visual-recovery-after-000006.safetensors"
    optimizer = checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")
    evidence = [
        descriptor(path)
        for number in range(2)
        for path in sorted((env.output / f"batch-{number:04d}").rglob("*.json"))
    ]
    failed = [descriptor(path) for path in sorted((env.output / "batch-0002").rglob("*.json"))]
    audit_path = env.path.with_name("stop-audit.json")
    audit = {
        "schema": "umi-studio-original-word-stopped-terminal-audit/1",
        "audit_complete": True,
        "same_original_worker_exit1": True,
        "original_PID_absent": True,
        "plan": descriptor(env.path),
        "worker": descriptor(env.output / "worker.json"),
        "exit": descriptor(env.output / "exit.json"),
        "stopped": descriptor(env.output / "stopped.json"),
        "failure": descriptor(env.output / "failure.json"),
        "training_examples": 6,
        "optimizer_updates": 2,
        "completed_batches": 2,
        "failed_batch": 2,
        "failed_batch_rewards_admitted_or_optimizer_updated": False,
        "failed_batch_control_violations": [{"actual_score": 9, "range": [0, 0]}],
        "curriculum_complete": False,
        "fixed_final_evaluation_eligible": False,
        "all_pinned_sources_and_model_assets_match": True,
        "audit_source": descriptor(source),
        "previous_accepted_observation": descriptor(observed),
        "completed_steps_prefix": descriptor(snapshot),
        "actual_completed_evidence": evidence,
        "actual_failed_evidence": failed,
        "recovery_visual": descriptor(checkpoint),
        "recovery_optimizer": descriptor(optimizer),
        "recovery_commit": descriptor(checkpoint.with_suffix(".json")),
    }
    put(audit_path, audit)
    monkeypatch.setattr(recovery, "STOP_AUDIT_SHA256", recovery.sha256(audit_path))
    for name in controller.CONTROL_SOURCES:
        original_path = Path(controller.__file__).parent / name
        target = tmp_path / "tools" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_path, target)
    restored_mps = []
    monkeypatch.setattr(
        torch.mps, "set_rng_state", lambda value: restored_mps.append(value.clone())
    )
    return SimpleNamespace(
        env=env,
        checkpoint=checkpoint,
        audit_path=audit_path,
        audit=audit,
        descriptor=descriptor,
        restored_mps=restored_mps,
    )


def freeze(stopped):
    env = stopped.env
    path = env.path.parent.parent / "resume/plan.json"
    controller.freeze(path, env.path, stopped.checkpoint, stopped.audit_path)
    plan = controller.checked(path)
    return path, plan


def test_recovery_controller_retains_original_numeric_loop_and_optimizer_constructor():
    original = ast.parse(inspect.getsource(fixtures.trainer.fit)).body[0]
    derived = controller.derived_fit_ast()
    before = next(node for node in original.body if isinstance(node, ast.Try)).body[0]
    after = next(node for node in derived.body if isinstance(node, ast.Try)).body[0]
    assert ast.dump(ast.Module(body=before.body, type_ignores=[])) == ast.dump(
        ast.Module(body=after.body[1:], type_ignores=[])
    )
    assert (
        ast.unparse(after.body[0])
        == "if batch_number < plan['resume']['completed_batches']:\n    continue"
    )
    old_adam = next(
        node
        for node in original.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "optimizer" for target in node.targets
        )
    )
    new_adam = next(
        node
        for node in derived.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "optimizer" for target in node.targets
        )
    )
    assert ast.dump(old_adam) == ast.dump(new_adam)
    assert recovery.restore is roi_semantic_resume.restore


def test_independent_lineage_auditor_reads_real350_tensor_Adam_and_preserves_RNG(
    tmp_path, monkeypatch
):
    state = epoch_fixtures.state.__wrapped__(tmp_path)
    path = tmp_path / "checkpoints/optimizer.safetensors"
    path.parent.mkdir()
    shutil.copyfile(state.path, path)
    monkeypatch.setattr(audit_c5_word_lineage, "artifact", partial(artifact, root=tmp_path))
    before = torch.get_rng_state().clone()
    result = audit_c5_word_lineage.optimizer_inventory(path, state.shapes, 1, fixtures.word.RECIPE)
    assert result["parameter_tensors"] == 350 and result["all_Adam_steps"] == 1
    assert torch.equal(before, torch.get_rng_state())


def test_complete_binding_and_prefix_copy_preserve_real_files_and_exclude_rejected_batch(stopped):
    path, plan = freeze(stopped)
    binding = plan["resume"]
    assert (
        binding["training_examples"],
        binding["optimizer_updates"],
        binding["completed_batches"],
    ) == (6, 2, 2)
    output = stopped.env.output.with_name("resumed")
    output.mkdir()
    record = recovery.inherit_prefix(output, plan)
    assert not (output / "batch-0002").exists()
    assert not (output / "failure.json").exists()
    assert (output / "steps.jsonl").read_bytes() == (
        stopped.env.output / "steps.jsonl"
    ).read_bytes()
    assert len(record["epochs"]) == 1
    checked, inherited = recovery.validate_inheritance(output, plan)
    assert checked == record and inherited[6]["source_plan_sha256"] == recovery.sha256(
        stopped.env.path
    )
    assert (output / "visual-after-000006.safetensors").stat().st_ino != (
        stopped.env.output / "visual-after-000006.safetensors"
    ).stat().st_ino
    put(output / "worker.json", {"pid": 222})
    recovery.claim(path, output, plan)
    with pytest.raises((ValueError, FileExistsError)):
        recovery.claim(path, output, plan)


@pytest.mark.parametrize(
    "mutation", ["stop", "source_file", "optimizer", "audit", "epoch", "resume_chain", "ledger"]
)
def test_changed_or_ambiguous_source_cannot_authorize_recovery(stopped, mutation):
    env = stopped.env
    if mutation == "stop":
        record = json.loads((env.output / "stopped.json").read_text())
        record["uncommitted_optimizer_mutation_possible"] = True
        put(env.output / "stopped.json", record)
    elif mutation == "source_file":
        first = stopped.audit["actual_completed_evidence"][0]
        (recovery.ROOT / first["path"]).write_text("changed")
    elif mutation == "optimizer":
        path = recovery.ROOT / stopped.audit["recovery_optimizer"]["path"]
        path.write_bytes(b"changed Adam bytes")
    elif mutation == "audit":
        put(stopped.audit_path, {})
    elif mutation == "epoch":
        (env.output / "visual-after-000006.json").write_text("{}")
    elif mutation == "resume_chain":
        env.plan["resume"] = {}
    else:
        path = env.output / "steps.jsonl"
        path.write_bytes(path.read_bytes() + b"{")
    with pytest.raises((ValueError, RuntimeError)):
        recovery.make_binding(env.path, stopped.checkpoint, stopped.audit_path)


def test_tiny_state_exact_continuation_matches_uninterrupted_weights_Adam_and_CPU_RNG(
    stopped, monkeypatch
):
    env = stopped.env
    path, plan = freeze(stopped)
    output = env.output.with_name("resumed")
    output.mkdir()
    put(output / "worker.json", {"pid": 222})
    next_pacer = fixtures.FakePacer()
    monkeypatch.setattr(fixtures.trainer, "RecoveryPacer", lambda: next_pacer)
    controller.fit(path, output, plan)
    continued_model = env.models[2]
    continued_optimizer = env.optimizers[1]
    actual_rng = torch.get_rng_state().clone()
    result = json.loads((output / "result.json").read_text())
    assert result["training_examples"] == 24 and result["optimizer_updates"] == 8
    assert len(result["checkpoints"]) == 4
    assert next_pacer.calls == 6 and len(stopped.restored_mps) == 1
    source_rng = load_file(
        str(stopped.checkpoint.with_name(stopped.checkpoint.stem + "-optimizer.safetensors"))
    )
    assert torch.equal(stopped.restored_mps[0], source_rng["rng.mps"])
    reference_path = env.path.parent.parent / "reference/plan.json"
    put(reference_path, env.plan)
    reference_output = env.output.with_name("reference")
    reference_output.mkdir()
    monkeypatch.setattr(fixtures.trainer, "RecoveryPacer", fixtures.FakePacer)
    fixtures.trainer.fit(reference_path, reference_output, env.plan)
    reference_model = env.models[4]
    reference_optimizer = env.optimizers[2]
    assert torch.equal(actual_rng, torch.get_rng_state())
    for name, parameter in continued_model.trainable_parameters().items():
        target = reference_model.trainable_parameters()[name]
        torch.testing.assert_close(parameter, target, rtol=0, atol=0)
        for field in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                continued_optimizer.state[parameter][field],
                reference_optimizer.state[target][field],
                rtol=0,
                atol=0,
            )


def test_source_request_cooldown_keeps_exact_sixty_seconds_and_short_waits(stopped):
    binding = recovery.make_binding(stopped.env.path, stopped.checkpoint, stopped.audit_path)
    current = datetime(2026, 10, 17, 20, tzinfo=UTC)
    waits = []

    def sleep(seconds):
        nonlocal current
        waits.append(seconds)
        current += timedelta(seconds=seconds)

    record = recovery.source_cooldown(binding, clock=lambda: current, sleep=sleep)
    assert waits == [30, 30]
    assert record["ready_at"] == record["not_before"] == "2026-10-17T20:01:00+00:00"


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "partial",
        "counter",
        "inherited_file",
        "copied_commit",
        "claim",
        "source_file",
        "control_source",
        "run_counter",
        "missing_new_batch",
    ],
)
def test_continued_final_requires_full_inherited_and_new_actual_evidence(
    stopped, monkeypatch, mutation
):
    from transformers import ByT5Tokenizer

    env = stopped.env
    path, plan = freeze(stopped)
    output = env.output.with_name("continued-final")
    output.mkdir()
    put(output / "worker.json", {"pid": 222})
    monkeypatch.setattr(fixtures.trainer, "RecoveryPacer", fixtures.FakePacer)
    monkeypatch.setattr(ByT5Tokenizer, "from_pretrained", lambda *a, **k: fixtures.Tokenizer())
    monkeypatch.setattr(
        fixtures.FakeDuty,
        "receipt",
        lambda self: {**resources(), "mps_active_seconds": 0.5, "mps_idle_seconds": 0.5},
    )
    controller.fit(path, output, plan)
    put(output / "exit.json", {"pid": 222, "exit_code": 0})
    result_path = output / "result.json"
    if mutation == "partial":
        result = json.loads(result_path.read_text())
        result["training_examples"] = 18
        put(result_path, result)
    elif mutation == "counter":
        steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
        steps[-1]["optimizer_updates"] = 7
        (output / "steps.jsonl").write_text("".join(json.dumps(step) + "\n" for step in steps))
    elif mutation == "inherited_file":
        put(output / "batch-0000/rewards.json", {})
    elif mutation == "copied_commit":
        put(output / "visual-after-000006.json", {})
    elif mutation == "claim":
        claim = env.path.with_name("word-manual-continuation.json")
        put(claim, {"plan": stopped.descriptor(env.path)})
    elif mutation == "source_file":
        source = recovery.ROOT / stopped.audit["actual_completed_evidence"][0]["path"]
        source.write_text("changed source evidence")
    elif mutation == "control_source":
        (controller.ROOT / "tools/c5_word_resume.py").write_text("changed control source")
    elif mutation == "run_counter":
        run = json.loads((output / "run.json").read_text())
        run["training_examples"] = 0
        put(output / "run.json", run)
    elif mutation == "missing_new_batch":
        shutil.rmtree(output / "batch-0002")
    if mutation is None:
        actual_plan, result, evidence = c5_word_final.checked_final(path, result_path)
        assert (
            actual_plan == plan
            and result["training_examples"] == 24
            and result["optimizer_updates"] == 8
        )
        assert len([item for item in evidence if item["path"].endswith("/response.json")]) == 8
        run = json.loads((output / "run.json").read_text())
        assert run["training_examples"] == 6 and run["optimizer_updates"] == 2
        assert run["resume"]["source_exit"] == stopped.descriptor(env.output / "exit.json")
        _, inherited = recovery.validate_inheritance(output, plan)
        steps = [json.loads(line) for line in (output / "steps.jsonl").read_text().splitlines()]
        with pytest.raises(ValueError, match="backwards"):
            c5_word_final.verify_steps(plan, steps, result)
        with pytest.raises(ValueError, match="handoff"):
            c5_word_final.verify_steps(plan, steps, result, resource_reset_batches={3})
        inherited[6]["source_plan_sha256"] = "wrong-plan"
        with pytest.raises(ValueError, match="another plan"):
            c5_word_final.checked_checkpoints(
                plan, result, output, recovery.sha256(path), steps, inherited=inherited
            )
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            c5_word_final.checked_final(path, result_path)
