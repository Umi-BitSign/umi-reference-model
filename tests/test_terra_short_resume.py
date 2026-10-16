from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import terra_short_resume as module
from benchmark_fleurs_dev import sha256
from terra_short_curriculum_learning import RECIPE


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def recovery(monkeypatch, tmp_path, request):
    root = tmp_path
    previous = root / "runs/previous/plan.json"
    snapshot = previous.with_name("source-snapshot")
    code = {}
    for name, contents in (
        ("train_terra_short_curriculum.py", "old-controller"),
        ("c5_terra_tail.py", "frozen-math"),
    ):
        path = snapshot / "tools" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
        code["tools/" + name] = sha256(path)
    data = root / "runs/data/plan.json"
    records = [
        {
            "role": "training" if i < 16 else "development",
            "training_allowed": i < 16,
            "original_pair": {"raw": {"source": "stem", "example_id": str(i)}},
        }
        for i in range(18)
    ]
    put(data, {"records": records})
    old = {
        "schema": "umi-studio-terra-short-pilot/1",
        "recipe": copy.deepcopy(RECIPE),
        "source_sha256": code,
        "data": {"path": str(data.relative_to(root)), "sha256": sha256(data)},
        "reference": {"translator_checkpoint_sha256": "C5"},
        "planned_training_examples": 64,
        "checks_after_examples": [8, 32, 64]
        if getattr(request, "param", None) == "pending_gate"
        else [16, 32, 64],
    }
    put(previous, old)
    put(snapshot / "manifest.json", {"plan_sha256": sha256(previous), "source_sha256": code})
    current = copy.deepcopy(old)
    current["source_sha256"]["tools/train_terra_short_curriculum.py"] = "new-controller"
    current["source_sha256"]["tools/terra_short_resume.py"] = "new-resume-controller"
    output = root / "checkpoints/stopped"
    output.mkdir(parents=True)
    checkpoint = output / "visual-recovery-after-000008.safetensors"
    optimizer = output / "visual-recovery-after-000008-optimizer.safetensors"
    provenance = {"plan_sha256": sha256(previous), "update": "8"}
    save_file(
        {"q": torch.ones(2)},
        str(checkpoint),
        metadata={**provenance, "optimizer_updates": "1", "fixed_winner_checkpoint": "C5"},
    )
    save_file({"q.step": torch.tensor(1.0)}, str(optimizer), metadata=provenance)
    put(
        checkpoint.with_suffix(".json"),
        {
            "complete": True,
            "training_examples": 8,
            "optimizer_updates": 1,
            "visual": module.artifact(checkpoint, root=root),
            "optimizer": module.artifact(optimizer, root=root),
        },
    )
    put(output / "run.json", {"plan_sha256": sha256(previous), "resume": None})
    put(
        output / "worker.json",
        {
            "pid": 999999,
            "uid": 501,
            "ps_started_local": "saved-start",
            "argv": [
                "python",
                "tools/train_terra_short_curriculum.py",
                "--plan",
                str(previous),
                "--output-dir",
                str(output),
            ],
        },
    )
    put(
        output / "exit.json",
        {"pid": 999999, "exit_code": 1, "observed_at": "2026-10-16T08:21:04.910547+00:00"},
    )
    put(
        output / "preupdate-stop.json",
        {
            "stage": "terra_scoring",
            "training_examples": 8,
            "optimizer_updates": 1,
            "failed_batch": 2,
            "failed_batch_optimizer_applied": False,
            "automatic_retries": 0,
        },
    )
    put(
        output / "terra/batch-000002/failure.json",
        {
            "http_status": 502,
            "ambiguous_completion": False,
            "message": json.dumps({"error": {"code": "server_is_overloaded"}}),
        },
    )
    put(output / "terra/batch-000002/request.json", {"model": "gpt-5.6-terra"})
    ordered = module.order(records)
    put(output / "steps.jsonl", {"training_examples": 8, "optimizer_update": 1})
    put(
        output / "candidate-pools.jsonl",
        {
            "optimizer_update": 1,
            "identities": [{k: p["raw"][k] for k in ("source", "example_id")} for p in ordered[:8]],
        },
    )
    monkeypatch.setattr(module, "worker_still_present", lambda _: False)
    return root, previous, checkpoint, current


def test_complete_recovery_binds_offsets_all_artifacts_and_cooldown(recovery):
    root, previous, checkpoint, current = recovery
    value = module.make_resume(previous, checkpoint, current, root=root)
    assert value["training_examples"] == 8 and value["optimizer_updates"] == 1
    assert value["unsaved_examples_to_recompute"] == 0
    assert value["minimum_request_gap_seconds"] == 60
    assert value["not_before"] == "2026-10-16T08:51:04.910547+00:00"
    assert value["automatic_retries"] == 0
    restored = module.paths({"resume": value}, root=root)
    assert restored["visual"] == checkpoint


@pytest.mark.parametrize("kind", ["loss", "data", "gate", "generation", "numerical_source"])
def test_recovery_cannot_change_scientific_recipe_or_components(recovery, kind):
    root, previous, checkpoint, current = recovery
    if kind == "loss":
        current["recipe"]["caption_weight"] = 0
    elif kind == "data":
        current["data"]["sha256"] = "changed"
    elif kind == "gate":
        current["checks_after_examples"] = [8]
    elif kind == "generation":
        current["reference"]["generation"] = {"num_beams": 1}
    else:
        current["source_sha256"]["tools/c5_terra_tail.py"] = "changed"
    with pytest.raises(ValueError):
        module.make_resume(previous, checkpoint, current, root=root)


@pytest.mark.parametrize(
    "kind",
    [
        "policy_bytes",
        "optimizer_bytes",
        "sampler",
        "counter",
        "worker",
        "terminal",
        "partial_step",
        "snapshot",
    ],
)
def test_partial_or_changed_state_and_sampler_are_not_silently_adopted(recovery, kind):
    root, previous, checkpoint, current = recovery
    output = checkpoint.parent
    if kind.endswith("bytes"):
        target = (
            checkpoint
            if kind == "policy_bytes"
            else checkpoint.with_name(checkpoint.stem + "-optimizer.safetensors")
        )
        target.write_bytes(target.read_bytes() + b"changed")
    elif kind == "sampler":
        p = output / "candidate-pools.jsonl"
        value = json.loads(p.read_text())
        value["identities"].reverse()
        put(p, value)
    elif kind == "counter":
        put(output / "steps.jsonl", {"training_examples": 7, "optimizer_update": 1})
    elif kind == "worker":
        p = output / "worker.json"
        value = json.loads(p.read_text())
        value["argv"][-1] = str(root / "checkpoints/unrelated")
        put(p, value)
    elif kind == "terminal":
        p = output / "exit.json"
        value = json.loads(p.read_text())
        value["pid"] = 1
        put(p, value)
    elif kind == "snapshot":
        (previous.with_name("source-snapshot") / "tools/c5_terra_tail.py").write_text(
            "changed math"
        )
    else:
        with (output / "steps.jsonl").open("a") as stream:
            stream.write('\n{"training_examples": 16, "optimizer_update": 2}')
    with pytest.raises(ValueError):
        module.make_resume(previous, checkpoint, current, root=root)


@pytest.mark.parametrize("kind", ["ambiguous", "wrong_status", "wrong_error", "applied", "live"])
def test_only_an_unambiguous_terminal_overload_can_be_continued(recovery, monkeypatch, kind):
    root, previous, checkpoint, current = recovery
    failure = checkpoint.parent / "terra/batch-000002/failure.json"
    value = json.loads(failure.read_text())
    if kind == "ambiguous":
        value["ambiguous_completion"] = True
    elif kind == "wrong_status":
        value["http_status"] = 429
    elif kind == "wrong_error":
        value["message"] = json.dumps({"error": {"code": "invalid_request"}})
    elif kind == "live":
        monkeypatch.setattr(module, "worker_still_present", lambda _: True)
    else:
        stop = checkpoint.parent / "preupdate-stop.json"
        saved = json.loads(stop.read_text())
        saved["failed_batch_optimizer_applied"] = True
        put(stop, saved)
    put(failure, value)
    with pytest.raises(ValueError):
        module.make_resume(previous, checkpoint, current, root=root)


@pytest.mark.parametrize("recovery", ["pending_gate"], indirect=True)
def test_pending_semantic_check_cannot_be_skipped(recovery):
    root, previous, checkpoint, current = recovery
    with pytest.raises(ValueError, match="pending semantic"):
        module.make_resume(previous, checkpoint, current, root=root)


@pytest.mark.parametrize("recovery", ["pending_gate"], indirect=True)
@pytest.mark.parametrize("kind", ["passed", "mastered", "uncalibrated"])
def test_existing_semantic_decision_is_preserved_and_stops_cannot_be_overridden(recovery, kind):
    root, previous, checkpoint, current = recovery
    decision = {
        "training_examples": 8,
        "complete_scoring": True,
        "calibration_passed": kind != "uncalibrated",
        "stop_for_semantic_decline": False,
        "stop_for_failed_calibration": False,
        "control_stage_mastered": kind == "mastered",
        "candidate_stage_mastered": False,
    }
    put(checkpoint.parent / "semantic-decisions.jsonl", decision)
    if kind == "passed":
        resumed = module.make_resume(previous, checkpoint, current, root=root)
        assert "source_decisions" in resumed
    else:
        with pytest.raises(ValueError, match="semantic stop"):
            module.make_resume(previous, checkpoint, current, root=root)


def test_cooldown_and_more_conservative_pacing_do_not_retry(monkeypatch, tmp_path):
    import terra_quality_training

    with pytest.raises(RuntimeError, match="cooldown"):
        module.check_cooldown({"resume": {"not_before": "9999-01-01T00:00:00+00:00"}})
    module.check_cooldown({"resume": {"not_before": "2000-01-01T00:00:00+00:00"}})
    clock = iter([0, 10, 60, 60])
    sleeps, calls = [], []
    pacer = module.RecoveryPacer(clock=lambda: next(clock), sleep=sleeps.append)
    monkeypatch.setattr(
        terra_quality_training, "score_with_terra", lambda rows, output: calls.append(rows)
    )
    pacer.score([{}], tmp_path / "one")
    pacer.score([{}], tmp_path / "two")
    assert sleeps == [50] and len(calls) == 2


def test_source_run_cannot_invent_a_restored_prefix(recovery):
    root, previous, checkpoint, current = recovery
    path = checkpoint.parent / "run.json"
    value = json.loads(path.read_text())
    value["resume"] = {"training_examples": 8, "optimizer_updates": 1}
    put(path, value)
    with pytest.raises(ValueError, match="restored prefix"):
        module.make_resume(previous, checkpoint, current, root=root)


def test_zero_new_updates_preserve_parent_prefix_and_use_latest_failure_cooldown(
    recovery, monkeypatch
):
    root, previous, checkpoint, current = recovery
    first = module.make_resume(previous, checkpoint, current, root=root)
    second_plan = root / "runs/second/plan.json"
    put(second_plan, {**current, "resume": first})
    second_output = root / "checkpoints/second-stop"
    shutil.copytree(checkpoint.parent, second_output)
    second_checkpoint = second_output / checkpoint.name
    second_optimizer = second_output / (checkpoint.stem + "-optimizer.safetensors")
    provenance = {"plan_sha256": sha256(second_plan), "update": "8"}
    save_file(
        {"q": torch.ones(2)},
        str(second_checkpoint),
        metadata={**provenance, "optimizer_updates": "1", "fixed_winner_checkpoint": "C5"},
    )
    save_file({"q.step": torch.tensor(1.0)}, str(second_optimizer), metadata=provenance)
    put(
        second_checkpoint.with_suffix(".json"),
        {
            "complete": True,
            "training_examples": 8,
            "optimizer_updates": 1,
            "visual": module.artifact(second_checkpoint, root=root),
            "optimizer": module.artifact(second_optimizer, root=root),
        },
    )
    put(second_output / "run.json", {"plan_sha256": sha256(second_plan), "resume": first})
    worker = json.loads((second_output / "worker.json").read_text())
    worker["argv"] = [
        "python",
        "tools/train_terra_short_curriculum.py",
        "--plan",
        str(second_plan),
        "--output-dir",
        str(second_output),
    ]
    put(second_output / "worker.json", worker)
    put(
        second_output / "exit.json",
        {"pid": worker["pid"], "exit_code": 1, "observed_at": "2026-10-16T09:21:04+00:00"},
    )
    (second_output / "steps.jsonl").write_text("")
    (second_output / "candidate-pools.jsonl").write_text("")
    # The snapshot mechanism is exercised above. This case isolates a second
    # controller's empty new logs while validating its real parent artifacts.
    original_verify = module.verify_snapshot
    monkeypatch.setattr(
        module, "verify_snapshot", lambda p, s: None if p == second_plan else original_verify(p, s)
    )
    resumed = module.make_resume(second_plan, second_checkpoint, current, root=root)
    assert resumed["training_examples"] == 8 and resumed["optimizer_updates"] == 1
    assert resumed["not_before"] == "2026-10-16T09:51:04+00:00"
