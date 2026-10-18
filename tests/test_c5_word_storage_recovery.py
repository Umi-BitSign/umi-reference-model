"""Synthetic authorization, prefix, and CPU/file orchestration tests only."""

from __future__ import annotations

import ast
import copy
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_word_storage_recovery as recovery
import train_c5_word_resume as old_controller
import train_c5_word_storage_recovery as controller
from terra_polarity_judge import Pacer, score_training_batch
from terra_training_recovery import artifact


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def approval(tmp_path, monkeypatch):
    proposal = tmp_path / "runs/proposal/proposal.json"
    data = {
        "visual": {
            "path": "checkpoints/parent/visual-after-000428.safetensors",
            "sha256": "a" * 64,
        },
        "inherited_exposures": 428,
        "rolled_back_accepted_exposures": 104,
        "actual_total_lineage_exposures_if_complete": 960,
    }
    put(proposal, data)
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "PROPOSAL", proposal)
    monkeypatch.setattr(recovery, "PROPOSAL_SHA", recovery.sha256(proposal))
    monkeypatch.setattr(recovery, "artifact", descriptor)
    path = tmp_path / "runs/proposal/authorization.json"
    authorization = {
        "scope": recovery.SCOPE,
        "approved": True,
        "source": "explicit_user_message",
        "user_message": "Synthetic fixture approval; never a production approval.",
        "proposal": descriptor(proposal),
        "source_checkpoint": data["visual"],
        "manual_continuation_limit": 1,
        "automatic_retries": 0,
        "automatic_further_continuations": 0,
        "received_at": "2026-10-18T00:00:00+00:00",
    }
    put(path, authorization)
    return SimpleNamespace(
        root=tmp_path, path=path, data=authorization, proposal=proposal, descriptor=descriptor
    )


def test_explicit_exact_scope_approval_is_bound_without_time_expiry(approval):
    descriptor, proposal = recovery.checked_authorization(approval.path)
    assert descriptor == approval.descriptor(approval.path)
    assert proposal["actual_total_lineage_exposures_if_complete"] == 960


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "delete-one-local-time-machine-snapshot"),
        ("approved", False),
        ("source", "automatic_goal_continuation"),
        ("user_message", ""),
        ("manual_continuation_limit", 2),
        ("manual_continuation_limit", True),
        ("automatic_retries", 1),
        ("automatic_retries", False),
        ("automatic_further_continuations", 1),
    ],
)
def test_other_approvals_or_relaxed_limits_cannot_authorize_training(approval, field, value):
    changed = {**approval.data, field: value}
    put(approval.path, changed)
    with pytest.raises(ValueError, match="explicit one-time rollback approval"):
        recovery.checked_authorization(approval.path)


def test_approval_cannot_change_checkpoint_or_proposal(approval):
    put(
        approval.path, {**approval.data, "source_checkpoint": {"path": "other", "sha256": "b" * 64}}
    )
    with pytest.raises(ValueError, match="rollback approval"):
        recovery.checked_authorization(approval.path)
    put(approval.path, approval.data)
    proposal = json.loads(approval.proposal.read_text())
    put(approval.proposal, {**proposal, "inherited_exposures": 532})
    with pytest.raises(ValueError, match="proposal changed"):
        recovery.checked_authorization(approval.path)


def test_timezone_is_required_for_the_approval_record(approval):
    put(approval.path, {**approval.data, "received_at": "2026-10-18T00:00:00"})
    with pytest.raises(ValueError, match="timezone-aware"):
        recovery.checked_authorization(approval.path)


def test_missing_approval_stops_before_source_or_model_work(approval, monkeypatch):
    approval.path.unlink()
    monkeypatch.setattr(
        recovery.training, "checked", lambda _: pytest.fail("source/model work reached")
    )
    with pytest.raises(ValueError, match="authorization is required"):
        recovery.make_binding(
            approval.root / "source",
            approval.root / "checkpoint",
            approval.root / "audit",
            approval.path,
        )


def test_missing_approval_freeze_creates_no_plan_or_output(approval, monkeypatch):
    approval.path.unlink()
    monkeypatch.setattr(controller, "ROOT", approval.root)
    monkeypatch.setattr(
        controller.word, "storage", lambda *a, **k: pytest.fail("storage/model work reached")
    )
    path = approval.root / "runs/not-created/plan.json"
    with pytest.raises(ValueError, match="authorization is required"):
        controller.freeze(path, Path("source"), Path("checkpoint"), Path("audit"), approval.path)
    assert not path.parent.exists()


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


def small_prefix():
    batches = [
        {"epoch": epoch, "training_record_indices": list(indices)}
        for epoch in range(2)
        for indices in ((0, 1), (2, 3))
    ]
    steps = [
        {
            **batch,
            "batch": index,
            "training_examples": 2 * (index + 1),
            "optimizer_updates": index + 1,
            "Terra_qualities": [[10, 0], [10, 0]],
            "contrasting_clips": 2,
            "gradient_norm_before_clip": 1.0,
            "resources": resources(),
        }
        for index, batch in enumerate(batches[:3])
    ]
    return {"batches": batches}, steps


def test_rollback_uses_a_complete_earlier_epoch_with_truthful_lost_work():
    plan, steps = small_prefix()
    result = recovery.rollback_prefix(plan, steps, 4)
    assert result == {
        "training_examples": 4,
        "optimizer_updates": 2,
        "completed_batches": 2,
        "actual_source_training_examples": 6,
        "actual_source_optimizer_updates": 3,
        "rolled_back_accepted_exposures": 2,
        "rolled_back_optimizer_updates": 1,
    }


@pytest.mark.parametrize("examples", [2, 6, 532])
def test_current_partial_or_nonexistent_boundary_cannot_be_inherited(examples):
    plan, steps = small_prefix()
    with pytest.raises(ValueError):
        recovery.rollback_prefix(plan, steps, examples)


def test_corrupt_parent_counter_prevents_prefix_selection():
    plan, steps = small_prefix()
    steps[-1]["optimizer_updates"] = 99
    with pytest.raises(ValueError, match="counter"):
        recovery.rollback_prefix(plan, steps, 4)


def test_derived_loop_only_adds_working_save_and_result_accounting():
    expected = old_controller.derived_fit_ast()
    actual = controller.derived_fit_ast()
    removed = {"save": 0, "result": 0}
    for node in ast.walk(actual):
        if isinstance(node, ast.For):
            kept = []
            for item in node.body:
                if (
                    isinstance(item, ast.Expr)
                    and isinstance(item.value, ast.Call)
                    and isinstance(item.value.func, ast.Name)
                    and item.value.func.id == "commit_working_checkpoint"
                ):
                    removed["save"] += 1
                else:
                    kept.append(item)
            node.body = kept
        if isinstance(node, ast.Dict):
            for index in range(len(node.keys) - 1, -1, -1):
                key = node.keys[index]
                if isinstance(key, ast.Constant) and key.value == "lineage_accounting":
                    del node.keys[index]
                    del node.values[index]
                    removed["result"] += 1
    assert removed == {"save": 1, "result": 1}
    assert ast.dump(actual, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def test_actual_bound_globals_keep_the_polarity_scorer_and_original_numerics():
    fit = controller.bound_fit()
    assert fit.__globals__["RecoveryPacer"] is Pacer
    assert fit.__globals__["score_batch"] is score_training_batch
    assert fit.__globals__["update_visual"] is controller.original.update_visual
    assert fit.__globals__["synchronize_visual"] is controller.original.synchronize_visual
    assert fit.__globals__["commit_working_checkpoint"] is controller.commit_working_checkpoint


def test_full_result_counts_both_actual_attempts_separately():
    plan = {
        "resume": {
            "actual_source_training_examples": 532,
            "training_examples": 428,
            "rolled_back_accepted_exposures": 104,
            "actual_source_optimizer_updates": 134,
            "optimizer_updates": 108,
            "rolled_back_optimizer_updates": 26,
        }
    }
    value = controller.lineage_counts(plan, 856, 216)
    assert value["actual_total_lineage_exposures"] == 960
    assert value["selected_model_trajectory_exposures"] == 856
    assert value["worker_exposures"] == 428
    assert value["actual_total_lineage_optimizer_updates"] == 242


class CpuDuty:
    def run(self, operation):
        return operation()


def test_real_tiny_cpu_adam_serialization_preserves_values_moments_and_rng(tmp_path, monkeypatch):
    """One-parameter CPU fixture; never a real C5 restore or MPS qualification."""
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller.word, "storage", lambda *a, **k: {})
    monkeypatch.setattr(controller.word, "artifact", partial(artifact, root=tmp_path))
    output = tmp_path / "checkpoints/cpu-fixture"
    output.mkdir(parents=True)
    plan_path = tmp_path / "runs/cpu-fixture/plan.json"
    put(plan_path, {"model_plan": {"reference": {"translator_checkpoint_sha256": "fixture"}}})
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    model = SimpleNamespace(trainable_parameters=lambda: {"vision.weight": parameter})
    optimizer = torch.optim.AdamW([parameter], lr=0.01, foreach=False)
    torch.manual_seed(17)
    mps_bytes = torch.tensor([4, 5, 6], dtype=torch.uint8)

    def update(examples, count):
        optimizer.zero_grad()
        (parameter**2).sum().backward()
        optimizer.step()
        with (output / "steps.jsonl").open("a") as stream:
            stream.write(
                json.dumps(
                    {"batch": count - 1, "training_examples": examples, "optimizer_updates": count}
                )
                + "\n"
            )

    update(4, 1)
    rng = {"cpu": torch.get_rng_state(), "mps": mps_bytes}
    seed = controller.word.save_checkpoint(
        model, optimizer, output, plan_path, "fixture", 4, 1, rng
    )
    controller.store_for(plan_path, output).seed(seed.with_suffix(".json"))
    update(8, 2)
    before_value = parameter.detach().clone()
    before_state = {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in optimizer.state[parameter].items()
    }
    before_rng = torch.get_rng_state().clone()
    result = controller.commit_working_checkpoint(
        plan_path, output, model, optimizer, CpuDuty(), 8, 2, {"cpu": before_rng, "mps": mps_bytes}
    )
    assert torch.equal(parameter, before_value) and torch.equal(torch.get_rng_state(), before_rng)
    assert all(
        torch.equal(optimizer.state[parameter][key], value) for key, value in before_state.items()
    )
    saved = load_file(str(tmp_path / result["visual"]["path"]))
    assert torch.equal(saved["vision.weight"], parameter)
    # Simulate loss of future-write headroom: recovery uses the existing pair,
    # without calling the normal checkpoint writer or another storage guard.
    monkeypatch.setattr(
        controller.word, "save_checkpoint", lambda *a, **k: pytest.fail("rewrote recovery")
    )
    monkeypatch.setattr(
        controller.word, "storage", lambda *a, **k: pytest.fail("needed more space")
    )
    recovered = controller.save_checkpoint(
        model, optimizer, output, plan_path, "fixture", 8, 2, {}, recovery=True
    )
    assert recovered == tmp_path / result["visual"]["path"]
    with pytest.raises(ValueError, match="behind current"):
        controller.save_checkpoint(
            model, optimizer, output, plan_path, "fixture", 12, 3, {}, recovery=True
        )


def test_restore_checks_both_rng_byte_vectors_after_the_old_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    path = tmp_path / "optimizer.safetensors"
    expected = {
        "cpu": torch.tensor([1, 2], dtype=torch.uint8),
        "mps": torch.tensor([3, 4], dtype=torch.uint8),
    }
    save_file({"rng." + key: value for key, value in expected.items()}, str(path))
    plan = {"resume": {"optimizer": {"path": path.name}}}
    monkeypatch.setattr(recovery.previous, "restore_committed", lambda *args: (428, 108))
    monkeypatch.setattr(torch, "get_rng_state", lambda: expected["cpu"].clone())
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: expected["mps"].clone())
    assert recovery.restore_committed(plan, None, None, CpuDuty()) == (428, 108)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: torch.tensor([9, 9], dtype=torch.uint8))
    with pytest.raises(ValueError, match="RNG bytes differ"):
        recovery.restore_committed(plan, None, None, CpuDuty())


@pytest.fixture
def inherited(tmp_path, monkeypatch):
    """Synthetic immutable134-batch source, with a chosen108-batch prefix."""
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    monkeypatch.setattr(recovery, "artifact", descriptor)
    monkeypatch.setattr(recovery, "storage", lambda *a, **k: {})
    parent, output = tmp_path / "checkpoints/parent", tmp_path / "checkpoints/child"
    parent.mkdir(parents=True)
    output.mkdir()
    source_plan = tmp_path / "runs/parent/plan.json"
    put(source_plan, {"synthetic": True})
    rows, evidence, examples = [], [], 0
    for number in range(134):
        examples += 2 if number in (53, 107) else 4
        rows.append(
            {"batch": number, "training_examples": examples, "optimizer_updates": number + 1}
        )
        path = parent / f"batch-{number:04d}/pool-00.json"
        put(path, {"synthetic_batch": number})
        if number < 108:
            evidence.append(descriptor(path))
    actual_steps = parent / "steps.jsonl"
    actual_steps.write_text("".join(json.dumps(row) + "\n" for row in rows))
    prefix_path = tmp_path / "runs/proposal/candidate-prefix428.jsonl"
    prefix_path.parent.mkdir(parents=True)
    prefix_path.write_text("".join(json.dumps(row) + "\n" for row in rows[:108]))
    epochs = []
    for exposure in (214, 428):
        visual, optimizer = (
            parent / f"visual-after-{exposure:06d}.safetensors",
            parent / f"visual-after-{exposure:06d}-optimizer.safetensors",
        )
        visual.write_bytes(f"synthetic visual {exposure}".encode())
        optimizer.write_bytes(f"synthetic optimizer {exposure}".encode())
        commit = visual.with_suffix(".json")
        put(
            commit,
            {
                "complete": True,
                "committed_batch_RNG": True,
                "training_examples": exposure,
                "optimizer_updates": exposure // 214 * 54,
                "visual": descriptor(visual),
                "optimizer": descriptor(optimizer),
            },
        )
        epochs.append(
            {
                "training_examples": exposure,
                "visual": descriptor(visual),
                "optimizer": descriptor(optimizer),
                "commit": descriptor(commit),
            }
        )
    binding = {
        "source_plan": descriptor(source_plan),
        "source_steps": descriptor(prefix_path),
        "actual_source_steps": descriptor(actual_steps),
        "source_output": str(parent.relative_to(tmp_path)),
        "source_evidence": evidence,
        "source_epochs": epochs,
    }
    monkeypatch.setattr(recovery, "checked_binding", lambda _: binding)
    plan = {"resume": binding}
    record = recovery.inherit_prefix(output, plan)
    return SimpleNamespace(
        root=tmp_path,
        parent=parent,
        output=output,
        plan=plan,
        record=record,
        binding=binding,
        descriptor=descriptor,
    )


def test_inheritance_copies_only_the_selected108_batch_prefix(inherited):
    record, epochs = recovery.validate_inheritance(inherited.output, inherited.plan)
    assert record == inherited.record and set(epochs) == {214, 428}
    assert len((inherited.output / "steps.jsonl").read_text().splitlines()) == 108
    assert len((inherited.parent / "steps.jsonl").read_text().splitlines()) == 134
    assert not (inherited.output / "batch-0108").exists()
    assert record["rolled_back_accepted_exposures"] == 104


def test_inherited_evidence_tampering_is_rejected(inherited):
    (inherited.output / "batch-0000/pool-00.json").write_text("changed")
    with pytest.raises(ValueError, match="raw evidence changed"):
        recovery.validate_inheritance(inherited.output, inherited.plan)


def test_an_extra_inherited_file_is_rejected(inherited):
    put(inherited.output / "batch-0107/extra.json", {"unapproved": True})
    with pytest.raises(ValueError, match="inventory changed"):
        recovery.validate_inheritance(inherited.output, inherited.plan)


def test_inherited_epoch_cannot_alias_the_parent_copy(inherited):
    record = copy.deepcopy(inherited.record)
    record["epochs"][0]["visual"] = inherited.binding["source_epochs"][0]["visual"]
    put(inherited.output / "inheritance.json", record)
    with pytest.raises(ValueError, match="epoch tensors changed"):
        recovery.validate_inheritance(inherited.output, inherited.plan)


def test_inheritance_cannot_hide_the_actual134_batch_parent(inherited):
    record = {**inherited.record, "actual_source_steps": inherited.binding["source_steps"]}
    put(inherited.output / "inheritance.json", record)
    with pytest.raises(ValueError, match="discarded accepted work"):
        recovery.validate_inheritance(inherited.output, inherited.plan)


def test_inherit_refuses_a_discarded_batch_even_if_given_as_an_evidence_item(
    inherited, monkeypatch
):
    binding = copy.deepcopy(inherited.binding)
    binding["source_evidence"] = [
        inherited.descriptor(inherited.parent / "batch-0108/pool-00.json")
    ]
    monkeypatch.setattr(recovery, "checked_binding", lambda _: binding)
    output = inherited.root / "checkpoints/another-child"
    output.mkdir()
    with pytest.raises(ValueError, match="discarded or unrelated"):
        recovery.inherit_prefix(output, {"resume": binding})
    assert not (output / "batch-0108").exists()
