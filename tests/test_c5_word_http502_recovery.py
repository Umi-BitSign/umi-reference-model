"""Portable admission fixtures; no production approval, model or API operation."""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_word_http502_recovery as recovery


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def approval(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    proposal = tmp_path / "runs/proposal/proposal.json"
    data = {
        "inherited_accepted_exposures": 512,
        "inherited_accepted_updates": 129,
        "inherited_completed_batches": 129,
        "remaining_exposures": 344,
        "remaining_batches": 87,
        "actual_accepted_exposures_before_new_work": 616,
        "actual_updates_before_new_work": 155,
        "actual_final_accepted_exposures_if_complete": 960,
        "newly_rolled_back_accepted_exposures": 0,
        "newly_rolled_back_accepted_updates": 0,
        "original_rolled_back_parent_exposures_preserved": 104,
        "original_rolled_back_parent_updates_preserved": 26,
        "starting_working_state": {
            "visual": {"path": "checkpoints/synthetic/512", "sha256": "a" * 64}
        },
    }
    put(proposal, data)
    monkeypatch.setattr(recovery, "PROPOSAL", proposal)
    monkeypatch.setattr(recovery, "PROPOSAL_SHA", recovery.sha(proposal))
    path = tmp_path / "runs/proposal/approval.json"
    value = {
        "scope": recovery.SCOPE,
        "approved": True,
        "source": "explicit_user_message",
        "user_message": "Synthetic fixture only. Never authorizes a real run.",
        "proposal": recovery.artifact(proposal),
        "source_checkpoint": data["starting_working_state"]["visual"],
        "additional_manual_continuation_limit": 1,
        "automatic_retries": 0,
        "automatic_further_continuations": 0,
        "received_at": "2026-10-18T00:00:00+00:00",
    }
    put(path, value)
    return SimpleNamespace(
        root=tmp_path, path=path, value=value, proposal=data, proposal_path=proposal
    )


def test_exact_new_user_approval_binds_checkpoint_proposal_and_no_expiry(approval):
    descriptor, proposal = recovery.checked_authorization(approval.path)
    assert descriptor == recovery.artifact(approval.path)
    assert proposal == approval.proposal


@pytest.mark.parametrize(
    "field,value",
    [
        ("scope", "one-polarity-word-storage-recovery-from-428"),
        ("source", "automatic_goal_continuation"),
        ("approved", False),
        ("user_message", " "),
        ("additional_manual_continuation_limit", 2),
        ("additional_manual_continuation_limit", True),
        ("automatic_retries", 1),
        ("automatic_retries", False),
        ("automatic_further_continuations", 1),
        ("source_checkpoint", {"path": "other", "sha256": "b" * 64}),
        ("proposal", {"path": "other", "sha256": "b" * 64}),
    ],
)
def test_prior_approval_automation_or_relaxed_limits_cannot_authorize_new_work(
    approval, field, value
):
    put(approval.path, {**approval.value, field: value})
    with pytest.raises(ValueError, match="own exact explicit HTTP502 approval"):
        recovery.checked_authorization(approval.path)


def test_missing_approval_refuses_before_any_source_checkpoint_or_model_work(approval, monkeypatch):
    approval.path.unlink()
    monkeypatch.setattr(recovery, "source_binding", lambda: pytest.fail("source work reached"))
    with pytest.raises(ValueError, match="new explicit HTTP502 recovery authorization is required"):
        recovery.make_binding(
            Path("missing-source"), Path("missing-weights"), Path("missing-audit"), approval.path
        )


def test_changed_proposal_and_timezone_free_approval_are_rejected(approval):
    put(approval.proposal_path, {**approval.proposal, "remaining_exposures": 343})
    with pytest.raises(ValueError, match="proposal changed"):
        recovery.checked_authorization(approval.path)
    put(approval.proposal_path, approval.proposal)
    put(approval.path, {**approval.value, "received_at": "2026-10-18T00:00:00"})
    with pytest.raises(ValueError, match="timezone-aware"):
        recovery.checked_authorization(approval.path)


def test_recovery_descriptor_rejects_changed_bytes_aliases_and_external_files(approval):
    descriptor = recovery.artifact(approval.path)
    assert recovery.bound(descriptor) == approval.path
    put(approval.path, {"changed": True})
    with pytest.raises(ValueError, match="source evidence changed"):
        recovery.bound(descriptor)
    alias = approval.path.with_name("alias.json")
    alias.symlink_to(approval.path)
    with pytest.raises(ValueError, match="canonical owned private file"):
        recovery.artifact(alias)
    outside = approval.root / "public.json"
    put(outside, {})
    with pytest.raises(ValueError, match="canonical owned private file"):
        recovery.artifact(outside)


@pytest.fixture
def terminal(approval):
    directory = approval.root / "checkpoints/synthetic"
    state = copy.deepcopy(approval.proposal["starting_working_state"])
    stopped = {
        "error": recovery.ERROR,
        "automatic_continuations": 0,
        "automatic_retries": 0,
        "recovery_error": None,
        "recovery_is_committed_prefix": True,
        "uncommitted_optimizer_mutation_possible": False,
        "training_examples": 512,
        "optimizer_updates": 129,
        "recovery": state["visual"],
    }
    for name, data in (
        ("stopped.json", stopped),
        ("failure.json", {"error": recovery.ERROR, "automatic_retries": 0}),
        ("worker.json", {"pid": 719, "uid": os.getuid()}),
        ("exit.json", {"pid": 719, "exit_code": 1}),
    ):
        put(directory / name, data)
    proposal = {**approval.proposal, "source_plan": {"synthetic_source": True}}
    source = {
        "manual_continuations_authorized": 1,
        "automatic_further_continuations": 0,
        "resume": {
            "training_examples": 428,
            "optimizer_updates": 108,
            "actual_source_training_examples": 532,
            "actual_source_optimizer_updates": 134,
            "rolled_back_accepted_exposures": 104,
            "rolled_back_optimizer_updates": 26,
        },
    }
    status = {
        key: recovery.artifact(directory / (key + ".json"))
        for key in ("exit", "stopped", "failure")
    }
    audit = {
        "schema": "umi-studio-polarity-storage-recovery-http502-stop-audit/1",
        "audit_complete": True,
        "complete": False,
        "same_approved_recovery_worker_terminal_exit1_authenticated": True,
        "plan": proposal["source_plan"],
        "durable_working_state": state,
        "training_examples": 512,
        "optimizer_updates": 129,
        "complete_batch_records": 129,
        "owned_process_ps": status,
        "owned_process_ps_after_authentication": status,
        "actual_total_lineage_exposures": 616,
        "actual_total_lineage_optimizer_updates": 155,
        "failed_batch": {
            "batch": 129,
            "accepted_exposures": 0,
            "optimizer_updates": 0,
            "completed_raw_requests": 1,
            "HTTP502_failed_requests": 1,
            "partial_scores_not_used_for_update_or_recovery": True,
        },
    }
    return SimpleNamespace(directory=directory, proposal=proposal, source=source, audit=audit)


def test_terminal_review_recomputes_nested_actual_work_without_new_rollback_or_writes(terminal):
    t = terminal
    before = {p: p.read_bytes() for p in t.directory.rglob("*") if p.is_file()}
    assert recovery.check_terminal(t.proposal, t.source, t.audit, t.directory) == (616, 155)
    assert before == {p: p.read_bytes() for p in t.directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_stop",
        "uncertain_mutation",
        "missing_checkpoint",
        "final_result",
        "wrong_exit",
        "changed_actual_work",
        "hidden_rollback",
        "admitted_partial_batch",
        "other_saved_prefix",
        "automatic_continuation",
    ],
)
def test_terminal_failures_cannot_be_reclassified_as_a_safe_resume(terminal, fault):
    t = terminal
    if fault in ("wrong_stop", "uncertain_mutation", "missing_checkpoint"):
        stopped = json.loads((t.directory / "stopped.json").read_text())
        if fault == "wrong_stop":
            stopped["error"] = "different failure"
        elif fault == "uncertain_mutation":
            stopped["uncommitted_optimizer_mutation_possible"] = True
        else:
            stopped["recovery"] = None
        put(t.directory / "stopped.json", stopped)
    elif fault == "final_result":
        put(t.directory / "result.json", {})
    elif fault == "wrong_exit":
        put(t.directory / "exit.json", {"pid": 719, "exit_code": 0})
    elif fault == "changed_actual_work":
        t.audit["actual_total_lineage_exposures"] = 512
    elif fault == "hidden_rollback":
        t.source["resume"]["rolled_back_accepted_exposures"] = 0
    elif fault == "admitted_partial_batch":
        t.audit["failed_batch"]["optimizer_updates"] = 1
    elif fault == "other_saved_prefix":
        t.audit["training_examples"] = 508
    else:
        t.source["automatic_further_continuations"] = 1
    with pytest.raises(ValueError):
        recovery.check_terminal(t.proposal, t.source, t.audit, t.directory)


@pytest.fixture
def inheritance(approval, monkeypatch):
    import c5_word_training

    monkeypatch.setattr(c5_word_training, "storage", lambda *a, **k: None)
    parent = approval.root / "checkpoints/parent"
    output = approval.root / "checkpoints/child"
    output.mkdir(parents=True)
    rows = [
        {"batch": i, "training_examples": 4 * (i + 1), "optimizer_updates": i + 1} for i in range(3)
    ]
    parent.mkdir(parents=True)
    (parent / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    evidence = []
    for index in range(3):
        receipt = parent / f"batch-{index:04d}/synthetic.json"
        put(receipt, {"synthetic": index})
        evidence.append(recovery.artifact(receipt))
    rejected = parent / "batch-0003/rewards.json"
    put(rejected, {"partial": True})

    def pair(stem, count, update):
        visual = parent / (stem + ".safetensors")
        optimizer = parent / (stem + "-optimizer.safetensors")
        visual.write_bytes(("synthetic visual " + stem).encode())
        optimizer.write_bytes(("synthetic optimizer " + stem).encode())
        commit = parent / (stem + ".json")
        value = {
            "visual": recovery.artifact(visual),
            "optimizer": recovery.artifact(optimizer),
            "training_examples": count,
            "optimizer_updates": update,
            "complete": True,
            "committed_batch_RNG": True,
        }
        put(commit, value)
        return {**value, "commit": recovery.artifact(commit), "tensor_plan_sha256": "a" * 64}

    epoch = pair("visual-after-000008", 8, 2)
    seed = pair("visual-after-000012", 12, 3)
    binding = {
        "source_plan": {"path": "runs/synthetic/plan.json", "sha256": "b" * 64},
        "source_steps": recovery.artifact(parent / "steps.jsonl"),
        "completed_batches": 3,
        "source_evidence": evidence,
        "source_epochs": [epoch],
        "source_working_states": [seed],
        "ancestor_bindings": [{"synthetic_ancestor": True}],
    }
    monkeypatch.setattr(recovery, "checked_binding", lambda value: value)
    return SimpleNamespace(parent=parent, output=output, binding=binding, plan={"resume": binding})


def test_inheritance_keeps_working_seed_out_of_epoch_inventory_and_rejected_batch(inheritance):
    t = inheritance
    record = recovery.inherit_prefix(t.output, t.plan)
    assert len(record["epochs"]) == 1 and record["working_seed_is_complete_epoch"] is False
    assert record["working_seed"]["source"] == t.binding["source_working_states"][0]
    assert not (t.output / "batch-0003").exists()
    assert (t.output / "steps.jsonl").read_bytes() == (t.parent / "steps.jsonl").read_bytes()
    checked, epochs = recovery.validate_inheritance(t.output, t.plan)
    assert checked == record
    assert epochs[8]["source_plan_sha256"] == "a" * 64
    assert 12 not in epochs
    assert not os.path.samefile(
        t.output / "resume-seed-000512.safetensors", t.parent / "visual-after-000012.safetensors"
    )
    with pytest.raises(FileExistsError):
        recovery.inherit_prefix(t.output, t.plan)


@pytest.mark.parametrize(
    "fault", ["seed_changed", "seed_as_epoch", "extra_accepted_receipt", "source_hardlink"]
)
def test_changed_or_aliased_inheritance_is_rejected(inheritance, fault):
    t = inheritance
    record = recovery.inherit_prefix(t.output, t.plan)
    if fault == "seed_changed":
        (t.output / "resume-seed-000512.safetensors").write_bytes(b"different")
    elif fault == "seed_as_epoch":
        record["working_seed_is_complete_epoch"] = True
        put(t.output / "inheritance.json", record)
    elif fault == "extra_accepted_receipt":
        put(t.output / "batch-0000/extra.json", {})
    else:
        target = t.output / "resume-seed-000512.safetensors"
        target.unlink()
        os.link(t.parent / "visual-after-000012.safetensors", target)
    with pytest.raises(ValueError):
        recovery.validate_inheritance(t.output, t.plan)


def test_second_recovery_claim_is_single_use_and_retains_original_rollback(inheritance, approval):
    t = inheritance
    path = approval.root / "runs/child/plan.json"
    put(path, {"synthetic": True})
    put(approval.root / "runs/synthetic/plan.json", {"synthetic_parent": True})
    put(t.output / "worker.json", {"synthetic_worker": True})
    t.binding["authorization"] = recovery.artifact(approval.path)
    recovery.claim(path, t.output, t.plan)
    target = approval.root / "runs/synthetic/word-manual-continuation.json"
    claim = json.loads(target.read_text())
    assert claim["manual_continuation_number"] == 2
    assert claim["additional_manual_continuation_limit"] == 1
    assert claim["automatic_continuations"] == claim["newly_rolled_back_accepted_exposures"] == 0
    assert claim["rolled_back_accepted_exposures"] == 104
    assert claim["authorization"] == recovery.artifact(approval.path)
    with pytest.raises(FileExistsError):
        recovery.claim(path, t.output, t.plan)


def test_restore_keeps_original_checked_restore_and_rejects_compound_artifact_fields(monkeypatch):
    import c5_word_storage_recovery

    calls = []

    def original(plan, model, optimizer, duty):
        calls.append((plan, model, optimizer, duty))
        return 512, 129

    monkeypatch.setattr(c5_word_storage_recovery, "restore_committed", original)
    plan = {
        "resume": {
            "visual": {"path": "synthetic", "sha256": "a" * 64},
            "ancestor_bindings": [{"synthetic": True}],
        }
    }
    assert recovery.restore_committed(plan, "model", "optimizer", "duty") == (512, 129)
    assert len(calls) == 1
    plan["resume"]["compound"] = {"not_an_artifact": True}
    with pytest.raises(ValueError, match="compound nonartifact dictionary"):
        recovery.restore_committed(plan, "model", "optimizer", "duty")
    assert len(calls) == 1
