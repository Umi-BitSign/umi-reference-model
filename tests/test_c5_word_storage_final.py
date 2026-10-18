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
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_c5_word_lineage as old_audit
import audit_c5_word_storage_lineage as audit
import c5_word_storage_final as final
import evaluate_c5_word_final as old_pipeline
import evaluate_c5_word_storage_final as pipeline
import evaluate_c5_word_storage_sentences as sentences
import evaluate_c5_word_storage_words as words
import test_evaluate_c5_word_final as handoff_fixtures
from c5_word_working_checkpoints import WorkingCheckpoints
from terra_training_recovery import artifact


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def history(tmp_path, monkeypatch):
    monkeypatch.setattr(final, "ROOT", tmp_path)
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(final, "artifact", descriptor)
    output = tmp_path / "checkpoints/synthetic"
    output.mkdir(parents=True)
    plan_sha = "a" * 64
    store = WorkingCheckpoints(
        output,
        workspace=tmp_path,
        plan_sha=plan_sha,
        storage_check=lambda _: None,
        pair_reserve_bytes=2048,
    )
    rows = []

    def append(number):
        row = {"batch": number - 1, "training_examples": number * 4, "optimizer_updates": number}
        rows.append(row)
        with (output / "steps.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def writer(directory, number):
        visual = directory / f"visual-after-{number * 4:06d}.safetensors"
        optimizer = visual.with_name(visual.stem + "-optimizer.safetensors")
        metadata = {
            "plan_sha256": plan_sha,
            "update": str(number * 4),
            "optimizer_updates": str(number),
        }
        save_file({"vision.weight": torch.tensor([float(number)])}, str(visual), metadata=metadata)
        save_file({"moment": torch.tensor([float(number)])}, str(optimizer), metadata=metadata)
        commit = visual.with_suffix(".json")
        put(
            commit,
            {
                "complete": True,
                "committed_batch_RNG": True,
                "training_examples": number * 4,
                "optimizer_updates": number,
                "visual": descriptor(visual),
                "optimizer": descriptor(optimizer),
            },
        )
        return commit

    for number in (1, 2):
        append(number)
    seed = writer(output, 2)
    store.seed(seed)
    seed_commit = json.loads(seed.read_text())
    put(
        output / "inheritance.json",
        {
            "epochs": [
                {
                    "commit": descriptor(seed),
                    "visual": seed_commit["visual"],
                    "optimizer": seed_commit["optimizer"],
                }
            ]
        },
    )
    for number in range(3, 8):
        append(number)
        store.commit(
            examples=number * 4,
            updates=number,
            writer=lambda directory, number=number: writer(directory, number),
        )
    final_commit = writer(output, 7)
    result = {"visual": json.loads(final_commit.read_text())["visual"]}
    return SimpleNamespace(
        root=tmp_path,
        output=output,
        store=store,
        plan={"resume": {"completed_batches": 2}},
        plan_sha=plan_sha,
        steps=rows,
        result=result,
        descriptor=descriptor,
    )


def checked(history):
    return final.checked_working_history(
        history.output, history.plan, history.plan_sha, history.steps, history.result
    )


def test_full_history_keeps_two_pairs_all_receipts_and_is_read_only(history):
    def snapshot():
        return {str(p): p.read_bytes() for p in history.output.rglob("*") if p.is_file()}

    before = snapshot()
    report, evidence = checked(history)
    assert report["new_committed_pairs"] == 5
    assert report["retired_working_pairs"] == 3
    assert report["retained_working_pairs"] == 2
    assert report["latest_equals_fixed_final_tensors_and_metadata"] is True
    assert len([a for a in evidence if a["path"].endswith(".json")]) == 7
    assert before == snapshot()


@pytest.mark.parametrize("kind", ["prepared", "committed", "retire_intent", "retired"])
def test_missing_transaction_event_prevents_complete_final(history, kind):
    path = history.store.journal
    events = [json.loads(line) for line in path.read_text().splitlines()]
    del events[next(i for i, event in enumerate(events) if event["kind"] == kind)]
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    with pytest.raises(ValueError):
        checked(history)


def test_torn_tail_is_recoverable_but_cannot_be_accepted_as_success(history):
    with history.store.journal.open("ab") as stream:
        stream.write(b'{"kind":')
    assert history.store.latest()["training_examples"] == 28
    with pytest.raises(ValueError, match="partial append"):
        checked(history)


@pytest.mark.parametrize("orphan", ["unexpected.json", "unused/extra.safetensors"])
def test_orphan_artifacts_block_final(history, orphan):
    path = history.store.directory / orphan
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"orphan")
    with pytest.raises(ValueError, match="orphan"):
        checked(history)


def test_extra_empty_working_directory_is_an_orphan(history):
    (history.store.directory / "unused").mkdir()
    with pytest.raises(ValueError, match="orphan"):
        checked(history)


def test_retired_tensor_cannot_reappear(history):
    state = next(
        event["state"]
        for event in history.store._events()
        if event["kind"] == "committed" and not event["state"]["inherited_seed"]
    )
    (history.root / state["visual"]["path"]).write_bytes(b"reappeared")
    with pytest.raises(ValueError, match="retired"):
        checked(history)


def test_missing_retained_pair_is_rejected(history):
    (history.root / history.store.latest()["visual"]["path"]).unlink()
    with pytest.raises(ValueError, match="bytes changed"):
        checked(history)


def test_final_epoch_cannot_differ_from_latest_working_tensor(history):
    path = history.root / history.result["visual"]["path"]
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as saved:
        metadata = saved.metadata()
    path.unlink()
    save_file({"vision.weight": torch.tensor([99.0])}, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="tensor values"):
        checked(history)


def test_same_tensor_values_with_changed_metadata_are_rejected(tmp_path):
    paths = [tmp_path / f"{index}.safetensors" for index in range(2)]
    for index, path in enumerate(paths):
        save_file({"x": torch.ones(2)}, str(path), metadata={"plan_sha256": str(index)})
    with pytest.raises(ValueError, match="metadata"):
        final.same_tensor_state(*paths)


def test_semantic_json_metadata_order_does_not_require_identical_file_sha(tmp_path):
    paths = [tmp_path / f"{index}.safetensors" for index in range(2)]
    for path, groups in zip(paths, ('[{"lr":1,"eps":2}]', '[{"eps":2,"lr":1}]'), strict=True):
        save_file({"x": torch.ones(2)}, str(path), metadata={"groups_without_parameters": groups})
    assert final.sha256(paths[0]) != final.sha256(paths[1])
    final.same_tensor_state(*paths)


def test_history_cannot_rebind_a_ledger_prefix(history):
    lines = (history.output / "steps.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row["optimizer_updates"] = 90
    lines[0] = json.dumps(row)
    (history.output / "steps.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        checked(history)


def test_independent_auditor_preserves_every_original_statement_before_final_write():
    before = ast.parse(inspect.getsource(old_audit.main)).body[0]
    after = audit.derived_main_ast()
    extras = [
        node
        for node in after.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "audit"
        and node.value.func.attr == "update"
    ]
    assert len(extras) == 1
    insertion = after.body.index(extras[0])
    assert isinstance(after.body[insertion + 1], ast.With)
    after.body.remove(extras[0])
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)
    bound = audit.bound_main()
    assert bound.__globals__["checked_final"] is final.checked_final
    assert bound.__globals__["__file__"] == audit.__file__
    assert bound.__globals__["storage_audit_fields"] is audit.storage_audit_fields


def test_final_prerequisites_change_only_the_reviewed_auditor_source_key():
    before = ast.parse(inspect.getsource(old_pipeline.prerequisites)).body[0]
    after = pipeline.derived_prerequisites_ast()
    changes = 0
    for node in ast.walk(after):
        if isinstance(node, ast.Constant) and node.value == pipeline.AUDITOR:
            node.value = "tools/audit_c5_word_lineage.py"
            changes += 1
    assert changes == 1
    assert ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False)


def test_evaluators_use_new_validator_scorer_and_child_programs():
    assert words.checked_final is sentences.checked_final is final.checked_final
    assert words.bound_baseline().__wrapped__.__globals__["checked"] is words.checked
    assert words.bound_inference().__wrapped__.__globals__["checked"] is words.checked
    assert sentences.bound_inference().__globals__["checked"] is sentences.checked
    assert sentences.authenticated_passes is words.authenticated_passes
    assert [item[2] for item in pipeline.locations().values()] == [
        "evaluate_c5_word_storage_words.py",
        "evaluate_c5_word_storage_sentences.py",
    ]
    sources = pipeline.control_sources()
    assert sources[pipeline.AUDITOR] == final.sha256(Path(audit.__file__))


def test_wrong_lineage_totals_cannot_pass_final():
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
    result = {"training_examples": 856, "optimizer_updates": 216}
    result["lineage_accounting"] = final.training.lineage_counts(plan, 856, 216)
    assert final.checked_accounting(plan, result)["actual_total_lineage_optimizer_updates"] == 242
    changed = copy.deepcopy(result)
    changed["lineage_accounting"]["actual_total_lineage_exposures"] = 856
    with pytest.raises(ValueError, match="accounting"):
        final.checked_accounting(plan, changed)


def test_final_dispatch_keeps_polarity_authentication_and_discarded_parent_evidence(
    tmp_path, monkeypatch
):
    """Dispatch fixture only; real completed-training validation remains unperformed."""
    monkeypatch.setattr(final, "ROOT", tmp_path)
    plan_path = tmp_path / "runs/child/plan.json"
    result_path = tmp_path / "checkpoints/child/result.json"
    put(plan_path, {"synthetic": True})
    result_path.parent.mkdir(parents=True)
    result_path.with_name("steps.jsonl").write_text("")
    authorization = {"synthetic": True}
    plan = {
        "resume": {
            "source_plan": {"path": "runs/parent/plan.json"},
            "authorization": authorization,
            "actual_source_evidence": [{"synthetic_discarded_parent": True}],
        }
    }

    def original_checked(*args):
        assert globals()["checked"] is final.training.checked
        assert globals()["validate_inheritance"] is final.validate_inheritance
        assert (
            globals()["original"].verified_batch_evidence is final.polarity.verified_batch_evidence
        )
        return plan, {}, [{"synthetic_selected_evidence": True}]

    monkeypatch.setattr(final.resumed, "checked_final", original_checked)
    monkeypatch.setattr(final, "checked_accounting", lambda *a: {})
    monkeypatch.setattr(
        final, "checked_working_history", lambda *a: ({}, [{"synthetic_working_evidence": True}])
    )
    put(
        tmp_path / "runs/parent/word-manual-continuation.json",
        {
            "authorization": authorization,
            "manual_continuation_limit": 1,
            "automatic_continuations": 0,
            "rolled_back_accepted_exposures": 104,
        },
    )
    _, _, evidence = final.checked_final(plan_path, result_path)
    assert evidence == [
        {"synthetic_selected_evidence": True},
        {"synthetic_discarded_parent": True},
        {"synthetic_working_evidence": True},
    ]


@pytest.fixture
def storage_handoff(tmp_path, monkeypatch):
    """Synthetic receipt fixture; completed-model and new inventory checks are stubs."""
    env = handoff_fixtures.completed.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "word_initialization", old_pipeline.word_initialization)
    descriptor = partial(artifact, root=tmp_path)
    put(
        env.plan,
        {
            "resume": {},
            "lineage_audit_source": descriptor(env.source),
            "source_sha256": {pipeline.AUDITOR: final.sha256(env.source)},
        },
    )
    env.binding["word_training_plan"] = descriptor(env.plan)
    env.audit["plan"] = descriptor(env.plan)
    fields = {
        "lineage_accounting": {"actual_total_lineage_exposures": 960},
        "manual_recovery_authorization": {"synthetic": True},
        "parent_storage_failure_audit": {"synthetic": True},
        "judge_protocol": {"synthetic": True},
        "working_checkpoint_history": {"synthetic": True},
        "retained_working_inventory": [{"synthetic": True}],
        "discarded_parent_work_preserved_and_not_reused": True,
        "original_public_initialization_with_explicit_rollback": True,
        "further_automatic_continuations": 0,
    }
    env.audit.update(fields)
    put(env.audit_path, env.audit)
    monkeypatch.setattr(pipeline, "storage_audit_fields", lambda *a: copy.deepcopy(fields))
    return env


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "lineage_accounting",
        "manual_recovery_authorization",
        "parent_storage_failure_audit",
        "judge_protocol",
        "working_checkpoint_history",
        "retained_working_inventory",
        "discarded_parent_work_preserved_and_not_reused",
        "original_public_initialization_with_explicit_rollback",
        "further_automatic_continuations",
    ],
)
def test_final_pipeline_requires_each_recovery_audit_field_before_children(
    storage_handoff, missing, monkeypatch
):
    env = storage_handoff
    if missing is None:
        assert pipeline.prerequisites(env.plan, env.result, env.audit_path) == env.binding
    else:
        env.audit.pop(missing)
        put(env.audit_path, env.audit)
        monkeypatch.setattr(
            pipeline.original.subprocess,
            "run",
            lambda *a, **k: pytest.fail("child launched before audit admission"),
        )
        output = env.result.parent.with_name("not-created")
        with pytest.raises(ValueError, match="storage final audit"):
            pipeline.pipeline(env.plan, env.result, env.audit_path, output)
        assert not output.exists()
