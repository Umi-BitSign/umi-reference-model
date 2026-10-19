"""Second-recovery control/evidence tests; no actual completed-model admission."""

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
from safetensors import SafetensorError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_c5_word_http502_lineage as audit
import audit_c5_word_storage_lineage as old_audit
import c5_word_http502_final as final
import c5_word_storage_final as old_final
import evaluate_c5_word_final as base_pipeline
import evaluate_c5_word_http502_final as pipeline
import evaluate_c5_word_http502_sentences as sentences
import evaluate_c5_word_http502_words as words
import evaluate_c5_word_storage_sentences as old_sentences
import evaluate_c5_word_storage_words as old_words
import test_c5_word_storage_final as fixtures
import train_c5_word_http502_recovery as controller
import train_c5_word_storage_recovery as old_controller
from terra_polarity_judge import Pacer, score_training_batch
from terra_training_recovery import artifact


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def dump(node):
    return ast.dump(node, include_attributes=False)


def test_training_loop_and_original_numerical_scoring_saving_functions_are_preserved():
    assert dump(controller.derived_fit_ast()) == dump(old_controller.derived_fit_ast())
    fit = controller.bound_fit()
    assert fit.__globals__["checked"] is controller.checked
    assert fit.__globals__["restore_committed"] is controller.restore_committed
    assert fit.__globals__["RecoveryPacer"] is Pacer
    assert fit.__globals__["score_batch"] is score_training_batch
    for name in ("commit_working_checkpoint", "save_checkpoint", "lineage_counts"):
        assert fit.__globals__[name] is getattr(old_controller, name)


def test_new_freeze_refuses_missing_approval_before_source_storage_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(
        controller.template.word, "storage", lambda *a, **k: pytest.fail("storage reached")
    )
    path = tmp_path / "runs/not-created/plan.json"
    with pytest.raises(ValueError, match="new explicit HTTP502 recovery authorization is required"):
        controller.freeze(
            path,
            Path("missing-source"),
            Path("missing-weights"),
            Path("missing-audit"),
            tmp_path / "missing-approval",
        )
    assert not path.parent.exists()


def test_fit_claims_inherits_and_seeds_working_prefix_before_entering_model_loop(
    tmp_path, monkeypatch
):
    events = []
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    monkeypatch.setattr(controller, "checked_binding", lambda b: events.append("checked"))
    monkeypatch.setattr(
        controller.template.resumed, "source_cooldown", lambda b: events.append("cooldown") or {}
    )
    monkeypatch.setattr(controller, "claim", lambda *a: events.append("claim"))
    monkeypatch.setattr(
        controller,
        "inherit_prefix",
        lambda *a: events.append("inherit")
        or {"working_seed": {"commit": {"path": "checkpoints/child/resume-seed.json"}}},
    )
    monkeypatch.setattr(
        controller,
        "store_for",
        lambda *a: SimpleNamespace(seed=lambda p: events.append(("seed", p.name))),
    )
    monkeypatch.setattr(
        controller, "bound_fit", lambda: lambda *a: events.append("fit") or "synthetic result"
    )
    output = tmp_path / "checkpoints/child"
    output.mkdir(parents=True)
    assert controller.fit(Path("synthetic-plan"), output, {"resume": {}}) == "synthetic result"
    assert events == [
        "checked",
        "cooldown",
        "claim",
        "inherit",
        ("seed", "resume-seed.json"),
        "fit",
    ]


@pytest.fixture
def specification(tmp_path, monkeypatch):
    """Synthetic approved-binding stub; never a production training authorization."""
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    path = tmp_path / "runs/child/plan.json"
    source_path = tmp_path / "runs/source/plan.json"
    original_file = tmp_path / "tools/original.py"
    original_file.parent.mkdir()
    original_file.write_text("# synthetic original\n")
    for name in controller.SOURCES:
        (tmp_path / "tools" / name).write_text("# synthetic reviewed consumer " + name + "\n")
    source = {
        "source_sha256": {"tools/original.py": final.sha256(original_file)},
        "recipe": {"epochs": 4, "seed": 719, "automatic_retries": 0},
        "records": [{"synthetic_training_record": True}],
        "batches": [{"original_order": True}],
        "donor_indices": [0],
        "judge_protocol": {"synthetic_frozen_protocol": True},
        "model_plan": {"synthetic_original_model": True},
        "working_checkpoint_policy": {
            "after_each_accepted_batch": True,
            "retained_working_pairs": 2,
        },
    }
    put(source_path, source)
    binding = {"authorization": {"synthetic_approval_descriptor": True}, "training_examples": 512}
    monkeypatch.setattr(controller, "make_binding", lambda *a: copy.deepcopy(binding))
    return SimpleNamespace(
        root=tmp_path, path=path, source_path=source_path, source=source, binding=binding
    )


def test_new_specification_keeps_exact_recipe_data_model_and_original_source_map(specification):
    f = specification
    plan = controller.specification(
        f.path, f.source_path, Path("weights"), Path("audit"), Path("approval")
    )
    for key in (
        "recipe",
        "records",
        "batches",
        "donor_indices",
        "judge_protocol",
        "model_plan",
        "working_checkpoint_policy",
    ):
        assert plan[key] == f.source[key]
    assert plan["resume"] == f.binding and plan["manual_continuations_authorized"] == 2
    assert plan["automatic_further_continuations"] == 0
    assert plan["source_sha256"] == {
        **f.source["source_sha256"],
        **{f"tools/{n}": final.sha256(f.root / "tools" / n) for n in controller.SOURCES},
    }
    assert not f.path.parent.exists()


def test_missing_final_adapter_stops_freeze_before_any_executable_plan(specification, monkeypatch):
    f = specification
    monkeypatch.setattr(controller, "checked_authorization", lambda *a: (None, None))
    monkeypatch.setattr(controller.template.word, "storage", lambda *a, **k: None)
    (f.root / "tools/evaluate_c5_word_http502_sentences.py").unlink()
    with pytest.raises(FileNotFoundError):
        controller.freeze(
            f.path, f.source_path, Path("weights"), Path("audit"), Path("synthetic-approval")
        )
    assert not f.path.parent.exists()


def test_history_changes_only_seed_lookup_and_auditor_keeps_original_tensor_loop():
    before = ast.parse(inspect.getsource(old_final.checked_working_history)).body[0]
    after = final.derived_working_history_ast()
    old = (
        ast.parse('json.loads((directory / "inheritance.json").read_text())["epochs"][-1]')
        .body[0]
        .value
    )
    new = (
        ast.parse('json.loads((directory / "inheritance.json").read_text())["working_seed"]')
        .body[0]
        .value
    )

    class RestoreSeed(ast.NodeTransformer):
        def visit_Subscript(self, node):
            return copy.deepcopy(old) if dump(node) == dump(new) else self.generic_visit(node)

    assert dump(RestoreSeed().visit(after)) == dump(before)
    original = ast.parse(inspect.getsource(old_audit.storage_audit_fields)).body[0]
    adapted = audit.derived_fields_ast()

    def loops(definition):
        return [dump(n) for n in definition.body if isinstance(n, ast.For)]

    assert loops(adapted) == loops(original)
    assert dump(audit.derived_main_ast()) == dump(old_audit.derived_main_ast())
    bound = audit.bound_main()
    assert bound.__globals__["checked_final"] is final.checked_final
    assert bound.__globals__["storage_audit_fields"] is audit.storage_audit_fields
    assert bound.__globals__["__file__"] == audit.__file__


@pytest.fixture
def history(tmp_path, monkeypatch):
    # Tiny existing real file-transaction fixture, not eligible856 model data.
    h = fixtures.history.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(final, "ROOT", tmp_path)
    path = h.output / "inheritance.json"
    record = json.loads(path.read_text())
    put(path, {"epochs": [], "working_seed": record["epochs"][0]})
    return h


def checked_history(h):
    return final.checked_working_history(h.output, h.plan, h.plan_sha, h.steps, h.result)


def test_mid_epoch_seed_history_retains_pairs_receipts_and_read_only_final_equality(history):
    h = history
    before = {p: p.read_bytes() for p in h.output.rglob("*") if p.is_file()}
    report, _ = checked_history(h)
    assert report["new_committed_pairs"] == 5 and report["retained_working_pairs"] == 2
    assert report["latest_equals_fixed_final_tensors_and_metadata"] is True
    assert before == {p: p.read_bytes() for p in h.output.rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["missing_event", "orphan", "changed_epoch", "changed_seed"])
def test_mid_epoch_history_does_not_relax_original_transaction_or_value_checks(history, fault):
    h = history
    if fault == "missing_event":
        events = h.store.journal.read_text().splitlines()
        h.store.journal.write_text("\n".join(events[:-1]) + "\n")
    elif fault == "orphan":
        (h.store.directory / "orphan.json").write_text("{}")
    elif fault == "changed_epoch":
        (h.root / h.result["visual"]["path"]).write_bytes(b"corrupt")
    else:
        data = json.loads((h.output / "inheritance.json").read_text())
        data["working_seed"]["visual"] = {"path": "other", "sha256": "b" * 64}
        put(h.output / "inheritance.json", data)
    with pytest.raises((ValueError, SafetensorError)):
        checked_history(h)


def test_resource_guard_adaptation_preserves_the_entire_original_verification_loop():
    before = ast.parse(inspect.getsource(final.original.verify_steps)).body[0]
    after = final.derived_verify_steps_ast()
    assert dump(before.body[1]) != dump(after.body[1])
    after.body[1] = copy.deepcopy(before.body[1])
    assert dump(before) == dump(after)


@pytest.fixture
def resource_sequence(monkeypatch):
    # Run the actual original risk/counter/resource loop. Only curriculum
    # coverage is stubbed; this synthetic1-clip-per-batch sequence is not data
    # for complete-model admission and has no pretrained tensors or API grades.
    from test_c5_word_storage_recovery import resources

    monkeypatch.setattr(final.original, "verify_coverage", lambda *a: None)
    batches, steps = [], []
    start = 0
    for number in range(216):
        if number in (108, 129):
            start = number
        batch = {"epoch": number // 54, "training_record_indices": [number % 214]}
        r = resources()
        r["mps_active_seconds"] = r["mps_idle_seconds"] = float(number - start + 1)
        batches.append(batch)
        steps.append(
            {
                **batch,
                "batch": number,
                "Terra_qualities": [[10, 0]],
                "before_update_risk": [{"expected_terra_error": 5.0}],
                "gradient_norm_before_clip": 1.0,
                "contrasting_clips": 1,
                "training_examples": number + 1,
                "optimizer_updates": number + 1,
                "resources": r,
            }
        )
    plan = {
        "resume": {"resource_reset_batches": [108, 129], "completed_batches": 129},
        "batches": batches,
    }
    result = {
        "optimizer_updates": 216,
        "resources": copy.deepcopy(steps[-1]["resources"]),
        "checkpoints": [],
    }
    return plan, steps, result


def test_actual_resource_loop_accepts_exact_two_authenticated_handoffs(resource_sequence):
    plan, steps, result = resource_sequence
    final.verify_steps(plan, steps, result, resource_reset_batches={129})
    with pytest.raises(ValueError, match="reset boundaries"):
        final.verify_steps(
            plan,
            steps,
            result,
            resource_reset_batches={1, 129},
        )
    plan["resume"]["resource_reset_batches"] = [129]
    with pytest.raises(ValueError, match="reset boundaries"):
        final.verify_steps(plan, steps, result)


@pytest.mark.parametrize("number", [100, 130])
def test_actual_resource_loop_rejects_any_additional_counter_reset(resource_sequence, number):
    plan, steps, result = resource_sequence
    steps[number]["resources"]["mps_active_seconds"] = 0.5
    steps[number]["resources"]["mps_idle_seconds"] = 0.5
    with pytest.raises(ValueError, match="counters moved backwards"):
        final.verify_steps(plan, steps, result, resource_reset_batches={129})


@pytest.mark.parametrize("number", [2, 1])
def test_final_dispatch_keeps_both_resource_resets_and_ancestor_evidence(
    tmp_path, monkeypatch, number
):
    """Dispatch fixture only; underlying completed-model authentication is stubbed."""
    monkeypatch.setattr(final, "ROOT", tmp_path)
    plan_path = tmp_path / "runs/child/plan.json"
    result_path = tmp_path / "checkpoints/child/result.json"
    put(plan_path, {"synthetic": True})
    result_path.parent.mkdir(parents=True)
    result_path.with_name("steps.jsonl").write_text("")
    authorization = {"synthetic_approval": True}
    plan = {
        "resume": {
            "source_plan": {"path": "runs/parent/plan.json"},
            "authorization": authorization,
            "actual_source_evidence": [{"synthetic_original_rollback_and_HTTP502_evidence": True}],
        }
    }

    def original_checked(*args):
        assert globals()["checked"] is final.training.checked
        assert globals()["validate_inheritance"] is final.validate_inheritance
        assert globals()["original"].verify_steps is final.verify_steps
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
            "additional_manual_continuation_limit": 1,
            "manual_continuation_number": number,
            "automatic_continuations": 0,
            "rolled_back_accepted_exposures": 104,
            "newly_rolled_back_accepted_exposures": 0,
        },
    )
    if number == 2:
        _, _, evidence = final.checked_final(plan_path, result_path)
        assert evidence == [
            {"synthetic_selected_evidence": True},
            {"synthetic_original_rollback_and_HTTP502_evidence": True},
            {"synthetic_working_evidence": True},
        ]
    else:
        with pytest.raises(ValueError, match="explicit second-recovery claim"):
            final.checked_final(plan_path, result_path)


@pytest.fixture
def counts(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(audit, "artifact", descriptor)
    rows = []
    for n in range(1, 217):
        epoch, position = divmod(n - 1, 54)
        rows.append(
            {
                "batch": n - 1,
                "training_examples": epoch * 214 + min((position + 1) * 4, 214),
                "optimizer_updates": n,
            }
        )
    original = tmp_path / "checkpoints/original/steps.jsonl"
    chosen = tmp_path / "runs/original/prefix.jsonl"
    parent = tmp_path / "checkpoints/parent/steps.jsonl"
    output = tmp_path / "checkpoints/child"
    output.mkdir(parents=True)
    for path, prefix in (
        (original, rows[:134]),
        (chosen, rows[:108]),
        (parent, rows[:129]),
        (output / "steps.jsonl", rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in prefix))
    resume = {
        "actual_source_training_examples": 616,
        "actual_source_optimizer_updates": 155,
        "training_examples": 512,
        "optimizer_updates": 129,
        "rolled_back_accepted_exposures": 104,
        "rolled_back_optimizer_updates": 26,
        "newly_rolled_back_accepted_exposures": 0,
        "newly_rolled_back_optimizer_updates": 0,
        "actual_source_steps": descriptor(parent),
        "ancestor_bindings": [
            {"actual_source_steps": descriptor(original), "source_steps": descriptor(chosen)}
        ],
    }
    plan = {"resume": resume}
    result = {
        "training_examples": 856,
        "optimizer_updates": 216,
        "lineage_accounting": controller.lineage_counts(plan, 856, 216),
    }
    return SimpleNamespace(
        plan=plan,
        result=result,
        output=output,
        original=original,
        chosen=chosen,
        parent=parent,
        descriptor=descriptor,
    )


def test_independent_ledger_accounting_preserves_original_rollback_and_all_new_work(counts):
    c = counts
    measured = audit.independent_counts(c.plan, c.result, c.output)
    assert measured["actual_total_lineage_exposures"] == 960
    assert measured["actual_total_lineage_optimizer_updates"] == 242
    assert measured["worker_exposures"] == 344
    assert final.checked_accounting(c.plan, c.result) == measured


@pytest.mark.parametrize(
    "fault",
    [
        "actual_work_hidden",
        "missing_rollback",
        "changed_parent",
        "changed_prefix",
        "lost_final_prefix",
        "partial_final",
    ],
)
def test_changed_nested_work_cannot_pass_independent_audit(counts, fault):
    c = counts
    if fault == "actual_work_hidden":
        c.result["lineage_accounting"]["actual_total_lineage_exposures"] = 856
    elif fault == "missing_rollback":
        c.result["lineage_accounting"]["rolled_back_accepted_exposures"] = 0
    elif fault in ("changed_parent", "changed_prefix"):
        path = c.parent if fault == "changed_parent" else c.chosen
        rows = path.read_text().splitlines()
        row = json.loads(rows[-1])
        row["training_examples"] -= 1
        rows[-1] = json.dumps(row)
        path.write_text("\n".join(rows) + "\n")
        if fault == "changed_parent":
            c.plan["resume"]["actual_source_steps"] = c.descriptor(path)
        else:
            c.plan["resume"]["ancestor_bindings"][0]["source_steps"] = c.descriptor(path)
    elif fault == "lost_final_prefix":
        path = c.output / "steps.jsonl"
        rows = path.read_text().splitlines()
        row = json.loads(rows[128])
        row["optimizer_updates"] -= 1
        rows[128] = json.dumps(row)
        path.write_text("\n".join(rows) + "\n")
    else:
        c.result["training_examples"] = 512
    with pytest.raises(ValueError):
        audit.independent_counts(c.plan, c.result, c.output)


def test_full_evaluators_preserve_every_algorithm_except_source_and_schema_bindings():
    for new, old in ((words, old_words), (sentences, old_sentences)):
        source = (
            inspect.getsource(new)
            .replace("HTTP502-recovery", "storage-recovery")
            .replace("http502", "storage")
        )
        new_tree, old_tree = ast.parse(source), ast.parse(inspect.getsource(old))
        assert [dump(n) for n in new_tree.body if isinstance(n, ast.FunctionDef)] == [
            dump(n) for n in old_tree.body if isinstance(n, ast.FunctionDef)
        ]
        assert new.checked_final is final.checked_final
    assert words.bound_baseline().__wrapped__.__globals__["checked"] is words.checked
    assert words.bound_inference().__wrapped__.__globals__["checked"] is words.checked
    assert sentences.bound_inference().__globals__["checked"] is sentences.checked
    assert sentences.authenticated_passes is words.authenticated_passes
    before = ast.parse(inspect.getsource(base_pipeline.prerequisites)).body[0]
    after = pipeline.derived_prerequisites_ast()
    matches = 0
    for node in ast.walk(after):
        if isinstance(node, ast.Constant) and node.value == pipeline.AUDITOR:
            node.value = "tools/audit_c5_word_lineage.py"
            matches += 1
    assert matches == 1 and dump(before) == dump(after)
    assert [v[2] for v in pipeline.locations().values()] == [
        "evaluate_c5_word_http502_words.py",
        "evaluate_c5_word_http502_sentences.py",
    ]
    assert pipeline.control_sources()[pipeline.AUDITOR] == final.sha256(Path(audit.__file__))


FIELDS = (
    "lineage_accounting",
    "manual_recovery_authorization",
    "parent_HTTP502_failure_audit",
    "judge_protocol",
    "working_checkpoint_history",
    "retained_working_inventory",
    "discarded_parent_work_preserved_and_not_reused",
    "original_public_initialization_with_explicit_rollback",
    "further_automatic_continuations",
    "original_storage_failure_audit",
    "manual_recovery_authorizations",
    "no_newly_rolled_back_accepted_work",
    "resource_reset_batches",
    "failed_HTTP502_batch_partial_evidence",
)


@pytest.fixture
def handoff(tmp_path, monkeypatch):
    h = fixtures.handoff_fixtures.completed.__wrapped__(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "word_initialization", base_pipeline.word_initialization)
    descriptor = partial(artifact, root=tmp_path)
    put(
        h.plan,
        {
            "resume": {},
            "lineage_audit_source": descriptor(h.source),
            "source_sha256": {pipeline.AUDITOR: final.sha256(h.source)},
        },
    )
    h.binding["word_training_plan"] = descriptor(h.plan)
    h.audit["plan"] = descriptor(h.plan)
    fields = {key: {"synthetic": key} for key in FIELDS}
    h.audit.update(fields)
    put(h.audit_path, h.audit)
    monkeypatch.setattr(pipeline, "storage_audit_fields", lambda *a: copy.deepcopy(fields))
    return h


@pytest.mark.parametrize("missing", [None, *FIELDS])
def test_every_nested_recovery_audit_field_is_required_before_evaluation_children(
    handoff, missing, monkeypatch
):
    h = handoff
    if missing is None:
        assert pipeline.prerequisites(h.plan, h.result, h.audit_path) == h.binding
    else:
        h.audit.pop(missing)
        put(h.audit_path, h.audit)
        monkeypatch.setattr(
            pipeline.template.original.subprocess,
            "run",
            lambda *a, **k: pytest.fail("child launched"),
        )
        output = h.result.parent.with_name("not-created")
        with pytest.raises(ValueError, match="HTTP502 final audit"):
            pipeline.pipeline(h.plan, h.result, h.audit_path, output)
        assert not output.exists()
