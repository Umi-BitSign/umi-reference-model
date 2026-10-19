"""Full16 synthetic qualification and terminal evidence; no real model or grader."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_timing_feature_data as data
import probe_c5_caption_timing as probe
import probe_c5_timing_semantic as original
import test_probe_c5_timing_semantic as original_tests
from test_c5_timing_semantic_model import student

cases = original_tests.cases


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def execution(cases, tmp_path, monkeypatch):
    from contextlib import contextmanager

    import c5_timing_semantic_model as semantic
    import prepare_c5_timing_features as features
    import train_terra_c5_tail as owner

    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    model = student()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    model_plan = {"synthetic": True}
    pools = original_tests.pools_for(cases)
    pool_path = tmp_path / "checkpoints/parent/pools.json"
    save(pool_path, pools)
    paired_path = pool_path.with_name("paired.json")
    save(paired_path, original.paired_inputs(cases, pools))
    audit_path = tmp_path / "runs/data-audit.json"
    save(audit_path, {"synthetic": True})
    plan = {
        "data_audit": data.artifact(audit_path),
        "model_plan": model_plan,
        "pools": data.artifact(pool_path),
        "paired_inputs": data.artifact(paired_path),
        "records": [
            {"index": index, "kind": "word" if i < 2 else "phrase", "row": case["row"]}
            for i, (index, case) in enumerate(zip(probe.POLICY["case_indices"], cases, strict=True))
        ],
        "grading": [{"pass": number, "qualities": [[10, 0] for _ in cases]} for number in (1, 2)],
        "grading_provenance": {
            "complete_authentic_source_passes": 2,
            "inherited_API_attempts_including_predecessor": 3,
            "inherited_orchestrated_failed_request_replays": 1,
            "new_Terra_requests": 0,
        },
    }
    path = tmp_path / "runs/proof/plan.json"
    save(path, plan)

    class Dataset:
        def __init__(self, path):
            assert path == audit_path

        def __len__(self):
            return 532

        def load(self, index):
            return copy.deepcopy(cases[index % 4])

    monkeypatch.setattr(probe, "TimingFeatureDataset", Dataset)
    monkeypatch.setattr(probe, "checked", lambda path: copy.deepcopy(plan))
    monkeypatch.setattr(semantic, "load_student", lambda plan: (model, original_tests.Tokenizer()))
    monkeypatch.setattr(semantic, "require_model", lambda m, p: m.require_fixed())
    monkeypatch.setattr(features, "require_environment", lambda: None)
    monkeypatch.setattr(original, "guard", lambda *args: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    lease = [False]

    @contextmanager
    def exclusive():
        assert not lease[0]
        lease[0] = True
        try:
            yield
        finally:
            lease[0] = False

    monkeypatch.setattr(owner, "gpu_lease", exclusive)
    load = semantic.load_student

    def load_under_lease(plan):
        assert lease[0]
        return load(plan)

    monkeypatch.setattr(semantic, "load_student", load_under_lease)
    output = tmp_path / "checkpoints/proof"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(tmp_path / "tools/probe_c5_caption_timing.py"),
            "--plan",
            str(path),
            "--output",
            str(output),
        ],
    )
    return path, output, plan, model, before, lease


def test_complete_synthetic_qualification_and_consumer_reject_changed_evidence(execution):
    path, output, plan, model, before, lease = execution
    probe.run(path, output)
    assert not lease[0]
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())
    result_path = output / "qualification.json"
    result = data.read(result_path)
    assert len(result["comparisons"]) == 16 and result["optimizer_updates"] == 0
    assert result["new_Terra_requests"] == 0
    assert data.read(output / "feature-identity.json")["clips"] == 532
    poll_path = path.with_name("terminal-poll.json")
    poll = {
        "tool": "write_stdin",
        "requested_session_id": 123,
        "actual_owned_pid": data.read(output / "worker.json")["pid"],
        "tool_result": {"exit_code": 0},
    }
    save(poll_path, poll)
    checked, result_checked, evidence = probe.checked_qualification(path, result_path, poll_path)
    assert checked == plan and result_checked == result and len(evidence) == 23
    case_path = output / "case-0-1-0.json"
    for change in (
        "gradient",
        "caption",
        "tokens",
        "state",
        "grades",
        "partial",
        "live",
        "lineage",
    ):
        original_case = case_path.read_bytes()
        original_result = result_path.read_bytes()
        original_poll = poll_path.read_bytes()
        report = data.read(case_path)
        altered = copy.deepcopy(result)
        if change == "gradient":
            report["optimized_gradient"]["gradient_values"][0] += 1
        elif change == "caption":
            report["optimized"]["caption_weight"] = 0
        elif change == "tokens":
            report["optimized"]["candidate_token_counts"][0] += 1
            report["reference"]["candidate_token_counts"][0] += 1
        elif change == "state":
            report["weights"][0] = 0.01
        elif change == "grades":
            report["actual_grades"][0] = 9
        elif change == "partial":
            altered["comparisons"].pop()
        elif change == "live":
            save(poll_path, {**poll, "tool_result": {"session_id": 123}})
        else:
            altered["grading_provenance"]["inherited_orchestrated_failed_request_replays"] = 0
        save(case_path, report)
        altered["comparisons"][0] = data.artifact(case_path)
        save(result_path, altered)
        with pytest.raises(ValueError):
            probe.checked_qualification(path, result_path, poll_path)
        case_path.write_bytes(original_case)
        result_path.write_bytes(original_result)
        poll_path.write_bytes(original_poll)
    extra = output / "unbound.json"
    save(extra, {})
    with pytest.raises(ValueError, match="extra or aliased"):
        probe.checked_qualification(path, result_path, poll_path)
    extra.unlink()
    probe.checked_qualification(path, result_path, poll_path)
    with pytest.raises(ValueError, match="fresh execution"):
        probe.run(path, output.with_name("another"))
    assert not output.with_name("another").exists()


def test_early_freeze_refuses_missing_grading_before_creating_plan(tmp_path, monkeypatch):
    import train_c5_timing as training

    monkeypatch.setattr(data, "ROOT", tmp_path)

    def refuse(*args):
        raise ValueError("complete authentic grading absent")

    monkeypatch.setattr(training, "qualification", refuse)
    plan = tmp_path / "runs/new/plan.json"
    with pytest.raises(ValueError, match="authentic grading absent"):
        probe.freeze(plan, *(tmp_path / name for name in ("parent", "result", "poll")))
    assert not plan.parent.exists()


def test_changed_objective_rejects_before_any_consumer_model_access():
    with pytest.raises(ValueError, match="objective"):
        probe.compare_losses(
            {"loss": 1, "expected_terra_error": 1, "caption_loss": 1},
            {"loss": 2, "expected_terra_error": 1, "caption_loss": 1},
        )


def test_specification_preserves_raw_grading_lineage_and_rejects_other_roles(
    execution, monkeypatch
):
    import train_c5_timing as training

    path, _, original_plan, _, _, _ = execution
    parent_paths = [
        path.with_name(name)
        for name in ("parent-plan.json", "parent-result.json", "parent-poll.json")
    ]
    for parent_path in parent_paths:
        save(parent_path, {"synthetic": True})
    parent = {**copy.deepcopy(original_plan), "source_sha256": {}}
    result = {
        "pools": parent["pools"],
        "comparison": parent["paired_inputs"],
        "grading": parent["grading"],
        "raw_grading_evidence": [{"synthetic": True}],
        "API_attempts_including_predecessor": 3,
        "new_Terra_requests": 2,
        "orchestrated_failed_request_replays": 1,
    }
    source = probe.ROOT / "tools"
    source.mkdir()
    for name in ("train_c5_timing.py", "c5_caption_timing_risk.py", "probe_c5_caption_timing.py"):
        (source / name).write_text("# synthetic source")
    monkeypatch.setattr(training, "qualification", lambda *args: (parent, result, {}))
    plan = probe.specification(*parent_paths)
    assert plan["grading_provenance"] == original_plan["grading_provenance"]
    assert plan["policy"]["comparisons"] == 16
    assert plan["policy"]["optimizer_updates"] == plan["policy"]["new_Terra_requests"] == 0
    assert plan["raw_grading_evidence"] == result["raw_grading_evidence"]
    assert len(plan["source_sha256"]) == 3
    parent["records"][0]["row"]["role"] = "development"
    with pytest.raises(ValueError, match="original training cases"):
        probe.specification(*parent_paths)
