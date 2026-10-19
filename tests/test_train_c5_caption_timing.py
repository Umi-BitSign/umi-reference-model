"""Full synthetic caption-timing optimization and exact replay; real raw-grade parser."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_caption_timing_final as final
import c5_timing_feature_data as data
import test_train_c5_timing as original_tests
import train_c5_caption_timing as training
from c5_monotone_feature_timing import WIDTHS
from test_c5_timing_semantic_model import student

records = original_tests.records
rewards = original_tests.rewards


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(rewards, records, tmp_path, monkeypatch):
    from contextlib import contextmanager

    import c5_timing_semantic_model as semantic
    import prepare_c5_timing_features as features
    import terra_polarity_judge as judge
    import train_terra_c5_tail as owner
    from transformers import ByT5Tokenizer

    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(final, "ROOT", tmp_path)
    tool_directory = tmp_path / "tools"
    tool_directory.mkdir()
    for name in ("train_c5_caption_timing.py", "c5_caption_timing_final.py"):
        (tool_directory / name).write_text("# synthetic source")
    monkeypatch.setattr(final, "__file__", str(tool_directory / "c5_caption_timing_final.py"))
    paired, semantic_directory = rewards
    graded = training.original.authenticated_rewards(paired, semantic_directory)
    reward_path = tmp_path / "rewards.json"
    save(reward_path, graded)
    paired_path = tmp_path / "comparison.json"
    save(paired_path, paired)
    donors = training.original.donor_indices(records)
    pools = []
    for index, record in enumerate(records):
        path = tmp_path / f"pools/clip-{index:04d}.json"
        save(path, original_tests.pool_for(record, records[donors[index]]))
        pools.append(data.artifact(path))
    qualification_path = tmp_path / "runs/qualification/plan.json"
    save(qualification_path, {"synthetic": True})
    audit_path = tmp_path / "runs/data-audit.json"
    save(audit_path, {"synthetic": True})
    model = student()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    plan = {
        "qualification_plan": data.artifact(qualification_path),
        "data_audit": data.artifact(audit_path),
        "model_plan": {"reference": {"bundle_root": "synthetic-model"}},
        "maximum_log_rate": model.timing.maximum_log_rate,
        "records": records,
        "pool_assets": pools,
        "rewards": data.artifact(reward_path),
        "comparison": data.artifact(paired_path),
        "raw_grading_evidence": graded["raw"],
        "recipe": training.RECIPE,
        "batches": training.original.batches(),
        "source_sha256": {},
        "grading_provenance": {
            "reused_training_requests": len(paired["requests"]),
            "qualification_source": {
                "complete_authentic_source_passes": 2,
                "inherited_API_attempts_including_predecessor": 3,
                "inherited_orchestrated_failed_request_replays": 1,
                "new_Terra_requests": 0,
            },
            "unique_historical_API_attempts_including_failed_predecessor": len(paired["requests"])
            + 3,
            "new_Terra_requests": 0,
        },
    }
    path = tmp_path / "runs/caption-training/plan.json"
    save(path, plan)
    monkeypatch.setattr(training, "checked", lambda path: copy.deepcopy(plan))
    generator = np.random.default_rng(731)
    streams = {
        name: (generator.normal(size=(5, width)) * np.linspace(0.1, 2, 5)[:, None]).astype(
            np.float32
        )
        for name, width in WIDTHS.items()
    }
    seen = []

    class Dataset:
        def __init__(self, path):
            assert path == audit_path

        def load(self, index):
            seen.append(index)
            return {
                "streams": copy.deepcopy(streams),
                "valid_intervals": np.zeros(4, dtype=np.bool_)
                if index == 5
                else np.ones(4, dtype=np.bool_),
            }

    monkeypatch.setattr(training, "TimingFeatureDataset", Dataset)
    monkeypatch.setattr(semantic, "require_model", lambda m, p: m.require_fixed())
    monkeypatch.setattr(features, "require_environment", lambda: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda count: None)
    monkeypatch.setattr(training.original, "guard", lambda *args: None)
    monkeypatch.setattr(
        ByT5Tokenizer, "from_pretrained", lambda *args, **kwargs: original_tests.Tokenizer()
    )
    monkeypatch.setattr(
        judge, "Pacer", lambda *args: pytest.fail("caption training attempted a new grade")
    )
    lease = [False]

    @contextmanager
    def exclusive():
        assert not lease[0]
        lease[0] = True
        try:
            yield
        finally:
            lease[0] = False

    def load(plan):
        assert lease[0]
        return model, original_tests.Tokenizer()

    monkeypatch.setattr(owner, "gpu_lease", exclusive)
    monkeypatch.setattr(semantic, "load_student", load)
    output = tmp_path / "checkpoints/caption-training"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(tmp_path / "tools/train_c5_caption_timing.py"),
            "--plan",
            str(path),
            "--output",
            str(output),
        ],
    )
    return path, output, plan, model, before, seen, lease


def test_full1064_exposures_and134_actual_synthetic_updates_replay_exactly(experiment):
    path, output, plan, model, before, seen, lease = experiment
    training.run(path, output)
    assert not lease[0] and len(seen) == 1064
    assert sorted(seen) == sorted(list(range(532)) * 2)
    assert seen.count(5) == 2
    for name, value in model.state_dict().items():
        if name != "timing.rate_weights":
            assert torch.equal(value, before[name])
    result_path = output / "result.json"
    result = data.read(result_path)
    assert result["optimizer_updates"] == 134 and result["new_Terra_requests"] == 0
    assert result["exposures"] == 1064 and result["completed_epochs"] == 2
    assert not list(output.rglob("response.json"))
    poll_path = path.with_name("terminal-poll.json")
    poll = {
        "tool": "write_stdin",
        "requested_session_id": 123,
        "actual_owned_pid": data.read(output / "worker.json")["pid"],
        "tool_result": {"exit_code": 0},
    }
    save(poll_path, poll)
    audited = final.audit(path, result_path, poll_path)
    assert audited["optimizer_updates_replayed"] == 134
    assert audited["weights"] == model.timing.rate_weights.detach().tolist()
    assert audited["grading_provenance"] == plan["grading_provenance"]
    target = output / "step-0001.json"
    first_bytes = target.read_bytes()
    result_bytes = result_path.read_bytes()
    for change in ("caption", "counts", "normalization", "loss", "gradient", "after", "partial"):
        step, altered = json.loads(first_bytes), json.loads(result_bytes)
        risk = step["samples"][0]["risk"]
        if change == "caption":
            risk["caption_weight"] = 0
        elif change == "counts":
            risk["candidate_token_counts"][0] += 1
        elif change == "normalization":
            risk["candidate_normalized_log_probabilities"][0] += 0.01
        elif change == "loss":
            risk["loss"] += 1
        elif change == "gradient":
            step["gradient"]["gradient_values"] = [-x for x in step["gradient"]["gradient_values"]]
        elif change == "after":
            step["after"][0] += 0.01
        else:
            altered["steps"].pop()
        save(target, step)
        altered["steps"][0] = data.artifact(target)
        save(result_path, altered)
        with pytest.raises(ValueError):
            final.audit(path, result_path, poll_path)
        target.write_bytes(first_bytes)
        result_path.write_bytes(result_bytes)
    adapter_path = output / "final-adapter.pt"
    adapter_bytes = adapter_path.read_bytes()
    saved = torch.load(adapter_path, map_location="cpu", weights_only=True)
    saved["optimizer"]["state"][0]["exp_avg"][0] += 0.01
    torch.save(saved, adapter_path)
    altered = json.loads(result_bytes)
    altered["adapter"] = data.artifact(adapter_path)
    save(result_path, altered)
    with pytest.raises(ValueError, match="AdamW state"):
        final.audit(path, result_path, poll_path)
    adapter_path.write_bytes(adapter_bytes)
    result_path.write_bytes(result_bytes)
    save(poll_path, {**poll, "tool_result": {"session_id": 123}})
    with pytest.raises(ValueError, match="actual successful terminal"):
        final.audit(path, result_path, poll_path)
    save(poll_path, poll)
    extra = output / "unbound.json"
    save(extra, {})
    with pytest.raises(ValueError, match="extra or aliased"):
        final.audit(path, result_path, poll_path)
    extra.unlink()
    final.audit(path, result_path, poll_path)
    with pytest.raises(ValueError, match="one fresh complete optimizer"):
        training.run(path, output.with_name("another"))
    assert not output.with_name("another").exists()


def test_changed_source_calibration_stops_before_optimizer_creation(experiment, monkeypatch):
    path, output, _, _, _, seen, lease = experiment
    request = path.parents[2] / "semantic/request-000/response.json"
    response = data.read(request)
    scores = json.loads(response["output"][0]["content"][0]["text"])
    scores["scores"][-1]["score"] = 4
    response["output"][0]["content"][0]["text"] = json.dumps(scores)
    save(request, response)
    saved_path = request.with_name("scores.json")
    saved = data.read(saved_path)
    saved["scores"][-1] = 4
    saved["response_sha256"] = data.sha(request)
    save(saved_path, saved)
    monkeypatch.setattr(
        torch.optim,
        "AdamW",
        lambda *args, **kwargs: pytest.fail("optimizer created before calibrated rewards"),
    )
    with pytest.raises(ValueError, match="original controls"):
        training.run(path, output)
    assert not seen and not lease[0] and not list(output.glob("step-*.json"))
    assert data.read(output / "exit.json")["exit_code"] == 1


def test_missing_actual_caption_qualification_blocks_freeze_before_parent_admission(
    tmp_path, monkeypatch
):
    import c5_timing_final as parent
    import probe_c5_caption_timing as qualification

    monkeypatch.setattr(data, "ROOT", tmp_path)

    def refuse(*args):
        raise ValueError("actual caption qualification absent")

    monkeypatch.setattr(qualification, "checked_qualification", refuse)
    monkeypatch.setattr(
        parent,
        "checked_final",
        lambda *args: pytest.fail("source admitted before actual caption qualification"),
    )
    plan = tmp_path / "runs/new/plan.json"
    with pytest.raises(ValueError, match="actual caption qualification absent"):
        training.freeze(
            plan, *(tmp_path / name for name in ("qplan", "qresult", "qpoll", "parent"))
        )
    assert not plan.parent.exists()


def test_admission_joins_public_data_and_shared_grade_ancestor_without_checkpoint_inheritance(
    experiment, monkeypatch
):
    import c5_timing_final as parent_auditor
    import probe_c5_caption_timing as qualification

    path, _, runtime, _, _, _, _ = experiment
    root = training.ROOT
    ancestors = [
        root / f"runs/source-qualification/{name}.json" for name in ("plan", "result", "poll")
    ]
    for ancestor in ancestors:
        save(ancestor, {"synthetic": True})
    source = {
        "model_plan": runtime["model_plan"],
        "data_audit": runtime["data_audit"],
        "records": runtime["records"],
        "qualification_plan": data.artifact(ancestors[0]),
        "qualification_result": data.artifact(ancestors[1]),
        "qualification_terminal_poll": data.artifact(ancestors[2]),
    }
    source_path = path.with_name("source-plan.json")
    save(source_path, source)
    # Source pool files are stored next to this synthetic result's directory.
    source_result_path = root / "source-result.json"
    save(
        source_result_path,
        {
            "comparison": runtime["comparison"],
            "rewards": runtime["rewards"],
            "new_Terra_requests": runtime["grading_provenance"]["reused_training_requests"],
        },
    )
    parent_path = root / "runs/source-final-audit.json"
    save(parent_path, {"synthetic": True})
    parent = {
        "plan": data.artifact(source_path),
        "result": data.artifact(source_result_path),
        "model_plan": runtime["model_plan"],
        "maximum_log_rate": runtime["maximum_log_rate"],
        "source_sha256": {},
    }
    qualified = {
        "model_plan": runtime["model_plan"],
        "data_audit": runtime["data_audit"],
        "parent_plan": source["qualification_plan"],
        "parent_result": source["qualification_result"],
        "parent_terminal_poll": source["qualification_terminal_poll"],
        "records": [runtime["records"][i] for i in qualification.POLICY["case_indices"]],
        "policy": qualification.POLICY,
        "source_sha256": {},
        "grading_provenance": runtime["grading_provenance"]["qualification_source"],
    }
    monkeypatch.setattr(parent_auditor, "checked_final", lambda *args: copy.deepcopy(parent))
    monkeypatch.setattr(qualification, "checked_qualification", lambda *args: (qualified, {}, []))
    plan = training.specification(*ancestors, parent_path)
    assert plan["recipe"]["parent_model_or_optimizer_inheritance"] is False
    assert plan["recipe"]["risk_alpha"] == 1 and plan["recipe"]["caption_weight"] == 0.1
    assert plan["grading_provenance"] == runtime["grading_provenance"]
    assert len(plan["pool_assets"]) == 532 and len(plan["batches"]) == 134
    assert len(plan["source_sha256"]) == 2
    qualified["parent_result"] = {"different": True}
    with pytest.raises(ValueError, match="qualified objective"):
        training.specification(*ancestors, parent_path)
