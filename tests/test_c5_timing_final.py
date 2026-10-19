"""Reconstruct all134 tiny optimizer steps and reject altered completion evidence."""

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
import c5_timing_final as final
import test_train_c5_timing as training_tests
import train_c5_timing as training

fit_fixture = training_tests.fit_fixture
records = training_tests.records


def save(path, value):
    path.write_text(json.dumps(value))


def test_all_synthetic_steps_and_adamw_state_replay_exactly_and_reject_mutations(
    fit_fixture, monkeypatch
):
    import math
    import os

    from transformers import ByT5Tokenizer

    path, output, plan, model, _, _ = fit_fixture
    plan["recipe"] = copy.deepcopy(training.RECIPE)
    qualification = path.with_name("qualification-plan.json")
    save(qualification, {"synthetic": True})
    plan["qualification_plan"] = data.artifact(qualification)
    plan["model_plan"] = {"reference": {"bundle_root": "synthetic-public-model"}}
    plan["maximum_log_rate"] = math.log(2)
    plan["source_sha256"] = {}
    save(path, plan)
    save(
        output / "worker.json",
        {
            "uid": os.getuid(),
            "pid": 123,
            "argv": [
                sys.executable,
                str(path.parent / "tools/train_c5_timing.py"),
                "--plan",
                str(path),
                "--output",
                str(output),
            ],
        },
    )
    training.train(path, output, plan)
    result = data.read(output / "result.json")
    rewards = data.read(output / "rewards.json")
    weights, optimizer = final.replay_steps(plan, result, output, rewards)
    assert torch.equal(weights, model.timing.rate_weights)
    saved = torch.load(output / "final-adapter.pt", weights_only=True)
    assert final.same_state(optimizer, saved["optimizer"])
    assert torch.equal(saved["timing"]["rate_weights"], weights)
    altered = copy.deepcopy(saved["optimizer"])
    altered["state"][0]["exp_avg"][0] += 0.01
    assert not final.same_state(optimizer, altered)
    step_path = data.bound(result["steps"][0])
    original_bytes = step_path.read_bytes()
    original = data.read(step_path)
    for change in (
        "before",
        "after",
        "gradient",
        "order",
        "loss",
        "counter",
        "final_weights",
        "final_count",
    ):
        step = copy.deepcopy(original)
        candidate = copy.deepcopy(result)
        if change in ("before", "after"):
            step[change][0] += 0.01
        elif change == "gradient":
            step["gradient"]["gradient_values"] = [-x for x in step["gradient"]["gradient_values"]]
        elif change == "order":
            step["samples"][0]["index"] = (step["samples"][0]["index"] + 1) % 532
        elif change == "loss":
            step["samples"][0]["risk"]["loss"] += 1
        elif change == "counter":
            step["exposures"] -= 1
        elif change == "final_weights":
            candidate["weights"][0] += 0.01
        else:
            candidate["exposures"] -= 1
        save(step_path, step)
        candidate["steps"][0] = data.artifact(step_path)
        with pytest.raises(ValueError):
            final.replay_steps(plan, candidate, output, rewards)
        step_path.write_bytes(original_bytes)
    # Full orchestration uses the fake training fixture's authenticated-reward
    # boundary; actual raw score parsing is tested in test_train_c5_timing.py.
    save(output / "exit.json", {"pid": 123, "exit_code": 0})
    poll = path.with_name("poll.json")
    save(
        poll,
        {
            "tool": "write_stdin",
            "requested_session_id": 9,
            "actual_owned_pid": 123,
            "tool_result": {"exit_code": 0},
        },
    )
    claim = {"plan": data.artifact(path), "worker": data.artifact(output / "worker.json")}
    save(path.with_name("execution-claim.json"), claim)
    save(qualification.with_name("optimizer-child-claim.json"), claim)
    source = Path(final.__file__)
    copied_source = path.parent / "tools/c5_timing_final.py"
    copied_source.parent.mkdir()
    copied_source.write_bytes(source.read_bytes())
    monkeypatch.setattr(final, "ROOT", path.parent)
    monkeypatch.setattr(final, "__file__", str(copied_source))
    monkeypatch.setattr(final, "owned", lambda *args: None)
    monkeypatch.setattr(
        ByT5Tokenizer, "from_pretrained", lambda *args, **kwargs: training_tests.Tokenizer()
    )
    audit = final.audit(path, output / "result.json", poll)
    assert audit["optimizer_updates_replayed"] == 134
    assert audit["complete_training_exposures"] == 1064
    assert audit["weights"] == model.timing.rate_weights.detach().tolist()
    assert audit["quality_gain_established"] is False


def test_live_final_tool_poll_refuses_before_tokenizer_or_model(tmp_path, monkeypatch):
    import os

    from transformers import ByT5Tokenizer

    monkeypatch.setattr(data, "ROOT", tmp_path)
    monkeypatch.setattr(final, "ROOT", tmp_path)
    monkeypatch.setattr(training, "checked", lambda path: {})
    monkeypatch.setattr(
        ByT5Tokenizer,
        "from_pretrained",
        lambda *args, **kwargs: pytest.fail("tokenizer read before actual terminal release"),
    )
    plan, poll = tmp_path / "runs/plan.json", tmp_path / "runs/poll.json"
    plan.parent.mkdir()
    save(plan, {})
    save(
        poll,
        {
            "tool": "write_stdin",
            "requested_session_id": 9,
            "actual_owned_pid": 123,
            "tool_result": {"exit_code": 0, "session_id": 9},
        },
    )
    output = tmp_path / "checkpoints/test"
    output.mkdir(parents=True)
    save(
        output / "worker.json",
        {
            "uid": os.getuid(),
            "pid": 123,
            "argv": [
                sys.executable,
                str(tmp_path / "tools/train_c5_timing.py"),
                "--plan",
                str(plan),
                "--output",
                str(output),
            ],
        },
    )
    save(output / "exit.json", {"pid": 123, "exit_code": 0})
    result = output / "result.json"
    save(result, {})
    with pytest.raises(ValueError, match="actual terminal tool"):
        final.audit(plan, result, poll)


def test_state_equality_checks_types_shapes_and_scalar_options():
    assert final.same_state({"a": [torch.ones(4)]}, {"a": [torch.ones(4)]})
    assert not final.same_state({"a": torch.ones(4)}, {"a": torch.ones(3)})
    assert not final.same_state([1, 2], (1, 2))
    assert not final.same_state({"weight_decay": 0.01}, {"weight_decay": 0.02})
    assert not final.same_state(torch.ones(4), torch.ones(4, dtype=torch.float64))
