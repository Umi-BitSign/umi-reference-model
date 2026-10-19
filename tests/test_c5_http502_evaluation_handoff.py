"""Execute the real coordinator with synthetic admission and child processes.

No model, GPU, Terra request, real training qualification or evaluation is run.
"""

from __future__ import annotations

import copy
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_c5_word_final as base
import evaluate_c5_word_http502_final as pipeline
from benchmark_fleurs_dev import sha256
from terra_training_recovery import artifact


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    source = tmp_path / "tools/evaluate_c5_word_http502_final.py"
    source.parent.mkdir()
    source.write_text("# synthetic coordinator source\n")
    monkeypatch.setattr(pipeline, "__file__", str(source))
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(base, "artifact", descriptor)
    monkeypatch.setattr(base, "storage", lambda **kwargs: None)
    monkeypatch.setattr(base, "identity", lambda pid, argv: {"pid": pid, "synthetic_argv": argv})
    monkeypatch.setattr(base, "now", lambda: "2026-10-19T04:20:00+00:00")
    initial = {"synthetic_same_final": True}
    plans = tmp_path / "runs/training"
    results = tmp_path / "checkpoints/training"
    plans.mkdir(parents=True)
    results.mkdir(parents=True)
    training, result, audit = plans / "plan.json", results / "result.json", plans / "audit.json"
    for path in (training, result, audit):
        path.write_text('{"synthetic":true}')
    result.with_name("exit.json").write_text(
        json.dumps({"observed_at": "2026-10-19T04:00:00+00:00"})
    )
    calls = []
    preconditions = []
    waits = []
    state = {"fault": None}

    def prerequisites(*args):
        assert args == (training, result, audit)
        preconditions.append(args)
        return (
            {"changed": True}
            if state["fault"] == "changed_final" and len(preconditions) > 1
            else copy.deepcopy(initial)
        )

    monkeypatch.setattr(pipeline, "prerequisites", prerequisites)
    monkeypatch.setattr(
        pipeline,
        "control_sources",
        lambda: {"tools/evaluate_c5_word_http502_final.py": sha256(source)},
    )

    def wait(predecessor):
        waits.append(predecessor)
        return predecessor["API_start_not_before"]

    monkeypatch.setattr(base, "wait_for_predecessor", wait)

    def child(argv, check):
        assert check is False
        kind = ("word", "sentence")[len(calls)]
        calls.append(argv)
        assert (
            Path(argv[1]).name
            == f"evaluate_c5_word_http502_{'words' if kind == 'word' else 'sentences'}.py"
        )
        assert argv[argv.index("--training-plan") + 1] == str(training)
        assert argv[argv.index("--training-result") + 1] == str(result)
        if state["fault"] == kind + "_child_failure":
            return SimpleNamespace(returncode=1)
        plan = Path(argv[argv.index("--freeze") + 1])
        plan.parent.mkdir(parents=True)
        plan.write_text('{"synthetic":true}')
        out = Path(argv[argv.index("--output") + 1])
        out.mkdir(parents=True)
        (out / "complete.json").write_text(json.dumps({"kind": kind}))
        if state["fault"] == "source_changed" and kind == "word":
            source.write_text("# changed source\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(base.subprocess, "run", child)
    authenticated = []

    def previous(path, kind, initialization, *, evaluation_only=False):
        assert initialization == initial and evaluation_only is True
        assert path.is_file()
        if state["fault"] == "calibration_failure" and kind == "word":
            raise ValueError("synthetic failed calibration")
        value = {
            "kind": kind,
            "parent_complete": descriptor(path),
            "API_start_not_before": "2026-10-19T04:10:00+00:00",
            "semantic_decline_recorded_for_evaluation_only": state["fault"] == "word_decline"
            and kind == "word",
        }
        authenticated.append(kind)
        return value

    monkeypatch.setattr(pipeline, "previous_evaluation", previous)
    return SimpleNamespace(
        root=tmp_path,
        source=source,
        training=training,
        result=result,
        audit=audit,
        output=tmp_path / "checkpoints/coordinator",
        state=state,
        calls=calls,
        preconditions=preconditions,
        waits=waits,
        authenticated=authenticated,
        initial=initial,
        descriptor=descriptor,
    )


def test_real_serial_handoff_keeps_source_identity_both_gates_and_spacing(fixture):
    f = fixture
    pipeline.pipeline(f.training, f.result, f.audit, f.output)
    complete = json.loads((f.output / "complete.json").read_text())
    handoff = json.loads((f.output / "handoff.json").read_text())
    assert len(f.calls) == 2 and len(f.preconditions) == 2
    assert f.authenticated == ["word", "sentence", "word", "sentence"]
    assert complete["initialization"] == f.initial and set(complete["evaluations"]) == {
        "word",
        "sentence",
    }
    assert complete["automatic_training_continuation"] is False
    assert handoff["coordinator_source_sha256"] == sha256(f.source)
    assert handoff["complete_evaluation_order"] == ["word96", "sentence34"]
    assert f.waits[0]["API_start_not_before"] == "2026-10-19T04:01:00+00:00"
    assert f.waits[1] == complete["evaluations"]["word"]
    assert len(complete["API_spacing"]) == 2
    assert json.loads((f.output / "exit.json").read_text())["exit_code"] == 0
    assert not (f.output / "failure.json").exists()
    with pytest.raises(ValueError, match="fresh owned output"):
        pipeline.pipeline(f.training, f.result, f.audit, f.output)
    assert len(f.calls) == 2


def test_word_decline_still_runs_full_sentence_gate_without_training_continuation(fixture):
    f = fixture
    f.state["fault"] = "word_decline"
    pipeline.pipeline(f.training, f.result, f.audit, f.output)
    complete = json.loads((f.output / "complete.json").read_text())
    assert len(f.calls) == 2
    assert complete["evaluations"]["word"]["semantic_decline_recorded_for_evaluation_only"] is True
    assert (
        complete["evaluations"]["sentence"]["semantic_decline_recorded_for_evaluation_only"]
        is False
    )
    assert complete["automatic_training_continuation"] is False
    assert complete["promotion_authorized"] is False


@pytest.mark.parametrize(
    "fault,children",
    [
        ("word_child_failure", 1),
        ("sentence_child_failure", 2),
        ("calibration_failure", 1),
        ("source_changed", 1),
        ("changed_final", 2),
    ],
)
def test_failed_child_calibration_or_changed_evidence_cannot_publish_success(
    fixture, fault, children
):
    f = fixture
    f.state["fault"] = fault
    with pytest.raises((ValueError, RuntimeError)):
        pipeline.pipeline(f.training, f.result, f.audit, f.output)
    assert len(f.calls) == children
    assert not (f.output / "complete.json").exists()
    failure = json.loads((f.output / "failure.json").read_text())
    assert failure["automatic_retries"] == 0
    assert json.loads((f.output / "exit.json").read_text())["exit_code"] == 1
    assert (f.output / "handoff.json").is_file()
