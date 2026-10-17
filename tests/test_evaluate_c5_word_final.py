from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_c5_word_final as handoff
import test_c5_mixed_risk_evidence as evidence_fixtures
from terra_training_recovery import artifact


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def completed(tmp_path, monkeypatch):
    """Synthetic terminal bindings; completed-model authentication is a stub in this fixture."""
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(handoff, "artifact", descriptor)
    plan = tmp_path / "runs/training/plan.json"
    result_path = tmp_path / "checkpoints/training/result.json"
    write(plan, {})
    write(result_path.with_name("worker.json"), {"pid": 123})
    write(result_path.with_name("exit.json"), {"pid": 123, "exit_code": 0})
    result = {
        "optimizer_updates": 216,
        "visual_initialization_sha256": "initial",
        "visual_final_sha256": "final",
        "checkpoints": [],
    }
    checkpoints, epochs = [], []
    for index in range(4):
        examples, updates = (index + 1) * 214, (index + 1) * 54
        visual = result_path.with_name(f"visual-after-{examples:06d}.safetensors")
        optimizer = visual.with_name(visual.stem + "-optimizer.safetensors")
        visual.write_bytes(b"synthetic visual binding")
        optimizer.write_bytes(b"synthetic Adam binding")
        commit = visual.with_suffix(".json")
        write(
            commit,
            {
                "visual": descriptor(visual),
                "optimizer": descriptor(optimizer),
                "training_examples": examples,
                "optimizer_updates": updates,
            },
        )
        checkpoints.append(
            {
                "visual": descriptor(visual),
                "optimizer": descriptor(optimizer),
                "commit": descriptor(commit),
                "training_examples": examples,
                "all_Adam_steps": updates,
                "parameter_tensors": 350,
            }
        )
        epochs.append(
            {
                "epoch": index,
                "unique_clips": 214,
                "exposures": 214,
                "optimizer_updates": 54,
                "tail_clips": 2,
            }
        )
        result["checkpoints"].append(descriptor(visual))
    result["visual"] = result["checkpoints"][-1]
    write(result_path, result)
    binding = {
        "word_training_plan": descriptor(plan),
        "word_training_result": descriptor(result_path),
        "word_training_evidence": [],
        "initial_visual_state_sha256": "final",
    }
    monkeypatch.setattr(handoff, "word_initialization", lambda *a: copy.deepcopy(binding))
    source = plan.with_name("terminal-audit.py")
    source.write_text("synthetic audit source, not an actual completed model audit")
    monkeypatch.setattr(handoff, "TERMINAL_AUDIT_SOURCE_SHA256", handoff.sha256(source))
    audit = {
        "schema": "umi-studio-complete-word-training-terminal-audit/1",
        "complete": True,
        "same_worker_exit0": True,
        "all_original_350_tensor_checkpoints_and_actual_reward_evidence_validated": True,
        "all_pinned_sources_and_model_assets_match": True,
        "raw_request_response_pairings_revalidated_by_checked_final": True,
        "training_examples": 856,
        "optimizer_updates": 216,
        "visual_initialization_sha256": "initial",
        "visual_final_sha256": "final",
        "plan": binding["word_training_plan"],
        "result": binding["word_training_result"],
        "training_evidence": binding["word_training_evidence"],
        "worker": descriptor(result_path.with_name("worker.json")),
        "exit": descriptor(result_path.with_name("exit.json")),
        "audit_source": descriptor(source),
        "new_Terra_requests_during_audit": 0,
        "optimizer_updates_during_audit": 0,
        "held_out_semantic_quality_evaluated": False,
        "promotion_authorized": False,
        "epochs": epochs,
        "checkpoints": checkpoints,
    }
    audit_path = plan.with_name("terminal-audit.json")
    write(audit_path, audit)
    return SimpleNamespace(
        plan=plan,
        result=result_path,
        audit_path=audit_path,
        audit=audit,
        source=source,
        binding=binding,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "partial",
        "count",
        "fixed_values",
        "source",
        "worker",
        "exit",
        "epoch",
        "step",
        "optimizer",
        "tail",
        "moment_file",
        "stopped",
    ],
)
def test_terminal_handoff_requires_all856_exposures_and_four_actual_Adam_audit_bindings(
    completed, mutation
):
    env = completed
    if mutation in ("partial", "count", "fixed_values", "epoch", "step", "optimizer", "tail"):
        if mutation == "partial":
            env.audit["complete"] = False
        elif mutation == "count":
            env.audit["training_examples"] = 642
        elif mutation == "fixed_values":
            env.audit["visual_final_sha256"] = "other"
        elif mutation == "epoch":
            env.audit["epochs"].pop()
        elif mutation == "step":
            env.audit["checkpoints"][0]["all_Adam_steps"] = 53
        elif mutation == "optimizer":
            env.audit["checkpoints"][0]["optimizer"] = env.audit["checkpoints"][1]["optimizer"]
        elif mutation == "tail":
            env.audit["epochs"][0]["tail_clips"] = 4
        write(env.audit_path, env.audit)
    elif mutation == "source":
        env.source.write_text("changed audit")
    elif mutation == "worker":
        write(env.result.with_name("worker.json"), {"pid": 124})
    elif mutation == "exit":
        write(env.result.with_name("exit.json"), {"pid": 123, "exit_code": 1})
    elif mutation == "moment_file":
        (handoff.ROOT / env.audit["checkpoints"][0]["optimizer"]["path"]).write_bytes(
            b"changed Adam bytes"
        )
    elif mutation == "stopped":
        write(env.result.with_name("stopped.json"), {})
    if mutation is None:
        assert handoff.prerequisites(env.plan, env.result, env.audit_path) == env.binding
    else:
        with pytest.raises(ValueError):
            handoff.prerequisites(env.plan, env.result, env.audit_path)


def test_unfinished_training_stops_before_terminal_audit_read_child_or_any_output(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(handoff, "ROOT", tmp_path)

    def unfinished(*args):
        raise ValueError("word training is unfinished")

    monkeypatch.setattr(handoff, "word_initialization", unfinished)
    monkeypatch.setattr(handoff.subprocess, "run", lambda *a, **k: pytest.fail("child launched"))
    output = tmp_path / "checkpoints/handoff"
    with pytest.raises(ValueError, match="unfinished"):
        handoff.pipeline(tmp_path / "plan", tmp_path / "result", tmp_path / "missing-audit", output)
    assert not output.exists()


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "word",
        "sentence",
        "auth_word",
        "auth_sentence",
        "missing_word",
        "wrong_exit",
        "changed_final",
        "changed_evaluation",
        "control_source",
    ],
)
def test_complete_word96_then_sentence34_children_are_serial_authenticated_paced_and_never_retried(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    monkeypatch.setattr(handoff, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(handoff, "storage", lambda *a, **k: None)
    monkeypatch.setattr(handoff, "identity", lambda pid, command: {"pid": pid})
    events = []
    initial = {"fixed_final": "binding"}
    audit = tmp_path / "runs/training/terminal-audit.json"
    write(audit, {})
    write(audit.with_name("exit.json"), {"observed_at": "2026-10-17T20:00:00+00:00"})
    for name in handoff.CONTROL_SOURCES:
        path = tmp_path / "tools" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic control source")
    checks = 0

    def prerequisites(*args):
        nonlocal checks
        checks += 1
        events.append("verified final")
        return {"changed": True} if failure == "changed_final" and checks == 2 else initial

    def child(command, check):
        assert check is False and "--freeze" in command and "--pipeline" in command
        assert command[command.index("--training-plan") + 1] == str(audit.with_name("plan.json"))
        kind = "sentence" if "sentences" in command[1] else "word"
        events.append(kind)
        if kind == "sentence":
            assert events[:5] == ["verified final", "cooldown", "word", "auth_word", "cooldown"]
        plan = Path(command[command.index("--freeze") + 1])
        output = Path(command[command.index("--output") + 1])
        write(plan, {})
        write(output / "worker.json", {"pid": 101})
        write(
            output / "exit.json", {"pid": 102 if failure == "wrong_exit" else 101, "exit_code": 0}
        )
        if failure != "missing_word":
            write(output / "complete.json", {"complete": True})
        if failure == "control_source" and kind == "word":
            (tmp_path / "tools" / handoff.CONTROL_SOURCES[0]).write_text("changed control source")
        return SimpleNamespace(returncode=int(failure == kind))

    authenticated = {"word": 0, "sentence": 0}

    def authenticate(path, kind, initialization, *, evaluation_only):
        assert initialization is initial
        assert evaluation_only is True
        handoff.require_terminal(path)
        assert json.loads(path.read_text())["complete"] is True
        events.append("auth_" + kind)
        authenticated[kind] += 1
        if failure == "auth_" + kind:
            raise ValueError("original calibration or semantic decline stop")
        return {
            "cases": 96 if kind == "word" else 34,
            "API_start_not_before": "2026-10-17T20:01:00+00:00",
            "changed": failure == "changed_evaluation" and authenticated[kind] > 1,
        }

    monkeypatch.setattr(handoff, "prerequisites", prerequisites)
    monkeypatch.setattr(handoff.subprocess, "run", child)
    monkeypatch.setattr(handoff, "previous_evaluation", authenticate)
    monkeypatch.setattr(
        handoff,
        "wait_for_predecessor",
        lambda plan: events.append("cooldown") or "2026-10-17T20:01:00+00:00",
    )
    output = tmp_path / "checkpoints/handoff"
    arguments = audit.with_name("plan.json"), audit.with_name("result.json"), audit, output
    if failure is None:
        handoff.pipeline(*arguments)
        assert events == [
            "verified final",
            "cooldown",
            "word",
            "auth_word",
            "cooldown",
            "sentence",
            "auth_sentence",
            "verified final",
            "auth_word",
            "auth_sentence",
        ]
        result = json.loads((output / "complete.json").read_text())
        assert result["handoff"] == handoff.artifact(output / "handoff.json")
        assert result["API_spacing"] == [
            handoff.artifact(output / name)
            for name in ("training-to-word-spacing.json", "word-to-sentence-spacing.json")
        ]
        assert result["evaluations"]["word"]["cases"] == 96
        assert result["evaluations"]["sentence"]["cases"] == 34
        assert (
            result["automatic_training_continuation"] is False
            and result["promotion_authorized"] is False
        )
    else:
        with pytest.raises((ValueError, RuntimeError, FileNotFoundError)):
            handoff.pipeline(*arguments)
        assert not (output / "complete.json").exists()
        assert events.count("word") <= 1 and events.count("sentence") <= 1
        if failure in ("word", "auth_word", "missing_word", "wrong_exit", "control_source"):
            assert "sentence" not in events
    assert json.loads((output / "exit.json").read_text())["exit_code"] == int(failure is not None)


@pytest.mark.parametrize("existing", ["plan", "scores", "output"])
def test_handoff_refuses_any_previous_plan_score_plan_or_output_before_children(
    tmp_path, monkeypatch, existing
):
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    monkeypatch.setattr(handoff, "prerequisites", lambda *a: {})
    monkeypatch.setattr(handoff.subprocess, "run", lambda *a, **k: pytest.fail("replayed child"))
    plan, output, _ = handoff.locations()["word"]
    folder = {
        "plan": plan.parent,
        "scores": plan.parent.with_name(plan.parent.name + "-scores"),
        "output": output,
    }[existing]
    folder.mkdir(parents=True)
    with pytest.raises(ValueError, match="replay"):
        handoff.pipeline(
            tmp_path / "plan",
            tmp_path / "result",
            tmp_path / "audit",
            tmp_path / "checkpoints/handoff",
        )


def test_handoff_uses_existing_authenticator_and_cross_process_cooldown():
    import c5_mixed_risk_evidence
    import probe_c5_mixed_semantic

    assert handoff.previous_evaluation is c5_mixed_risk_evidence.previous_evaluation
    assert handoff.wait_for_predecessor is probe_c5_mixed_semantic.wait_for_predecessor


@pytest.mark.parametrize(
    "timestamp", ["2026-10-17T20:00:00+00:00", "2026-10-17T16:00:00-04:00", "2026-10-17T20:00:00"]
)
def test_training_boundary_preserves_sixty_seconds_after_actual_parent_exit(tmp_path, timestamp):
    path = tmp_path / "result.json"
    write(path.with_name("exit.json"), {"observed_at": timestamp})
    if datetime.fromisoformat(timestamp).tzinfo is None:
        with pytest.raises(ValueError, match="timezone-aware"):
            handoff.training_boundary(path)
    else:
        boundary = handoff.training_boundary(path)
        assert (
            datetime.fromisoformat(boundary["API_start_not_before"])
            - datetime.fromisoformat(timestamp)
        ).total_seconds() == 60


def test_calibrated_word_decline_keeps_complete_sentence_measurement_and_never_authorizes_training(
    tmp_path, monkeypatch
):
    """Real local scoring-file authentication, with synthetic decoded plans and no model/API."""
    build = evidence_fixtures.predecessor.__wrapped__(tmp_path, monkeypatch)
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(handoff, "ROOT", tmp_path)
    monkeypatch.setattr(handoff, "artifact", descriptor)
    monkeypatch.setattr(handoff, "storage", lambda *a, **k: None)
    monkeypatch.setattr(handoff, "identity", lambda pid, command: {"pid": pid})
    monkeypatch.setattr(handoff, "wait_for_predecessor", lambda *a: "2026-10-17T20:01:00+00:00")
    training_result = tmp_path / "checkpoints/train/result.json"
    write(training_result, {"fixture": True})
    write(training_result.with_name("exit.json"), {"observed_at": "2026-10-17T20:00:00+00:00"})
    binding = {
        "word_training_plan": {"path": "runs/train/plan.json", "sha256": "training"},
        "word_training_result": descriptor(training_result),
        "initial_visual": {"path": "checkpoints/train/final", "sha256": "visual"},
        "initial_visual_state_sha256": "state",
    }
    audit = tmp_path / "runs/train/terminal-audit.json"
    write(audit, {})
    monkeypatch.setattr(handoff, "prerequisites", lambda *a: binding)
    monkeypatch.setattr(
        handoff,
        "locations",
        lambda: {
            "word": (tmp_path / "runs/word/plan.json", tmp_path / "checkpoints/word", "words"),
            "sentence": (
                tmp_path / "runs/sentence/plan.json",
                tmp_path / "checkpoints/sentence",
                "sentences",
            ),
        },
    )
    for name in handoff.CONTROL_SOURCES:
        path = tmp_path / "tools" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic control source")
    order = []

    def child(command, check):
        kind = "word" if command[1].endswith("words") else "sentence"
        order.append(kind)
        build(kind, decline=kind == "word")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(handoff.subprocess, "run", child)
    output = tmp_path / "checkpoints/handoff"
    handoff.pipeline(audit.with_name("plan.json"), training_result, audit, output)
    assert order == ["word", "sentence"]
    result = json.loads((output / "complete.json").read_text())
    assert result["evaluations"]["word"]["decision_values"]["stop_for_semantic_decline"] is True
    assert result["evaluations"]["word"]["semantic_decline_recorded_for_evaluation_only"] is True
    assert result["evaluations"]["sentence"]["decision_values"]["clips"] == 34
    assert result["evaluations"]["word"]["decision_values"]["clips"] == 96
    assert result["automatic_training_continuation"] is False
    assert all(
        evaluation["stage_advance_or_promotion_authorized"] is False
        for evaluation in result["evaluations"].values()
    )
