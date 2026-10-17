from __future__ import annotations

import json
import math
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import c5_mixed_risk_evidence as evidence
import evaluate_c5_words as words
import test_c5_word_training as fixtures
import test_terra_mixed_semantic_cases as case_fixtures
from launch_terra_visual_job import write
from terra_short_curriculum_learning import comparison_items, score_comparison, semantic_decision
from terra_training_recovery import artifact


@pytest.fixture
def predecessor(tmp_path, monkeypatch):
    """Synthetic complete paired files; no actual model or quality evidence."""
    descriptor = partial(artifact, root=tmp_path)
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(evidence, "artifact", descriptor)

    def build(kind="word", *, decline=False, failed_control=False):
        count = 96 if kind == "word" else 34
        output = tmp_path / "checkpoints" / kind
        output.mkdir(parents=True)
        training = tmp_path / "checkpoints/train/result.json"
        if not training.exists():
            training.parent.mkdir()
            write(training, {"fixture": True})
        inference_path = tmp_path / f"runs/{kind}/plan.json"
        inference_path.parent.mkdir(parents=True)
        write(inference_path, {})
        initializer = {
            "word_training_plan": {"path": "runs/train/plan.json", "sha256": "training"},
            "word_training_result": descriptor(training),
            "initial_visual": {"path": "checkpoints/train/final", "sha256": "visual"},
            "initial_visual_state_sha256": "state",
        }
        inference = {
            "training_plan": initializer["word_training_plan"],
            "training_result": initializer["word_training_result"],
            "candidate_visual": initializer["initial_visual"],
            "candidate_visual_state_sha256": initializer["initial_visual_state_sha256"],
            "source_sha256": {},
        }
        decoded = output / "inference/receipt.json"
        decoded.parent.mkdir()
        write(decoded, {"fixture": True})
        rows = [
            {
                "source": "fixture",
                "example_id": str(index),
                "source_group": str(index // 4),
                "signer": "reserved",
                "reference": f"Reference {index % 3}",
                "role": "development",
                "training_allowed": False,
            }
            for index in range(count)
        ]
        baseline = [{**row, "hypothesis": row["reference"], "WER_diagnostic": 0} for row in rows]
        predictions = [
            {
                **row,
                "hypothesis": "Wrong" if decline else row["reference"],
                "WER_diagnostic": int(decline),
            }
            for row in rows
        ]
        comparison = comparison_items(rows, baseline, predictions)
        score_path = tmp_path / f"runs/{kind}-scores/plan.json"
        score_path.parent.mkdir()
        plan = {
            "inference_plan": descriptor(inference_path),
            "decoded": descriptor(decoded),
            "comparison": comparison,
            "baseline": baseline,
            "predictions": predictions,
            "source_sha256": {},
            "planned_Terra_requests": 2 * math.ceil(len(comparison["items"]) / 16),
        }
        write(score_path, plan)
        directory = output / "scoring"
        directory.mkdir()

        class Pacer(fixtures.FakePacer):
            def score(self, rows, directory):
                super().score(rows, directory)
                return json.loads((directory / "scores.json").read_text())

        scores = score_comparison(
            comparison, directory / "semantic", Pacer(fail_call=1 if failed_control else None)
        )
        decision = {
            **semantic_decision(comparison, scores, baseline, predictions),
            "plan_sha256": evidence.sha256(score_path),
            "new_Terra_requests": plan["planned_Terra_requests"],
            "optimizer_updates": 0,
        }
        decision_path = directory / "decision.json"
        write(decision_path, decision)
        for index, folder in enumerate((output, directory, decoded.parent), start=1):
            write(folder / "worker.json", {"pid": index})
            write(
                folder / "exit.json",
                {"pid": index, "exit_code": 0, "observed_at": "2026-10-17T20:00:00+00:00"},
            )
        parent = output / "complete.json"
        write(
            parent,
            {
                "complete": True,
                "inference_plan": descriptor(inference_path),
                "scoring_plan": descriptor(score_path),
                "decision": descriptor(decision_path),
                "training_result": descriptor(training),
                "optimizer_updates": 0,
                "automatic_retries": 0,
                "promotion_authorized": False,
            },
        )
        module = evidence.word_evaluation if kind == "word" else evidence.sentence_evaluation
        monkeypatch.setattr(module, "checked", lambda path: inference)
        monkeypatch.setattr(module, "checked_scores", lambda path: plan)
        return SimpleNamespace(
            parent=parent,
            initialization=initializer,
            inference=inference,
            scores=plan,
            directory=directory,
            decision=decision_path,
        )

    return build


@pytest.mark.parametrize("kind", ["word", "sentence"])
def test_complete_predecessors_reauthenticate_both_raw_passes_and_keep_failed_gain_separate(
    predecessor, kind
):
    env = predecessor(kind)
    result = evidence.previous_evaluation(env.parent, kind, env.initialization)
    assert result["decision_values"]["passes_semantic_research_gate"] is False
    assert result["decision_values"]["stop_for_semantic_decline"] is False
    assert result["stage_advance_or_promotion_authorized"] is False
    assert result["API_start_not_before"] == "2026-10-17T20:01:00+00:00"
    assert len(result["raw_scoring_evidence"]) >= 11


@pytest.mark.parametrize(
    "mutation",
    [
        "decline",
        "calibration",
        "response",
        "failure",
        "stopped",
        "exit",
        "worker",
        "initializer",
        "count",
        "decision",
        "timezone",
    ],
)
def test_predecessor_rejects_original_stops_partial_coverage_and_actual_evidence_drift(
    predecessor, mutation
):
    env = predecessor(decline=mutation == "decline", failed_control=mutation == "calibration")
    if mutation == "response":
        request = next((env.directory / "semantic").glob("pass-1-batch-*"))
        (request / "response.json").write_text("{}")
    elif mutation in ("failure", "stopped"):
        write(env.directory / f"{mutation}.json", {})
    elif mutation in ("exit", "timezone"):
        path = env.parent.with_name("exit.json")
        record = json.loads(path.read_text())
        record["exit_code" if mutation == "exit" else "observed_at"] = (
            1 if mutation == "exit" else "2026-10-17T20:00:00"
        )
        path.write_text(json.dumps(record))
    elif mutation == "worker":
        for name in ("worker.json", "exit.json"):
            path = env.parent.with_name(name)
            record = json.loads(path.read_text())
            record["pid"] = None
            path.write_text(json.dumps(record))
    elif mutation == "initializer":
        env.inference["candidate_visual_state_sha256"] = "other"
    elif mutation == "count":
        env.scores["comparison"]["cases"].pop()
    elif mutation == "decision":
        record = json.loads(env.decision.read_text())
        record["absolute_gain"] = 1
        env.decision.write_text(json.dumps(record))
        record = json.loads(env.parent.read_text())
        record["decision"] = evidence.artifact(env.decision)
        env.parent.write_text(json.dumps(record))
    with pytest.raises((ValueError, KeyError)):
        evidence.previous_evaluation(env.parent, "word", env.initialization)


def test_incomplete_training_blocks_evaluations_inputs_and_any_plan_output(tmp_path, monkeypatch):
    def unfinished(*args):
        raise ValueError("training incomplete")

    monkeypatch.setattr(evidence, "word_initialization", unfinished)
    monkeypatch.setattr(
        evidence, "previous_evaluation", lambda *a: pytest.fail("evaluation reached")
    )
    monkeypatch.setattr(evidence, "checked_result", lambda *a: pytest.fail("inputs reached"))
    with pytest.raises(ValueError, match="training incomplete"):
        evidence.specification(*([tmp_path] * 6))


def test_prior_raw_scoring_authenticator_is_the_prepared_word_evaluator():
    assert evidence.word_evaluation.authenticated_passes is words.authenticated_passes


def test_significant_decline_can_be_recorded_for_complete_evaluation_but_still_blocks_stage_advance(
    predecessor,
):
    env = predecessor(decline=True)
    result = evidence.previous_evaluation(
        env.parent, "word", env.initialization, evaluation_only=True
    )
    assert result["decision_values"]["stop_for_semantic_decline"] is True
    assert result["semantic_decline_recorded_for_evaluation_only"] is True
    assert result["stage_advance_or_promotion_authorized"] is False
    with pytest.raises(ValueError, match="stop condition"):
        evidence.previous_evaluation(env.parent, "word", env.initialization)


def test_evaluation_only_mode_never_bypasses_failed_calibration(predecessor):
    env = predecessor(failed_control=True)
    with pytest.raises(ValueError, match="stop condition"):
        evidence.previous_evaluation(env.parent, "word", env.initialization, evaluation_only=True)


@pytest.mark.parametrize("failure", [None, "word", "sentence", "inputs", "selector"])
def test_specification_joins_all_complete_prerequisites_in_order_without_model_work(
    tmp_path, monkeypatch, failure
):
    monkeypatch.setattr(evidence, "ROOT", tmp_path)
    monkeypatch.setattr(evidence, "artifact", partial(artifact, root=tmp_path))
    selected, inputs = case_fixtures.complete_binding.__wrapped__()
    inputs.update(authority={"fixture": True}, verification_source_sha256="reader")
    original = {
        "reference": {},
        "initial_reference_hashes": {},
        "expected_visual_gradient_names": [f"vision.{index}" for index in range(350)],
        "resource_policy": "runs/budget.json",
        "resource_policy_sha256": "budget",
        "assets": {},
    }
    initialization = {"original_model_plan": original}
    events = []

    def initializer(*args):
        events.append("complete training")
        return initialization

    def evaluation(path, kind, binding):
        assert binding is initialization
        events.append(kind)
        if failure == kind:
            raise ValueError("unfinished evaluation")
        return {
            "source_sha256": {},
            "API_start_not_before": f"2026-10-17T20:0{int(kind == 'sentence')}:00+00:00",
        }

    def checked_inputs(*args):
        events.append("inputs")
        if failure == "inputs":
            raise ValueError("unfinished inputs")
        return inputs

    monkeypatch.setattr(evidence, "word_initialization", initializer)
    monkeypatch.setattr(evidence, "previous_evaluation", evaluation)
    monkeypatch.setattr(evidence, "checked_result", checked_inputs)
    train = tmp_path / "runs/train.json"
    train.parent.mkdir()
    write(train, {"source_sha256": {}})
    data = tmp_path / "runs/data.json"
    write(data, {"source_sha256": {}, "assets": {}})
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in (
        "probe_c5_mixed_semantic.py",
        "c5_mixed_risk_evidence.py",
        "c5_mixed_semantic_model.py",
        "terra_mixed_semantic_protocol.py",
        "terra_mixed_semantic_cases.py",
        "terra_mixed_native_input.py",
        "prepare_terra_mixed_native.py",
        "c5_word_final.py",
        "evaluate_c5_words.py",
        "evaluate_c5_word_sentences.py",
        "c5_word_evaluation_io.py",
    ):
        (tools / name).write_text("fixture")
    archived = tmp_path / "runs/archive.py"
    archived.write_text("fixture")
    proposal = tmp_path / "runs/proposal.json"
    write(
        proposal,
        {
            "case_selector_source": evidence.artifact(archived),
            "input_selector_source": evidence.artifact(archived),
            "selected_training_cases": selected,
        },
    )
    if failure == "selector":
        (tools / "terra_mixed_semantic_cases.py").write_text("changed")
    if failure is None:
        result = evidence.specification(train, train, data, data, train, train, proposal)
        assert [bound["case"] for bound in result["records"]] == selected
        assert result["API_start_not_before"] == "2026-10-17T20:01:00+00:00"
        assert result["stage_advance_or_promotion_authorized"] is False
        assert result["limits"] is evidence.LIMITS
        assert len(result["source_sha256"]) == 11
    else:
        with pytest.raises(ValueError):
            evidence.specification(train, train, data, data, train, train, proposal)
    assert (
        events
        == ["complete training", "word", "sentence", "inputs"][
            : {None: 4, "word": 2, "sentence": 3, "inputs": 4, "selector": 4}[failure]
        ]
    )
