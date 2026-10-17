"""Authenticate mixed-risk predecessors and fixed training inputs before qualification."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import evaluate_c5_word_sentences as sentence_evaluation
import evaluate_c5_words as word_evaluation
from benchmark_fleurs_dev import sha256
from c5_mixed_semantic_model import word_initialization
from evaluate_roi_short_translation import terminal
from prepare_terra_mixed_native import checked_result
from terra_mixed_semantic_cases import bind_cases
from terra_short_curriculum_learning import semantic_decision
from terra_training_recovery import artifact
from tiny_rgb_capacity_protocol import LIMITS

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "runs/c5-mixed-semantic-proposal-v1/cases.json"


def require_terminal(path):
    terminal(path)
    worker = json.loads(path.with_name("worker.json").read_text())
    if type(worker.get("pid")) is not int or worker["pid"] <= 0:
        raise ValueError("mixed risk requires an actual predecessor worker identity")
    if list(path.parent.rglob("failure.json")) or list(path.parent.rglob("stopped.json")):
        raise ValueError("mixed risk cannot consume a failed or stopped predecessor")


def previous_evaluation(path, kind, initialization):
    path = path.resolve()
    require_terminal(path)
    parent = json.loads(path.read_text())
    if (
        parent.get("complete") is not True
        or parent.get("training_result") != initialization["word_training_result"]
        or parent.get("optimizer_updates") != 0
        or parent.get("automatic_retries") != 0
        or parent.get("promotion_authorized") is not False
    ):
        raise ValueError("mixed risk requires complete terminal word-final evaluation pipelines")
    for key in ("inference_plan", "scoring_plan", "decision", "training_result"):
        if artifact(ROOT / parent[key]["path"]) != parent[key]:
            raise ValueError("mixed risk predecessor lost an actual artifact binding")
    if kind == "word":
        checked_inference, checked_scores, count = (
            word_evaluation.checked,
            word_evaluation.checked_scores,
            96,
        )
    elif kind == "sentence":
        checked_inference, checked_scores, count = (
            sentence_evaluation.checked,
            sentence_evaluation.checked_scores,
            34,
        )
    else:
        raise ValueError("mixed risk predecessor must be word96 or sentence34")
    inference = checked_inference(ROOT / parent["inference_plan"]["path"])
    if (
        inference["training_plan"] != initialization["word_training_plan"]
        or inference["training_result"] != initialization["word_training_result"]
        or inference["candidate_visual"] != initialization["initial_visual"]
        or inference["candidate_visual_state_sha256"]
        != initialization["initial_visual_state_sha256"]
    ):
        raise ValueError("mixed risk predecessor evaluated another visual initializer")
    score_path = ROOT / parent["scoring_plan"]["path"]
    plan = checked_scores(score_path)
    decision_path = path.parent / "scoring/decision.json"
    if (
        parent["decision"] != artifact(decision_path)
        or plan["inference_plan"] != parent["inference_plan"]
        or plan["decoded"] != artifact(path.parent / "inference/receipt.json")
        or len(plan["comparison"]["cases"]) != count
    ):
        raise ValueError("mixed risk predecessor changed full decoded coverage or lineage")
    require_terminal(decision_path)
    actual = word_evaluation.authenticated_passes(
        plan["comparison"], decision_path.parent / "semantic"
    )
    expected = semantic_decision(plan["comparison"], actual, plan["baseline"], plan["predictions"])
    decision = json.loads(decision_path.read_text())
    if (
        any(decision.get(key) != value for key, value in expected.items() if key != "scope")
        or decision.get("plan_sha256") != sha256(score_path)
        or decision.get("new_Terra_requests") != plan["planned_Terra_requests"]
        or decision.get("optimizer_updates") != 0
    ):
        raise ValueError("mixed risk predecessor decision differs from authentic paired scores")
    if not expected["calibration_passed"] or expected["stop_for_semantic_decline"]:
        raise ValueError("mixed risk predecessor triggered an original semantic stop condition")
    exit_record = json.loads(path.with_name("exit.json").read_text())
    exited = datetime.fromisoformat(exit_record["observed_at"])
    if exited.tzinfo is None:
        raise ValueError("mixed risk predecessor exit needs a timezone-aware timestamp")
    return {
        "kind": kind,
        "parent_complete": artifact(path),
        "parent_worker": artifact(path.with_name("worker.json")),
        "parent_exit": artifact(path.with_name("exit.json")),
        "inference_plan": parent["inference_plan"],
        "scoring_plan": parent["scoring_plan"],
        "decision": parent["decision"],
        "decision_values": decision,
        "raw_scoring_evidence": [
            artifact(item) for item in sorted((decision_path.parent / "semantic").rglob("*.json"))
        ],
        "API_start_not_before": (exited + timedelta(seconds=60)).isoformat(),
        "source_sha256": {**inference["source_sha256"], **plan["source_sha256"]},
        "stage_advance_or_promotion_authorized": False,
    }


def specification(
    training_path,
    result_path,
    input_plan,
    input_result,
    word_parent,
    sentence_parent,
    proposal_path=PROPOSAL,
):
    initialization = word_initialization(training_path, result_path)
    predecessors = {
        "word": previous_evaluation(word_parent, "word", initialization),
        "sentence": previous_evaluation(sentence_parent, "sentence", initialization),
    }
    inputs = checked_result(input_plan, input_result)
    proposal = json.loads(proposal_path.read_text())
    for key, current in (
        ("case_selector_source", ROOT / "tools/terra_mixed_semantic_cases.py"),
        ("input_selector_source", ROOT / "tools/terra_mixed_native_input.py"),
    ):
        if (
            artifact(ROOT / proposal[key]["path"]) != proposal[key]
            or sha256(current) != proposal[key]["sha256"]
        ):
            raise ValueError("mixed risk changed the predeclared case or input selector")
    records = bind_cases(proposal["selected_training_cases"], inputs)
    original = initialization["original_model_plan"]
    fit = json.loads(training_path.read_text())
    data = json.loads(input_plan.read_text())
    sources = {**fit["source_sha256"], **data["source_sha256"]}
    for predecessor in predecessors.values():
        sources.update(predecessor["source_sha256"])
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
        sources[f"tools/{name}"] = sha256(ROOT / "tools" / name)
    return {
        "initialization": initialization,
        "preceding_evaluations": predecessors,
        "input_plan": artifact(input_plan),
        "input_result": artifact(input_result),
        "input_authority": inputs["authority"],
        "input_verification_source_sha256": inputs["verification_source_sha256"],
        "metadata_proposal": artifact(proposal_path),
        "records": records,
        "limits": LIMITS,
        "reference": original["reference"],
        "initial_reference_hashes": original["initial_reference_hashes"],
        "expected_visual_gradient_names": original["expected_visual_gradient_names"],
        "resource_policy": original["resource_policy"],
        "resource_policy_sha256": original["resource_policy_sha256"],
        "source_sha256": sources,
        "assets": {**original["assets"], **data["assets"]},
        "API_start_not_before": max(
            (item["API_start_not_before"] for item in predecessors.values()),
            key=datetime.fromisoformat,
        ),
        "stage_advance_or_promotion_authorized": False,
    }
