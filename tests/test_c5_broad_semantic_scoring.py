"""Synthetic text and score evidence only; no real reserved predictions or API calls."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import c5_broad_semantic_scoring as scoring
from semantic_quality_judge import calibration_controls


def make_data(layout=(("how2sign", "a", 2), ("how2sign", "b", 2),
                      ("openasl", "c", 1), ("openasl", "d", 1))):
    rows = []
    for family, group, count in layout:
        for i in range(count):
            identity = f"synthetic-{family}-{group}-{i}"
            rows.append({"clip_id": identity, "family": family,
                         "source_group": family + ":synthetic-" + group,
                         "reference": "A synthetic reference for " + identity,
                         "video": {"path": "data/synthetic/" + identity, "sha256": "synthetic"},
                         "training_allowed": False})
    arms = [[{**r, "hypothesis": prefix + r["clip_id"]} for r in rows]
            for prefix in ("original-", "candidate-")]
    return rows, *arms


def comparison():
    return scoring.comparison_items(*make_data())


def grades(comp, original=4, candidate=6):
    result = [[None] * len(comp["items"]) for _ in range(2)]
    for case in comp["cases"]:
        for values in result:
            values[case["indices"][0]] = original
            values[case["indices"][1]] = candidate
    for control in comp["controls"]:
        for values in result:
            values[control["index"]] = control["range"][0]
    return result


def protocol():
    text = " Synthetic append-only clarification."
    instruction = scoring.original_payload([])["instructions"] + text
    return {"schema": "umi-studio-Terra-polarity-clarification/1",
            "model": "gpt-5.6-terra", "reasoning_effort": "medium",
            "original_controls": calibration_controls(), "clarification_append_only": text,
            "full_instructions_sha256": hashlib.sha256(instruction.encode()).hexdigest()}


def save(path, value):
    path.write_text(json.dumps(value))


def raw_evidence(root, comp, values):
    p = protocol()
    save(root / "blinded-inputs.json", comp)
    for repeat, batches in enumerate(scoring.batches(comp), start=1):
        save(root / f"pass-{repeat}-scores.json", values[repeat - 1])
        for batch, chosen in enumerate(batches):
            directory = root / f"pass-{repeat}-batch-{batch:03d}"
            directory.mkdir()
            save(directory / "request.json", scoring.polarity_payload([comp["items"][i] for i in chosen], p))
            save(directory / "started.json", {
                "model": p["model"], "reasoning_effort": "medium",
                "pairs": len(chosen), "automatic_retries": 0,
                "request_sha256": hashlib.sha256((directory / "request.json").read_bytes()).hexdigest()})
            scores = [values[repeat - 1][i] for i in chosen]
            response = {"model": p["model"], "reasoning": {"effort": "medium"},
                        "status": "completed", "output": [{"type": "message", "content": [
                            {"type": "output_text", "text": json.dumps({"scores": [
                                {"id": f"pair_{i:02d}", "score": value} for i, value in enumerate(scores)]})}]}]}
            save(directory / "response.json", response)
            save(directory / "scores.json", {"model": p["model"], "reasoning_effort": "medium",
                                            "scores": scores, "response_sha256": hashlib.sha256(
                                                (directory / "response.json").read_bytes()).hexdigest()})
    return p


def test_constant_positive_gains_pass_and_wer_has_no_veto():
    comp = comparison()
    result = scoring.summarize(comp, grades(comp), {"original": 0.1, "candidate": 99})
    assert result["absolute_gain"] == 2 and result["paired_source_group_bootstrap95_interval"] == [2, 2]
    assert result["passes_semantic_statistics"] is True
    assert result["raw_response_authentication_performed"] is False
    assert result["candidate_or_runtime_admission_verified"] is False


def test_family_regression_cannot_hide_behind_a_larger_other_family_gain():
    comp = comparison()
    values = grades(comp, 4, 8)
    for case in comp["cases"]:
        if case["family"] == "openasl":
            for repeat in values:
                repeat[case["indices"][1]] = 3
    result = scoring.summarize(comp, values)
    assert result["absolute_gain"] == 1.5
    assert result["no_family_mean_regression"] is False
    assert result["passes_semantic_statistics"] is False


def test_equal_family_and_group_means_do_not_weight_larger_groups_as_more_sources():
    comp = scoring.comparison_items(*make_data((("how2sign", "a", 12), ("how2sign", "b", 1),
                                               ("openasl", "c", 1))))
    values = grades(comp, 4, 4)
    for case in comp["cases"]:
        if case["source_group"].endswith("-a"):
            for repeat in values:
                repeat[case["indices"][1]] = 8
    result = scoring.summarize(comp, values)
    assert result["by_family"]["how2sign"]["absolute_gain"] == 2
    assert result["absolute_gain"] == 1
    assert result["paired_source_group_bootstrap95_interval"][0] == 0
    assert result["passes_semantic_statistics"] is False


def test_two_grading_passes_are_averaged_before_source_bootstrap():
    comp = comparison()
    values = grades(comp, 4, 4)
    for case in comp["cases"]:
        values[0][case["indices"][1]] = 3
        values[1][case["indices"][1]] = 7
    result = scoring.summarize(comp, values)
    assert result["absolute_gain"] == 1
    assert result["paired_source_group_bootstrap95_interval"] == [1, 1]


def test_weak_mean_gain_cannot_pass_a_positive_interval():
    comp = comparison()
    values = grades(comp, 8, 8)
    for case in comp["cases"][:1]:
        for repeat in values:
            repeat[case["indices"][1]] = 9
    result = scoring.summarize(comp, values)
    assert result["absolute_gain"] < result["required_absolute_gain"]
    assert result["passes_semantic_statistics"] is False


@pytest.mark.parametrize("control_index,bad", [(0, 9), (1, 9), (2, 1), (3, 4)])
def test_any_actual_control_failure_in_either_complete_pass_rejects(control_index, bad):
    comp = comparison()
    values = grades(comp)
    values[1][comp["controls"][control_index]["index"]] = bad
    result = scoring.summarize(comp, values)
    assert result["calibration_controls_passed_both_passes"] is False
    assert result["passes_semantic_statistics"] is False
    assert values[1][comp["controls"][control_index]["index"]] == bad


@pytest.mark.parametrize("kind", ["missing_pass", "short_pass", "bool", "float", "nan", "range"])
def test_partial_or_transformed_scores_cannot_enter_statistics(kind):
    comp = comparison()
    values = grades(comp)
    if kind == "missing_pass":
        values.pop()
    elif kind == "short_pass":
        values[1].pop()
    else:
        values[1][0] = {"bool": True, "float": 4.0, "nan": float("nan"), "range": 11}[kind]
    with pytest.raises(ValueError):
        scoring.summarize(comp, values)


def test_identical_transcripts_share_one_item_and_empty_hypotheses_are_retained():
    rows, original, candidate = make_data()
    candidate[0]["hypothesis"] = original[0]["hypothesis"]
    candidate[1]["hypothesis"] = ""
    comp = scoring.comparison_items(rows, original, candidate)
    assert comp["cases"][0]["indices"][0] == comp["cases"][0]["indices"][1]
    assert comp["items"][comp["cases"][1]["indices"][1]]["candidate"] == ""


@pytest.mark.parametrize("field,value", [("video", {}), ("reference", "changed"),
                                         ("source_group", "openasl:other"), ("training_allowed", True)])
def test_arm_video_reference_source_and_role_must_match(field, value):
    rows, original, candidate = make_data()
    candidate[0][field] = value
    with pytest.raises(ValueError):
        scoring.comparison_items(rows, original, candidate)


def test_controls_cannot_be_rewritten_or_referenced_as_clip_grades():
    comp = comparison()
    comp["controls"][2]["range"] = [False, False]
    with pytest.raises(ValueError, match="original calibration"):
        scoring.summarize(comp, grades(comp))
    comp = comparison()
    comp["cases"][0]["indices"][0] = len(comp["items"]) - 1
    with pytest.raises(ValueError, match="paired indices"):
        scoring.summarize(comp, grades(comp))


def test_real_raw_authentication_code_rejoins_complete_synthetic_requests(tmp_path):
    comp = comparison()
    values = grades(comp)
    p = raw_evidence(tmp_path, comp, values)
    assert scoring.authenticated_passes(comp, tmp_path, p) == values
    assert scoring.batches(comp) == scoring.batches(comp)
    assert sorted(i for batch in scoring.batches(comp)[0] for i in batch) == list(range(len(comp["items"])))


@pytest.mark.parametrize("mutation", ["model", "prompt", "raw_score", "saved_score", "aggregate", "extra_batch"])
def test_changed_raw_request_response_or_coverage_cannot_authenticate(tmp_path, mutation):
    comp = comparison()
    p = raw_evidence(tmp_path, comp, grades(comp))
    directory = tmp_path / "pass-1-batch-000"
    if mutation == "extra_batch":
        (tmp_path / "pass-1-batch-999").mkdir()
    else:
        name = {"model": "started.json", "prompt": "request.json", "raw_score": "response.json",
                "saved_score": "scores.json", "aggregate": "../pass-1-scores.json"}[mutation]
        path = directory / name
        data = json.loads(path.read_text())
        if mutation == "model":
            data["model"] = "other"
        elif mutation == "prompt":
            data["instructions"] += " changed"
        elif mutation == "raw_score":
            data["output"][0]["content"][0]["text"] = '{"scores":[]}'
        elif mutation == "saved_score":
            data["scores"][0] = (data["scores"][0] + 1) % 11
        else:
            data[0] = (data[0] + 1) % 11
        save(path, data)
    with pytest.raises(ValueError):
        scoring.authenticated_passes(comp, tmp_path, p)


def test_protocol_rejects_full_instruction_hash_or_original_control_changes():
    p = protocol()
    p["clarification_append_only"] += " changed"
    with pytest.raises(ValueError, match="instructions"):
        scoring.polarity_payload([], p)
    p = protocol()
    p["original_controls"][3]["range"] = [0, 5]
    with pytest.raises(ValueError, match="controls"):
        scoring.polarity_payload([], p)


def test_bootstrap_matches_exact_distribution_for_two_source_groups():
    comp = scoring.comparison_items(*make_data((("how2sign", "a", 10), ("how2sign", "b", 1),
                                               ("openasl", "c", 1), ("openasl", "d", 1))))
    values = grades(comp, 4, 5)
    for case in comp["cases"]:
        if case["family"] == "how2sign":
            for repeat in values:
                repeat[case["indices"][1]] = 4 if case["source_group"].endswith("-a") else 8
    result = scoring.summarize(comp, values)
    # Sampling two How2Sign sources with replacement gives gains0,2,4,
    # with probabilities1/4,1/2,1/4. OpenASL gain is always1.
    assert result["absolute_gain"] == 1.5
    assert result["paired_source_group_bootstrap95_interval"] == [0.5, 2.5]
    assert result["passes_semantic_statistics"] is True


def test_zero_original_quality_uses_absolute_threshold_without_dividing_by_zero():
    comp = comparison()
    result = scoring.summarize(comp, grades(comp, 0, 1))
    assert result["required_absolute_gain"] == 0.1
    assert result["relative_gain"] is None and result["passes_semantic_statistics"] is True


def test_saved_boolean_cannot_impersonate_raw_integer_grade(tmp_path):
    comp = comparison()
    p = raw_evidence(tmp_path, comp, grades(comp, 0, 1))
    path = tmp_path / "pass-1-batch-000/scores.json"
    saved = json.loads(path.read_text())
    index = next(i for i, v in enumerate(saved["scores"]) if v in (0, 1))
    saved["scores"][index] = bool(saved["scores"][index])
    save(path, saved)
    with pytest.raises(ValueError, match="unchanged raw grades"):
        scoring.authenticated_passes(comp, tmp_path, p)


def test_prompt_json_type_change_cannot_hide_behind_recomputed_request_hash(tmp_path):
    comp = comparison()
    p = raw_evidence(tmp_path, comp, grades(comp))
    directory = tmp_path / "pass-1-batch-000"
    path = directory / "request.json"
    request = json.loads(path.read_text())
    request["store"] = 0
    save(path, request)
    started = json.loads((directory / "started.json").read_text())
    started["request_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    save(directory / "started.json", started)
    with pytest.raises(ValueError, match="exact prompt"):
        scoring.authenticated_passes(comp, tmp_path, p)


def test_protocol_controls_cannot_use_boolean_score_ranges():
    p = protocol()
    p["original_controls"][2]["range"] = [False, False]
    with pytest.raises(ValueError, match="controls"):
        scoring.polarity_payload([], p)


