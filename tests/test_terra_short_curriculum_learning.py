from __future__ import annotations

import copy
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import terra_short_curriculum_learning as module


def cases():
    rows = [
        {
            "source": "asl_stem",
            "example_id": str(i),
            "source_group": "group" + str(i),
            "signer": "P4",
            "reference": "Phrase number " + str(i),
            "role": "development",
            "training_allowed": False,
        }
        for i in range(3)
    ]
    a = [{**r, "hypothesis": "baseline" + r["example_id"], "WER_diagnostic": 0.1} for r in rows]
    b = [{**r, "hypothesis": "candidate" + r["example_id"], "WER_diagnostic": 0.9} for r in rows]
    return rows, a, b


def scores(comparison, a=2, b=4):
    values = [0] * len(comparison["items"])
    for case in comparison["cases"]:
        i, j = case["indices"]
        values[i], values[j] = a, b
    for control in comparison["controls"]:
        values[control["index"]] = control["range"][0]
    return [values.copy(), values.copy()]


def test_semantic_gain_wins_despite_worse_WER_and_keeps_stage_scope():
    rows, a, b = cases()
    comparison = module.comparison_items(rows, a, b)
    result = module.semantic_decision(comparison, scores(comparison), a, b)
    assert result["passes_semantic_research_gate"]
    assert not result["WER_veto_used"]
    assert result["scope"] == "reserved_STEM_short_phrases_only"
    assert result["curriculum_stage_only"]
    assert not result["promotion_authorized"]


def test_semantic_decline_stops_despite_better_WER():
    rows, a, b = cases()
    comparison = module.comparison_items(rows, a, b)
    result = module.semantic_decision(comparison, scores(comparison, 6, 2), b, a)
    assert result["stop_for_semantic_decline"]
    assert not result["passes_semantic_research_gate"]


def test_exact_duplicate_transcripts_share_one_scoring_item():
    rows, a, b = cases()
    b[0]["hypothesis"] = a[0]["hypothesis"]
    result = module.comparison_items(rows, a, b)
    assert result["cases"][0]["indices"][0] == result["cases"][0]["indices"][1]
    assert len(result["items"]) == 9


@pytest.mark.parametrize("kind", ["missing", "duplicate", "reference", "role", "signer"])
def test_unpaired_changed_or_training_predictions_cannot_be_selected(kind):
    rows, a, b = cases()
    if kind == "missing":
        b.pop()
    elif kind == "duplicate":
        b.append(copy.deepcopy(b[0]))
    elif kind == "reference":
        b[0]["reference"] = "new label"
    elif kind == "role":
        b[0]["role"] = "training"
    else:
        b[0]["signer"] = "P0"
    with pytest.raises(ValueError):
        module.comparison_items(rows, a, b)


def test_failed_controls_stop_the_run_without_a_quality_win():
    rows, a, b = cases()
    comparison = module.comparison_items(rows, a, b)
    values = scores(comparison)
    values[0][comparison["controls"][0]["index"]] = 9
    result = module.semantic_decision(comparison, values, a, b)
    assert result["stop_for_failed_calibration"]
    assert not result["passes_semantic_research_gate"]


def test_high_baseline_mastery_does_not_become_a_candidate_win():
    rows, a, b = cases()
    comparison = module.comparison_items(rows, a, b)
    result = module.semantic_decision(comparison, scores(comparison, 10, 9), a, b)
    assert result["control_stage_mastered"] and result["candidate_stage_mastered"]
    assert not result["passes_semantic_research_gate"]
    assert result["stop_for_semantic_decline"]


def test_sampling_replays_each_training_identity_each_epoch_and_never_development():
    records = [
        {
            "role": role,
            "training_allowed": role == "training",
            "original_pair": {"raw": {"source": "stem", "example_id": str(i)}},
        }
        for i, role in enumerate(("training", "development", "training"))
    ]
    actual = [p["raw"]["example_id"] for p in module.order(records)]
    assert Counter(actual) == {"0": 4, "2": 4}
    assert module.order(records) == module.order(records)


def test_request_pacing_is_shared_with_evaluation_and_has_no_retry(monkeypatch, tmp_path):
    import terra_quality_training

    clock = iter([0, 10, 30])
    waits, calls = [], []
    pacer = module.JudgePacer(clock=lambda: next(clock), sleep=waits.append)
    monkeypatch.setattr(terra_quality_training, "score_with_terra", lambda r, p: calls.append(r))
    pacer.score([{}], tmp_path / "one")
    pacer.score([{}], tmp_path / "two")
    assert waits == [20] and len(calls) == 2
    with pytest.raises(ValueError):
        pacer.score([{}] * 17, tmp_path / "too_many")


def test_partial_or_boolean_scores_cannot_be_used():
    rows, a, b = cases()
    comparison = module.comparison_items(rows, a, b)
    with pytest.raises(ValueError):
        module.semantic_decision(comparison, scores(comparison)[:1], a, b)
    values = scores(comparison)
    values[0][0] = True
    with pytest.raises(ValueError):
        module.semantic_decision(comparison, values, a, b)


def test_failed_scoring_request_is_not_retried(monkeypatch, tmp_path):
    import terra_quality_training

    calls = []

    def fail(rows, output):
        calls.append(rows)
        raise RuntimeError("overloaded")

    monkeypatch.setattr(terra_quality_training, "score_with_terra", fail)
    with pytest.raises(RuntimeError, match="overloaded"):
        module.JudgePacer().score([{}], tmp_path / "failed")
    assert len(calls) == 1
