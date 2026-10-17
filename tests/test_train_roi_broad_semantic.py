from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_roi_broad_semantic as training


def row(number, words):
    return {
        "roi_role": "training",
        "roi_training_allowed": True,
        "original_row": {
            "source": "asl_stem_original",
            "example_id": str(number),
            "role": "training",
            "training_allowed": True,
            "reference": " ".join([f"word{number}"] * words),
        },
    }


def test_curriculum_has_all_short_then_full_epochs_with_complete_exact_coverage():
    rows = [row(i, 3 if i < 99 else 6) for i in range(318)]
    batches = training.batches_for(rows)
    assert len(batches) == 420
    assert sum(len(b["training_record_indices"]) for b in batches) == 1668
    assert len(batches[:100]) == 4 * 25
    assert {b["stage"] for b in batches[:100]} == {"two_to_three_words"}
    assert {b["stage"] for b in batches[100:]} == {"two_to_six_words_with_short_replay"}
    for epoch in range(4):
        short = [
            i for b in batches[:100] if b["epoch"] == epoch for i in b["training_record_indices"]
        ]
        broad = [
            i for b in batches[100:] if b["epoch"] == epoch for i in b["training_record_indices"]
        ]
        assert sorted(short) == list(range(99))
        assert sorted(broad) == list(range(318))
    counts = Counter(i for b in batches for i in b["training_record_indices"])
    assert all(counts[i] == (8 if i < 99 else 4) for i in range(318))
    assert batches == training.batches_for(copy.deepcopy(rows))
    assert all(0 < len(b["training_record_indices"]) <= 4 for b in batches)


@pytest.mark.parametrize(
    "mutation",
    ["nontraining", "role", "outer-permission", "outer-role", "batch-zero", "epoch-zero"],
)
def test_curriculum_rejects_unauthorized_inputs_or_empty_work(mutation):
    rows, recipe = [row(0, 3)], copy.deepcopy(training.RECIPE)
    if mutation == "nontraining":
        rows[0]["roi_training_allowed"] = False
    elif mutation == "role":
        rows[0]["roi_role"] = "feature_diagnostic"
    elif mutation == "outer-permission":
        rows[0]["original_row"]["training_allowed"] = False
    elif mutation == "outer-role":
        rows[0]["original_row"]["role"] = "development"
    elif mutation == "batch-zero":
        recipe["batch_clips"] = 0
    else:
        recipe["stages"][0]["epochs"] = 0
    with pytest.raises(ValueError):
        training.batches_for(rows, recipe)


def test_curriculum_cannot_drop_short_stage_when_only_longer_captions_are_present():
    with pytest.raises(ValueError, match="empty"):
        training.batches_for([row(0, 6)])


def test_final_coverage_checks_order_and_planned_repeats_not_just_total_count():
    rows = [row(0, 3), row(1, 6)]
    plan = {
        "records": rows,
        "batches": [{"training_record_indices": [0]}, {"training_record_indices": [0, 1]}],
        "planned_training_examples": 3,
    }
    training.verify_coverage(plan, 3, [0, 0, 1], ["final"])
    for examples, order, saved in [
        (2, [0, 1], ["final"]),
        (3, [0, 1, 0], ["final"]),
        (3, [0, 0, 0], ["final"]),
        (3, [0, 0, 1], []),
    ]:
        with pytest.raises(ValueError):
            training.verify_coverage(plan, examples, order, saved)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("complete", False),
        ("numerics_passed", False),
        ("plan_sha256", "wrong"),
        ("optimizer_updates", 1),
        ("new_Terra_requests", 1),
        ("pure_Terra_objective", False),
        ("all_fixed_weights_unchanged", False),
        ("trainable_tensors", 115),
        ("vision_initialization_sha256", "wrong"),
    ],
)
def test_training_cannot_bypass_actual_cross_source_qualification(
    tmp_path, monkeypatch, flag, value
):
    path = tmp_path / "qualification.json"
    proof = {
        "complete": True,
        "numerics_passed": True,
        "plan_sha256": "plan-sha",
        "optimizer_updates": 0,
        "new_Terra_requests": 0,
        "pure_Terra_objective": True,
        "all_fixed_weights_unchanged": True,
        "trainable_tensors": 116,
        "comparisons": [],
        "vision_initialization_sha256": "init-sha",
    }
    proof[flag] = value
    path.write_text(json.dumps(proof))
    monkeypatch.setattr(training, "PROOF", path)
    monkeypatch.setattr(training, "terminal", lambda *args: None)
    monkeypatch.setattr(training, "sha256", lambda *args: "plan-sha")
    monkeypatch.setattr(
        training,
        "checked_proof_plan",
        lambda *args: {
            "records": [],
            "original_initialization_sha256": "init-sha",
        },
    )
    with pytest.raises(ValueError, match="complete cross-source"):
        training.qualified()


def test_missing_authoritative_qualification_exit_stops_before_any_model_or_scoring(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(training, "PROOF", tmp_path / "qualification.json")
    monkeypatch.setattr(training, "checked_proof_plan", lambda *args: {})
    with pytest.raises(FileNotFoundError):
        training.qualified()


def test_recipe_keeps_Terra_decisive_fixed_final_and_service_limits():
    recipe = training.RECIPE
    assert recipe["caption_weight"] == 0
    assert recipe["risk_alpha"] == training.NUMERICAL_POLICY["risk_alpha"]
    assert recipe["WER_is_diagnostic_only"] is True
    assert recipe["automatic_retries"] == 0
    assert recipe["maximum_pairs_per_request"] == 16
    assert recipe["minimum_request_start_gap_seconds"] == 60
    assert recipe["promotion_authorized"] is False
    assert "fixed final" in recipe["checkpoint_selection"]
