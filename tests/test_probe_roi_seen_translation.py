from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_roi_seen_translation as probe


def inputs():
    records = [
        {
            "roi_training_allowed": True,
            "roi_role": "training",
            "original_row": {
                "source": "asl_stem_original",
                "example_id": str(i),
                "signer": f"P{i + 100}",
                "source_group": f"g{i}",
                "reference": f"Word {i}",
                "role": "training",
                "training_allowed": True,
            },
        }
        for i in range(6)
    ]
    return {"records": records}, {"records": copy.deepcopy(records)}


def predictions(records):
    return {
        label: [
            probe.prediction(r["original_row"], r["original_row"]["reference"]) for r in records
        ]
        for label in probe.LABELS
    }


def test_all_six_predeclared_inputs_must_be_seen_by_both_models():
    data, expanded = inputs()
    assert probe.selected_training(data, expanded) == data["records"]
    expanded["records"].pop()
    with pytest.raises(ValueError):
        probe.selected_training(data, expanded)


def test_cannot_relabel_development_as_a_training_diagnostic():
    data, expanded = inputs()
    data["records"][0]["original_row"]["role"] = "development"
    data["records"][0]["original_row"]["training_allowed"] = False
    with pytest.raises(ValueError):
        probe.selected_training(data, expanded)


def test_cannot_replace_the_six_case_scope_or_use_a_single_repeated_signer():
    data, expanded = inputs()
    data["records"].pop()
    with pytest.raises(ValueError):
        probe.selected_training(data, expanded)
    data, expanded = inputs()
    data["records"][1]["original_row"]["signer"] = data["records"][0]["original_row"]["signer"]
    with pytest.raises(ValueError):
        probe.selected_training(data, expanded)


def test_identical_text_pairs_deduplicate_without_hiding_any_training_case():
    data, _ = inputs()
    paired = probe.comparison(data["records"], predictions(data["records"]))
    assert len(paired["cases"]) == 6
    assert len(paired["items"]) == 10  # six shared texts, four original calibration controls
    assert all(len(set(c["indices"].values())) == 1 for c in paired["cases"])


def test_scoring_cannot_drop_a_model_or_change_training_references():
    data, _ = inputs()
    outputs = predictions(data["records"])
    outputs.pop(probe.LABELS[-1])
    with pytest.raises(ValueError):
        probe.comparison(data["records"], outputs)
    outputs = predictions(data["records"])
    outputs[probe.LABELS[0]][0]["reference"] = "changed"
    with pytest.raises(ValueError):
        probe.comparison(data["records"], outputs)


def test_high_training_scores_never_create_a_development_gate_or_model_win():
    data, _ = inputs()
    paired = probe.comparison(data["records"], predictions(data["records"]))
    scores = [10] * len(paired["items"])
    for control in paired["controls"]:
        scores[control["index"]] = control["range"][0]
    result = probe.summarize(paired, [scores, scores])
    assert result["calibration_passed_both_passes"]
    assert all(v == 10 for v in result["training_mean_quality"].values())
    assert result["training_only"] is True
    assert result["development_quality_gate_computed"] is False
    assert result["model_win_claimed"] is False
    assert result["promotion_authorized"] is False


def test_incomplete_or_out_of_rubric_scores_are_rejected():
    data, _ = inputs()
    paired = probe.comparison(data["records"], predictions(data["records"]))
    size = len(paired["items"])
    for scores in ([[10] * size], [[10] * size, [10] * (size - 1)], [[11] * size] * 2):
        with pytest.raises(ValueError):
            probe.summarize(paired, scores)


def test_failed_calibration_remains_visible_even_on_perfect_training_text():
    data, _ = inputs()
    paired = probe.comparison(data["records"], predictions(data["records"]))
    result = probe.summarize(paired, [[10] * len(paired["items"])] * 2)
    assert result["calibration_passed_both_passes"] is False
    assert result["model_win_claimed"] is False


def test_scoring_requires_valid_actual_tokens_and_the_true_training_role():
    data, _ = inputs()
    result = predictions(data["records"])[probe.LABELS[0]][0]
    tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: result["hypothesis"])
    case = {**result, "tokens": [0, 3, 1]}
    probe.verify_case(result, case, tokenizer)
    for key, value in (
        ("tokens", []),
        ("tokens", [384]),
        ("tokens", [True]),
        ("tokens", [0] * 2049),
        ("role", "development"),
        ("hypothesis", "changed"),
    ):
        with pytest.raises(ValueError):
            probe.verify_case(result, {**case, key: value}, tokenizer)
