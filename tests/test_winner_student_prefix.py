from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_winner_student_prefix import quality_bounds


def fixture():
    evaluation = [
        {
            "example_id": str(i),
            "domain": "how2sign" if i < 135 else "fleurs",
            "signer": str(i % 8),
            "source_group": str(i),
            "reference": "red blue green white black",
            "role": "open_development",
            "training_allowed": False,
        }
        for i in range(142)
    ]
    control = [
        dict(row, reference_words=5, edits=4, wer=0.8, hypothesis="red") for row in evaluation
    ]
    observed = [
        dict(row, reference_words=5, edits=100, wer=20.0, hypothesis=" ".join(["wrong"] * 100))
        for row in evaluation[:6]
    ]
    return evaluation, control, observed


def test_optimistic_nonnegative_bound_proves_failure_without_complete_score():
    evaluation, control, observed = fixture()
    result = quality_bounds(evaluation, control, observed)
    assert result["full_quality_gate_mathematically_impossible"]
    assert not result["complete_benchmark_score"]
    how = result["domains"]["how2sign"]
    assert how["mean_wer_lower_bound_even_if_remaining_perfect"] == pytest.approx(120 / 135)
    assert how["unchanged_gate_mean_wer_limit"] == pytest.approx(0.76)
    perfect_remainder = [
        dict(row, reference_words=5, edits=0, wer=0.0, hypothesis=row["reference"])
        for row in evaluation[6:]
    ]
    all_predictions = observed + perfect_remainder
    full_mean = sum(row["wer"] for row in all_predictions if row["domain"] == "how2sign") / 135
    assert full_mean == how["mean_wer_lower_bound_even_if_remaining_perfect"]


def test_prefix_that_can_still_pass_is_not_rejected():
    evaluation, control, observed = fixture()
    result = quality_bounds(evaluation, control, observed[:1])
    assert not result["full_quality_gate_mathematically_impossible"]


@pytest.mark.parametrize(
    "changed", ["duplicate", "domain", "reference_words", "edits", "wer", "training"]
)
def test_invalid_partial_metrics_or_identity_are_rejected(changed):
    evaluation, control, observed = fixture()
    observed = copy.deepcopy(observed)
    if changed == "duplicate":
        observed.append(observed[0])
    elif changed == "training":
        observed[0]["training_allowed"] = True
    else:
        observed[0][changed] = "fleurs" if changed == "domain" else -1
    with pytest.raises(ValueError):
        quality_bounds(evaluation, control, observed)
