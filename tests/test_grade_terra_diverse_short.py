from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from grade_terra_diverse_short import apply_scores, bookend_batches, unknown_items


@pytest.mark.parametrize("count", [1, 12, 13, 16, 149])
def test_bookends_preserve_coverage_pacing_batch_cap_and_two_calibration_sets(count):
    batches = bookend_batches(count)
    assert all(1 <= len(batch) <= 16 for batch in batches)
    indices = [r["index"] for batch in batches for r in batch if r["kind"] == "candidate"]
    assert sorted(indices) == list(range(count))
    assert [r["index"] for r in batches[0] if r["kind"] == "control"] == list(range(4))
    assert [r["index"] for r in batches[-1] if r["kind"] == "control"] == list(range(4))
    if count == 149:
        assert len(batches) == 10


def records():
    return [
        {
            "data_row": {"reference": "Gold"},
            "pool": [{"text": "Gold"}, {"text": "New"}],
            "quality": [10, None],
        }
    ]


def test_only_missing_grades_are_requested_and_original_labels_remain_unchanged():
    rows = records()
    items, positions = unknown_items(rows)
    assert items == [{"reference": "Gold", "candidate": "New"}]
    assert positions == [[0, 1]]
    assert apply_scores(rows, positions, [0])[0]["quality"] == [10, 0]
    assert rows[0]["quality"] == [10, None]


@pytest.mark.parametrize("score", [None, True, -1, 11, 3.5])
def test_unverified_or_non_rubric_scores_cannot_make_training_data(score):
    with pytest.raises(ValueError):
        apply_scores(records(), [[0, 1]], [score])


def test_new_grades_cannot_replace_prior_authentic_scores_or_leave_gaps():
    with pytest.raises(ValueError, match="overwrite"):
        apply_scores(records(), [[0, 0]], [9])
    with pytest.raises(ValueError, match="ungraded"):
        apply_scores(records(), [], [])


def test_repeated_new_texts_are_rejected_before_spending():
    with pytest.raises(ValueError, match="deduplicated"):
        unknown_items([*records(), *records()])
