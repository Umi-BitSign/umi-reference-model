from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_word_lineage_fold import Segment, fold


def fixture():
    sizes = (4, 2, 4, 2, 4)
    batches = [
        {"epoch": i // 2, "training_record_indices": list(range(size))}
        for i, size in enumerate(sizes)
    ]
    rows = []
    examples = updates = 0
    for i, batch in enumerate(batches):
        examples += len(batch["training_record_indices"])
        updates += i != 3
        qualities = [[10, 0] if i != 3 else [10, 10] for _ in batch["training_record_indices"]]
        rows.append(
            {
                **batch,
                "batch": i,
                "training_examples": examples,
                "optimizer_updates": updates,
                "Terra_qualities": qualities,
                "contrasting_clips": 0 if i == 3 else len(qualities),
                "gradient_norm_before_clip": None if i == 3 else 1.0,
            }
        )
    return batches, rows


def raw(rows):
    return b"".join((json.dumps(r) + "\n").encode() for r in rows)


def test_multiple_continuations_preserve_one_original_rollback_and_tail_ties():
    batches, rows = fixture()
    report = fold(
        [
            Segment(raw(rows[:3]), 2),
            Segment(raw(rows[:3]), 3),
            Segment(raw(rows[:4]), 4),
            Segment(raw(rows), 5),
        ],
        batches,
    )
    assert report["selected_model_trajectory_exposures"] == 16
    assert report["actual_total_lineage_exposures"] == 20
    assert report["rolled_back_accepted_exposures"] == 4
    assert report["selected_model_optimizer_updates"] == 4
    assert report["actual_total_lineage_optimizer_updates"] == 5
    assert report["rolled_back_optimizer_updates"] == 1
    assert report["resource_reset_batches_with_accepted_work"] == [2, 3, 4]
    assert report["resource_reset_batches_if_resumed"] == [2, 3, 4]
    assert report["segments"][2]["new_optimizer_updates"] == 0
    assert report["segments"][2]["new_accepted_exposures"] == 2


def test_repeated_zero_progress_attempts_add_no_accepted_work_or_duplicate_resets():
    batches, rows = fixture()
    report = fold(
        [
            Segment(raw(rows[:3]), 2),
            Segment(raw(rows[:3]), 3),
            Segment(raw(rows[:3]), 3),
            Segment(raw(rows[:3]), 3),
        ],
        batches,
    )
    assert report["actual_total_lineage_exposures"] == 14
    assert report["selected_model_trajectory_exposures"] == 10
    assert report["actual_total_lineage_optimizer_updates"] == 4
    assert report["selected_model_optimizer_updates"] == 3
    assert report["resource_reset_batches_with_accepted_work"] == [2]
    assert report["resource_reset_batches_if_resumed"] == [2, 3]
    assert all(
        s["new_accepted_exposures"] == s["new_optimizer_updates"] == 0
        for s in report["segments"][-2:]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "partial",
        "prefix_bytes",
        "sampler",
        "hidden_update",
        "new_rollback",
        "counter",
        "gradient",
        "qualities",
        "bool_count",
    ],
)
def test_invalid_or_replayed_prefix_cannot_hide_accepted_work(fault):
    batches, rows = fixture()
    changed = copy.deepcopy(rows)
    first = Segment(raw(rows[:3]), 2)
    second = Segment(raw(changed[:4]), 4)
    if fault == "partial":
        second = Segment(second.ledger[:-1], 4)
    elif fault == "prefix_bytes":
        second = Segment(second.ledger.replace(b'"batch": 0', b'"batch":0'), 4)
    elif fault == "sampler":
        changed[3]["training_record_indices"] = [1, 0]
    elif fault == "hidden_update":
        changed[3]["optimizer_updates"] += 1
    elif fault == "new_rollback":
        second = Segment(raw(changed[:4]), 3)
    elif fault == "counter":
        changed[3]["training_examples"] -= 1
    elif fault == "gradient":
        changed[2]["gradient_norm_before_clip"] = float("nan")
    elif fault == "qualities":
        changed[3]["Terra_qualities"][0][0] = 9
    elif fault == "bool_count":
        second = Segment(raw(changed[:4]), True)
    if fault not in ("partial", "prefix_bytes", "new_rollback", "bool_count"):
        second = Segment(raw(changed[:4]), 4)
    with pytest.raises(ValueError):
        fold([first, second], batches)


@pytest.mark.parametrize("field", ["batch", "optimizer_updates"])
def test_boolean_batch_or_update_fields_cannot_masquerade_as_integer_counts(field):
    batches, rows = fixture()
    row_index = 1 if field == "batch" else 0
    rows[row_index][field] = True
    with pytest.raises(ValueError):
        fold([Segment(raw(rows[:3]), 2)], batches)
