from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_fsboard_byt5 import dominant_hand, validate_training_steps, validation_rows


def test_frozen_validation_receipts_are_complete_and_signer_disjoint() -> None:
    selections = [
        ROOT / "runs/fsboard-validation-pilot-v1/selection.json",
        ROOT / "runs/fsboard-followup-v1/selection.json",
    ]
    landmarks = [
        ROOT / "data/fsboard/validation-pilot-landmarks",
        ROOT / "data/fsboard/followup-landmarks",
    ]
    receipts = [
        ROOT / "runs/fsboard-validation-pilot-v1/landmark-receipts.jsonl",
        ROOT / "runs/fsboard-followup-v1/landmark-receipts.jsonl",
    ]
    index = ROOT / "runs/fsboard-train-shard00-v3/training-index.jsonl"
    rows = validation_rows(selections, landmarks, receipts, index)
    assert len(rows) == 60
    assert len({row[0]["clip_filename"] for row in rows}) == 60
    assert len({row[0]["signer_id"] for row in rows}) == 15
    with pytest.raises(ValueError, match="frozen receipt"):
        validation_rows(selections[:1] * 2, landmarks[:1] * 2, receipts[:1] * 2, index)


def test_training_step_ledger_must_match_frozen_schedule() -> None:
    rows = (
        {"clip_filename": "a.mp4", "signer_id": "a", "shard_row": 0,
         "left_hand_detection_fraction": 0.7, "right_hand_detection_fraction": 0.0},
        {"clip_filename": "b.mp4", "signer_id": "b", "shard_row": 1,
         "left_hand_detection_fraction": 0.0, "right_hand_detection_fraction": 0.2},
        {"clip_filename": "c.mp4", "signer_id": "c", "shard_row": 2,
         "left_hand_detection_fraction": 0.0, "right_hand_detection_fraction": 0.6},
    )
    from train_fsboard_byt5_smoke import eligible_positions

    steps = [
        {"step": number, "clip_filename": rows[position]["clip_filename"],
         "signer_id": rows[position]["signer_id"], "source_row": rows[position]["shard_row"]}
        for number, position in enumerate(eligible_positions(rows, 17), start=1)
    ]
    validate_training_steps(steps, rows, 17)
    with pytest.raises(ValueError, match="audited schedule"):
        validate_training_steps(steps[:-1], rows, 17)
    steps[0]["signer_id"] = "wrong"
    with pytest.raises(ValueError, match="audited schedule"):
        validate_training_steps(steps, rows, 17)


def test_dominant_hand_from_frozen_receipt_counts() -> None:
    assert dominant_hand({"left_hand_finite_scalars": 4, "right_hand_finite_scalars": 1}) == "left"
    assert dominant_hand({"left_hand_finite_scalars": 1, "right_hand_finite_scalars": 4}) == "right"
    assert dominant_hand({"left_hand_finite_scalars": 2, "right_hand_finite_scalars": 2}) == "tie"
