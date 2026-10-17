from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import train_roi_semantic as train


def records():
    return [
        {
            "roi_role": "training",
            "roi_training_allowed": True,
            "frames": 89 if i < 87 else 122,
            "original_row": {
                "source": "asl_stem_original",
                "example_id": str(i),
                "signer": f"P{100 + i % 25}",
                "role": "training",
                "training_allowed": True,
                "reference": f"Words {i}",
            },
        }
        for i in range(88)
    ]


def paired():
    return {
        "items": [{"reference": "a", "candidate": str(i)} for i in range(6)],
        "cases": [{"indices": [0, 1]}, {"indices": [2, 3]}],
        "controls": [{"index": 4, "range": [10, 10]}, {"index": 5, "range": [0, 0]}],
    }


def test_inventory_is_complete_and_diagnostics_never_enter_training():
    rows = records()
    assert train.records_for_training({"records": rows}, []) == rows
    for field, value in (("role", "development"), ("training_allowed", False), ("signer", "P4")):
        changed = copy.deepcopy(rows)
        changed[0]["original_row"][field] = value
        with pytest.raises(ValueError):
            train.records_for_training({"records": changed}, ["P4"])
    rows[0]["roi_role"] = "feature_diagnostic"
    with pytest.raises(ValueError):
        train.records_for_training({"records": rows}, [])


def test_duplicate_or_incomplete_training_inventory_is_rejected():
    rows = records()
    with pytest.raises(ValueError):
        train.records_for_training({"records": rows[:-1]}, [])
    changed = copy.deepcopy(rows)
    changed[1]["original_row"]["example_id"] = changed[0]["original_row"]["example_id"]
    with pytest.raises(ValueError):
        train.records_for_training({"records": changed}, [])


def test_donor_is_deterministic_and_skips_same_normalized_caption():
    rows = records()
    rows[1]["original_row"]["reference"] = "WORDS 0!"
    assert train.donor_for(rows, 0) == rows[2]
    for row in rows:
        row["original_row"]["reference"] = "same"
    with pytest.raises(ValueError):
        train.donor_for(rows, 0)


def test_training_can_have_valid_zero_signal_without_fabricated_reward():
    assert train.training_rewards(paired(), [10, 10, 10, 0, 10, 0]) == [[10, 10], [10, 0]]
    assert train.training_rewards(paired(), [10, 10, 10, 10, 10, 0]) == [[10, 10], [10, 10]]


@pytest.mark.parametrize(
    "scores",
    (
        [10],
        [9, 0, 10, 0, 10, 0],
        [10, 0, 10, 0, 9, 0],
        [10, 0, 10, 0, 10, 1],
        [10, 0.5, 10, 0, 10, 0],
    ),
)
def test_invalid_rewards_or_failed_calibration_stop_before_optimizer(scores):
    with pytest.raises(ValueError):
        train.training_rewards(paired(), scores)


def test_visual_sync_copies_only_the_116_declared_parameters():
    class Vision(nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.values = nn.ParameterList(
                [nn.Parameter(torch.tensor([float(i + offset)])) for i in range(116)]
            )

    source = SimpleNamespace(vision=Vision(1))
    destination = SimpleNamespace(vision=Vision(0))
    train.synchronize_visual(source, destination)
    assert all(
        torch.equal(a, b)
        for a, b in zip(source.vision.parameters(), destination.vision.parameters(), strict=True)
    )
    with pytest.raises(ValueError):
        train.synchronize_visual(source, SimpleNamespace(vision=nn.Linear(1, 1)))


def test_experiment_is_pure_semantic_and_fixed_final():
    assert train.RECIPE["caption_weight"] == 0
    assert train.RECIPE["epochs"] == 1
    assert train.RECIPE["checkpoint_after_training_examples"][-1] == 88
    assert train.RECIPE["WER_is_diagnostic_only"]
    assert train.RECIPE["automatic_retries"] == 0
    assert train.RECIPE["promotion_authorized"] is False
