from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import freeze_c5_anchored_projection_plan as plan
from train_c5_anchored_projection import relative_anchor_penalty


def test_balanced_selector_cycles_across_groups() -> None:
    rows = [
        {"id": f"{group}-{index}", "participant": group}
        for group, count in (("a", 3), ("b", 2), ("c", 1))
        for index in range(count)
    ]
    selected = plan.take_balanced(rows, "participant", 5, "test")
    assert len(selected) == 5
    assert Counter(row["participant"] for row in selected).most_common()[0][1] <= 2


def test_relative_anchor_penalty_is_zero_then_positive(monkeypatch) -> None:
    monkeypatch.setattr(
        "train_c5_anchored_projection.DIRECT_NAMES", {"weight", "bias"}
    )
    selected = {
        "weight": torch.nn.Parameter(torch.ones(2, 2)),
        "bias": torch.nn.Parameter(torch.ones(2)),
    }
    anchors = {name: value.detach().clone() for name, value in selected.items()}
    assert relative_anchor_penalty(selected, anchors, 1e-4).item() == 0
    selected["weight"].data.add_(1)
    assert relative_anchor_penalty(selected, anchors, 1e-4).item() > 0
