"""Corrective visual LoRA scheduling is deterministic and target-balanced."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_visual_lora_corrective_plan import RECIPE, TARGETS, corrective_schedule


def test_corrective_schedule_matches_targets_and_is_deterministic():
    rows = [
        {"source": "fleurs", "id": f"f{index}", "signer": 3}
        for index in range(5)
    ]
    for signer in (1, 2, 3, 4, 5, 8, 9, 11):
        rows.extend(
            {
                "source": "how2sign_raw",
                "id": f"h{signer}-{index}",
                "signer": signer,
            }
            for index in range(5)
        )
    first = corrective_schedule(rows)
    second = corrective_schedule(list(reversed(rows)))
    assert first == second
    assert len(first) == RECIPE["train_examples"]
    counts = Counter(
        "fleurs" if row["source"] == "fleurs" else str(row["signer"])
        for row in first
    )
    assert dict(counts) == TARGETS
