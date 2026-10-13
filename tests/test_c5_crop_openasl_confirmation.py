from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from decide_c5_crop_openasl_confirmation import decide
from freeze_c5_crop_openasl_confirmation import select_one_per_source


def test_selection_keeps_one_seeded_clip_per_source() -> None:
    rows = [
        {"source_group": f"source-{source}", "clip_id": f"{source}-{clip}"}
        for source in range(60)
        for clip in range(2)
    ]
    selected = select_one_per_source(rows)
    assert len(selected) == 60
    assert len({row["source_group"] for row in selected}) == 60
    assert selected == select_one_per_source(list(reversed(rows)))


def test_decision_requires_mean_weighted_and_paired_gates() -> None:
    examples = [{"clip_id": str(index)} for index in range(60)]
    plan = {
        "examples": examples,
        "quality_gate": {
            "maximum_mean_wer_delta": -0.005,
            "maximum_word_weighted_wer_delta": 0.0,
            "minimum_paired_improved_minus_regressed": 0,
        },
    }
    native = {
        str(index): {
            "single_reference_wer": 1.0,
            "single_reference_word_edits": 10,
            "reference_word_count": 10,
            "active_seconds": 2.0,
            "hypothesis": f"native-{index}",
        }
        for index in range(60)
    }
    candidate = {
        key: {
            **row,
            "single_reference_wer": 0.9,
            "single_reference_word_edits": 9,
            "active_seconds": 1.8,
            "hypothesis": f"candidate-{key}",
        }
        for key, row in native.items()
    }
    result = decide(plan, native, candidate)
    assert result["quality_gate_passed"] is True
    assert result["paired"] == {"improved": 60, "tied": 0, "regressed": 0}
    candidate["0"] = {**candidate["0"], "single_reference_word_edits": 1000}
    assert decide(plan, native, candidate)["quality_gate_passed"] is False
