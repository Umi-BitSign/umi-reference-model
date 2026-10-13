"""Frozen full-data recipe invariants."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from freeze_full_cross_attention_plan import cross_attention_plan


def test_full_cross_attention_plan_changes_only_the_training_recipe():
    base = {
        "schema": "umi-studio-aligned-mixed-plan/1",
        "training_examples": {"fleurs": 313, "how2sign": 1024, "flores2m": 0},
        "fleurs_training_index_sha256": "fleurs",
        "how2sign_training_index_sha256": "how2sign",
        "rank": 8,
        "alpha": 16,
        "flores2m_repeat": 1,
    }
    plan = cross_attention_plan(base, "rejected-pilot")
    assert plan["schema"] == "umi-studio-full-cross-attention-plan/1"
    assert plan["how2sign_training_index_sha256"] == "how2sign"
    assert plan["prior_rejected_pilot_decision_sha256"] == "rejected-pilot"
    assert plan["recipes"][0]["id"] == "full_cross_attention_e1"
    assert "rank" not in plan and "alpha" not in plan
    assert "flores2m_repeat" not in plan
    assert "rank" in base and base["schema"] == "umi-studio-aligned-mixed-plan/1"


def test_full_cross_attention_plan_requires_all_1024_clips():
    with pytest.raises(ValueError, match="all aligned"):
        cross_attention_plan(
            {
                "schema": "umi-studio-aligned-mixed-plan/1",
                "training_examples": {"fleurs": 313, "how2sign": 168, "flores2m": 0},
            },
            "rejected-pilot",
        )
