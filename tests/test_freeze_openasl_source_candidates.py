"""OpenASL candidate selection excludes official held-out source videos."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import freeze_openasl_source_candidates as candidates


def test_selects_two_unique_clips_per_source(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(candidates, "TARGET_SOURCES", 2)
    rows = []
    for yid in ("a", "b", "held"):
        rows.extend(
            {
                "vid": f"{yid}-{index}",
                "yid": yid,
                "normalized_reference": f"{yid} reference {index}",
            }
            for index in range(3)
        )
    selected, sources = candidates.select_candidates(rows, {"held"})
    assert set(sources) == {"a", "b"}
    assert len(selected) == 4
    assert {row["yid"] for row in selected} == {"a", "b"}
