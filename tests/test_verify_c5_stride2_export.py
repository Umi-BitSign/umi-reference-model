"""Merged C5 export equivalence requires exact confirmed hypotheses."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_c5_stride2_export import compare_results


def _row(hypothesis: str) -> dict:
    return {
        "example_id": "clip",
        "reference": "reference",
        "hypothesis": hypothesis,
        "single_reference_wer": 0.5,
        "single_reference_word_edits": 1,
        "source_frame_count": 5,
        "evaluated_frame_count": 3,
        "temporal_stride": 2,
    }


def test_accepts_exact_decode_equivalence():
    compare_results([_row("same")], [_row("same")])


def test_rejects_changed_hypothesis():
    with pytest.raises(ValueError, match="decode differs"):
        compare_results([_row("expected")], [_row("changed")])
