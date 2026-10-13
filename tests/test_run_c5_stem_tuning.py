"""ASL STEM continuation tuning compares paired stride-2 predictions."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_c5_stem_tuning import summarize_candidate


def _row(adapter: str, wer: float) -> dict:
    return {
        "example_id": "clip",
        "reference": "reference",
        "index_sha256": "index",
        "checkpoint_sha256": "checkpoint",
        "adapter_sha256": adapter,
        "temporal_stride": 2,
        "single_reference_wer": wer,
        "signer": 3,
    }


def test_summarizes_paired_stride_two_results():
    result = summarize_candidate([_row("start", 0.9)], [_row("candidate", 0.8)])
    assert result["candidate_minus_baseline_mean_wer"] == pytest.approx(-0.1)
    assert result["signer_means"]["3"]["candidate_mean_wer"] == 0.8


def test_rejects_other_stride():
    candidate = _row("candidate", 0.8)
    candidate["temporal_stride"] = 1
    with pytest.raises(ValueError, match="inputs differ"):
        summarize_candidate([_row("start", 0.9)], [candidate])
