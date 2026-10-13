"""Existing visual LoRA hypotheses have a reproducible oracle frontier."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_visual_lora_frontier import build_analysis


def _row(example_id: str, signer: int, wer: float) -> dict:
    return {
        "example_id": example_id,
        "signer": signer,
        "reference": f"reference {example_id}",
        "index_sha256": "index",
        "split": "validation",
        "single_reference_wer": wer,
    }


def test_oracle_selects_best_existing_hypothesis_and_checks_gate():
    baseline = [_row("a", 1, 0.9), _row("b", 2, 0.8)]
    candidates = {
        "first": [_row("a", 1, 0.7), _row("b", 2, 0.9)],
        "second": [_row("a", 1, 0.8), _row("b", 2, 0.7)],
    }
    result = build_analysis(baseline, candidates)
    assert result["oracle_mean_single_reference_wer"] == pytest.approx(0.7)
    assert result["required_how2sign_mean_wer"] == pytest.approx(0.8)
    assert result["existing_hypothesis_oracle_passes_gate"] is True
    assert result["records"][0]["oracle_sources"] == ["first"]
    assert result["records"][1]["oracle_sources"] == ["second"]


def test_oracle_rejects_mismatched_inputs():
    baseline = [_row("a", 1, 0.9)]
    candidate = [{**_row("a", 1, 0.8), "reference": "different"}]
    with pytest.raises(ValueError, match="different inputs"):
        build_analysis(baseline, {"candidate": candidate})


def test_oracle_allows_separately_frozen_indexes_for_same_examples():
    baseline = [_row("a", 1, 0.9)]
    candidate = [{**_row("a", 1, 0.8), "index_sha256": "stream-index"}]
    result = build_analysis(baseline, {"candidate": candidate})
    assert result["oracle_mean_single_reference_wer"] == 0.8
