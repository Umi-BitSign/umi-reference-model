"""Matched FSboard comparison stays signer balanced and excludes missing clips."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from select_fsboard_validation import SOURCE_PREFIX
from summarize_fsboard_pilot import summarize


def _fixture():
    rows = [
        {
            "clip_filename": f"{signer}{index}.mp4",
            "source_path": f"{SOURCE_PREFIX}/{signer}{index}.mp4",
            "md5": "a" * 32,
            "size_bytes": 100,
            "reference": "abc",
            "signer_id": signer,
        }
        for signer in ("a", "b")
        for index in (0, 1)
    ]
    selection = {
        "schema": "umi-studio-fsboard-validation-pilot/1",
        "dataset": "googleai/fsboard",
        "source_split": "validation",
        "test_used": False,
        "signer_count": 2,
        "clips_per_signer": 2,
        "rows": rows,
    }
    candidate = {
        row["clip_filename"]: {
            "status": "right_hand_evaluated",
            "reference": "abc",
            "signer_id": row["signer_id"],
            "single_reference_cer": cer,
        }
        for row, cer in zip(rows, (0, 0.5, 0.25, 0.25), strict=True)
    }
    baseline = {
        row["clip_filename"]: {"reference": "abc", "single_reference_cer": 0.5} for row in rows
    }
    return selection, candidate, baseline


def test_summary_is_paired_and_signer_balanced():
    selection, candidate, baseline = _fixture()
    result = summarize(selection, candidate, baseline, bootstrap_draws=100)
    assert result["clips"] == 4
    assert result["signers"] == 2
    assert result["candidate_mean_cer"] == 0.25
    assert result["baseline_mean_cer"] == 0.5
    assert result["candidate_wins"] == 3
    assert result["ties"] == 1
    assert result["clamped_similarity_improvement_95pct_signer_bootstrap"] == [0.25, 0.25]


def test_missing_or_unscored_clip_fails():
    selection, candidate, baseline = _fixture()
    candidate.pop("a0.mp4")
    with pytest.raises(ValueError, match="every selected"):
        summarize(selection, candidate, baseline, bootstrap_draws=10)
