"""The visual-sensitivity probe uses only frozen training examples."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from probe_visual_conditioning import SIGNERS, choose_rows, summarize


def _row(signer: int, name: str) -> dict:
    return {
        "schema": "umi-studio-how2sign-encoder-cache-example/1",
        "split": "training",
        "signer": signer,
        "sentence_name": name,
        "encoder_shape": [90, 1536],
        "reference": "one two three four five six",
    }


def test_choose_one_per_signer_and_reject_validation():
    rows = [_row(signer, f"b-{signer}") for signer in SIGNERS]
    rows.extend(_row(signer, f"a-{signer}") for signer in SIGNERS)
    rows.append(_row(9, "other-training-signer"))
    assert [row["sentence_name"] for row in choose_rows(rows)] == [
        f"a-{signer}" for signer in SIGNERS
    ]
    rows[0]["split"] = "validation"
    with pytest.raises(ValueError, match="training-only"):
        choose_rows(rows)


def test_summary_counts_changed_hypotheses():
    rows = [
        {
            "own_hypothesis": "a",
            "swapped_hypothesis": "b" if index < 3 else "a",
            "hypothesis_word_edits": 1 if index < 3 else 0,
            "own_wer": 0.8,
            "swapped_wer": 1.0,
        }
        for index in range(len(SIGNERS))
    ]
    summary = summarize(rows)
    assert summary["changed_hypotheses"] == 3
    assert summary["mean_hypothesis_word_edits"] == 0.6
    assert summary["mean_own_wer"] == 0.8
