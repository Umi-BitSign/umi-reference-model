from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import score_roi_short_translation as scorer


def fixture():
    row = {
        "source": "asl_stem_original",
        "example_id": "a",
        "source_group": "g",
        "signer": "P4",
        "reference": "A test",
        "role": "development",
        "training_allowed": False,
    }
    case = {**scorer.prediction(row, "A"), "tokens": [0, 68, 1], "native_crop_parity": True}
    return row, case, SimpleNamespace(decode=lambda tokens, **kw: "A")


def test_actual_candidate_tokens_and_outer_roles_are_required():
    row, case, tokenizer = fixture()
    assert scorer.verified_prediction(row, case, tokenizer) == scorer.prediction(row, "A")
    for field, value in (
        ("training_allowed", True),
        ("native_crop_parity", False),
        ("reference", "Changed"),
        ("hypothesis", "B"),
        ("WER_diagnostic", 0),
    ):
        bad = copy.deepcopy(case)
        bad[field] = value
        with pytest.raises(ValueError):
            scorer.verified_prediction(row, bad, tokenizer)


def test_empty_invalid_or_over_horizon_token_evidence_is_rejected():
    row, case, tokenizer = fixture()
    for tokens in ([], [False], [-1], [384], [0] * 2049):
        with pytest.raises(ValueError):
            scorer.verified_prediction(row, {**case, "tokens": tokens}, tokenizer)


def test_owner_rubric_and_conservative_judge_limits_are_unchanged():
    assert scorer.POLICY["judge_model"] == "gpt-5.6-terra"
    assert scorer.POLICY["reasoning_effort"] == "medium"
    assert scorer.POLICY["passes"] == 2
    assert scorer.POLICY["minimum_seconds_between_request_starts"] == 60
    assert scorer.POLICY["maximum_pairs_per_request"] == 16
    assert scorer.POLICY["automatic_retries"] == 0
    assert scorer.POLICY["WER_is_diagnostic_only"] is True
    assert scorer.POLICY["promotion_authorized"] is False
