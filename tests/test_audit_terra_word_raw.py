from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_terra_word_raw as audit


def inputs():
    records = [
        {"row": {"role": role}, "video_sha256": str(i), "probe": {"duration_sec": 3.0}}
        for i, role in enumerate(("training", "development"))
    ]
    shape = (12, 32, 32, 3)
    signatures = [np.full(shape, value, dtype=np.uint8) for value in (20, 100)]
    return records, signatures


def test_byte_duplicate_across_roles_is_reported_and_cannot_pass():
    records, signatures = inputs()
    records[1]["video_sha256"] = records[0]["video_sha256"]
    result = audit.duplicate_screen(records, signatures)
    assert result["exact_duplicate_groups"] == [[0, 1]]
    assert result["cross_role_screen_passed"] is False


def test_small_codec_color_changes_are_flagged_for_review_without_changing_roles():
    records, signatures = inputs()
    before = copy.deepcopy(records)
    signatures[1] = signatures[0] + 1
    result = audit.duplicate_screen(records, signatures)
    assert result["cross_role_perceptual_review_flags"] == [
        {"indices": [0, 1], "RGB_mean_absolute_difference": 1.0}
    ]
    assert result["cross_role_screen_passed"] is False
    assert records == before


def test_distinct_cross_role_appearances_have_no_close_reencode_flag():
    records, signatures = inputs()
    result = audit.duplicate_screen(records, signatures)
    assert result["exact_duplicate_groups"] == []
    assert result["cross_role_perceptual_review_flags"] == []
    assert result["cross_role_screen_passed"] is True
    assert "does not prove" in result["limitations"]


def test_equal_appearance_on_different_duration_is_not_claimed_as_a_close_reencode():
    records, signatures = inputs()
    signatures[1] = signatures[0].copy()
    records[1]["probe"]["duration_sec"] = 4
    assert audit.duplicate_screen(records, signatures)["cross_role_perceptual_review_flags"] == []


def test_same_role_exact_duplicates_still_remain_visible_in_the_audit():
    records, signatures = inputs()
    records[1]["row"]["role"] = "training"
    records[1]["video_sha256"] = records[0]["video_sha256"]
    result = audit.duplicate_screen(records, signatures)
    assert result["exact_duplicate_groups"] == [[0, 1]]
    assert result["cross_role_screen_passed"] is True
    assert audit.POLICY["model_training_ready"] is False


@pytest.mark.parametrize("mutation", ["missing", "dtype", "shape", "role", "memory"])
def test_incomplete_or_changed_signature_inputs_cannot_pass(mutation, monkeypatch):
    records, signatures = inputs()
    if mutation == "missing":
        signatures.pop()
    elif mutation == "dtype":
        signatures[0] = signatures[0].astype(np.float32)
    elif mutation == "shape":
        signatures[0] = signatures[0][:11]
    elif mutation == "role":
        records[0]["row"]["role"] = "private_service"
    else:
        monkeypatch.setitem(audit.POLICY, "maximum_signature_RAM_bytes", 1)
    with pytest.raises(ValueError):
        audit.duplicate_screen(records, signatures)
