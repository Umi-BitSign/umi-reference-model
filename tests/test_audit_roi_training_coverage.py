from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_roi_training_coverage import eligible_pairs, lexical_coverage, word_band


def pair(source, signer, ident):
    return {
        "raw": {
            "source": source,
            "signer": signer,
            "example_id": ident,
            "reference": "A phrase",
            "role": "training",
            "training_allowed": True,
        },
        "teacher": {
            "source": source,
            "id": ident,
            "reference": "A phrase",
            "role": "training",
            "training_allowed": True,
        },
        "feature_pretraining_allowed": True,
    }


def test_reserved_STEM_signers_are_excluded_across_both_sources_only():
    rows = [
        pair("asl_stem_original", "P4", "a"),
        pair("asl_stem_supplemental", "P4", "b"),
        pair("asl_stem_original", "P19", "c"),
        pair("asl_stem_original", "P5", "d"),
        pair("how2sign_raw", "P4", "e"),
    ]
    eligible, excluded = eligible_pairs(rows, ["P4", "P19"])
    assert eligible == rows[3:]
    assert excluded == {"asl_stem_original": 2, "asl_stem_supplemental": 1}


def test_nontraining_misaligned_or_duplicate_metadata_is_rejected():
    original = pair("asl_stem_original", "P7", "a")
    for field, value in (("role", "development"), ("training_allowed", False)):
        changed = copy.deepcopy(original)
        changed["raw"][field] = value
        with pytest.raises(ValueError):
            eligible_pairs([changed], [])
    changed = copy.deepcopy(original)
    changed["teacher"]["id"] = "other"
    with pytest.raises(ValueError):
        eligible_pairs([changed], [])
    with pytest.raises(ValueError):
        eligible_pairs([original, original], [])


def test_curriculum_bands_are_only_metadata_difficulty_counts():
    assert [word_band(n) for n in (1, 2, 3, 4, 6, 7, 10, 11)] == [
        "1",
        "2-3",
        "2-3",
        "4-6",
        "4-6",
        "7-10",
        "7-10",
        "11+",
    ]
    with pytest.raises(ValueError):
        word_band(0)


def test_lexical_coverage_counts_do_not_claim_ASL_or_semantic_knowledge():
    train = [{"reference": "The blue car"}, {"reference": "A bike"}]
    dev = [{"reference": "The blue boat"}, {"reference": "A bike"}]
    result = lexical_coverage(train, dev)
    assert result["development_word_types_absent_from_training_references"] == 1
    assert result["development_clips_with_at_least_one_absent_word"] == 1
    assert result["development_exact_normalized_references_seen"] == 1
    assert result["not_ASL_sign_coverage_or_semantic_generalization"] is True
