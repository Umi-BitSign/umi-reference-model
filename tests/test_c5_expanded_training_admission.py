"""Text overlap exclusions are separate from the intended known-word task."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_expanded_training_admission import caption_key, exclusion_reasons, require_training_row


def row():
    return {
        "source": "asl_stem_original",
        "example_id": "a",
        "source_group": "group",
        "signer": "P2",
        "reference": "Solid acid catalysts.",
        "role": "training",
        "training_allowed": True,
    }


def test_excludes_normalized_protected_sentence_with_original_role_preserved():
    original = row()
    assert exclusion_reasons(original, set(), {caption_key("SOLID ACID CATALYSTS")}) == [
        "protected_sentence_reference"
    ]
    assert original["role"] == "training"
    with pytest.raises(ValueError, match="protected_sentence_reference"):
        require_training_row(original, set(), {caption_key("Solid acid catalysts")})


def test_reports_all_reasons_without_silently_relabelling_a_row():
    original = {**row(), "signer": "P6"}
    assert exclusion_reasons(original, {"group"}, {caption_key(original["reference"])}) == [
        "protected_source_group",
        "whole_protected_STEM_signer",
        "protected_sentence_reference",
    ]


def test_known_word_class_is_allowed_when_no_protected_sentence_matches():
    original = {**row(), "source": "asl_stem_word", "reference": "Health"}
    assert (
        require_training_row(original, set(), {caption_key("Public health is important.")})
        == original
    )


@pytest.mark.parametrize(
    "signer", ["P0", "P19", "P10", "P13", "P14", "P21", "P22", "P31", "P4", "P6"]
)
def test_whole_reserved_or_diagnostic_signers_remain_excluded(signer):
    with pytest.raises(ValueError, match="whole_protected_STEM_signer"):
        require_training_row({**row(), "signer": signer}, set(), set())


@pytest.mark.parametrize(
    "change",
    [
        {"role": "development"},
        {"training_allowed": False},
        {"training_allowed": 1},
        {"example_id": ""},
    ],
)
def test_role_and_identity_are_strict(change):
    with pytest.raises(ValueError, match="expanded admission"):
        require_training_row({**row(), **change}, set(), set())
