"""Explicit source, signer and sentence-reference exclusions for future training.

Historical feature qualification and original role labels stay unchanged.
Known isolated-word classes may be shared; protected sentence text may not.
"""

from __future__ import annotations

from benchmark_fleurs_dev import normalize
from terra_word_native_input import DIAGNOSTIC, RESERVED


def caption_key(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("training caption must be nonempty original text")
    return tuple(normalize(text))


def exclusion_reasons(row, protected_groups, protected_sentence_references):
    """Pure metadata check; invoke before loading any candidate training arrays."""
    if row.get("role") != "training" or row.get("training_allowed") is not True:
        raise ValueError("expanded admission requires affirmative original training roles")
    if not all(
        isinstance(row.get(key), str) and row[key]
        for key in ("source", "example_id", "source_group")
    ):
        raise ValueError("expanded admission requires complete original source identity")
    reasons = []
    if row["source_group"] in protected_groups:
        reasons.append("protected_source_group")
    if row["source"].startswith("asl_stem") and row["signer"] in RESERVED | DIAGNOSTIC:
        reasons.append("whole_protected_STEM_signer")
    if caption_key(row["reference"]) in protected_sentence_references:
        reasons.append("protected_sentence_reference")
    return reasons


def require_training_row(row, protected_groups, protected_sentence_references):
    reasons = exclusion_reasons(row, protected_groups, protected_sentence_references)
    if reasons:
        raise ValueError("expanded training input excluded: " + ", ".join(reasons))
    return row
