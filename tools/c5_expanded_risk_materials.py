"""Keep authentic source rewards while excluding future-protected target text."""

from __future__ import annotations

from c5_expanded_training_admission import caption_key, require_training_row


def filtered_pool(row, pool, quality, protected_groups, protected_sentence_references):
    require_training_row(row, protected_groups, protected_sentence_references)
    candidates = pool["candidates"]
    if (
        not candidates
        or len(candidates) != len(quality)
        or candidates[0]["kind"] != "human_anchor"
        or candidates[0]["text"] != row["reference"]
        or quality[0] != 10
        or any(type(value) is not int or not 0 <= value <= 10 for value in quality)
    ):
        raise ValueError("expanded risk material requires complete original candidate/grade joins")
    kept, excluded = [], []
    for index, candidate in enumerate(candidates):
        # An authentic generated empty answer is a valid negative target (EOS).
        key = caption_key(candidate["text"]) if candidate["text"].strip() else ()
        if key in protected_sentence_references:
            excluded.append(index)
        else:
            kept.append(index)
    if not kept or kept[0] != 0:
        raise ValueError("expanded risk filtering lost its admitted human anchor")
    return {
        "candidates": [candidates[index] for index in kept],
        "quality": [quality[index] for index in kept],
        "original_candidate_indices": kept,
        "excluded_protected_target_indices": excluded,
        "source_grades_changed_or_invented": False,
    }
