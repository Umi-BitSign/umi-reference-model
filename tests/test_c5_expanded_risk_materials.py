"""Protected donors cannot reenter a clean run as negative target strings."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_expanded_risk_materials import filtered_pool
from c5_expanded_training_admission import caption_key


def inputs():
    row = {
        "source": "asl_stem_word",
        "example_id": "a",
        "source_group": "a",
        "signer": "P2",
        "reference": "Health",
        "role": "training",
        "training_allowed": True,
    }
    pool = {
        "candidates": [
            {"kind": "human_anchor", "text": "Health", "tokens": [5, 1]},
            {"kind": "other_training_caption", "text": "Solid acid catalysts.", "tokens": [6, 1]},
            {"kind": "greedy", "text": "", "tokens": [1]},
        ]
    }
    return row, pool, [10, 0, 0]


def test_keeps_actual_reward_and_token_joins_and_retains_empty_generated_negative():
    row, pool, quality = inputs()
    actual = filtered_pool(row, pool, quality, set(), {caption_key("Solid acid catalysts")})
    assert actual["original_candidate_indices"] == [0, 2]
    assert actual["excluded_protected_target_indices"] == [1]
    assert actual["candidates"][1]["tokens"] == [1]
    assert actual["quality"] == [10, 0]
    assert len(pool["candidates"]) == 3


def test_rejects_excluded_anchor_before_negative_target_handling():
    row, pool, quality = inputs()
    with pytest.raises(ValueError, match="protected_sentence_reference"):
        filtered_pool(row, pool, quality, set(), {caption_key("Health")})


def test_never_assigns_or_repairs_missing_rewards():
    row, pool, quality = inputs()
    with pytest.raises(ValueError, match="complete original"):
        filtered_pool(row, pool, quality[:2], set(), set())
