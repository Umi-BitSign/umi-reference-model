from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_terra_diverse_short import GENERATION, case_seed, merge_pool


def test_seed_is_deterministic_identity_only_and_not_conditioned_on_reference():
    row = {"source": "stem", "example_id": "one", "reference": "Gold caption"}
    assert case_seed(row) == case_seed({**row, "reference": "Different label"})
    assert case_seed(row) != case_seed({**row, "example_id": "two"})
    assert case_seed(row) != case_seed({**row, "source": "other"})


def test_sampling_profile_is_bounded_and_not_the_benchmark_decoder():
    assert GENERATION["do_sample"] is True
    assert GENERATION["num_return_sequences"] == 4
    assert GENERATION["max_length"] == 513
    assert GENERATION["num_beams"] == 1


def test_merge_preserves_original_rewards_and_deduplicates_tokens_without_fabricating_scores():
    base = {
        "pool": [{"tokens": [3, 1], "text": "Gold"}, {"tokens": [4, 1], "text": "Old wrong"}],
        "quality": [10, 2],
        "source_response": "original provenance",
    }
    before = copy.deepcopy(base)
    samples = [
        {"tokens": [3, 1], "text": "Gold"},
        {"tokens": [5, 1], "text": "New hypothesis"},
        {"tokens": [5, 1], "text": "New hypothesis"},
        {"tokens": [6, 1], "text": "Old wrong"},
    ]
    merged = merge_pool(base, samples)
    assert base == before
    assert merged["pool"][:2] == base["pool"]
    assert merged["quality"] == [10, 2, None, 2]
    assert merged["original_pool_size"] == 2
    assert merged["source_response"] == "original provenance"


def test_different_text_is_not_normalized_into_a_cached_grade():
    base = {"pool": [{"tokens": [3, 1], "text": "Gold"}], "quality": [10]}
    merged = merge_pool(base, [{"tokens": [4, 1], "text": "gold"}])
    assert merged["quality"] == [10, None]
