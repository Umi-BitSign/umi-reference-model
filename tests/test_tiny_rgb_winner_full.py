from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from tiny_rgb_winner_student_protocol import RECIPE as PROOF_RECIPE
from train_tiny_rgb_winner_full import RECIPE, lr_factor, order_key


def test_each_frozen_epoch_contains_every_identity_once_in_reproducible_new_order():
    pairs = [
        {"raw": {"source": source, "example_id": str(i)}}
        for source in ("how2sign", "fleurs", "asl_stem", "openasl")
        for i in range(20)
    ]
    orders = []
    for epoch in range(1, RECIPE["epochs"] + 1):
        first = sorted(pairs, key=lambda pair: order_key(pair, epoch))
        second = sorted(reversed(pairs), key=lambda pair: order_key(pair, epoch))
        assert first == second
        keys = [(p["raw"]["source"], p["raw"]["example_id"]) for p in first]
        assert len(keys) == len(set(keys)) == len(pairs)
        orders.append(keys)
    assert all(left != right for left, right in pairwise(orders))


def test_frozen_learning_schedule_keeps_warmup_cosine_floor_and_quality_checks():
    assert lr_factor(0) == pytest.approx(1 / 128)
    assert lr_factor(127) == 1
    assert lr_factor(128) == 1
    assert lr_factor(30092) == pytest.approx(0.1)
    assert all(0.1 <= lr_factor(update) <= 1 for update in range(128, 30093))
    assert RECIPE["checks_after_updates"] == [2048, 8192, 16384, 7523 * 4]


def test_full_recipe_retains_qualified_objective_inputs_and_frozen_teacher_language():
    for name in (
        "short_side",
        "target_fps",
        "max_frames",
        "max_text_tokens",
        "latent_coefficient",
        "feature_coefficient",
        "initial_diagnostic_weights_reused",
    ):
        assert RECIPE[name] == PROOF_RECIPE[name]
    assert RECIPE["frozen_language"]
    assert RECIPE["teacher_states"] == "fresh_winner_online_no_disk_cache"
    assert RECIPE["sampler"] == "each_native_identity_once_per_epoch_no_minority_repeats"
    assert RECIPE["seed"] == 317
