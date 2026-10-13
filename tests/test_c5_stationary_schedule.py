from __future__ import annotations

import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_stationary_schedule import mixture, stationary_schedule

MASS = {"how2sign": 0.6, "fleurs": 0.2, "asl_stem": 0.15, "openasl": 0.05}


def examples():
    groups = [("how2sign_new", 1, 3), ("how2sign_new", 8, 40)]
    groups += [("fleurs", "participant", 6), ("asl_stem_original", "participant", 8)]
    groups += [("openasl_scale", "participant", 4)]
    return [
        {
            "source": source,
            "signer": signer,
            "id": f"{source}-{signer}-{index}",
            "role": "training",
            "training_allowed": True,
            "loss_weight": 2.0,
        }
        for source, signer, count in groups
        for index in range(count)
    ]


def test_stationary_replay_covers_every_example_with_exact_unit_weight_objective():
    rows = examples()
    original = deepcopy(rows)
    replay = stationary_schedule(rows, MASS, 287)
    assert rows == original
    assert len(replay) == 140
    assert {(r["source"], r["id"]) for r in replay} == {(r["source"], r["id"]) for r in rows}
    assert all(r["loss_weight"] == 1 and r["original_corpus_loss_weight"] == 2 for r in replay)
    assert mixture(replay)["objective_domain_mass"] == MASS
    assert mixture(replay)["how_signer_counts"] == {1: 42, 8: 42}


def test_stationary_replay_keeps_every_domain_and_signer_after_minority_exhaustion():
    replay = stationary_schedule(examples(), MASS, 287)
    for window in (replay[:40], replay[-40:]):
        summary = mixture(window)
        assert summary["objective_domain_mass"] == MASS
        assert summary["how_signer_counts"] == {1: 12, 8: 12}
    minority = [r for r in replay if r["source"] == "how2sign_new" and r["signer"] == 1]
    assert set(Counter(r["id"] for r in minority).values()) == {14}


def test_stationary_replay_is_deterministic_independent_of_input_order():
    rows = examples()
    assert stationary_schedule(rows, MASS, 287) == stationary_schedule(
        list(reversed(rows)), MASS, 287
    )
    assert stationary_schedule(rows, MASS, 288) != stationary_schedule(rows, MASS, 287)


def test_stationary_replay_rejects_duplicate_or_evaluation_material():
    rows = examples()
    with pytest.raises(ValueError, match="unique"):
        stationary_schedule([*rows, rows[0]], MASS, 287)
    rows[0].update(role="open_development", training_allowed=False)
    with pytest.raises(ValueError, match="training-only"):
        stationary_schedule(rows, MASS, 287)


def test_stationary_replay_requires_normalized_masses_and_bounded_exposures():
    with pytest.raises(ValueError, match="normalized"):
        stationary_schedule(examples(), {**MASS, "fleurs": 0.3}, 287)
    with pytest.raises(ValueError, match="exposure budget"):
        stationary_schedule(examples(), MASS, 287, maximum_exposures=100)


def test_mixture_audit_rejects_empty_or_nonfinite_objective_weights():
    with pytest.raises(ValueError, match="nonempty"):
        mixture([])
    rows = examples()
    rows[0]["loss_weight"] = float("nan")
    with pytest.raises(ValueError, match="finite positive"):
        mixture(rows)
