from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_full_coverage_mps import comparison, learning_rate_factor, ordered_training


def examples():
    rows = [
        {"source": "how2sign_new", "id": f"how-{signer}-{index}", "signer": signer}
        for signer in (1, 2, 3, 4, 5, 8, 9, 11)
        for index in range(32)
    ]
    rows += [
        {"source": source, "id": f"{source}-{index}", "signer": "participant"}
        for source, count in (("fleurs", 64), ("asl_stem_original", 64), ("openasl_scale", 32))
        for index in range(count)
    ]
    return rows


def records():
    signers = (1, 2, 3, 4, 5, 8, 9, 11)
    return [
        {
            "example_id": str(index),
            "domain": "how2sign" if index < 135 else "fleurs",
            "source_group": f"group-{index // 4}",
            "signer": signers[index % 8],
            "hypothesis": f"valid hypothesis {index}",
            "wer": 0.8,
            "edits": 8,
            "reference_words": 10,
        }
        for index in range(142)
    ]


def test_coverage_is_complete_deterministic_and_exposes_all_how_signers_early():
    rows = examples()
    ordered = ordered_training(rows, 287)
    assert len(ordered) == len(rows)
    assert {(r["source"], r["id"]) for r in ordered} == {(r["source"], r["id"]) for r in rows}
    assert {r["signer"] for r in ordered[:16] if r["source"].startswith("how2sign")} == {
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        11,
    }
    assert ordered_training(list(reversed(rows)), 287) == ordered


def test_full_run_learning_rate_warmup_and_cosine_floor():
    assert learning_rate_factor(0, 1883) == 0.01
    assert learning_rate_factor(99, 1883) == 1.0
    assert learning_rate_factor(1883, 1883) == 0.1
    with pytest.raises(ValueError, match="warmup"):
        learning_rate_factor(0, 100)


def test_open_gate_requires_how_gain_and_no_fleurs_regression():
    parent = records()
    candidate = deepcopy(parent)
    for row in candidate[:135]:
        row.update(wer=0.6, edits=6)
    assert comparison(parent, candidate)["passes_open_development_gate"]
    candidate[-1].update(wer=0.9, edits=9)
    assert not comparison(parent, candidate)["passes_open_development_gate"]


def test_open_gate_rejects_signer_harm_empty_predictions_and_missing_cases():
    parent = records()
    candidate = deepcopy(parent)
    for row in candidate[:135]:
        row.update(wer=0.6, edits=6)
    for row in candidate[:135]:
        if row["signer"] == 9:
            row.update(wer=0.9, edits=9)
    assert not comparison(parent, candidate)["passes_open_development_gate"]
    candidate = deepcopy(parent)
    for row in candidate[:135]:
        row.update(wer=0.6, edits=6)
    candidate[0]["hypothesis"] = ""
    assert not comparison(parent, candidate)["passes_open_development_gate"]
    with pytest.raises(ValueError, match="coverage differs"):
        comparison(parent, candidate[:-1])
