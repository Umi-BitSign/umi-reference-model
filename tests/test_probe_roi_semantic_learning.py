from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_roi_semantic_learning as probe


def records():
    return [
        {
            "roi_role": "training",
            "roi_training_allowed": True,
            "cache": {"path": f"checkpoints/test/{i}.npz", "sha256": str(i)},
            "original_row": {
                "source": "asl_stem_original",
                "example_id": str(i),
                "signer": f"P{i + 100}",
                "role": "training",
                "training_allowed": True,
                "reference": f"Words {i}",
            },
        }
        for i in range(3)
    ]


def pools():
    return [
        [
            {
                "tokens": [2 + i, 1],
                "text": f"Words {i}",
                "kind": "human_anchor",
                "truncated": False,
                "termination": "eos",
            },
            {
                "tokens": [10 + i, 1],
                "text": "unrelated",
                "kind": "sampled",
                "truncated": False,
                "termination": "eos",
            },
        ]
        for i in range(3)
    ]


def calibrated(paired):
    scores = [10 if row["candidate"].startswith("Words") else 0 for row in paired["items"]]
    for control in paired["controls"]:
        scores[control["index"]] = control["range"][0]
    return scores


def test_training_selection_rejects_reserved_or_relabelled_cases():
    rows = records()
    assert probe.training_records({"records": rows}, [0, 1, 2], []) == rows
    with pytest.raises(ValueError):
        probe.training_records({"records": rows}, [0, 1, 2], ["P100"])
    rows[1]["original_row"]["role"] = "development"
    with pytest.raises(ValueError):
        probe.training_records({"records": rows}, [0, 1, 2], [])


def test_candidate_associations_keep_every_real_training_identity():
    paired = probe.comparison(records(), pools())
    assert len(paired["cases"]) == 3
    assert len(paired["controls"]) == 4
    assert probe.qualities(paired, calibrated(paired)) == [[10, 0]] * 3


def test_rewards_cannot_relabel_diagnostic_or_development_clips():
    rows = records()
    rows[0]["roi_role"] = "feature_diagnostic"
    with pytest.raises(ValueError, match="heldout"):
        probe.comparison(rows, pools())


def test_anchor_reference_and_complete_tokens_cannot_change():
    changed = pools()
    changed[0][0]["text"] = "other"
    with pytest.raises(ValueError):
        probe.comparison(records(), changed)
    changed = pools()
    changed[0][1]["tokens"] = [10]
    with pytest.raises(ValueError):
        probe.comparison(records(), changed)


def test_tied_rewards_do_not_claim_actual_nonzero_semantic_gradient_proof():
    paired = probe.comparison(records(), pools())
    scores = calibrated(paired)
    scores[paired["cases"][0]["indices"][1]] = 10
    with pytest.raises(ValueError, match="contrasting"):
        probe.qualities(paired, scores)


def test_failed_calibration_or_noninteger_scores_prevent_proof():
    paired = probe.comparison(records(), pools())
    scores = calibrated(paired)
    for value in (True, 1.5, -1, 11):
        changed = copy.deepcopy(scores)
        changed[0] = value
        with pytest.raises(ValueError):
            probe.qualities(paired, changed)
    scores[paired["controls"][0]["index"]] = 11
    with pytest.raises(ValueError):
        probe.qualities(paired, scores)


def test_proof_policy_cannot_claim_training_or_promotion():
    assert probe.POLICY["caption_weight"] == 0
    assert probe.POLICY["optimizer_updates"] == 0
    assert probe.POLICY["model_win_claimed"] is False
    assert probe.POLICY["promotion_authorized"] is False
