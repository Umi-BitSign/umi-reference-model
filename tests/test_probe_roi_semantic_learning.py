from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_contrastive_caption_is_explicit_and_has_an_approved_distinct_donor():
    class Tokenizer:
        def __call__(self, reference, truncation):
            assert truncation is False
            return SimpleNamespace(input_ids=[10, 11, 1])

    rows = records()
    candidate = probe.training_negative(rows[0], rows[1], Tokenizer())
    assert candidate["kind"] == "other_training_caption"
    assert candidate["donor_record"] == rows[1]
    assert candidate["text"] == rows[1]["original_row"]["reference"]
    assert "quality" not in candidate
    with pytest.raises(ValueError):
        probe.training_negative(rows[0], rows[0], Tokenizer())
    rows[1]["roi_training_allowed"] = False
    with pytest.raises(ValueError):
        probe.training_negative(rows[0], rows[1], Tokenizer())


@pytest.mark.parametrize("first_exit", (0, 1))
def test_pipeline_waits_for_scoring_exit_and_never_retries(tmp_path, monkeypatch, first_exit):
    path = tmp_path / "runs/plan.json"
    path.parent.mkdir()
    path.write_text(
        json.dumps({"source_sha256": {"tools/probe_roi_semantic_learning.py": "bound"}})
    )
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "verify_snapshot", lambda *args: None)
    monkeypatch.setattr(probe, "storage", lambda: None)
    monkeypatch.setattr(probe, "sha256", lambda *args: "bound")
    monkeypatch.setattr(probe, "identity", lambda *args: {"pid": 1})
    monkeypatch.setattr(probe, "artifact", lambda path: {"path": str(path), "sha256": "bound"})
    calls = []

    def child(command, check):
        assert check is False
        calls.append(command)
        return SimpleNamespace(returncode=first_exit if len(calls) == 1 else 0)

    monkeypatch.setattr(probe.subprocess, "run", child)
    output = tmp_path / "checkpoints/pipeline"
    if first_exit:
        with pytest.raises(RuntimeError, match="no retry"):
            probe.pipeline(path, output)
        assert len(calls) == 1
        assert not (output / "complete.json").exists()
    else:
        probe.pipeline(path, output)
        assert len(calls) == 2
        assert calls[0][calls[0].index("--mode") + 1] == "candidates"
        assert calls[1][calls[1].index("--mode") + 1] == "qualify"
        assert "--candidates" in calls[1]
        assert json.loads((output / "complete.json").read_text())["optimizer_updates"] == 0
    assert json.loads((output / "exit.json").read_text())["exit_code"] == first_exit
