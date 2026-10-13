from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from verify_c5_face_recovery_runtime import ARMS, ObservedEmbedder, landmark_fingerprint, summarize
from c5_face_recovery_runtime import RecoveryLandmarks


def fixture():
    plan = {"examples": [{"example_id": "h", "domain": "how2sign"}, {"example_id": "f", "domain": "fleurs"}],
            "repeats": 2, "runtime_qualification": "hybrid CPU test"}
    records = []
    for case in plan["examples"]:
        for repeat in range(2):
            for arm in ARMS:
                candidate = arm == ARMS[1]
                records.append({"example_id": case["example_id"], "arm": arm, "repeat": repeat,
                    "hypothesis": f"{case['example_id']} {arm}", "legacy_fingerprint": "legacy", "nonface_fingerprint": "nonface",
                    "recovered_source_indices": [0] if candidate else [], "face_frame_sha256": ["new" if candidate else "old", "untouched"],
                    "end_to_end_seconds": 1., "wer": .6 if candidate else .8, "edits": 6 if candidate else 8, "reference_words": 10,
                    "expected_hypothesis_parity": True, "expected_legacy_parity": True, "expected_feature_parity": True})
    return records, plan


def test_runtime_and_cross_domain_gate_passes_only_with_preserved_inputs():
    records, plan = fixture(); decision = summarize(records, plan)
    assert decision["runtime_parity_passes"] and decision["open_cross_domain_passes"]
    assert decision["speed_keep_passes"] and not decision["promoted"] and not decision["confirmation_decoded"]
    bad = deepcopy(records); bad[-1]["nonface_fingerprint"] = "changed"
    with pytest.raises(ValueError, match="hand or pose"): summarize(bad, plan)
    bad = deepcopy(records); bad[-1]["face_frame_sha256"][1] = "changed"
    with pytest.raises(ValueError, match="unrecovered face"): summarize(bad, plan)


def test_cross_domain_regression_and_expected_parity_failure_are_rejected():
    records, plan = fixture()
    for item in records:
        if item["example_id"] == "f" and item["arm"] == ARMS[1]: item.update(wer=.9, edits=9)
    assert not summarize(records, plan)["open_cross_domain_passes"]
    records, plan = fixture(); records[0]["expected_hypothesis_parity"] = False
    assert not summarize(records, plan)["runtime_parity_passes"]
    records, plan = fixture(); records[-1]["hypothesis"] = "nondeterministic"
    with pytest.raises(ValueError, match="vary"): summarize(records, plan)


def test_observer_calls_underlying_embedder_once_and_reset_is_request_local():
    class Base:
        calls = 0
        def extract_embeddings_from_frames(self, frames):
            self.calls += 1; return np.ones((len(frames), 384), dtype=np.float32)
    base = Base(); observed = ObservedEmbedder(base)
    assert observed.extract_embeddings_from_frames([1, 2]).shape == (2, 384)
    assert base.calls == 1 and observed.frame_counts == [2]
    observed.reset(); assert observed.values == [] and observed.frame_counts == []


def test_request_local_recovery_does_not_change_legacy_fingerprint():
    legacy = {0: {"face_landmarks": None, "hand_landmarks": None, "pose_landmarks": [np.ones((33, 3)).tolist()]}}
    recovered = RecoveryLandmarks(legacy, {0: np.ones((478, 3))})
    assert landmark_fingerprint(legacy) == landmark_fingerprint(recovered)
