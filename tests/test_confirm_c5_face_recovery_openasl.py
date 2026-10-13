from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from confirm_c5_face_recovery_openasl import GATES, summarize
from verify_c5_face_recovery_runtime import ARMS


def fixture():
    plan = {"examples": [{"example_id": str(i)} for i in range(60)], "gates": GATES, "repeats": 1,
            "upstream_overlap_qualification": "upstream unknown", "runtime_qualification": "hybrid only"}
    records = []
    for item in plan["examples"]:
        for arm in ARMS:
            records.append({"example_id": item["example_id"], "arm": arm, "legacy_fingerprint": "same", "nonface_fingerprint": "same",
                "face_frame_sha256": ["same"], "recovered_source_indices": [], "end_to_end_seconds": 1.,
                "wer": .8, "edits": 8, "reference_words": 10, "hypothesis": "valid", "expected_legacy_parity": True})
    return records, plan


def test_confirmation_accepts_nonregression_but_consumes_cohort_without_promotion():
    records, plan = fixture(); result = summarize(records, plan)
    assert result["passes"] and result["cohort_consumed"] and result["confirmation_decoded"]
    assert not result["promoted"] and result["paired_tied"] == 60


def test_confirmation_rejects_quality_regression_and_duplicate_decode():
    records, plan = fixture(); records[-1].update(wer=1., edits=10)
    assert not summarize(records, plan)["passes"]
    records, plan = fixture(); records[-1] = deepcopy(records[-2])
    with pytest.raises(ValueError, match="more than once"): summarize(records, plan)


def test_confirmation_requires_preserved_unrelated_inputs_and_empty_output_gate():
    records, plan = fixture(); records[-1]["face_frame_sha256"] = ["changed"]
    with pytest.raises(ValueError, match="unrecovered"): summarize(records, plan)
    records, plan = fixture(); records[-1]["hypothesis"] = ""
    assert not summarize(records, plan)["passes"]
