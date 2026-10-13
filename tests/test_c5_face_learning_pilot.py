from copy import deepcopy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from c5_face_learning_pilot import SIGNERS, balanced_epoch, select, source_group
from train_c5_face_learning_pilot import DutyCycle, comparison, learning_rate_factor


def rows(shared=False):
    result = []
    for signer in SIGNERS:
        for group in range(8):
            content = f"{0 if shared else signer:02d}{group:09d}"
            for clip in range(10):
                result.append({"id": f"{content}_{clip}-{signer}-rgb_front", "source": "how2sign_test",
                    "signer": signer, "reference": f"unique actor {signer} content {group} clip {clip}"})
    for group in range(32):
        for clip in range(4):
            result.append({"id": f"dev_document_{group}_sentence_{clip}_signer_3", "source": "fleurs",
                "signer": 3, "reference": f"fleurs content {group} clip {clip}"})
    return result


def test_balanced_selection_excludes_entire_content_groups_across_actors():
    training, holdout, groups = select(rows(shared=True))
    assert len(training) == 320 and len(holdout) == 64
    assert Counter(r["signer"] for r in training if r["source"] != "fleurs") == {s:32 for s in SIGNERS}
    assert Counter(r["signer"] for r in holdout) == {s:8 for s in SIGNERS}
    assert {source_group(r) for r in training}.isdisjoint(groups)
    assert len(groups) == 2  # Shared content must remain globally held out.
    assert select(list(reversed(rows(shared=True)))) == (training, holdout, groups)


def test_holdout_exact_references_are_excluded_from_other_training_sources():
    fixture = rows(); training, holdout, _ = select(fixture)
    duplicate = training[0]
    next(r for r in fixture if r["id"] == duplicate["id"])["reference"] = holdout[0]["reference"]
    training, holdout, _ = select(fixture)
    assert {r["reference"] for r in training}.isdisjoint({r["reference"] for r in holdout})


def test_epoch_schedule_exposes_every_actor_early_and_preserves_counts():
    training, _, _ = select(rows())
    inputs = [{**r, "example_id": r["id"], "domain": "fleurs" if r["source"] == "fleurs" else "how2sign"} for r in training]
    schedule = balanced_epoch(inputs, 1)
    assert len(schedule) == 320 and len({r["example_id"] for r in schedule}) == 320
    assert Counter(r["signer"] for r in schedule[:10] if r["domain"] == "how2sign") == {s:1 for s in SIGNERS}
    assert sum(r["domain"] == "fleurs" for r in schedule[:10]) == 2
    assert balanced_epoch(inputs, 1) == schedule and balanced_epoch(inputs, 2) != schedule


def test_duty_cycle_waits_equal_active_time_and_handles_failed_operations():
    times = iter([0., 2.]); sleeps = []
    mps = SimpleNamespace(synchronize=lambda:None, current_allocated_memory=lambda:1024)
    duty = DutyCycle(SimpleNamespace(mps=mps), clock=lambda:next(times), sleep=sleeps.append)
    assert duty.run(lambda:"result") == "result"
    assert duty.active_seconds == duty.idle_seconds == 2. and sleeps == [2.]
    times = iter([0., 3.])
    duty = DutyCycle(SimpleNamespace(mps=mps), clock=lambda:next(times), sleep=sleeps.append)
    def fail(): raise ValueError("operation failed")
    with pytest.raises(ValueError): duty.run(fail)
    assert duty.active_seconds == duty.idle_seconds == 3.


def test_learning_rate_warmup_and_floor():
    assert learning_rate_factor(0, 320) == .125
    assert learning_rate_factor(8, 320) == 1.
    assert learning_rate_factor(320, 320) == .1


def test_diagnostic_gate_requires_generalized_gain_not_training_fit():
    parent = [{"example_id": str(i), "source_group": f"group-{i//4}", "signer": SIGNERS[i//8],
               "hypothesis": "valid", "wer": .8, "edits": 8, "reference_words": 10} for i in range(64)]
    candidate = [{**r, "wer": .6, "edits": 6} for r in parent]
    assert comparison(parent, candidate)["passes_diagnostic_gate"]
    candidate[-1]["hypothesis"] = ""
    assert not comparison(parent, candidate)["passes_diagnostic_gate"]
    candidate = deepcopy(parent)
    for r in candidate[:16]: r.update(wer=.5, edits=5)
    assert not comparison(parent, candidate)["passes_diagnostic_gate"]  # Only two actors improve.
