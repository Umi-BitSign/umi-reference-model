from __future__ import annotations

import copy
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_native_terra_semantics as module


def fixture_rows():
    rows = [
        {
            "example_id": str(i),
            "reference": f"reference{i}",
            "source_group": f"source{i}",
            "signer": i,
            "domain": "how2sign",
            "role": "open_development",
            "training_allowed": False,
        }
        for i in range(2)
    ]
    initial = [
        {**r, "hypothesis": f"initial{i}", "checkpoint_examples": 0, "wer": 0.8}
        for i, r in enumerate(rows)
    ]
    current = [
        {
            **r,
            "hypothesis": ("initial0" if i == 0 else "new1"),
            "checkpoint_examples": 4096,
            "wer": 0.9,
        }
        for i, r in enumerate(rows)
    ]
    return rows, initial, current, {r["example_id"]: "official_val" for r in rows}


def test_exact_transcripts_share_item_and_no_model_identity_reaches_judge():
    cases, items = module.make_cases(*fixture_rows())
    assert len(cases) == 2 and len(items) == 3
    assert cases[0]["item_indices_C5_then_native4096"] == [0, 0]
    assert all(set(i) == {"reference", "candidate"} for i in items)
    assert all(c["training_allowed"] is False for c in cases)


@pytest.mark.parametrize("change", ["training", "signer", "duplicate", "checkpoint", "split"])
def test_mismatched_or_protected_evidence_is_rejected(change):
    values = fixture_rows()
    if change == "training":
        values[0][0]["training_allowed"] = True
    if change == "signer":
        values[2][0]["signer"] = 999
    if change == "duplicate":
        values[2].append(copy.deepcopy(values[2][0]))
    if change == "checkpoint":
        values[2][0]["checkpoint_examples"] = 2048
    if change == "split":
        values[3].clear()
    with pytest.raises(ValueError):
        module.make_cases(*values)


def test_both_shuffled_passes_cover_every_item_exactly_once_in_small_batches():
    actual = module.batches(226)
    assert actual == module.batches(226) and len(actual) == 2
    assert actual[0] != actual[1]
    for groups in actual:
        assert max(map(len, groups)) <= 16
        assert sorted(i for group in groups for i in group) == list(range(226))


def spec():
    cases, items = module.make_cases(*fixture_rows())
    return {
        "cases": cases,
        "items": [*items, {"reference": "calibration", "candidate": "calibration"}],
        "calibration_controls": [{"item_index": 3, "range": [10, 10]}],
    }


def test_paired_summary_keeps_identical_outputs_tied_and_reports_judge_noise():
    result = module.summarize(spec(), [[3, 4, 6, 10], [4, 5, 7, 10]])
    assert result["complete"] and result["calibration_controls_passed_both_passes"]
    assert result["mean_quality_delta_native_minus_C5"] == 1
    assert result["improved_clips"] == 1 and result["tied_clips"] == 1
    assert result["mean_absolute_repeat_score_difference"] == 0.75
    assert not result["promotion_authorized"] and not result["independent_human_validation"]
    assert not result["complete_benchmark_score"] and not result["existing_WER_gates_changed"]


def test_bad_controls_forbid_interpretation_and_incomplete_passes_cannot_score():
    result = module.summarize(spec(), [[3, 4, 6, 9], [4, 5, 7, 10]])
    assert not result["interpretation_permitted"]
    for scores in ([[3, 4, 6, 10]], [[3, 4, 6, 10], [4, 5, 10]], [[3, 4, 6, 10], [4, 5, True, 10]]):
        with pytest.raises(ValueError):
            module.summarize(spec(), scores)


def test_cooldown_prevents_any_request_or_output_creation(monkeypatch, tmp_path):
    import terra_quality_training

    monkeypatch.setattr(module, "ROOT", tmp_path)
    output = tmp_path / "runs" / "output"
    monkeypatch.setattr(module, "checked", lambda _: {"not_before": "2099-01-01T00:00:00+00:00"})
    monkeypatch.setattr(
        terra_quality_training, "score_with_terra", lambda *a: pytest.fail("no API before cooldown")
    )
    with pytest.raises(RuntimeError, match="cooldown"):
        module.run(tmp_path / "plan.json", output)
    assert not output.exists()


@pytest.mark.parametrize("failure", [False, True])
def test_runner_is_serial_has_no_retry_and_never_trains(monkeypatch, tmp_path, failure):
    import terra_quality_training
    import train_terra_c5_tail

    monkeypatch.setattr(module, "ROOT", tmp_path)
    plan = {**spec(), "not_before": "2000-01-01T00:00:00+00:00", "batches": [[[0, 1, 2, 3]]] * 2}
    path = tmp_path / "plan.json"
    path.write_text("{}")
    monkeypatch.setattr(module, "checked", lambda _: plan)
    monkeypatch.setattr(train_terra_c5_tail, "gpu_lease", nullcontext)
    clock = iter(range(0, 10000, 100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    calls = []

    def score(rows, destination):
        calls.append((rows, destination))
        if failure:
            raise RuntimeError("simulated overload")
        return {"scores": ([3, 4, 6, 10] if len(calls) == 1 else [4, 5, 7, 10]), "seconds": 1}

    monkeypatch.setattr(terra_quality_training, "score_with_terra", score)
    output = tmp_path / "runs" / "output"
    if failure:
        with pytest.raises(RuntimeError, match="overload"):
            module.run(path, output)
        assert len(calls) == 1 and not (output / "decision.json").exists()
    else:
        module.run(path, output)
        result = json.loads((output / "decision.json").read_text())
        assert len(calls) == 2 and result["complete"]
        assert not result["GPU_use"] and not result["training_allowed"]
    assert all(set(row) == {"reference", "candidate"} for rows, _ in calls for row in rows)


def test_modified_plan_is_rejected_before_any_work(monkeypatch, tmp_path):
    path = tmp_path / "plan.json"
    path.write_text('{"original": false}')
    monkeypatch.setattr(module, "specification", lambda: {"original": True})
    with pytest.raises(ValueError, match="binding changed"):
        module.checked(path)
