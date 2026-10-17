from __future__ import annotations

import copy
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import prepare_terra_word_native as prepare
from terra_training_recovery import artifact


def inventory():
    records = []
    for index in range(324):
        role = "training" if index < 228 else "development"
        participant = (
            f"P{1 + index // 60}"
            if index < 214
            else ("P0" if index % 2 else "P19")
            if index < 228
            else "P4"
        )
        # P4 is reserved, so keep the fourth training repetition with P1.
        if index < 214 and participant == "P4":
            participant = "P1"
        records.append(
            {
                "original": {
                    "row": {
                        "example_id": str(index),
                        "label": f"word{index % 60}",
                        "role": role,
                        "participant": participant,
                        "training_allowed": role == "training",
                    }
                },
                "ROI_lineage_training_allowed": index < 214,
            }
        )
    return records


def test_all_original_roles_survive_the_explicit_training_eligibility_filter():
    records = inventory()
    original = copy.deepcopy(records)
    eligible, excluded = prepare.eligible_records(records)
    assert len(eligible) == 310 and len(excluded) == 14
    assert all(r["original"]["row"]["role"] == "training" for r in excluded)
    assert records == original
    assert len({r["original"]["row"]["label"] for r in eligible[:214]}) == 60


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "reserved", "diagnostic", "labels"])
def test_preflight_cannot_silently_drop_labels_clips_or_reserved_exclusions(mutation):
    records = inventory()
    if mutation == "duplicate":
        records[1] = copy.deepcopy(records[0])
    elif mutation == "missing":
        records.pop()
    elif mutation == "reserved":
        records[0]["original"]["row"]["participant"] = "P10"
    elif mutation == "diagnostic":
        records[214]["ROI_lineage_training_allowed"] = True
    else:
        for record in records:
            record["original"]["row"]["label"] = "one"
    with pytest.raises(ValueError):
        prepare.eligible_records(records)


def test_preflight_uses_eighty_GiB_floor_and_the_complete_receipt_reservation(monkeypatch):
    required = sum(
        prepare.POLICY[key]
        for key in ("physical_free_floor_bytes", "headroom_bytes", "maximum_output_bytes")
    )
    assert prepare.POLICY["physical_free_floor_bytes"] == 80 * 1024**3
    monkeypatch.setattr(prepare.shutil, "disk_usage", lambda path: SimpleNamespace(free=required))
    prepare.guard()
    monkeypatch.setattr(
        prepare.shutil, "disk_usage", lambda path: SimpleNamespace(free=required - 1)
    )
    with pytest.raises(RuntimeError, match="80GiB"):
        prepare.guard()


@pytest.mark.parametrize("fail_second", [False, True])
def test_preflight_stops_at_first_bad_clip_and_retains_authoritative_exit_receipts(
    tmp_path, monkeypatch, fail_second
):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")
    plan = tmp_path / "runs/plan.json"
    plan.parent.mkdir()
    plan.write_text("{}")
    rows = [{"original": {"row": {"role": "training", "example_id": str(i)}}} for i in range(3)]
    checked = []

    def checked_plan(path):
        checked.append(path)
        return {"records": rows, "eligible_counts": {"training": 3}}

    monkeypatch.setattr(prepare, "checked", checked_plan)
    monkeypatch.setattr(prepare, "eligible_records", lambda records: (records, []))
    monkeypatch.setattr(prepare, "guard", lambda output=None: None)
    observed = []

    def read(record, purpose):
        index = int(record["original"]["row"]["example_id"])
        observed.append(index)
        assert purpose == "training"
        if index == 1 and fail_second:
            raise ValueError("actual native pixel mismatch fixture")
        return {
            "pixels": np.full((1, 3, 2, 2, 3), index, dtype=np.uint8),
            "pose_features": np.zeros((1, 14), dtype=np.float32),
            "frame_indices": np.array([0], dtype=np.int64),
            "header": {"sampled_frames": 1},
        }

    monkeypatch.setattr(prepare, "read_input", read)
    output = tmp_path / "checkpoints/word"
    if fail_second:
        with pytest.raises(ValueError, match="native pixel mismatch"):
            prepare.run(plan, output)
        assert observed == [0, 1] and len(checked) == 1
        assert (output / "input-0000.json").exists()
        assert not (output / "data.json").exists()
        assert json.loads((output / "failure.json").read_text())["automatic_retries"] == 0
    else:
        prepare.run(plan, output)
        assert observed == [0, 1, 2] and len(checked) == 2
        saved = json.loads((output / "data.json").read_text())
        assert saved["complete"] and saved["sampled_frames_by_role"] == {"training": 3}
        assert saved["model_training_ready"] is False and saved["optimizer_updates"] == 0
    worker = json.loads((output / "worker.json").read_text())
    terminal = json.loads((output / "exit.json").read_text())
    assert worker["pid"] == terminal["pid"]
    assert terminal["exit_code"] == (1 if fail_second else 0)
