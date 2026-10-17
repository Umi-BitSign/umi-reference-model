from __future__ import annotations

import copy
import hashlib
import json
import sys
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_terra_mixed_native as prepare
import terra_mixed_native_input as mixed
from terra_training_recovery import artifact
from test_terra_mixed_native_input import phrase_record, word_case


@pytest.fixture
def tiny_preflight(tmp_path, monkeypatch):
    import audit_terra_word_landmarks

    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(audit_terra_word_landmarks, "artifact", partial(artifact, root=tmp_path))
    monkeypatch.setattr(prepare, "guard", lambda output=None: None)
    cases = [
        {"kind": "word", "word": word_case()},
        {"kind": "phrase", "phrase": phrase_record("phrase1")},
        {"kind": "phrase", "phrase": phrase_record("phrase2", frames=6)},
    ]
    values = []
    for index, case in enumerate(cases):
        header = mixed.expected_header(case)
        arrays = {
            "pixels": np.full((3, 3, 112, 112, 3), index, dtype=np.uint8),
            "pose_features": np.full((3, 14), -9999, dtype=np.float32),
            "frame_indices": np.asarray([0, 2, 4], dtype=np.int64),
            "header": header,
        }
        if case["kind"] == "word":
            case["word"]["verified_input"] = {
                "arrays": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                        "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                    }
                    for name, value in arrays.items()
                    if name != "header"
                }
            }
        values.append(arrays)
    plan = {
        "records": cases,
        "counts": {"word": 1, "phrase": 2},
        "sampled_frames_by_kind": {"word": 3, "phrase": 6},
    }
    checked = []
    monkeypatch.setattr(prepare, "checked", lambda path: checked.append(path) or plan)
    accesses = []

    def read(case):
        index = cases.index(case)
        accesses.append(index)
        return copy.deepcopy(values[index])

    monkeypatch.setattr(prepare, "read_input", read)
    path = tmp_path / "runs/mixed/plan.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan))
    output = tmp_path / "checkpoints/mixed"
    return SimpleNamespace(
        path=path,
        output=output,
        cases=cases,
        values=values,
        plan=plan,
        checked=checked,
        accesses=accesses,
        read=read,
    )


def test_complete_preflight_is_read_back_through_actual_terminal_and_per_input_receipts(
    tiny_preflight,
):
    env = tiny_preflight
    prepare.prepare(env.path, env.output)
    assert env.accesses == [0, 1, 2] and len(env.checked) == 2
    result = json.loads((env.output / "data.json").read_text())
    assert result["sampled_frames_by_kind"] == {"word": 3, "phrase": 6}
    assert result["model_training_ready"] is False and result["optimizer_recipe_selected"] is False
    assert not list(env.output.glob("*.npz"))
    binding = prepare.checked_result(env.path, env.output / "data.json")
    assert len(binding["records"]) == 3 and binding["model_training_qualification_still_required"]
    for index, case in enumerate(binding["records"]):
        actual = prepare.load_verified_input(case)
        for name in ("pixels", "pose_features", "frame_indices"):
            np.testing.assert_array_equal(actual[name], env.values[index][name])


def test_bad_second_clip_stops_without_retry_or_complete_result(tiny_preflight, monkeypatch):
    env = tiny_preflight

    def read(case):
        index = env.cases.index(case)
        if index == 1:
            env.accesses.append(index)
            raise ValueError("native pixel mismatch fixture")
        return env.read(case)

    monkeypatch.setattr(prepare, "read_input", read)
    with pytest.raises(ValueError, match="native pixel mismatch"):
        prepare.prepare(env.path, env.output)
    assert env.accesses == [0, 1]
    assert (env.output / "input-0000.json").exists() and not (env.output / "data.json").exists()
    assert json.loads((env.output / "exit.json").read_text())["exit_code"] == 1
    assert json.loads((env.output / "failure.json").read_text())["automatic_retries"] == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "counts",
        "frames",
        "pose",
        "GPU",
        "quality",
        "terminal",
        "missing",
        "extra",
        "bytes",
        "rebound_shape",
    ],
)
def test_complete_result_rejects_incomplete_or_changed_actual_evidence(tiny_preflight, mutation):
    env = tiny_preflight
    prepare.prepare(env.path, env.output)
    result_path = env.output / "data.json"
    result = json.loads(result_path.read_text())
    if mutation == "counts":
        result["counts"]["word"] = 0
    elif mutation == "frames":
        result["sampled_frames_by_kind"]["phrase"] = 5
    elif mutation == "pose":
        result["all_original_pose_and_easier_word_inputs_preserved"] = False
    elif mutation == "GPU":
        result["GPU_use"] = True
    elif mutation == "quality":
        result["translation_quality_evaluated"] = True
    elif mutation == "terminal":
        (env.output / "exit.json").write_text(json.dumps({"pid": -1, "exit_code": 0}))
    elif mutation == "missing":
        (env.output / "input-0001.json").unlink()
    elif mutation == "extra":
        (env.output / "input-0999.json").write_text("{}")
    elif mutation == "bytes":
        (env.output / "input-0001.json").write_text("{}")
    else:
        path = env.output / "input-0001.json"
        receipt = json.loads(path.read_text())
        receipt["arrays"]["pixels"]["shape"][2:4] = [64, 64]
        path.write_text(json.dumps(receipt))
        result["receipts"][1] = prepare.artifact(path)
    result_path.write_text(json.dumps(result))
    with pytest.raises((ValueError, FileNotFoundError)):
        prepare.checked_result(env.path, result_path)


@pytest.mark.parametrize("array", ["pixels", "pose_features", "frame_indices"])
def test_every_array_is_reauthenticated_on_later_learning_read(tiny_preflight, array):
    env = tiny_preflight
    prepare.prepare(env.path, env.output)
    binding = prepare.checked_result(env.path, env.output / "data.json")
    env.values[1][array].reshape(-1)[0] += 1
    with pytest.raises(ValueError):
        prepare.load_verified_input(binding["records"][1])


def test_forbidden_role_fails_before_receipt_access(tiny_preflight, monkeypatch):
    env = tiny_preflight
    prepare.prepare(env.path, env.output)
    binding = prepare.checked_result(env.path, env.output / "data.json")
    record = binding["records"][1]
    record["case"]["phrase"]["roi_training_allowed"] = False
    monkeypatch.setattr(prepare, "artifact", lambda *a: pytest.fail("read forbidden receipt"))
    with pytest.raises(ValueError):
        prepare.load_verified_input(record)


def test_shared_lease_refuses_overlap_before_source_or_output_access(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;2")

    @contextmanager
    def busy():
        raise RuntimeError("another owned C5 Terra GPU job holds the lease")
        yield

    monkeypatch.setattr(prepare, "gpu_lease", busy)
    monkeypatch.setattr(prepare, "prepare", lambda *a: pytest.fail("started overlapping decoding"))
    output = tmp_path / "checkpoints/new"
    with pytest.raises(RuntimeError, match="holds the lease"):
        prepare.run(tmp_path / "plan.json", output)
    assert not output.exists()


def test_receipt_storage_keeps_the_eighty_GiB_floor_and_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    required = sum(
        prepare.POLICY[name]
        for name in ("physical_free_floor_bytes", "headroom_bytes", "maximum_output_bytes")
    )
    prepare.guard(disk_usage=lambda root: SimpleNamespace(free=required))
    with pytest.raises(RuntimeError, match="80GiB"):
        prepare.guard(disk_usage=lambda root: SimpleNamespace(free=required - 1))
    output = tmp_path / "checkpoints/preflight"
    output.mkdir(parents=True)
    with (output / "too-large").open("wb") as stream:
        stream.truncate(prepare.POLICY["maximum_output_bytes"] + 1)
    with pytest.raises(RuntimeError, match="32MiB"):
        prepare.guard(output, disk_usage=lambda root: SimpleNamespace(free=required))


def test_output_alias_is_rejected_before_preflight_writes(tiny_preflight):
    env = tiny_preflight
    target = env.output.parent / "target"
    target.mkdir(parents=True)
    alias = env.output.parent / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        prepare.prepare(env.path, alias / "new")
    assert not (target / "new").exists()
