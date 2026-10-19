"""Synthetic runtime/data wiring; no real export, model, detector or GPU execution."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import probe_c5_native112_full_runtime as runtime
from extract_mediapipe_landmarks import write_landmarks


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setattr(runtime.exporter, "ROOT", tmp_path)
    return tmp_path


def test_missing_actual_export_rejects_before_any_output_or_model_import(owned):
    before = "torch" in sys.modules
    with pytest.raises(ValueError, match="complete visual export"):
        runtime.run(owned / "checkpoints/missing/EXPORT.json",
                    owned / "checkpoints/missing-parity/result.json", owned / "checkpoints/probe")
    assert not (owned / "checkpoints/probe").exists()
    assert not (owned / "data/probe").exists()
    assert ("torch" in sys.modules) == before


@pytest.fixture
def predecessor(owned, monkeypatch):
    training = owned / "runs/training/plan.json"
    cases = [{"record": {"original": {"row": {"role": "training", "training_allowed": True,
                                              "example_id": f"synthetic-{i}"}}}} for i in range(2)]
    save(training, {"records": cases})
    export = owned / "checkpoints/export/EXPORT.json"
    save(export, {"synthetic": True})
    record = {"evidence": {"training_plan": runtime.exporter.artifact(training)}}
    monkeypatch.setattr(runtime, "authenticate_export", lambda path: (record, {"synthetic": True}, {}))
    directory = owned / "checkpoints/parity"
    save(directory / "worker.json", {"pid": 1234})
    save(directory / "exit.json", {"pid": 1234, "exit_code": 0})
    source = owned / "tools/source.py"
    source.parent.mkdir()
    source.write_text("# synthetic")
    snapshot = directory / "source-snapshot/tools/source.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(source.read_bytes())
    plan = {"export": runtime.exporter.artifact(export), "case_indices": [0, 1], "records": cases,
            "source_sha256": {"tools/source.py": runtime.exporter.sha(source)},
            "CPU_threads": 2, "GPU_duty_fraction": 0.5, "optimizer_updates": 0, "new_Terra_requests": 0}
    save(directory / "plan.json", plan)
    predictions = [{"tokens": [0, i + 2, 1], "text": f"synthetic-{i}", "exact_token_match": True}
                   for i in range(2)]
    result = {"plan": runtime.exporter.artifact(directory / "plan.json"), "complete": True,
              "CPU": predictions, "Metal": predictions,
              "worker": runtime.exporter.artifact(directory / "worker.json")}
    save(directory / "result.json", result)
    for i, prediction in enumerate(predictions):
        feature = directory / f"reference-{i:02d}.npz"
        feature.write_bytes(b"synthetic placeholder; not decoded as an array in admission")
        save(directory / f"reference-{i:02d}.json",
             {**prediction, "features": runtime.exporter.artifact(feature)})
    return SimpleNamespace(export=export, result=directory / "result.json", directory=directory,
                           plan=plan, training=training, snapshot=snapshot, record=record)


def test_same_export_complete_synthetic_predecessor_is_bound(predecessor):
    record, model, cases, references, authority = runtime.admit(predecessor.export, predecessor.result)
    assert len(cases) == len(references) == 2
    assert authority["worker"]["path"].endswith("worker.json")


@pytest.mark.parametrize("kind", ["exit", "failure", "source", "snapshot", "tokens", "scope", "role"])
def test_incomplete_changed_or_reserved_predecessor_is_rejected(predecessor, kind):
    d = predecessor.directory
    if kind == "exit":
        save(d / "exit.json", {"pid": 1234, "exit_code": 1})
    elif kind == "failure":
        save(d / "failure.json", {"synthetic": True})
    elif kind in ("source", "snapshot"):
        path = predecessor.snapshot if kind == "snapshot" else runtime.ROOT / "tools/source.py"
        path.write_text("# changed")
    elif kind == "tokens":
        value = json.loads((d / "result.json").read_text())
        value["Metal"] = [dict(p) for p in value["Metal"]]
        value["Metal"][0]["tokens"] = [0, 99, 1]
        save(d / "result.json", value)
    else:
        plan = json.loads((d / "plan.json").read_text())
        if kind == "scope":
            plan["case_indices"] = [1, 2]
        else:
            plan["records"][0]["record"]["original"]["row"] = {
                "role": "development", "training_allowed": False, "example_id": "synthetic-reserved"}
            save(predecessor.training, {"records": plan["records"]})
            predecessor.record["evidence"]["training_plan"] = runtime.exporter.artifact(predecessor.training)
        save(d / "plan.json", plan)
        result = json.loads((d / "result.json").read_text())
        result["plan"] = runtime.exporter.artifact(d / "plan.json")
        save(d / "result.json", result)
    with pytest.raises(ValueError, match="reserved" if kind == "role" else None):
        runtime.admit(predecessor.export, predecessor.result)


def test_reference_copy_preserves_bytes_without_aliasing_or_changing_source(tmp_path):
    source, target = tmp_path / "source", tmp_path / "private"
    source.mkdir()
    (source / "nested").mkdir()
    file = source / "nested/asset"
    file.write_bytes(b"tiny immutable synthetic asset")
    file.chmod(0o444)
    before = file.stat()
    reservations = []
    result = runtime.private_copy(source, target, reservations.append)
    assert result["nested/asset"] == runtime.exporter.sha(file)
    assert file.stat().st_ino == before.st_ino and file.stat().st_nlink == 1
    assert (target / "nested/asset").stat().st_ino != before.st_ino
    assert (target / "nested/asset").stat().st_nlink == 1
    assert file.read_bytes() == b"tiny immutable synthetic asset"
    assert reservations == [len(b"tiny immutable synthetic asset")]
    (target / ".cache").mkdir()


def test_reference_copy_rejects_symlinks_before_copying(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "real").write_bytes(b"synthetic")
    (source / "alias").symlink_to(source / "real")
    with pytest.raises(ValueError, match="aliases"):
        runtime.private_copy(source, tmp_path / "private", lambda _: None)
    assert not (tmp_path / "private").exists()


def test_observed_embeddings_preserve_complete_pixel_order_and_reject_extra_calls():
    expected = np.arange(2 * 112 * 112 * 3, dtype=np.uint8).reshape(2, 112, 112, 3)
    called = []

    def embed(frames):
        called.append(frames)
        return np.ones((2, 384), dtype=np.float32)

    observed = runtime.ObservedEmbedder(SimpleNamespace(extract_embeddings_from_frames=embed),
                                        [expected, expected + np.uint8(1)])
    assert observed.extract_embeddings_from_frames(list(expected)).shape == (2, 384)
    observed.extract_embeddings_from_frames(expected + np.uint8(1))
    assert len(called) == 2 and len(observed.calls) == 2
    with pytest.raises(ValueError, match="repeated"):
        observed.extract_embeddings_from_frames(expected)
    assert len(called) == 2


@pytest.mark.parametrize("kind", ["pixel", "dtype", "order", "truncated"])
def test_changed_runtime_crop_cannot_reach_visual_model(kind):
    expected = np.stack([np.full((112, 112, 3), i, dtype=np.uint8) for i in (1, 2)])
    actual = expected.copy()
    if kind == "pixel":
        actual[0, 0, 0, 0] = 99
    elif kind == "dtype":
        actual = actual.astype(np.float32)
    elif kind == "order":
        actual = actual[::-1]
    else:
        actual = actual[:1]
    model = SimpleNamespace(extract_embeddings_from_frames=lambda _: pytest.fail("model must not run"))
    with pytest.raises(ValueError, match="crop pixels"):
        runtime.ObservedEmbedder(model, [expected]).extract_embeddings_from_frames(actual)


def test_landmark_fingerprint_distinguishes_missing_frame_and_empty_detections():
    empty = {"face_landmarks": None, "hand_landmarks": None, "pose_landmarks": None}
    assert runtime.fingerprint({0: None}) != runtime.fingerprint({0: empty})
    assert runtime.fingerprint({0: empty, 1: None}) == runtime.fingerprint({0: empty, 1: None})


@pytest.fixture
def detection(owned, monkeypatch):
    clip = owned / "data/clip.mp4"
    clip.parent.mkdir()
    clip.write_bytes(b"synthetic compressed video; not decoded by these tests")
    model_root = owned / "artifacts/baseline-6fe8df59/models"
    model_root.mkdir(parents=True)
    face = model_root / "face_landmarker_v2_with_blendshapes.task"
    hand = model_root / "hand_landmarker.task"
    face.write_bytes(b"synthetic face")
    hand.write_bytes(b"synthetic hand")
    rows = {0: {"face_landmarks": None, "hand_landmarks": None, "pose_landmarks": None}, 1: None}
    old = owned / "data/original.npz"
    write_landmarks(old, rows, source=clip, width=4, height=4, fps=25, mediapipe_version="0.10.14")
    source = {"clip": runtime.exporter.artifact(clip),
              "landmarks": runtime.exporter.artifact(old),
              "normalized_probe": {"width": 4, "height": 4, "decoded_frames": 2}}
    output = owned / "data/fresh"
    image, digest = "sha256:" + "a" * 64, "b" * 64
    monkeypatch.setattr(runtime, "container_command", lambda **kwargs: ["docker", "run", runtime.IMAGE, "python"])
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        write_landmarks(output / "landmarks.npz", rows, source=clip, width=4, height=4,
                        fps=25, mediapipe_version="0.10.14")
        return SimpleNamespace(returncode=0, stdout="synthetic stdout", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    detector = runtime.FreshDetector(clip=clip, source=source, output=output, image=image,
                                     digest=digest, number=0, guard=lambda extra=0: None)
    return SimpleNamespace(detector=detector, face=face, hand=hand, output=output, rows=rows,
                           clip=clip, calls=calls, image=image, digest=digest)


def test_fresh_detector_checks_original_graph_and_never_reuses_sidecar(detection):
    d = detection
    result = d.detector(np.zeros((2, 4, 4, 3), dtype=np.uint8), d.face, d.hand)
    assert result == d.rows and len(d.calls) == 1
    command, options = d.calls[0]
    assert command[command.index("--pull") + 1] == "never"
    assert d.image in command and d.digest in command[command.index("--label") + 1]
    assert options["timeout"] == 600 and "DOCKER_HOST" in options["env"]
    assert d.detector.receipt["fresh_detection"] is True
    assert d.detector.receipt["cached_landmark_substitution"] is False
    with pytest.raises(ValueError, match="replay"):
        d.detector(np.zeros((2, 4, 4, 3), dtype=np.uint8), d.face, d.hand)
    assert len(d.calls) == 1


def test_wrong_decoded_timeline_rejects_before_detector_process(detection):
    with pytest.raises(ValueError, match="decoded timeline"):
        detection.detector(np.zeros((1, 4, 4, 3), dtype=np.uint8), detection.face, detection.hand)
    assert detection.calls == [] and not detection.output.exists()


def test_changed_detector_graph_cannot_be_qualified(detection, monkeypatch):
    d = detection

    def run(*args, **kwargs):
        changed = dict(d.rows)
        changed[1] = {"face_landmarks": None, "hand_landmarks": None, "pose_landmarks": None}
        write_landmarks(d.output / "landmarks.npz", changed, source=d.clip, width=4, height=4,
                        fps=25, mediapipe_version="0.10.14")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    with pytest.raises(ValueError, match="complete legacy"):
        d.detector(np.zeros((2, 4, 4, 3), dtype=np.uint8), d.face, d.hand)


@pytest.mark.parametrize("owned", [True, False])
def test_timeout_cleanup_only_removes_a_named_container_with_matching_label_and_image(monkeypatch, owned):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "inspect":
            assert command[2] == "--format" and command[-1] == "specific-name"
            assert "Config.Env" not in command[3]
            return SimpleNamespace(returncode=0, stdout=json.dumps({
                "image": "image", "label": "digest" if owned else "other"
            }), stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runtime.subprocess, "run", run)
    if owned:
        runtime.clean_timed_out_container("specific-name", "digest", "image", {})
        assert calls[-1] == ["docker", "rm", "--force", "specific-name"]
    else:
        with pytest.raises(ValueError, match="unrelated"):
            runtime.clean_timed_out_container("specific-name", "digest", "image", {})
        assert len(calls) == 1


def test_preparation_binds_reviewed_source_bytes_before_heavy_execution(owned, monkeypatch):
    source = owned / "tools/probe.py"
    source.parent.mkdir()
    source.write_text("# synthetic tested source")
    path = owned / "runs/preparation/validation.json"
    value = {"schema": "umi-studio-native112-full-runtime-preparation/1",
             "implementation_validated": True, "tests_passed": 22,
             "actual_runtime_model_or_detector_executed": False,
             "source_sha256": {"tools/probe.py": runtime.exporter.sha(source)}}
    save(path, value)
    monkeypatch.setattr(runtime, "PREPARATION", path)
    assert runtime.checked_preparation() == runtime.exporter.artifact(path)
    source.write_text("# changed source")
    with pytest.raises(ValueError, match="preparation source changed"):
        runtime.checked_preparation()


def test_private_class_import_isolated_and_refuses_duplicate_module_name(tmp_path):
    name = "synthetic_private_native_runtime_test"
    source = tmp_path / "helper.py"
    source.write_text("from dataclasses import dataclass\n@dataclass\nclass Example:\n    value: int\n")
    try:
        module = runtime.load_private_class(tmp_path, "helper.py", name)
        assert module.Example(7).value == 7
        with pytest.raises(ValueError, match="already loaded"):
            runtime.load_private_class(tmp_path, "helper.py", name)
    finally:
        sys.modules.pop(name, None)


