from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    path = ROOT / "community" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime():
    return load("native_runtime")


@pytest.fixture
def roots(tmp_path):
    result = {}
    for name in ("code", "environment", "python"):
        path = tmp_path / name
        path.mkdir(mode=0o700)
        file = path / "fixture.py"
        file.write_text("raise AssertionError('never execute inventory input')\n")
        file.chmod(0o600)
        result[name] = path
    return result


def stage(runtime, roots, tmp_path):
    manifest = tmp_path / "runtime.json"
    revision = runtime.stage_manifest(
        roots, bundle_sha256="ab" * 32, overlay_sha256="cd" * 32, output=manifest
    )
    return manifest, revision


def test_inventory_roundtrip_never_executes_inputs(runtime, roots, tmp_path):
    manifest, revision = stage(runtime, roots, tmp_path)
    doc = runtime.verify_runtime(roots, manifest=manifest, expected_revision=revision)
    assert doc["model_bundle_sha256"] == "ab" * 32
    assert len(doc["entries"]) == 6
    assert hashlib.sha256(runtime.DOMAIN + manifest.read_bytes()).hexdigest() == revision
    assert manifest.stat().st_mode & 0o777 == 0o400


@pytest.mark.parametrize("change", ["content", "additional", "removed", "mode"])
def test_runtime_drift_refuses_readiness(runtime, roots, tmp_path, change):
    manifest, revision = stage(runtime, roots, tmp_path)
    path = roots["environment"] / "fixture.py"
    if change == "content":
        path.write_text("changed executable")
    elif change == "additional":
        (roots["environment"] / "extra.py").write_text("unreviewed")
    elif change == "removed":
        path.unlink()
    else:
        path.chmod(0o400)
    with pytest.raises(ValueError, match="installed runtime"):
        runtime.verify_runtime(roots, manifest=manifest, expected_revision=revision)


def test_symlink_to_inventoried_python_is_bound(runtime, roots, tmp_path):
    link = roots["environment"] / "python"
    link.symlink_to(roots["python"] / "fixture.py")
    manifest, revision = stage(runtime, roots, tmp_path)
    runtime.verify_runtime(roots, manifest=manifest, expected_revision=revision)
    link.unlink()
    link.symlink_to(roots["code"] / "fixture.py")
    with pytest.raises(ValueError, match="installed runtime"):
        runtime.verify_runtime(roots, manifest=manifest, expected_revision=revision)


@pytest.mark.parametrize("kind", ["escape", "fifo", "hardlink", "group_writable"])
def test_bad_runtime_layout_rejected(runtime, roots, tmp_path, kind):
    path = roots["environment"] / "bad"
    if kind == "escape":
        outside = tmp_path / "outside"
        outside.write_text("outside runtime")
        path.symlink_to(outside)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "hardlink":
        os.link(roots["code"] / "fixture.py", path)
    else:
        roots["environment"].chmod(0o770)
    with pytest.raises(ValueError):
        runtime.inventory(roots)


def test_manifest_cannot_authorize_itself(runtime, roots, tmp_path):
    manifest, _ = stage(runtime, roots, tmp_path)
    with pytest.raises(ValueError, match="reviewed release"):
        runtime.verify_runtime(roots, manifest=manifest, expected_revision="ff" * 32)
    with pytest.raises(FileExistsError):
        stage(runtime, roots, tmp_path)


def test_runtime_size_and_count_bounds(runtime, roots, monkeypatch):
    monkeypatch.setattr(runtime, "MAXIMUM_BYTES", 1)
    with pytest.raises(ValueError, match="bounded"):
        runtime.inventory(roots)
    monkeypatch.setattr(runtime, "MAXIMUM_BYTES", 1024)
    monkeypatch.setattr(runtime, "MAXIMUM_ENTRIES", 1)
    with pytest.raises(ValueError, match="too many"):
        runtime.inventory(roots)


def test_runtime_roots_must_not_overlap(runtime, roots):
    roots["code"] = roots["environment"]
    with pytest.raises(ValueError, match="overlap"):
        runtime.inventory(roots)


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "community"))
    monkeypatch.syspath_prepend(str(ROOT / "community/native-build"))
    return load("native_worker")


def test_worker_directories_must_be_canonical_and_nonoverlapping(worker, tmp_path):
    roots = {}
    for name in ("code", "environment", "python", "bundle", "overlay", "scratch"):
        roots[name] = tmp_path / name
        roots[name].mkdir()
    assert worker.checked_directories(roots) == roots

    alias = tmp_path / "bundle-alias"
    alias.symlink_to(roots["bundle"], target_is_directory=True)
    with pytest.raises(ValueError, match="without aliases"):
        worker.checked_directories({**roots, "bundle": alias})

    nested = roots["bundle"] / "scratch"
    nested.mkdir()
    with pytest.raises(ValueError, match="non-overlapping"):
        worker.checked_directories({**roots, "scratch": nested})


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin F_GETPATH contract")
def test_worker_rejects_apfs_case_alias(worker, tmp_path):
    canonical = tmp_path / "CaseSensitiveSpelling"
    canonical.mkdir()
    alias = Path(str(canonical).replace("/Users/", "/users/", 1))
    if alias == canonical or not alias.is_dir():
        pytest.skip("temporary volume has no alternate-case root spelling")
    with pytest.raises(ValueError, match="aliases"):
        worker._opened_directory(alias)


def test_live_worker_bundle_contract_is_exact_and_preserves_historical_bundle():
    contract_root = ROOT / "community/native-worker-bundle"
    contract = json.loads((contract_root / "contract.json").read_bytes())
    historical = json.loads((ROOT / "community/native/manifest.json").read_bytes())
    historical_raw = json.dumps(
        historical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    assert (
        hashlib.sha256(historical_raw).hexdigest()
        == contract["historical_bundle"]["manifest_file_sha256"]
    )
    assert (
        hashlib.sha256(b"umi-open-competition-v1\0" + historical_raw).hexdigest()
        == contract["historical_bundle"]["model_bundle_sha256"]
    )

    entrypoint = (contract_root / "umi_inference.py").read_bytes()
    live = contract["live_worker_bundle"]
    assert len(entrypoint) == live["entrypoint_size_bytes"]
    assert hashlib.sha256(entrypoint).hexdigest() == live["entrypoint_sha256"]
    record = next(item for item in historical["files"] if item["path"] == live["entrypoint_path"])
    assert record["sha256"] == contract["historical_bundle"]["entrypoint_sha256"]
    assert record["size_bytes"] == contract["historical_bundle"]["entrypoint_size_bytes"]
    record.update(sha256=live["entrypoint_sha256"], size_bytes=live["entrypoint_size_bytes"])
    live_raw = json.dumps(
        historical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    assert hashlib.sha256(live_raw).hexdigest() == live["manifest_file_sha256"]
    assert (
        hashlib.sha256(b"umi-open-competition-v1\0" + live_raw).hexdigest()
        == live["model_bundle_sha256"]
    )
    assert not contract["bundle_entrypoint_executed_by_worker"]


@pytest.mark.parametrize("fail", [False, True])
def test_worker_uses_private_scratch_and_cleans_input(worker, tmp_path, fail):
    deadline = time.time_ns() + 1_000_000_000

    def translate(path, *, deadline_unix_ns):
        assert path.parent == tmp_path
        assert path.read_bytes() == b"exact video bytes"
        assert path.stat().st_mode & 0o777 == 0o600
        assert deadline_unix_ns == deadline
        if fail:
            raise TimeoutError("fixture deadline")
        return "English output"

    model = types.SimpleNamespace(translate_path=translate)
    if fail:
        with pytest.raises(TimeoutError):
            worker.translate_video(model, tmp_path, b"exact video bytes", deadline)
    else:
        assert (
            worker.translate_video(model, tmp_path, b"exact video bytes", deadline)
            == "English output"
        )
    assert list(tmp_path.iterdir()) == []


def test_loader_requires_expected_native_binding(worker, tmp_path, monkeypatch):
    overlay = tmp_path / "overlay"
    bundle = tmp_path / "bundle"
    calls = []
    torch = types.SimpleNamespace(
        set_num_threads=lambda count: calls.append(("threads", count)),
        set_num_interop_threads=lambda count: calls.append(("interop", count)),
    )
    model = types.SimpleNamespace()

    def constructor(root, **arguments):
        assert root == bundle / "model"
        assert arguments["device"] == "mps"
        assert arguments["generation_num_beams"] == 5
        assert arguments["generation_max_length"] == 2048
        assert arguments["verify_assets"] is True
        calls.append(("model", arguments))
        return model

    monkeypatch.setitem(sys.modules, "torch", torch)
    bindings = types.SimpleNamespace(
        __file__=str(overlay / "mediapipe/python/_framework_bindings.cpython-310-darwin.so")
    )
    monkeypatch.setitem(
        sys.modules, "mediapipe.python", types.SimpleNamespace(_framework_bindings=bindings)
    )
    monkeypatch.setitem(
        sys.modules, "runtime", types.SimpleNamespace(SHuBERTInferenceRuntime=constructor)
    )
    reader, landmarks = object(), object()
    monkeypatch.setitem(sys.modules, "umi_video_reader", types.SimpleNamespace(VideoReader=reader))
    monkeypatch.setitem(
        sys.modules, "umi_landmarks", types.SimpleNamespace(video_holistic=landmarks)
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    assert worker.load_model(bundle, overlay, tmp_path / "scratch") is model
    assert model._video_reader is reader and model._video_holistic is landmarks
    assert calls[0:2] == [("threads", 4), ("interop", 1)]
    bindings.__file__ = str(tmp_path / "wrong.so")
    with pytest.raises(RuntimeError, match="not loaded"):
        worker.load_model(bundle, overlay, tmp_path / "scratch")
