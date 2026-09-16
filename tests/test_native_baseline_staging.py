from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from .test_community_baseline import ROOT, archive_fixture, load


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    return load("tools/stage_native_baseline.py")


def test_native_templates_match_the_measured_bundle(tool):
    manifest = tool.expected_manifest()
    entries = {record["path"]: record for record in manifest["files"]}
    assert len(entries) == len(manifest["files"])
    for name in tool.OVERLAYS:
        content = tool.overlay_bytes(name)
        assert len(content) == entries[name]["size_bytes"]
        assert hashlib.sha256(content).hexdigest() == entries[name]["sha256"]
    assert [entry["path"] for entry in manifest["files"] if entry["role"] == "inference"] == [
        "umi_inference.py"
    ]


def fixture_bundle(tmp_path, tool, monkeypatch):
    archive = archive_fixture(tmp_path, tool.intake, monkeypatch)
    original = tmp_path / "original"
    tool.intake.stage_baseline(archive, original)
    manifest = json.loads((original / "manifest.json").read_bytes())
    entries = {record["path"]: record for record in manifest["files"]}
    for name in tool.OVERLAYS:
        body = tool.overlay_bytes(name)
        entries[name] = {
            "path": name,
            "sha256": hashlib.sha256(body).hexdigest(),
            "size_bytes": len(body),
            "role": tool.intake._role(name),
        }
    manifest["files"] = sorted(entries.values(), key=lambda entry: entry["path"])
    digest = hashlib.sha256(
        b"umi-open-competition-v1\0" + tool.intake.canonical(manifest)
    ).hexdigest()
    monkeypatch.setattr(tool, "BUNDLE_SHA256", digest)
    monkeypatch.setattr(tool, "expected_manifest", lambda: manifest)
    return archive, manifest, original


def test_reconstruction_preserves_tensors_and_is_not_an_activation(tmp_path, tool, monkeypatch):
    archive, manifest, original = fixture_bundle(tmp_path, tool, monkeypatch)
    destination = tmp_path / "native"
    report = tool.stage_native_baseline(archive, destination)
    assert (destination / "manifest.json").read_bytes() == tool.intake.canonical(manifest)
    assert report["device"] == "mps" and not report["tensor_bytes_changed"]
    assert not report["native_runtime_installed"] and not report["inference_verified"]
    assert not report["rights_reviewed"] and not report["chain_submission_authorized"]
    assert report["contributor_attribution"] is None
    for entry in manifest["files"]:
        path = destination / "model" / entry["path"]
        body = path.read_bytes()
        assert hashlib.sha256(body).hexdigest() == entry["sha256"]
        assert path.stat().st_mode & 0o777 == 0o400
        if entry["role"] == "weights":
            assert body == (original / "model" / entry["path"]).read_bytes()
    with pytest.raises(ValueError, match="new absolute path"):
        tool.stage_native_baseline(archive, destination)
    assert not list(tmp_path.glob(".native-baseline-pending-*"))


def test_changed_overlay_is_rejected_before_extraction(tmp_path, tool, monkeypatch):
    archive, _, _ = fixture_bundle(tmp_path, tool, monkeypatch)
    monkeypatch.setattr(tool, "overlay_bytes", lambda name: b"changed")
    with pytest.raises(ValueError, match="overlay differs"):
        tool.stage_native_baseline(archive, tmp_path / "native")
    assert not (tmp_path / "native").exists()
    assert not list(tmp_path.glob(".native-baseline-pending-*"))


def test_changed_inventory_is_rejected_and_temporary_copy_removed(tmp_path, tool, monkeypatch):
    archive, manifest, _ = fixture_bundle(tmp_path, tool, monkeypatch)
    next(entry for entry in manifest["files"] if entry["role"] == "weights")["sha256"] = "00" * 32
    with pytest.raises(ValueError, match="checksum differs"):
        tool.stage_native_baseline(archive, tmp_path / "native")
    assert not (tmp_path / "native").exists()
    assert not list(tmp_path.glob(".native-baseline-pending-*"))


def test_destination_alias_is_rejected(tmp_path, tool):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="without aliases"):
        tool.stage_native_baseline(Path("unused.zip"), alias / "native")
