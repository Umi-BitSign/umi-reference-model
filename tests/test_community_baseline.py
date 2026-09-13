from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
import types
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(relative: str):
    spec = importlib.util.spec_from_file_location("community_test_module", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archive_fixture(tmp_path, tool, monkeypatch, *, change=None):
    payloads = {
        "LICENSE": b"MIT test notice",
        "PROVENANCE.md": b"fixture only",
        "models/byt5_base/tokenizer_config.json": b"{}",
        "models/checkpoint-11625/config.json": b"{}",
        "models/checkpoint-11625/model.safetensors": b"inert test bytes",
        "requirements.txt": b"",
        "runtime.py": b"raise AssertionError('intake must never execute model code')",
    }
    inventory = {
        "files": [
            {"path": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in sorted(payloads.items())
        ]
    }
    payloads["FILE_LIST.json"] = json.dumps(inventory).encode()
    payloads["SHA256SUMS"] = "".join(
        f"{hashlib.sha256(value).hexdigest()}  {name}\n" for name, value in sorted(payloads.items())
    ).encode()
    if change is not None:
        change(payloads)
    path = tmp_path / "source.zip"
    with zipfile.ZipFile(path, "x") as archive:
        for name, content in payloads.items():
            info = zipfile.ZipInfo(tool.ARCHIVE_ROOT + name)
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, content)
    monkeypatch.setattr(tool, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return path


def test_stage_preserves_source_and_adds_plaintext_cpu_entrypoint(tmp_path, monkeypatch):
    tool = load("tools/import_community_baseline.py")
    source = archive_fixture(tmp_path, tool, monkeypatch)
    destination = tmp_path / "staged"
    report = tool.stage_baseline(source, destination)
    manifest = (destination / "manifest.json").read_bytes()
    document = json.loads(manifest)
    assert (
        report["model_bundle_sha256"]
        == hashlib.sha256(b"umi-open-competition-v1\0" + manifest).hexdigest()
    )
    assert report["manifest_file_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert not report["inference_verified"] and not report["rights_reviewed"]
    assert report["contributor_attribution"] is None
    assert not report["chain_submission_authorized"]
    assert document["parent_baseline_sha256"] is None
    assert [r["path"] for r in document["files"] if r["role"] == "inference"] == [
        "umi_inference.py"
    ]
    assert {r["role"] for r in document["files"]} >= {
        "weights",
        "inference",
        "config",
        "processor",
        "environment",
        "license",
        "provenance",
    }
    for record in document["files"]:
        file = destination / "model" / record["path"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == record["sha256"]
        assert file.stat().st_size == record["size_bytes"]
        assert stat.S_IMODE(file.stat().st_mode) == 0o400
    with pytest.raises(ValueError, match="new absolute"):
        tool.stage_baseline(source, destination)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a//b", "a\\b", "a/./b"])
def test_archive_rejects_unsafe_paths(tmp_path, monkeypatch, name):
    tool = load("tools/import_community_baseline.py")
    source = archive_fixture(
        tmp_path, tool, monkeypatch, change=lambda payloads: payloads.update({name: b"bad"})
    )
    with pytest.raises(ValueError, match="path"):
        tool.stage_baseline(source, tmp_path / "staged")
    assert not (tmp_path / "staged").exists()


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda p: p.update({"undeclared.py": b"bad"}), "inventory mismatch"),
        (lambda p: p.update({"runtime.py": b"changed"}), "inventories disagree"),
        (
            lambda p: p.update({"runtime.py": p["runtime.py"].replace(b"intake", b"INTAKE")}),
            "checksum mismatch",
        ),
        (lambda p: p.update({"LICENSE/child": b"bad"}), "also a directory"),
        (lambda p: p.update({"license": b"bad"}), "duplicate"),
    ],
)
def test_archive_rejects_untrusted_inventory(tmp_path, monkeypatch, change, reason):
    tool = load("tools/import_community_baseline.py")
    source = archive_fixture(tmp_path, tool, monkeypatch, change=change)
    with pytest.raises(ValueError, match=reason):
        tool.stage_baseline(source, tmp_path / "staged")
    assert not (tmp_path / "staged").exists()
    assert not list(tmp_path.glob(".community-pending-*"))


def test_archive_requires_pinned_bytes_and_private_destination(tmp_path, monkeypatch):
    tool = load("tools/import_community_baseline.py")
    source = archive_fixture(tmp_path, tool, monkeypatch)
    monkeypatch.setattr(tool, "ARCHIVE_SHA256", "00" * 32)
    with pytest.raises(ValueError, match="SHA-256"):
        tool.stage_baseline(source, tmp_path / "staged")
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match="private"):
        tool.stage_baseline(source, tmp_path / "staged")


def test_entrypoint_uses_cpu_without_json_or_diagnostic_stdout(tmp_path, monkeypatch, capsys):
    adapter = load("community/umi_inference.py")
    calls = []

    class FakeRuntime:
        def __init__(self, root, **options):
            calls.append((root, options))
            print("model loading diagnostic")

        def translate_path(self, video):
            calls.append(video)
            print("model inference diagnostic")
            return "A test sentence."

    monkeypatch.setitem(
        sys.modules, "runtime", types.SimpleNamespace(SHuBERTInferenceRuntime=FakeRuntime)
    )
    # Restore the environment after this test.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    result = adapter.translate(tmp_path / "input.mp4", tmp_path)
    assert result == "A test sentence."
    assert calls[0][1] == {
        "device": "cpu",
        "generation_num_beams": 5,
        "generation_max_length": 2048,
        "model_execution_concurrency": 1,
        "verify_assets": True,
    }
    assert os.environ["HF_HOME"] == "/tmp/hf"
    assert os.environ["SHUBERT_DEVICE"] == "cpu"
    output = capsys.readouterr()
    assert output.out == ""
    assert "model loading diagnostic" in output.err
    assert "model inference diagnostic" in output.err


@pytest.mark.parametrize("result", [None, "", " \n", "a" * 4097, "é" * 2049])
def test_entrypoint_rejects_invalid_or_oversized_hypotheses(tmp_path, monkeypatch, result):
    adapter = load("community/umi_inference.py")

    class FakeRuntime:
        def __init__(self, *args, **kwargs):
            pass

        def translate_path(self, video):
            return result

    monkeypatch.setitem(
        sys.modules, "runtime", types.SimpleNamespace(SHuBERTInferenceRuntime=FakeRuntime)
    )
    monkeypatch.setattr(os, "environ", os.environ.copy())
    with pytest.raises(ValueError):
        adapter.translate(tmp_path / "input.mp4", tmp_path)


@pytest.mark.parametrize("entry_mode", [stat.S_IFLNK | 0o777, stat.S_IFIFO | 0o600])
def test_archive_rejects_nonregular_member(tmp_path, monkeypatch, entry_mode):
    tool = load("tools/import_community_baseline.py")
    source = tmp_path / "nonregular.zip"
    with zipfile.ZipFile(source, "x") as archive:
        info = zipfile.ZipInfo(tool.ARCHIVE_ROOT + "payload")
        info.external_attr = entry_mode << 16
        archive.writestr(info, b"outside")
    monkeypatch.setattr(tool, "ARCHIVE_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="non-regular"):
        tool.stage_baseline(source, tmp_path / "staged")
    assert not (tmp_path / "staged").exists()
