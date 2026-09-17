from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "community/native-build" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def verifier():
    return load("verify_macos_overlay")


def publish(root, document):
    root.chmod(0o700)
    manifest = root / "overlay.json"
    if manifest.exists():
        manifest.chmod(0o600)
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    manifest.write_bytes(raw)
    manifest.chmod(0o400)
    root.chmod(0o500)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def overlay(tmp_path):
    root = tmp_path / "overlay"
    (root / "mediapipe/python").mkdir(parents=True)
    name = "mediapipe/python/placeholder.so"
    value = b"inert fixture, never imported"
    (root / name).write_bytes(value)
    (root / name).chmod(0o400)
    for folder in (root, root / "mediapipe", root / "mediapipe/python"):
        folder.chmod(0o500)
    document = {
        "schema": "umi-mediapipe-cpu-overlay/1",
        "platform": "macos/arm64",
        "python_abi": "cp310",
        "mediapipe_version": "0.10.14",
        "files": [
            {"path": name, "sha256": hashlib.sha256(value).hexdigest(), "size_bytes": len(value)}
        ],
    }
    digest = publish(root, document)
    try:
        yield root, document, digest
    finally:
        for folder in (root, root / "mediapipe", root / "mediapipe/python"):
            folder.chmod(0o700)


def check(verifier, overlay):
    root, _, digest = overlay
    return verifier.verify_overlay(root, expected_manifest_sha256=digest)


def test_read_only_verification_does_not_execute_fixture(verifier, overlay):
    result = check(verifier, overlay)
    assert result == {"manifest_sha256": overlay[2], "files": 1, "bytes": 29}
    assert check(verifier, overlay) == result


def test_digest_must_be_supplied_outside_overlay(verifier, overlay):
    with pytest.raises(ValueError, match="checksum"):
        verifier.verify_overlay(overlay[0], expected_manifest_sha256="0" * 64)


def test_overlay_root_alias_is_rejected(verifier, overlay, tmp_path):
    alias = tmp_path / "overlay-alias"
    alias.symlink_to(overlay[0], target_is_directory=True)
    with pytest.raises(ValueError, match="without links"):
        verifier.verify_overlay(alias, expected_manifest_sha256=overlay[2])


def test_path_replacement_while_hashing_is_rejected(verifier, overlay, tmp_path, monkeypatch):
    root = overlay[0]
    manifest = root / "overlay.json"
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(manifest.read_bytes())
    replacement.chmod(0o400)
    real_fstat = verifier.os.fstat
    calls = 0

    def replace_after_second_stat(descriptor):
        nonlocal calls
        result = real_fstat(descriptor)
        calls += 1
        if calls == 2:
            root.chmod(0o700)
            replacement.replace(manifest)
            root.chmod(0o500)
        return result

    monkeypatch.setattr(verifier.os, "fstat", replace_after_second_stat)
    with pytest.raises(ValueError, match="changed during verification"):
        check(verifier, overlay)


@pytest.mark.parametrize(
    "mutation",
    [
        "contents",
        "missing",
        "extra",
        "symlink",
        "hardlink",
        "fifo",
        "writable_file",
        "writable_directory",
    ],
)
def test_refuses_modified_tree(verifier, overlay, tmp_path, mutation):
    root, _, _ = overlay
    file = root / "mediapipe/python/placeholder.so"
    file.parent.chmod(0o700)
    if mutation == "contents":
        file.chmod(0o600)
        file.write_bytes(b"modified executable")
        file.chmod(0o400)
    elif mutation == "missing":
        file.unlink()
    elif mutation == "extra":
        extra = file.parent / "extra.py"
        extra.write_bytes(b"no extra executable allowed")
        extra.chmod(0o400)
    elif mutation == "symlink":
        file.rename(tmp_path / "outside")
        file.symlink_to(tmp_path / "outside")
    elif mutation == "hardlink":
        os.link(file, tmp_path / "outside")
    elif mutation == "fifo":
        file.unlink()
        os.mkfifo(file, mode=0o400)
    elif mutation == "writable_file":
        file.chmod(0o600)
    if mutation != "writable_directory":
        file.parent.chmod(0o500)
    with pytest.raises(ValueError):
        check(verifier, overlay)


@pytest.mark.parametrize(
    "name",
    [
        "../outside",
        "/outside",
        "mediapipe/../outside",
        "mediapipe//file",
        "other/file",
        "mediapipe",
    ],
)
def test_rejects_non_package_paths(verifier, overlay, name):
    root, document, _ = overlay
    document["files"][0]["path"] = name
    digest = publish(root, document)
    with pytest.raises(ValueError, match="path"):
        verifier.verify_overlay(root, expected_manifest_sha256=digest)


def test_rejects_duplicate_record(verifier, overlay):
    root, document, _ = overlay
    document["files"].append(document["files"][0])
    digest = publish(root, document)
    with pytest.raises(ValueError, match="duplicate"):
        verifier.verify_overlay(root, expected_manifest_sha256=digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("platform", "linux/arm64"),
        ("python_abi", "cp312"),
        ("mediapipe_version", "other"),
    ],
)
def test_rejects_different_runtime_contract(verifier, overlay, field, value):
    root, document, _ = overlay
    document[field] = value
    digest = publish(root, document)
    with pytest.raises(ValueError, match="unsupported"):
        verifier.verify_overlay(root, expected_manifest_sha256=digest)


@pytest.mark.parametrize(
    "name",
    [
        "/tmp/libopencv_core.3.4.dylib",
        "@rpath/unreviewed.dylib",
        "@executable_path/libopencv_core.3.4.dylib",
        "@rpath/../libopencv_core.3.4.dylib",
    ],
)
def test_packager_rejects_unreviewed_dependency(name):
    with pytest.raises(ValueError, match="dependency"):
        load("stage_macos_overlay").private_dependencies([name], relocated=False)


def test_packager_accepts_system_and_only_private_opencv_dependencies():
    packager = load("stage_macos_overlay")
    names = [
        "/usr/lib/libc++.1.dylib",
        "/System/Library/Frameworks/Cocoa.framework/Cocoa",
        "@rpath/libopencv_core.3.4.dylib",
    ]
    assert packager.private_dependencies(names, relocated=False) == [names[-1]]
    with pytest.raises(ValueError):
        packager.private_dependencies(names, relocated=True)
    names[-1] = "@loader_path/libopencv_core.3.4.dylib"
    assert packager.private_dependencies(names, relocated=True) == [names[-1]]


def test_packager_inputs_bind_live_overlay_and_every_native_input():
    packager = load("stage_macos_overlay")
    document = packager.build_inputs()
    expected = {f"mediapipe/python/{packager.EXTENSION}"} | {
        f"mediapipe/python/{name}" for name in packager.LIBRARIES
    }
    assert set(document["unrelocated_native_sha256"]) == expected
    assert document["overlay_manifest_sha256"] == (
        "a9aaeb93f0b28f7823448212ecefb1c4f6375da0e74d62b494ed50e39677a312"
    )
    assert document["installed_mediapipe_inventory"]["file_count"] == 581
    assert document["installed_mediapipe_inventory"]["content_bytes"] == 120_539_339


def test_installed_package_identity_covers_files_but_not_ignored_caches(tmp_path):
    packager = load("stage_macos_overlay")
    package = tmp_path / "mediapipe"
    python = package / "python"
    cache = python / "__pycache__"
    cache.mkdir(parents=True)
    (python / packager.EXTENSION).write_bytes(b"original installed binding")
    module = package / "calculators.py"
    module.write_bytes(b"reviewed source")
    ignored = cache / "calculators.pyc"
    ignored.write_bytes(b"host-local cache")
    first = packager.installed_package_identity(package)
    assert first["file_count"] == 2
    assert first["content_bytes"] == len(b"original installed bindingreviewed source")
    ignored.write_bytes(b"different cache bytes")
    assert packager.installed_package_identity(package) == first
    module.write_bytes(b"changed source")
    assert packager.installed_package_identity(package)["sha256"] != first["sha256"]
