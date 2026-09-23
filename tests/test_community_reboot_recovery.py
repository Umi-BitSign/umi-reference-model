"""Recovery state machine with simulated boot identities and inert artifacts."""

from __future__ import annotations

import importlib
import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIRST = {"platform": "darwin", "host": "11" * 16, "boot": "22" * 16}
SECOND = {**FIRST, "boot": "33" * 16}
SHA = "aa" * 32


@pytest.fixture
def private_root():
    # Darwin's default pytest path exceeds the Unix socket pathname limit.
    with tempfile.TemporaryDirectory(prefix="umi-reboot-", dir="/tmp") as directory:
        yield Path(directory).resolve()


@pytest.fixture
def deployment(private_root, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "community"))
    recovery = importlib.import_module("reboot_recovery")
    root = private_root
    root.chmod(0o700)
    scratch = root / "scratch"
    scratch.mkdir(mode=0o700)
    document = {"workers": [{"scratch_directory": str(scratch)}]}
    endpoint = root / "model.sock"
    config = root / "config.json"
    config.write_bytes(b"retained configuration")
    config.chmod(0o600)
    monkeypatch.setattr(recovery, "kernel_identity", lambda: dict(FIRST))
    monkeypatch.setattr(recovery, "darwin_volume_uuid", lambda _path: "55" * 16)

    def instance(sha=SHA):
        return recovery.RebootRecovery(endpoint, config, sha, document)

    initial = instance()
    initial.prepare()
    initial.begin()
    return recovery, root, endpoint, scratch, instance


def leave_artifacts(endpoint, scratch):
    with socket.socket(socket.AF_UNIX) as channel:
        channel.bind(str(endpoint))
        endpoint.chmod(0o600)
    capacity = Path(f"{endpoint}.capacity.json")
    capacity.write_bytes(b"retained capacity")
    capacity.chmod(0o600)
    (scratch / "clip.bin").write_bytes(b"retained video")


def test_kernel_identity_is_available_and_stable(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "community"))
    module = importlib.import_module("reboot_recovery")
    first = module.kernel_identity()
    module.validate_identity(first)
    assert module.kernel_identity() == first


def test_same_boot_active_is_rejected_even_without_a_socket(deployment):
    _module, _root, endpoint, scratch, instance = deployment
    assert not endpoint.exists() and not list(scratch.iterdir())
    with pytest.raises(RuntimeError, match="current boot"):
        instance().prepare()


def test_changed_boot_quarantines_without_following_links(deployment, monkeypatch):
    module, root, endpoint, scratch, instance = deployment
    leave_artifacts(endpoint, scratch)
    retained = root / "nonce.sqlite"
    retained.write_bytes(b"never touch HTTP state")
    (scratch / "outside-link").symlink_to(retained)
    old_inode = scratch.stat().st_ino
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    current = instance()
    current.prepare()
    assert not endpoint.exists() and not Path(f"{endpoint}.capacity.json").exists()
    assert not list(scratch.iterdir()) and scratch.stat().st_ino != old_inode
    moved = [p for p in root.iterdir() if p.name.startswith(".umi-reboot-")]
    assert len(moved) == 3
    saved_scratch = next(p for p in moved if p.is_dir())
    assert (saved_scratch / "clip.bin").read_bytes() == b"retained video"
    assert (saved_scratch / "outside-link").is_symlink()
    assert retained.read_bytes() == b"never touch HTTP state"
    # Re-running a completed plan before begin is idempotent.
    instance().prepare()
    current.begin()
    assert current.state["identity"] == SECOND and current.state["plan"] is None
    with pytest.raises(RuntimeError, match="current boot"):
        instance().prepare()
    current.finish()
    assert instance().state["phase"] == "clean"
    instance().prepare()


@pytest.mark.parametrize("operation", ["rename", "mkdir"])
def test_interrupted_quarantine_resumes_its_durable_plan(deployment, monkeypatch, operation):
    module, root, endpoint, scratch, instance = deployment
    leave_artifacts(endpoint, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    if operation == "rename":
        original = module.os.rename
        calls = 0

        def fail(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated interrupted recovery")
            return original(source, target)

        monkeypatch.setattr(module.os, "rename", fail)
    else:
        original = Path.mkdir

        def fail(path, *args, **kwargs):
            if path == scratch:
                raise OSError("simulated interrupted recovery")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail)
    with pytest.raises(OSError, match="interrupted"):
        instance().prepare()
    if operation == "rename":
        monkeypatch.setattr(module.os, "rename", original)
    else:
        monkeypatch.setattr(Path, "mkdir", original)
    resumed = instance()
    resumed.prepare()
    resumed.begin()
    assert not list(scratch.iterdir())
    assert len(list(root.glob(".umi-reboot-*"))) == 3


@pytest.mark.parametrize(
    "change", ["boot_unavailable", "host", "config", "scratch", "socket", "capacity", "journal"]
)
def test_ambiguous_state_preserves_every_artifact(deployment, monkeypatch, change):
    module, root, endpoint, scratch, instance = deployment
    leave_artifacts(endpoint, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    sha = SHA
    if change == "boot_unavailable":
        monkeypatch.setattr(module, "kernel_identity", lambda: {**SECOND, "boot": ""})
    elif change == "host":
        monkeypatch.setattr(module, "kernel_identity", lambda: {**SECOND, "host": "44" * 16})
    elif change == "config":
        sha = "bb" * 32
    elif change == "scratch":
        scratch.rename(root / "original-scratch")
        scratch.mkdir(mode=0o700)
    elif change == "socket":
        endpoint.unlink()
        endpoint.symlink_to(root / "missing")
    elif change == "capacity":
        os.link(Path(f"{endpoint}.capacity.json"), root / "hardlink")
    else:
        Path(f"{endpoint}.reboot.json").chmod(0o644)
    before = sorted(p.name for p in root.iterdir())
    with pytest.raises((ValueError, OSError)):
        instance(sha).prepare()
    assert sorted(p.name for p in root.iterdir()) == before
    assert not list(root.glob(".umi-reboot-*"))
    assert Path(f"{endpoint}.capacity.json").read_bytes() == b"retained capacity"


def test_retained_files_without_journal_are_not_claimed(deployment):
    _module, _root, endpoint, scratch, instance = deployment
    Path(f"{endpoint}.reboot.json").unlink()
    leave_artifacts(endpoint, scratch)
    with pytest.raises(FileExistsError):
        instance().prepare()
    assert (scratch / "clip.bin").read_bytes() == b"retained video"


def test_corrupt_plan_cannot_redirect_quarantine(deployment, monkeypatch):
    module, root, endpoint, scratch, instance = deployment
    leave_artifacts(endpoint, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    pending = instance()
    pending._write({**pending.state, "plan": pending._entries()})
    journal = Path(f"{endpoint}.reboot.json")
    document = json.loads(journal.read_bytes())
    document["plan"][0]["target"] = str(root / "unrelated")
    journal.write_bytes(module.canonical(document))
    with pytest.raises(ValueError, match="binding"):
        instance().prepare()
    assert endpoint.exists() and (scratch / "clip.bin").exists()
    assert not (root / "unrelated").exists()


def test_recovery_journal_cannot_be_bypassed_by_default_startup(deployment):
    _module, _root, endpoint, _scratch, _instance = deployment
    service = importlib.import_module("service")
    with pytest.raises(FileExistsError), service.service_lock(endpoint):
        raise AssertionError("recovery journal bypassed")


def change_device_number(monkeypatch, scratch):
    original = Path.lstat

    def changed(path, *args, **kwargs):
        info = original(path, *args, **kwargs)
        if path == scratch:
            values = {name: getattr(info, name) for name in dir(info) if name.startswith("st_")}
            return SimpleNamespace(**{**values, "st_dev": info.st_dev + 1})
        return info

    monkeypatch.setattr(Path, "lstat", changed)


def test_new_boot_accepts_renumbered_device_on_same_volume(deployment, monkeypatch):
    module, root, endpoint, scratch, instance = deployment
    leave_artifacts(endpoint, scratch)
    change_device_number(monkeypatch, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    current = instance()
    current.prepare()
    assert not list(scratch.iterdir())
    assert (
        next(p for p in root.glob(".umi-reboot-*") if p.is_dir()).joinpath("clip.bin").read_bytes()
        == b"retained video"
    )
    current.begin()
    current.finish()
    assert current.state["schema"] == module.SCHEMA


@pytest.mark.parametrize("phase", ["active", "clean"])
def test_different_volume_does_not_claim_same_inode(deployment, monkeypatch, phase):
    module, root, endpoint, scratch, instance = deployment
    if phase == "clean":
        instance().finish()
    else:
        leave_artifacts(endpoint, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    monkeypatch.setattr(module, "darwin_volume_uuid", lambda _path: "66" * 16)
    before = sorted(p.name for p in root.iterdir())
    with pytest.raises(ValueError):
        instance().prepare()
    assert sorted(p.name for p in root.iterdir()) == before


@pytest.mark.parametrize("phase", ["active", "clean"])
def test_legacy_journal_keeps_device_checks_and_migrates_on_begin(deployment, monkeypatch, phase):
    module, _root, endpoint, scratch, instance = deployment
    legacy = instance()
    legacy._write(
        {
            **legacy.state,
            "schema": module.LEGACY_SCHEMA,
            "phase": phase,
            "scratches": [
                {"path": str(scratch), "metadata": module.metadata(scratch, "directory")}
            ],
        }
    )
    if phase == "active":
        leave_artifacts(endpoint, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    current = instance()
    current.prepare()
    current.begin()
    assert current.state["schema"] == module.SCHEMA
    assert current.state["scratches"][0]["metadata"][0] == "55" * 16


def test_legacy_journal_cannot_invent_old_volume_identity(deployment, monkeypatch):
    module, root, endpoint, scratch, instance = deployment
    legacy = instance()
    legacy._write(
        {
            **legacy.state,
            "schema": module.LEGACY_SCHEMA,
            "scratches": [
                {"path": str(scratch), "metadata": module.metadata(scratch, "directory")}
            ],
        }
    )
    leave_artifacts(endpoint, scratch)
    change_device_number(monkeypatch, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    with pytest.raises(ValueError, match="identity changed"):
        instance().prepare()
    assert not list(root.glob(".umi-reboot-*"))
    assert (scratch / "clip.bin").read_bytes() == b"retained video"


def test_second_reboot_resumes_partly_quarantined_plan(deployment, monkeypatch):
    module, root, endpoint, scratch, instance = deployment
    leave_artifacts(endpoint, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    pending = instance()
    pending._write({**pending.state, "plan": pending._entries()})
    pending._apply(pending.state["plan"][0], execute=True)
    change_device_number(monkeypatch, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: {**SECOND, "boot": "77" * 16})
    resumed = instance()
    resumed.prepare()
    resumed.begin()
    assert len(list(root.glob(".umi-reboot-*"))) == 3
    assert not list(scratch.iterdir())


@pytest.mark.parametrize("identity", [None, "", "00" * 16, "55" * 15, 1, True])
def test_invalid_volume_identity_never_moves_artifacts(deployment, monkeypatch, identity):
    module, root, endpoint, scratch, instance = deployment
    leave_artifacts(endpoint, scratch)
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    monkeypatch.setattr(module, "darwin_volume_uuid", lambda _path: identity)
    with pytest.raises(ValueError):
        instance().prepare()
    assert not list(root.glob(".umi-reboot-*"))
    assert (scratch / "clip.bin").read_bytes() == b"retained video"


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin kernel API")
def test_native_volume_uuid_survives_directory_rename(private_root, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "community"))
    module = importlib.import_module("reboot_recovery")
    old = private_root / "before"
    old.mkdir(mode=0o700)
    identity = module.metadata(old, "directory", volume=True)
    assert identity[0] == module.darwin_volume_uuid(private_root)
    new = private_root / "after"
    old.rename(new)
    assert module.metadata(new, "directory", volume=True) == identity


def test_linux_journal_does_not_require_darwin_api(deployment, monkeypatch):
    module, _root, endpoint, scratch, _instance = deployment
    Path(f"{endpoint}.reboot.json").unlink()
    monkeypatch.setattr(module, "kernel_identity", lambda: {**FIRST, "platform": "linux"})

    def no_darwin(_path):
        raise AssertionError("Linux must not query Darwin")

    monkeypatch.setattr(module, "darwin_volume_uuid", no_darwin)
    current = module.RebootRecovery(
        endpoint,
        scratch.parent / "config.json",
        SHA,
        {"workers": [{"scratch_directory": str(scratch)}]},
    )
    current.prepare()
    current.begin()
    current.finish()
    assert current.state["scratches"][0]["metadata"] == [
        scratch.stat().st_dev,
        scratch.stat().st_ino,
    ]


@pytest.mark.parametrize("boundary", ["capacity_after_rename", "scratch_after_mkdir"])
def test_retry_repeats_failed_parent_durability_barrier(private_root, monkeypatch, boundary):
    monkeypatch.syspath_prepend(str(ROOT / "community"))
    module = importlib.import_module("reboot_recovery")
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(FIRST))
    monkeypatch.setattr(module, "darwin_volume_uuid", lambda _path: "55" * 16)
    sockets, models = private_root / "sockets", private_root / "models"
    sockets.mkdir(mode=0o700)
    models.mkdir(mode=0o700)
    scratch = models / "scratch"
    scratch.mkdir(mode=0o700)
    endpoint = sockets / "model.sock"
    config = private_root / "config.json"
    config.write_bytes(b"retained configuration")
    config.chmod(0o600)
    document = {"workers": [{"scratch_directory": str(scratch)}]}

    def instance():
        return module.RebootRecovery(endpoint, config, SHA, document)

    instance().begin()
    leave_artifacts(endpoint, scratch)
    old_scratch_inode = scratch.stat().st_ino
    monkeypatch.setattr(module, "kernel_identity", lambda: dict(SECOND))
    expected_parent = sockets if boundary == "capacity_after_rename" else models
    original = module.sync_directory
    failed = False

    def fail_once(path):
        nonlocal failed
        at_boundary = (
            not Path(f"{endpoint}.capacity.json").exists()
            if boundary == "capacity_after_rename"
            else scratch.exists() and scratch.stat().st_ino != old_scratch_inode
        )
        if path == expected_parent and at_boundary and not failed:
            failed = True
            raise OSError("parent sync interrupted")
        return original(path)

    monkeypatch.setattr(module, "sync_directory", fail_once)
    with pytest.raises(OSError, match="parent sync interrupted"):
        instance().prepare()
    assert failed
    synced = []

    def recording(path):
        original(path)
        synced.append(path)

    monkeypatch.setattr(module, "sync_directory", recording)
    recovered = instance()
    recovered.prepare()
    assert expected_parent in synced
    assert not list(scratch.iterdir())
    assert next(models.glob(".umi-reboot-*")).joinpath("clip.bin").read_bytes() == b"retained video"
    recovered.begin()
    recovered.finish()
