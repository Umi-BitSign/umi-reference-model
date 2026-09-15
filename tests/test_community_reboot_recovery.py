"""Recovery state machine with simulated boot identities and inert artifacts."""

from __future__ import annotations

import importlib
import json
import os
import socket
import tempfile
from pathlib import Path

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
