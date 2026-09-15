"""Service configuration, exclusive startup and real-process restart checks.

Workers are inert. These tests do not establish model capacity or accuracy.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVISION = "ab" * 32
POLICY = "cd" * 32


@pytest.fixture
def service(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "community"))
    spec = importlib.util.spec_from_file_location(
        "community_service_test", ROOT / "community/service.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def configuration(service):
    with tempfile.TemporaryDirectory(prefix="umi-service-", dir="/tmp") as directory:
        root = Path(directory).resolve(strict=True)
        scratch = root / "scratch"
        scratch.mkdir(mode=0o700)
        document = {
            "schema": "umi-community-model-service/1",
            "socket_path": str(root / "model.sock"),
            "scoring_policy_sha256": POLICY,
            "validator_slot_count": 1,
            "workers": [
                {
                    "command": [sys.executable, "-c", "raise AssertionError('not a real worker')"],
                    "environment": {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                    "cwd": str(root),
                    "model_revision": REVISION,
                    "startup_seconds": 5,
                    "inference_seconds": 2,
                    "scratch_directory": str(scratch),
                }
            ],
        }
        path = root / "config.json"
        checksum = save(service, path, document)
        yield root, path, document, checksum


def save(service, path, document):
    raw = service.canonical(document)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def test_reviewed_configuration_roundtrip_no_worker_start(service, configuration, monkeypatch):
    _root, path, document, checksum = configuration
    calls = []
    monkeypatch.setattr(service, "run_sidecar_service", lambda *a, **kw: calls.append((a, kw)))
    assert service.read_configuration(path, checksum) == document
    service.run_configuration(path, checksum)
    (((sidecar, socket), arguments),) = calls
    assert sidecar.model_revision == REVISION
    assert sidecar.validator_slot_count == 1
    assert sidecar.workers[0]._process is None
    assert socket == Path(document["socket_path"])
    assert arguments == {"scoring_policy_sha256": POLICY}


@pytest.mark.parametrize("kind", ["policy", "slots", "workers", "boolean", "live_null"])
def test_standby_cannot_claim_a_policy_or_extra_workers(service, configuration, kind):
    _root, path, document, _checksum = configuration
    document["schema"] = "umi-community-model-standby/1"
    document["scoring_policy_sha256"] = None
    if kind == "policy":
        document["scoring_policy_sha256"] = POLICY
    elif kind == "slots":
        document["validator_slot_count"] = 2
    elif kind == "workers":
        document["workers"].append(dict(document["workers"][0]))
    elif kind == "boolean":
        document["validator_slot_count"] = True
    else:
        document["schema"] = "umi-community-model-service/1"
    checksum = save(service, path, document)
    with pytest.raises(ValueError):
        service.read_configuration(path, checksum)


@pytest.mark.parametrize(
    "kind",
    [
        "hash",
        "readable",
        "symlink",
        "hardlink",
        "fifo",
        "oversized",
        "extra",
        "noncanonical",
        "wrong_schema",
        "policy",
    ],
)
def test_unsafe_configuration_rejected(service, configuration, kind):
    root, path, document, checksum = configuration
    if kind == "hash":
        checksum = "00" * 32
    elif kind == "readable":
        path.chmod(0o644)
    elif kind == "symlink":
        alias = root / "alias.json"
        alias.symlink_to(path)
        path = alias
    elif kind == "hardlink":
        os.link(path, root / "link.json")
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path, 0o600)
    elif kind == "oversized":
        path.write_bytes(b"x" * (service.MAXIMUM_CONFIG_BYTES + 1))
    elif kind == "extra":
        document["extra"] = True
        checksum = save(service, path, document)
    elif kind == "noncanonical":
        path.write_bytes(path.read_bytes() + b"\n")
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    elif kind == "wrong_schema":
        document["schema"] = "wrong"
        checksum = save(service, path, document)
    elif kind == "policy":
        document["scoring_policy_sha256"] = "invalid"
        checksum = save(service, path, document)
    with pytest.raises((ValueError, OSError)):
        service.read_configuration(path, checksum)


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "extra",
        "shared_scratch",
        "nested_scratch",
        "slots",
        "relative_command",
        "missing_scratch",
    ],
)
def test_worker_configuration_rejected_before_loading(service, configuration, kind):
    root, _path, document, _checksum = configuration
    worker = document["workers"][0]
    if kind == "empty":
        document["workers"] = []
    elif kind == "extra":
        worker["unknown"] = True
    elif kind in {"shared_scratch", "nested_scratch"}:
        second = dict(worker)
        if kind == "nested_scratch":
            nested = root / "scratch/child"
            nested.mkdir(mode=0o700)
            # The first worker must refuse its nonempty scratch too.
            second["scratch_directory"] = str(nested)
        document["workers"].append(second)
    elif kind == "slots":
        document["validator_slot_count"] = 2
    elif kind == "relative_command":
        worker["command"][0] = "python"
    elif kind == "missing_scratch":
        worker["scratch_directory"] = None
    with pytest.raises(ValueError):
        service.build_sidecar(document)


@pytest.mark.parametrize(
    "kind",
    ["symlink", "hardlink", "readable", "directory", "fifo", "socket_exists", "capacity_exists"],
)
def test_unsafe_or_occupied_service_lock_preserves_files(service, configuration, kind):
    root, _path, document, _checksum = configuration
    socket = Path(document["socket_path"])
    lock = Path(f"{socket}.service.lock")
    target = root / "retained"
    target.write_bytes(b"retained")
    target.chmod(0o600)
    if kind == "symlink":
        lock.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, lock)
    elif kind == "readable":
        lock.write_bytes(b"")
        lock.chmod(0o644)
    elif kind == "directory":
        lock.mkdir(mode=0o700)
    elif kind == "fifo":
        os.mkfifo(lock, 0o600)
    elif kind == "socket_exists":
        socket.symlink_to(root / "missing")
    else:
        Path(f"{socket}.capacity.json").write_bytes(b"retained")
    with pytest.raises((ValueError, OSError)), service.service_lock(socket):
        raise AssertionError("unsafe service lock accepted")
    assert target.read_bytes() == b"retained"


def test_lock_is_exclusive_and_inode_retained(service, configuration):
    _root, _path, document, _checksum = configuration
    socket = Path(document["socket_path"])
    lock = Path(f"{socket}.service.lock")
    with service.service_lock(socket):
        identity = lock.stat().st_ino
        with pytest.raises(BlockingIOError), service.service_lock(socket):
            raise AssertionError("duplicate service acquired the lock")
    assert lock.stat().st_ino == identity
    with service.service_lock(socket):
        assert lock.stat().st_ino == identity


@pytest.mark.parametrize(
    "stage", ["configuration", "exclusive_startup", "worker_configuration", "runtime"]
)
def test_startup_failure_identifies_stage_without_input_values(
    service, configuration, monkeypatch, stage
):
    _root, path, document, checksum = configuration
    if stage == "configuration":
        checksum = "00" * 32
    elif stage == "exclusive_startup":
        Path(document["socket_path"]).write_text("retained")
    elif stage == "worker_configuration":
        document["validator_slot_count"] = 0
        checksum = save(service, path, document)
    else:

        def fail(*args, **kwargs):
            raise RuntimeError("private model output")

        monkeypatch.setattr(service, "run_sidecar_service", fail)
    with pytest.raises(service.ModelServiceError) as failure:
        service.run_configuration(path, checksum)
    suffix = "invalid" if stage in {"configuration", "worker_configuration"} else "failed"
    assert failure.value.reason_code == f"model_service_{stage}_{suffix}"
    assert str(failure.value) == failure.value.reason_code


@pytest.mark.parametrize("shutdown_signal", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("standby", [False, True])
def test_real_service_request_duplicate_rejection_and_restart(
    service, configuration, shutdown_signal, standby
):
    api = pytest.importorskip("umi.model_sidecar", reason="requires reviewed UMI socket helper")
    root, path, document, _checksum = configuration
    if standby:
        document["schema"] = "umi-community-model-standby/1"
        document["scoring_policy_sha256"] = None
    worker = root / "worker.py"
    pid_path = root / "worker-pid"
    worker.write_text(
        "import sys, os\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(ROOT / 'community')!r})\n"
        "from worker_transport import serve_worker\n"
        f"Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "serve_worker(lambda video, deadline: video.decode(), "
        f"verified_model_revision={REVISION!r})\n"
    )
    document["workers"][0]["command"] = [sys.executable, str(worker)]
    checksum = save(service, path, document)
    command = [
        sys.executable,
        str(ROOT / "community/service.py"),
        "--config",
        str(path),
        "--expected-config-sha256",
        checksum,
    ]
    environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(Path(api.__file__).resolve().parents[1]),
    }
    socket = Path(document["socket_path"])

    async def request():
        video = b"exact input"
        metadata = {
            "video": {
                "sha256": hashlib.sha256(video).hexdigest(),
                "size_bytes": len(video),
                "url": "https://not-fetched.invalid/video.mp4",
            },
            "task": {"source_language": "ase", "target_language": "en", "stratum": "continuous"},
        }
        raw = service.canonical(metadata)
        request_hash = hashlib.sha256(b"umi-request-v1\0" + raw).digest()
        reader, writer = await asyncio.open_unix_connection(socket)
        try:
            writer.write(
                api.MODEL_REQUEST_MAGIC
                + len(raw).to_bytes(4, "big")
                + len(video).to_bytes(8, "big")
                + request_hash
                + bytes.fromhex(REVISION)
                + raw
                + video
            )
            await writer.drain()
            prefix = await reader.readexactly(api.MODEL_RESPONSE_PREFIX_BYTES)
            assert prefix[:-4] == api.MODEL_RESPONSE_MAGIC + request_hash + bytes.fromhex(REVISION)
            length = int.from_bytes(prefix[-4:], "big")
            assert 0 < length <= 4096
            assert await reader.readexactly(length) == video
        finally:
            writer.close()
            await writer.wait_closed()

    for _ in range(2):
        process = subprocess.Popen(
            command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        worker_pid = None
        try:
            deadline = time.monotonic() + 10
            while not Path(f"{socket}.capacity.json").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise AssertionError("service did not start")
                time.sleep(0.01)
            worker_pid = int(pid_path.read_text())
            if standby:
                descriptor = json.loads(Path(f"{socket}.capacity.json").read_bytes())
                assert descriptor["scoring_policy_sha256"] is None
                with pytest.raises(RuntimeError, match="binding does not match"):
                    api.validate_model_sidecar_capacity(
                        socket,
                        expected_model_revision=REVISION,
                        expected_scoring_policy_sha256=POLICY,
                        required_validator_slots=1,
                        maximum_inference_seconds=2,
                    )
            asyncio.run(asyncio.wait_for(request(), timeout=5))
            duplicate = subprocess.run(command, env=environment, capture_output=True, timeout=5)
            assert duplicate.returncode == 1
            assert duplicate.stdout == b""
            assert json.loads(duplicate.stderr)["status"] == "blocked"
            assert (
                json.loads(duplicate.stderr)["reason_code"]
                == "model_service_exclusive_startup_failed"
            )
            assert str(root).encode() not in duplicate.stderr
            assert int(pid_path.read_text()) == worker_pid
            process.send_signal(shutdown_signal)
            stdout, stderr = process.communicate(timeout=10)
            assert process.returncode == 0, stderr
            assert not stdout and not stderr
            with pytest.raises(ProcessLookupError):
                os.kill(worker_pid, 0)
            assert not socket.exists() and not Path(f"{socket}.capacity.json").exists()
            assert not list((root / "scratch").iterdir())
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=5)
            if worker_pid is not None:
                try:
                    os.killpg(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
