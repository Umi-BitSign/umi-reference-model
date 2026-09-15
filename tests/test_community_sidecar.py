"""Cross-repository socket tests; require UMI's dependency-light model helper.

Run with the reviewed UMI source on PYTHONPATH (or its wheel installed). Tests
which use the socket explicitly skip if UMI is absent; no model is imported.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import signal
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REVISION = "ab" * 32
POLICY = "cd" * 32
TASK = {"source_language": "ase", "target_language": "en", "stratum": "short_utterance"}


def load(relative):
    spec = importlib.util.spec_from_file_location("community_sidecar_test", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def setup(tmp_path):
    api = pytest.importorskip("umi.model_sidecar", reason="requires reviewed UMI socket helper")
    transport = load("community/worker_transport.py")
    sidecar = load("community/sidecar.py")
    script = tmp_path / "worker.py"
    transport_path = str(ROOT / "community/worker_transport.py")
    script.write_text(
        "import importlib.util, os, time\n"
        f"spec = importlib.util.spec_from_file_location('transport', {transport_path!r})\n"
        "transport = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(transport)\n"
        "def translate(video, deadline_ns):\n"
        "    if video == b'hang': time.sleep(30)\n"
        "    if video == b'crash': os._exit(7)\n"
        "    if video == b'print': os.write(1, b'native stdout diagnostic\\n')\n"
        "    return str(os.getpid()) + ':' + video.decode()\n"
        f"transport.serve_worker(translate, verified_model_revision={REVISION!r})\n"
    )
    workers = tuple(
        transport.WarmModelProcess(
            [sys.executable, str(script)],
            environment={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            cwd=tmp_path,
            model_revision=REVISION,
            startup_seconds=5,
            inference_seconds=2,
        )
        for _ in range(2)
    )
    # Keep the Unix socket below Darwin's pathname limit.
    with tempfile.TemporaryDirectory(prefix="umi-cs-", dir="/tmp") as socket_root:
        yield (
            api,
            sidecar.CommunityModelSidecar(workers, validator_slot_count=2),
            Path(socket_root) / "m.sock",
        )


def absent(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


async def until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(wait(), timeout=5)


def metadata(video, *, task=None):
    return {
        "task": dict(TASK) if task is None else task,
        "video": {
            "url": "https://not-fetched.invalid/video.mp4",
            "size_bytes": len(video),
            "sha256": hashlib.sha256(video).hexdigest(),
        },
    }


async def request(api, path, video, *, revision=REVISION, task=None):
    raw = json.dumps(metadata(video, task=task), sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(b"umi-request-v1\0" + raw).digest()
    reader, writer = await asyncio.open_unix_connection(path)
    try:
        writer.write(
            api.MODEL_REQUEST_MAGIC
            + len(raw).to_bytes(4, "big")
            + len(video).to_bytes(8, "big")
            + digest
            + bytes.fromhex(revision)
            + raw
            + video
        )
        await writer.drain()
        prefix = await reader.readexactly(api.MODEL_RESPONSE_PREFIX_BYTES)
        assert prefix == api.MODEL_RESPONSE_MAGIC + digest + bytes.fromhex(revision) + prefix[-4:]
        size = int.from_bytes(prefix[-4:], "big")
        assert 0 < size <= 4096
        return (await reader.readexactly(size)).decode()
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.parametrize("count,slots", [(0, 1), (1, 2), (3, 2), (2, 0), (2, True), (2, 257)])
def test_invalid_capacity(count, slots):
    module = load("community/sidecar.py")
    workers = [
        types.SimpleNamespace(model_revision=REVISION, inference_seconds=2) for _ in range(count)
    ]
    with pytest.raises(ValueError, match="distinct workers"):
        module.CommunityModelSidecar(workers, validator_slot_count=slots)


@pytest.mark.parametrize("mutation", ["duplicate", "revision", "deadline"])
def test_no_alias_or_mixed_worker_binding(mutation):
    module = load("community/sidecar.py")
    workers = [
        types.SimpleNamespace(model_revision=REVISION, inference_seconds=2) for _ in range(2)
    ]
    if mutation == "duplicate":
        workers[1] = workers[0]
    elif mutation == "revision":
        workers[1].model_revision = "ff" * 32
    else:
        workers[1].inference_seconds = 1
    with pytest.raises(ValueError):
        module.CommunityModelSidecar(workers, validator_slot_count=2)


def test_socket_returns_bound_text_and_exact_capacity(setup):
    api, sidecar, path = setup

    async def run():
        await sidecar.start(path, scoring_policy_sha256=POLICY)
        pids = {worker._process.pid for worker in sidecar.workers}
        assert len(pids) == 2
        try:
            first = await request(api, path, b"print")
            second = await request(api, path, b"second")
            assert first.endswith(":print") and second.endswith(":second")
            assert {int(first.split(":")[0]), int(second.split(":")[0])} == pids
            assert path.stat().st_mode & 0o777 == 0o600
            capacity = json.loads(Path(str(path) + ".capacity.json").read_bytes())
            assert capacity["maximum_concurrency"] == capacity["validator_slot_count"] == 2
            assert capacity["model_revision"] == REVISION
            assert capacity["scoring_policy_sha256"] == POLICY
            assert capacity["maximum_inference_milliseconds"] == 2000
        finally:
            await sidecar.close()
        for pid in pids:
            absent(pid)
        assert not path.exists() and not Path(str(path) + ".capacity.json").exists()
        await sidecar.close()

    asyncio.run(run())


def test_blocked_worker_does_not_block_other_slot_and_disconnect_reaps_it(setup):
    api, sidecar, path = setup

    async def run():
        await sidecar.start(path, scoring_policy_sha256=POLICY)
        first_pid, second_pid = [worker._process.pid for worker in sidecar.workers]
        pending = asyncio.create_task(request(api, path, b"hang"))
        try:
            await until(lambda: bool(sidecar.workers[0]._operations))
            assert await request(api, path, b"available") == f"{second_pid}:available"
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await until(
                lambda: sidecar.workers[0]._process is None
                or sidecar.workers[0]._process.pid != first_pid
            )
            absent(first_pid)
            assert sidecar.workers[1]._process.pid == second_pid
        finally:
            await sidecar.close()
            await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(run())


def test_socket_deadline_reaps_hung_process(setup):
    api, sidecar, path = setup

    async def run():
        await sidecar.start(path, scoring_policy_sha256=POLICY)
        pid = sidecar.workers[0]._process.pid
        try:
            with pytest.raises(asyncio.IncompleteReadError):
                await asyncio.wait_for(request(api, path, b"hang"), timeout=5)
            await until(
                lambda: sidecar.workers[0]._process is None
                or sidecar.workers[0]._process.pid != pid
            )
            absent(pid)
        finally:
            await sidecar.close()

    asyncio.run(run())


def test_close_cancels_queued_and_active_work(setup):
    api, sidecar, path = setup

    async def run():
        await sidecar.start(path, scoring_policy_sha256=POLICY)
        pids = [worker._process.pid for worker in sidecar.workers]
        meta = api.CanonicalModelRequest(canonical_json=b"{}", document=metadata(b"hang"))
        calls = [asyncio.create_task(sidecar.translate(b"hang", meta)) for _ in range(3)]
        await until(lambda: all(worker._operations for worker in sidecar.workers))
        await sidecar.close()
        results = await asyncio.gather(*calls, return_exceptions=True)
        assert all(isinstance(result, asyncio.CancelledError) for result in results)
        for pid in pids:
            absent(pid)
        with pytest.raises(RuntimeError, match="ready"):
            await sidecar.translate(b"test", meta)

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["wrong_revision", "startup_cancel", "occupied_socket"])
def test_startup_failure_reaps_workers_without_overwriting_socket(setup, failure):
    _, sidecar, path = setup

    async def run():
        if failure == "wrong_revision":
            sidecar.workers[1]._revision = bytes.fromhex("ff" * 32)
        elif failure == "startup_cancel":
            sidecar.workers[1].command = (sys.executable, "-c", "import time; time.sleep(30)")
        else:
            path.write_text("keep this file")
        startup = asyncio.create_task(sidecar.start(path, scoring_policy_sha256=POLICY))
        if failure == "startup_cancel":
            await until(lambda: sidecar.workers[1]._process is not None)
            startup.cancel()
        with pytest.raises((RuntimeError, FileExistsError, asyncio.CancelledError)):
            await startup
        assert all(worker._process is None and worker._closed for worker in sidecar.workers)
        if failure == "occupied_socket":
            assert path.read_text() == "keep this file"
        else:
            assert not path.exists()

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad_task",
    [
        {},
        {**TASK, "source_language": "bfi"},
        {**TASK, "target_language": "fr"},
        {**TASK, "stratum": "other"},
    ],
)
def test_unsupported_tasks_do_not_reach_workers(setup, bad_task):
    api, sidecar, path = setup

    async def run():
        await sidecar.start(path, scoring_policy_sha256=POLICY)
        try:
            with pytest.raises(asyncio.IncompleteReadError):
                await request(api, path, b"unused", task=bad_task)
            assert all(not worker._operations for worker in sidecar.workers)
        finally:
            await sidecar.close()

    asyncio.run(run())


def test_socket_cleanup_error_still_reaps_workers(setup, monkeypatch):
    _, sidecar, path = setup

    async def run():
        await sidecar.start(path, scoring_policy_sha256=POLICY)
        pids = [worker._process.pid for worker in sidecar.workers]
        real_close = sidecar._server.close

        async def fail():
            await real_close()
            raise OSError("injected socket cleanup failure")

        monkeypatch.setattr(sidecar._server, "close", fail)
        with pytest.raises(RuntimeError, match="socket cleanup"):
            await sidecar.close()
        for pid in pids:
            absent(pid)

    asyncio.run(run())


def test_worker_cleanup_uncertainty_closes_whole_sidecar(setup, monkeypatch):
    api, sidecar, path = setup

    async def run():
        await sidecar.start(path, scoring_policy_sha256=POLICY)
        pids = [worker._process.pid for worker in sidecar.workers]
        worker = sidecar.workers[0]
        real_discard = worker._discard

        async def uncertain():
            if worker._process is None:
                return
            # Reap the test process before injecting the terminal cleanup
            # state. Actual signal-denial behavior has its own transport test.
            await real_discard()
            worker._closed = True
            raise RuntimeError("injected cleanup uncertainty")

        monkeypatch.setattr(worker, "_discard", uncertain)
        try:
            with pytest.raises(asyncio.IncompleteReadError):
                await request(api, path, b"crash")
            await until(lambda: sidecar._closed)
        finally:
            await sidecar.close()
        assert all(worker.closed for worker in sidecar.workers)
        assert not path.exists()
        for pid in pids:
            absent(pid)

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["before_start", "startup", "serving", "inference"])
def test_service_stop_reaps_all_workers_and_removes_socket(setup, phase):
    api, sidecar, path = setup

    async def run():
        stop = asyncio.Event()
        if phase == "before_start":
            stop.set()
        elif phase == "startup":
            sidecar.workers[1].command = (sys.executable, "-c", "import time; time.sleep(30)")
        service = asyncio.create_task(
            sidecar.serve_until_stopped(path, scoring_policy_sha256=POLICY, stop=stop)
        )
        calls = []
        pids = []
        try:
            if phase == "startup":
                await until(lambda: sidecar.workers[1]._process is not None)
            elif phase in ("serving", "inference"):
                await until(lambda: sidecar._server is not None)
                if phase == "inference":
                    calls = [asyncio.create_task(request(api, path, b"hang")) for _ in range(2)]
                    await until(lambda: all(worker._operations for worker in sidecar.workers))
            pids = [w._process.pid for w in sidecar.workers if w._process is not None]
            stop.set()
            await asyncio.wait_for(service, timeout=5)
        finally:
            service.cancel()
            await asyncio.gather(service, *calls, return_exceptions=True)
            await sidecar.close()
        assert all(w.closed and w._process is None for w in sidecar.workers)
        for pid in pids:
            absent(pid)
        assert not path.exists() and not Path(str(path) + ".capacity.json").exists()

    asyncio.run(run())


def test_service_cancellation_during_startup_reaps_worker(setup):
    _, sidecar, path = setup

    async def run():
        sidecar.workers[1].command = (sys.executable, "-c", "import time; time.sleep(30)")
        service = asyncio.create_task(
            sidecar.serve_until_stopped(path, scoring_policy_sha256=POLICY, stop=asyncio.Event())
        )
        await until(lambda: sidecar.workers[1]._process is not None)
        pids = [w._process.pid for w in sidecar.workers]
        service.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(service, timeout=5)
        for pid in pids:
            absent(pid)
        assert not path.exists()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["startup", "serving"])
def test_service_preserves_failure_and_closes_workers(setup, monkeypatch, failure):
    _, sidecar, path = setup

    async def unexpected_exit():
        return

    async def run():
        if failure == "startup":
            sidecar.workers[1]._revision = bytes.fromhex("ff" * 32)
        else:
            monkeypatch.setattr(sidecar, "serve_forever", unexpected_exit)
        with pytest.raises(RuntimeError, match=r"revision differs|without a shutdown request"):
            await sidecar.serve_until_stopped(
                path, scoring_policy_sha256=POLICY, stop=asyncio.Event()
            )
        assert all(w.closed and w._process is None for w in sidecar.workers)
        assert not path.exists()

    asyncio.run(run())


@pytest.mark.parametrize("number", [signal.SIGTERM, signal.SIGINT])
@pytest.mark.parametrize("phase", ["startup", "serving"])
def test_process_service_signal_reaps_child_before_exit(setup, tmp_path, number, phase):
    api, _sidecar, path = setup
    script = tmp_path / "service.py"
    command = (
        [sys.executable, "-c", "import time; time.sleep(30)"]
        if phase == "startup"
        else [sys.executable, str(tmp_path / "worker.py")]
    )
    script.write_text(
        "import asyncio, sys\n"
        "from pathlib import Path\n"
        f"sys.path[:0] = {[str(ROOT / 'community'), str(Path(api.__file__).parents[1])]!r}\n"
        "from sidecar import CommunityModelSidecar, run_sidecar_service\n"
        "from worker_transport import WarmModelProcess\n"
        f"worker = WarmModelProcess({command!r},\n"
        "    environment={'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1'},\n"
        f"    cwd=Path({str(tmp_path)!r}), model_revision={REVISION!r},\n"
        "    startup_seconds=5, inference_seconds=2)\n"
        "original = worker.startup\n"
        "async def startup():\n"
        "    pending = asyncio.create_task(original())\n"
        "    try:\n"
        f"        if {phase!r} == 'startup':\n"
        "            while worker._process is None: await asyncio.sleep(0.01)\n"
        "            print(worker._process.pid, flush=True)\n"
        "        await pending\n"
        f"        if {phase!r} == 'serving': print(worker._process.pid, flush=True)\n"
        "    finally:\n"
        "        pending.cancel()\n"
        "        await asyncio.gather(pending, return_exceptions=True)\n"
        "worker.startup = startup\n"
        "sidecar = CommunityModelSidecar([worker], validator_slot_count=1)\n"
        f"run_sidecar_service(sidecar, Path({str(path)!r}), scoring_policy_sha256={POLICY!r})\n"
    )

    async def run():
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        )
        pid = None
        try:
            pid = int(await asyncio.wait_for(process.stdout.readline(), timeout=5))
            process.send_signal(number)
            _out, err = await asyncio.wait_for(process.communicate(), timeout=5)
            assert process.returncode == 0, err.decode()
            absent(pid)
            assert not path.exists() and not Path(str(path) + ".capacity.json").exists()
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()
            if pid is not None:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    asyncio.run(run())
