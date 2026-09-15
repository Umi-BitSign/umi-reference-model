from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import io
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "community/worker_transport.py"
REVISION = "ab" * 32


def load():
    spec = importlib.util.spec_from_file_location("worker_transport_test", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker(tmp_path):
    module = load()
    script = tmp_path / "worker.py"
    script.write_text(
        "import importlib.util, os, sys, time\n"
        f"spec = importlib.util.spec_from_file_location('transport', {str(MODULE_PATH)!r})\n"
        "transport = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(transport)\n"
        "def translate(video, deadline_ns):\n"
        "    if video == b'hang': time.sleep(30)\n"
        "    if video == b'slow': time.sleep(0.2)\n"
        "    if video == b'crash': os._exit(7)\n"
        "    if video == b'native-print': os.write(1, b'library diagnostic\\n')\n"
        "    if video == b'environment': return str('UMI_TEST_SECRET' in os.environ)\n"
        "    return video.decode('utf-8')\n"
        f"transport.serve_worker(translate, verified_model_revision={REVISION!r})\n"
    )

    def make(*, startup=3, inference=3, command=None, revision=REVISION):
        return module.WarmModelProcess(
            command or [sys.executable, str(script)],
            environment={"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
            cwd=tmp_path,
            model_revision=revision,
            startup_seconds=startup,
            inference_seconds=inference,
        )

    return module, make


def assert_absent(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_warm_process_reuses_one_loaded_worker_and_closes(worker, monkeypatch):
    _, make = worker
    monkeypatch.setenv("UMI_TEST_SECRET", "must-not-reach-child")

    async def run():
        backend = make()
        try:
            await backend.startup()
            pid = backend._process.pid
            assert await backend.translate(b"hello") == "hello"
            assert await backend.translate(b"native-print") == "native-print"
            assert await backend.translate(b"environment") == "False"
            assert backend._process.pid == pid
        finally:
            await backend.close()
        assert_absent(pid)
        with pytest.raises(RuntimeError, match="closed"):
            await backend.translate(b"hello")

    asyncio.run(run())


def test_claimed_scratch_cleared_only_after_worker_reaped(worker, tmp_path):
    module, _ = worker
    scratch = tmp_path / "owned-scratch"
    scratch.mkdir(mode=0o700)
    outside = tmp_path / "must-preserve"
    outside.write_text("not scratch")
    source = tmp_path / "scratch-worker.py"
    source.write_text(
        "import importlib.util, pathlib, sys, time\n"
        f"spec = importlib.util.spec_from_file_location('transport', {str(MODULE_PATH)!r})\n"
        "transport = importlib.util.module_from_spec(spec); spec.loader.exec_module(transport)\n"
        f"scratch = pathlib.Path({str(scratch)!r})\n"
        "def translate(video, deadline):\n"
        "    (scratch / 'nested').mkdir()\n"
        "    (scratch / 'nested/input.mp4').write_bytes(video)\n"
        f"    (scratch / 'outside-link').symlink_to({str(outside)!r})\n"
        "    time.sleep(30)\n"
        f"transport.serve_worker(translate, verified_model_revision={REVISION!r})\n"
    )

    async def run():
        backend = module.WarmModelProcess(
            [sys.executable, str(source)],
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            cwd=tmp_path,
            model_revision=REVISION,
            startup_seconds=3,
            inference_seconds=0.2,
            scratch_directory=scratch,
        )
        try:
            await backend.startup()
            pid = backend._process.pid
            with pytest.raises(TimeoutError):
                await backend.translate(b"temporary video")
            assert_absent(pid)
            assert list(scratch.iterdir()) == []
            assert outside.read_text() == "not scratch"
            await backend.startup()
        finally:
            await backend.close()

    asyncio.run(run())


def test_existing_or_replaced_scratch_is_not_deleted(worker, tmp_path):
    module, _ = worker
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    existing = scratch / "existing"
    existing.write_text("preserve")
    kwargs = dict(
        environment={},
        cwd=tmp_path,
        model_revision=REVISION,
        startup_seconds=1,
        inference_seconds=1,
        scratch_directory=scratch,
    )
    with pytest.raises(ValueError, match="new, empty"):
        module.WarmModelProcess([sys.executable, "-c", "pass"], **kwargs)
    assert existing.read_text() == "preserve"
    existing.unlink()
    backend = module.WarmModelProcess([sys.executable, "-c", "pass"], **kwargs)
    scratch.rename(tmp_path / "original")
    scratch.mkdir(mode=0o700)
    marker = scratch / "replacement"
    marker.write_text("preserve replacement")
    with pytest.raises(RuntimeError, match="identity changed"):
        backend._clear_scratch()
    assert marker.read_text() == "preserve replacement"


def test_scratch_cleanup_failure_forbids_replacement(worker, tmp_path, monkeypatch):
    module, _ = worker
    scratch = tmp_path / "scratch"
    scratch.mkdir(mode=0o700)
    script = tmp_path / "worker.py"

    async def run():
        backend = module.WarmModelProcess(
            [sys.executable, str(script)],
            environment={"PYTHONDONTWRITEBYTECODE": "1"},
            cwd=tmp_path,
            model_revision=REVISION,
            startup_seconds=3,
            inference_seconds=0.2,
            scratch_directory=scratch,
        )
        await backend.startup()
        pid = backend._process.pid

        def failed_cleanup():
            assert_absent(pid)
            raise OSError("cleanup failed")

        monkeypatch.setattr(backend, "_clear_scratch", failed_cleanup)
        with pytest.raises(OSError, match="cleanup failed"):
            await backend.translate(b"hang")
        assert backend.closed
        assert backend._process is None
        with pytest.raises(RuntimeError, match="closed"):
            await backend.startup()
        await backend.close()

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["hang", "crash"])
def test_failed_worker_is_reaped_before_replacement(worker, fault):
    _, make = worker

    async def run():
        backend = make(inference=0.5)
        try:
            await backend.startup()
            pid = backend._process.pid
            with pytest.raises((TimeoutError, asyncio.IncompleteReadError)):
                await backend.translate(fault.encode())
            assert_absent(pid)
            assert backend._process is None
            await backend.startup()
            assert await backend.translate(b"new") == "new"
        finally:
            await backend.close()

    asyncio.run(run())


def test_cancelling_active_request_kills_worker(worker):
    _, make = worker

    async def run():
        backend = make()
        await backend.startup()
        pid = backend._process.pid
        task = asyncio.create_task(backend.translate(b"hang"))
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert_absent(pid)
        await backend.close()

    asyncio.run(run())


def test_cancelling_queued_request_preserves_active_worker(worker):
    _, make = worker

    async def run():
        backend = make()
        try:
            await backend.startup()
            pid = backend._process.pid
            active = asyncio.create_task(backend.translate(b"slow"))
            await asyncio.sleep(0.05)
            queued = asyncio.create_task(backend.translate(b"unused"))
            await asyncio.sleep(0.02)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert await active == "slow"
            assert backend._process.pid == pid
            assert await backend.translate(b"next") == "next"
        finally:
            await backend.close()

    asyncio.run(run())


def test_close_cancels_inflight_and_queued_requests(worker):
    _, make = worker

    async def run():
        backend = make()
        await backend.startup()
        pid = backend._process.pid
        tasks = [asyncio.create_task(backend.translate(b"hang")) for _ in range(2)]
        await asyncio.sleep(0.1)
        await asyncio.wait_for(backend.close(), timeout=2)
        assert all(
            isinstance(x, asyncio.CancelledError)
            for x in await asyncio.gather(*tasks, return_exceptions=True)
        )
        assert_absent(pid)
        assert not backend._operations

    asyncio.run(run())


def test_wrong_ready_revision_is_rejected_and_reaped(worker):
    _, make = worker

    async def run():
        backend = make(revision="cd" * 32)
        try:
            with pytest.raises(RuntimeError, match="revision"):
                await backend.startup()
            assert backend._process is None
        finally:
            await backend.close()

    asyncio.run(run())


def test_startup_timeout_reaps_child(worker):
    _, make = worker

    async def run():
        backend = make(startup=0.1, command=[sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            with pytest.raises(TimeoutError):
                await backend.startup()
            assert backend._process is None
        finally:
            await backend.close()

    asyncio.run(run())


def test_cancellation_during_spawn_still_reaps_created_child(worker, monkeypatch):
    module, make = worker
    original = module.asyncio.create_subprocess_exec
    children = []

    async def delayed(*args, **kwargs):
        await asyncio.sleep(0.1)
        child = await original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", delayed)

    async def run():
        backend = make()
        task = asyncio.create_task(backend.startup())
        await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(children) == 1
        assert_absent(children[0].pid)
        await backend.close()

    asyncio.run(run())


@pytest.mark.parametrize("fault", ["marker", "nonce", "size", "utf8", "truncated"])
def test_parent_rejects_malformed_reply_and_reaps_child(worker, fault):
    _, make = worker
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('t', {str(MODULE_PATH)!r})\n"
        "t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)\n"
        "out = sys.stdout.buffer\n"
        f"out.write(t.READY + bytes.fromhex({REVISION!r})); out.flush()\n"
        "head = t._read_exact(sys.stdin.buffer, len(t.REQUEST) + t._REQUEST_FIELDS.size)\n"
        "nonce = head[len(t.REQUEST):len(t.REQUEST) + 32]\n"
        f"fault = {fault!r}\n"
        "marker = b'x' + t.RESPONSE[1:] if fault == 'marker' else t.RESPONSE\n"
        "nonce = b'x' * 32 if fault == 'nonce' else nonce\n"
        "size = 4097 if fault == 'size' else 5 if fault == 'truncated' else 1\n"
        "body = b'\\xff' if fault == 'utf8' else b'x'\n"
        "out.write(marker + t._RESPONSE_FIELDS.pack(nonce, size) + body); out.flush()\n"
    )

    async def run():
        backend = make(command=[sys.executable, "-c", code])
        try:
            await backend.startup()
            pid = backend._process.pid
            with pytest.raises((RuntimeError, UnicodeError, asyncio.IncompleteReadError)):
                await backend.translate(b"video")
            assert backend._process is None
            assert_absent(pid)
        finally:
            await backend.close()

    asyncio.run(run())


def test_flooded_stdout_is_drained_after_kill_without_hanging(worker):
    _, make = worker
    code = (
        "import sys, time\n"
        f"sys.stdout.buffer.write({load().READY!r} + bytes.fromhex({REVISION!r}))\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stdout.buffer.write(b'x' * 1048576); sys.stdout.buffer.flush()\n"
        "time.sleep(30)\n"
    )

    async def run():
        backend = make(command=[sys.executable, "-c", code])
        try:
            await backend.startup()
            pid = backend._process.pid
            await asyncio.sleep(0.1)
            with pytest.raises(RuntimeError, match="marker"):
                await asyncio.wait_for(backend.translate(b"video"), timeout=2)
            assert_absent(pid)
        finally:
            await backend.close()

    asyncio.run(run())


def test_uncertain_cleanup_forbids_replacement(worker, monkeypatch):
    module, make = worker
    original = module.os.killpg

    def denied(*args):
        raise PermissionError("injected signal denial")

    async def run():
        backend = make()
        await backend.startup()
        pid = backend._process.pid
        monkeypatch.setattr(module.os, "killpg", denied)
        try:
            with pytest.raises((PermissionError, TimeoutError)):
                await backend.close()
            with pytest.raises(RuntimeError, match="closed"):
                await backend.startup()
            assert backend._process.pid == pid
        finally:
            monkeypatch.setattr(module.os, "killpg", original)
            await backend.close()
        assert_absent(pid)

    asyncio.run(run())


def frame(module, video=b"video", *, size=None, digest=None, deadline=None):
    return (
        module.REQUEST
        + module._REQUEST_FIELDS.pack(
            b"n" * 32,
            hashlib.sha256(video).digest() if digest is None else digest,
            len(video) if size is None else size,
            time.time_ns() + 1_000_000_000 if deadline is None else deadline,
        )
        + video
    )


@pytest.mark.parametrize("fault", ["oversize", "digest", "expired", "truncated", "marker"])
def test_worker_rejects_invalid_frames_before_calling_model(fault):
    module = load()
    body = {
        "oversize": frame(module, size=module.VIDEO_LIMIT + 1),
        "digest": frame(module, digest=b"x" * 32),
        "expired": frame(module, deadline=1),
        "truncated": frame(module)[:-1],
        "marker": b"x" + frame(module)[1:],
    }[fault]
    calls = []
    with pytest.raises((ValueError, EOFError)):
        module.serve_worker(
            lambda *args: calls.append(args),
            verified_model_revision=REVISION,
            source=io.BytesIO(body),
            destination=io.BytesIO(),
        )
    assert calls == []


@pytest.mark.parametrize("result", [None, "", "   ", "x" * 4097, "\ud800"])
def test_worker_rejects_invalid_hypothesis(result):
    module = load()
    output = io.BytesIO()
    with pytest.raises((ValueError, UnicodeError)):
        module.serve_worker(
            lambda *args: result,
            verified_model_revision=REVISION,
            source=io.BytesIO(frame(module)),
            destination=output,
        )
    assert output.getvalue() == module.READY + bytes.fromhex(REVISION)


def test_worker_binds_each_response_and_preserves_exact_video():
    module = load()
    video = b"exact verified MP4 bytes"
    output = io.BytesIO()
    seen = []

    def translate(actual, deadline):
        seen.append(actual)
        assert deadline > time.time_ns()
        return "Books"

    module.serve_worker(
        translate,
        verified_model_revision=REVISION,
        source=io.BytesIO(frame(module, video)),
        destination=output,
    )
    assert seen == [video]
    assert output.getvalue() == (
        module.READY
        + bytes.fromhex(REVISION)
        + module.RESPONSE
        + module._RESPONSE_FIELDS.pack(b"n" * 32, 5)
        + b"Books"
    )
