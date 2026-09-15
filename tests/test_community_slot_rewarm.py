"""Deterministic recovery tests, without model imports or wall-clock benchmarks."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


class Worker:
    model_revision = "ab" * 32
    inference_seconds = 0.1

    def __init__(self):
        self.closed = False
        self.ready = False
        self.starts = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.recovery_cancelled = False
        self.fail_reload = False

    async def startup(self):
        self.starts += 1
        if self.starts > 1:
            self.entered.set()
            try:
                await asyncio.wait_for(self.release.wait(), 5)
            except asyncio.CancelledError:
                self.recovery_cancelled = True
                raise
            if self.fail_reload:
                raise ValueError("verification failed")
        self.ready = True

    async def translate(self, video):
        assert self.ready and not self.closed
        if video == b"crash":
            self.ready = False
            raise RuntimeError("worker exited")
        return video.decode()

    async def close(self):
        self.closed = True
        self.ready = False


def build(worker):
    spec = importlib.util.spec_from_file_location("rewarm_sidecar", ROOT / "community/sidecar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sidecar = module.CommunityModelSidecar([worker], validator_slot_count=1)

    class Server:
        async def close(self):
            pass

    sidecar._server = Server()
    return sidecar


REQUEST = SimpleNamespace(
    document={"task": {"source_language": "ase", "target_language": "en", "stratum": "continuous"}}
)


async def prepare():
    worker = Worker()
    await worker.startup()
    sidecar = build(worker)
    sidecar._available.put_nowait(worker)
    with pytest.raises(RuntimeError, match="worker exited"):
        await sidecar.translate(b"crash", REQUEST)
    await asyncio.wait_for(worker.entered.wait(), 1)
    assert sidecar._available.empty()
    return sidecar, worker


def test_waiting_request_timeout_does_not_cancel_reload():
    async def run():
        sidecar, worker = await prepare()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await sidecar.translate(b"waiting", REQUEST)
            assert worker.starts == 2 and not worker.recovery_cancelled
            assert len(sidecar._recoveries) == 1 and sidecar._available.empty()
            worker.release.set()
            await asyncio.wait_for(asyncio.gather(*sidecar._recoveries), 1)
            assert await sidecar.translate(b"warm", REQUEST) == "warm"
            assert worker.starts == 2 and sidecar._available.qsize() == 1
        finally:
            await sidecar.close()
        assert worker.closed and not sidecar._recoveries

    asyncio.run(run())


def test_shutdown_cancels_reload_and_closes_worker():
    async def run():
        sidecar, worker = await prepare()
        await asyncio.wait_for(sidecar.close(), 1)
        assert worker.closed and worker.recovery_cancelled
        assert not sidecar._recoveries and sidecar._available.empty()

    asyncio.run(run())


def test_failed_reload_closes_service_without_requeue():
    async def run():
        sidecar, worker = await prepare()
        worker.fail_reload = True
        worker.release.set()
        await asyncio.wait_for(asyncio.gather(*sidecar._recoveries), 1)
        await asyncio.wait_for(sidecar.close(), 1)
        assert worker.closed and sidecar._closed
        assert not sidecar._recoveries and sidecar._available.empty()

    asyncio.run(run())


def test_cancelling_waiting_request_does_not_cancel_reload():
    async def run():
        sidecar, worker = await prepare()
        try:
            pending = asyncio.create_task(sidecar.translate(b"waiting", REQUEST))
            await asyncio.sleep(0)
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            assert not worker.recovery_cancelled and worker.starts == 2
            worker.release.set()
            await asyncio.wait_for(asyncio.gather(*sidecar._recoveries), 1)
            assert await sidecar.translate(b"warm", REQUEST) == "warm"
        finally:
            await sidecar.close()

    asyncio.run(run())
