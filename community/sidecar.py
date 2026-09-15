"""Connect isolated warm workers to UMI's existing private model socket.

No model framework or wallet is imported here. Each advertised slot owns a
separate WarmModelProcess supplied by the reviewed launcher. Actual accelerator
capacity still needs calibration; a process count is not a throughput result.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Sequence
from pathlib import Path


async def _drain(awaitable):
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
    return task.result()


class CommunityModelSidecar:
    """One private socket with one separately cancelable process per model slot.

    The caller constructs workers with an explicit command/environment, sandbox,
    release identity and resource bounds. Each child must verify its loaded
    release before emitting readiness. This class does not generate identities,
    approve rights, select policy values or construct a sandbox.
    """

    def __init__(self, workers: Sequence, *, validator_slot_count: int) -> None:
        if (
            type(validator_slot_count) is not int
            or not 1 <= validator_slot_count <= 256
            or not validator_slot_count <= len(workers) <= 256
            or len(workers) % validator_slot_count
            or len({id(worker) for worker in workers}) != len(workers)
        ):
            raise ValueError(
                "distinct workers must provide an equal positive number of slots per validator"
            )
        if len({worker.model_revision for worker in workers}) != 1:
            raise ValueError("workers must use the same verified model revision")
        if len({worker.inference_seconds for worker in workers}) != 1:
            raise ValueError("workers must use the same inference deadline")
        self.workers = tuple(workers)
        self.validator_slot_count = validator_slot_count
        self.model_revision = self.workers[0].model_revision
        self.inference_seconds = self.workers[0].inference_seconds
        self._available = asyncio.Queue(maxsize=len(self.workers))
        self._operations: set[asyncio.Task] = set()
        self._recoveries: set[asyncio.Task] = set()
        self._recovery_lock = asyncio.Lock()
        self._startup: asyncio.Task | None = None
        self._server = None
        self._close_task: asyncio.Task | None = None
        self._closed = False
        self._starting = False

    async def start(self, socket_path: Path, *, scoring_policy_sha256: str | None) -> None:
        if self._closed or self._starting:
            raise RuntimeError("sidecar may be started only once")
        self._starting = True
        try:
            # Import only the dependency-light socket helper, never UMI's wallet
            # or chain clients. No socket is offered until every worker is ready.
            from umi.model_sidecar import start_model_sidecar

            async def startup():
                # Sequential loading avoids a burst of concurrent model loads.
                for worker in self.workers:
                    await worker.startup()
                    self._available.put_nowait(worker)
                self._server = await start_model_sidecar(
                    socket_path,
                    self.translate,
                    model_revision=self.model_revision,
                    scoring_policy_sha256=scoring_policy_sha256,
                    validator_slot_count=self.validator_slot_count,
                    maximum_concurrency=len(self.workers),
                    maximum_inference_seconds=self.inference_seconds,
                )

            self._startup = asyncio.create_task(startup())
            await self._startup
        except BaseException:
            await self.close()
            raise

    async def _translate(self, video: bytes):
        worker = await self._available.get()
        try:
            if self._closed:
                raise RuntimeError("sidecar is closed")
            result = await worker.translate(video)
        except BaseException:
            if worker.closed:
                # Schedule cleanup without awaiting ourselves through the set
                # of active operations. serve_forever/close drains the result.
                self._begin_close()
            elif not self._closed:
                # Reload outside any caller's inference deadline. A request
                # waiting for this slot may expire without canceling recovery.
                recovery = asyncio.create_task(self._rewarm(worker))
                self._recoveries.add(recovery)
                recovery.add_done_callback(self._recoveries.discard)
            raise
        else:
            self._available.put_nowait(worker)
            return result

    async def _rewarm(self, worker) -> None:
        try:
            # As at initial startup, avoid simultaneous model-loading bursts.
            async with self._recovery_lock:
                if self._closed:
                    return
                await worker.startup()
                if not self._closed:
                    self._available.put_nowait(worker)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Failed verification/reload must not leave advertised slots with
            # no usable workers. The service manager can observe this shutdown.
            self._begin_close()

    async def translate(self, video: bytes, request) -> str:
        if self._closed or self._server is None:
            raise RuntimeError("sidecar is not ready")
        task = request.document.get("task")
        if (
            not isinstance(task, dict)
            or set(task) != {"source_language", "target_language", "stratum"}
            or task["source_language"] != "ase"
            or task["target_language"] != "en"
            or task["stratum"] not in {"fingerspelling", "short_utterance", "continuous"}
        ):
            raise ValueError("community model supports only the declared ASL-to-English tasks")
        # Only exact video bytes reach the model. The worker never receives the
        # request's URL, validator identity or protected evaluation reference.
        operation = asyncio.create_task(self._translate(video))
        self._operations.add(operation)
        try:
            # This deadline includes queueing. Replacement workers reload in a
            # separate bounded task and rejoin the queue only after readiness.
            # The UMI socket helper separately enforces the same outer limit.
            return await asyncio.wait_for(operation, timeout=self.inference_seconds)
        finally:
            if not operation.done():
                operation.cancel()
                await _drain(asyncio.gather(operation, return_exceptions=True))
            self._operations.discard(operation)

    async def serve_forever(self) -> None:
        if self._server is None or self._closed:
            raise RuntimeError("sidecar is not ready")
        try:
            await self._server.serve_forever()
        finally:
            await self.close()

    async def serve_until_stopped(
        self, socket_path: Path, *, scoring_policy_sha256: str | None, stop: asyncio.Event
    ) -> None:
        """Own startup and serving until the service manager requests shutdown."""
        tasks = []
        try:
            if stop.is_set():
                return
            stopped = asyncio.create_task(stop.wait())
            startup = asyncio.create_task(
                self.start(socket_path, scoring_policy_sha256=scoring_policy_sha256)
            )
            tasks.extend((stopped, startup))
            await asyncio.wait((startup, stopped), return_when=asyncio.FIRST_COMPLETED)
            if stopped.done():
                return
            await startup  # Preserve a startup failure instead of reporting readiness.
            serving = asyncio.create_task(self.serve_forever())
            tasks.append(serving)
            await asyncio.wait((serving, stopped), return_when=asyncio.FIRST_COMPLETED)
            if not stopped.done():
                await serving
                raise RuntimeError("model socket stopped without a shutdown request")
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            try:
                await _drain(asyncio.gather(*tasks, return_exceptions=True))
            finally:
                await self.close()

    async def _close(self) -> None:
        if self._startup is not None and not self._startup.done():
            self._startup.cancel()
            await asyncio.gather(self._startup, return_exceptions=True)
        server_error = None
        if self._server is not None:
            try:
                await self._server.close()
            except BaseException as error:
                # A socket/descriptor cleanup failure must not leave native
                # worker processes running without a serving owner.
                server_error = error
        pending = tuple(self._operations)
        for operation in pending:
            operation.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        recoveries = tuple(self._recoveries)
        for recovery in recoveries:
            recovery.cancel()
        await asyncio.gather(*recoveries, return_exceptions=True)
        results = await asyncio.gather(
            *(worker.close() for worker in self.workers), return_exceptions=True
        )
        if any(isinstance(result, BaseException) for result in results):
            raise RuntimeError("one or more model workers could not be reaped")
        if server_error is not None:
            raise RuntimeError("model socket cleanup failed after worker cleanup") from server_error

    def _begin_close(self) -> asyncio.Task:
        self._closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close())
        return self._close_task

    async def close(self) -> None:
        await _drain(self._begin_close())


def run_sidecar_service(sidecar, socket_path: Path, *, scoring_policy_sha256: str | None) -> None:
    """Process entry point for a reviewed launcher, with graceful TERM/INT.

    The launcher selects the real policy, or None for local standby, and constructs
    isolated workers. This function neither installs a daemon nor grants model
    rights, public readiness or permission to write weights. It owns the main
    thread's event loop and signal handlers; do not embed it in another server.
    """

    async def run():
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        previous = {}
        try:
            for number in (signal.SIGTERM, signal.SIGINT):
                handler = signal.getsignal(number)
                loop.add_signal_handler(number, stop.set)
                previous[number] = handler
            await sidecar.serve_until_stopped(
                socket_path, scoring_policy_sha256=scoring_policy_sha256, stop=stop
            )
        finally:
            for number, handler in previous.items():
                loop.remove_signal_handler(number)
                signal.signal(number, handler)

    asyncio.run(run())
