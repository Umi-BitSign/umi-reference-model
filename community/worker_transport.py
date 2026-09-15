"""Bounded IPC for one warm model process, with no model imports in its caller.

The command and environment are reviewed operator inputs. Process groups provide
cancellation, not a security sandbox. The launcher must separately deny wallet
access and outbound networking and enforce memory/process limits.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import os
import re
import secrets
import signal
import stat
import struct
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

READY = b"umi-community-worker-v1\0"
REQUEST = b"umi-community-video-v1\0"
RESPONSE = b"umi-community-text-v1\0"
VIDEO_LIMIT = 16 * 1024 * 1024
TEXT_LIMIT = 4096
_REQUEST_FIELDS = struct.Struct(">32s32sQQ")  # nonce, video digest, size, deadline_ns
_RESPONSE_FIELDS = struct.Struct(">32sI")


def _revision(value: str) -> bytes:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("model revision must be a lowercase SHA-256 digest")
    return bytes.fromhex(value)


def _seconds(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("deadline must be numeric")
    if not math.isfinite(value) or not 0 < value <= 600:
        raise ValueError("deadline must be in (0, 600]")
    return float(value)


async def _finish(awaitable):
    """Finish bounded cleanup even if a caller cancels it again."""
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            pass
    return task.result()


class WarmModelProcess:
    """One serialized, replaceable worker; create one per enforced model slot.

    Startup waits for the worker's release-bound readiness frame. Each request
    has a fresh nonce and a queue-inclusive deadline. Failure destroys that
    worker before another request can launch its replacement. No error path
    invents a transcript. Readiness alone does not verify model artifacts: the
    worker must verify its release before calling ``serve_worker``.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        cwd: Path,
        model_revision: str,
        startup_seconds: float,
        inference_seconds: float,
        scratch_directory: Path | None = None,
    ) -> None:
        if os.name != "posix":
            raise RuntimeError("model process transport requires POSIX process groups")
        if (
            isinstance(command, str | bytes)
            or not command
            or any(not isinstance(item, str) or not item or "\0" in item for item in command)
            or not Path(command[0]).is_absolute()
        ):
            raise ValueError("worker command must have an absolute executable and fixed arguments")
        if not cwd.is_absolute() or not cwd.is_dir():
            raise ValueError("worker cwd must be an existing absolute directory")
        if any(
            not isinstance(key, str)
            or not key
            or "=" in key
            or "\0" in key
            or not isinstance(value, str)
            or "\0" in value
            for key, value in environment.items()
        ):
            raise ValueError("worker environment is invalid")
        self.command = tuple(command)
        self.environment = dict(environment)  # Never inherit the caller's credentials.
        self.cwd = cwd
        self.model_revision = model_revision
        self._revision = _revision(model_revision)
        self.startup_seconds = _seconds(startup_seconds)
        self.inference_seconds = _seconds(inference_seconds)
        self._lock = asyncio.Lock()
        self._process = None
        self._operations: set[asyncio.Task] = set()
        self._closed = False
        self._scratch = None
        if scratch_directory is not None:
            path = scratch_directory
            if not path.is_absolute() or path.resolve(strict=True) != path:
                raise ValueError("worker scratch must be an absolute directory without links")
            metadata = path.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o700
                or any(path.iterdir())
            ):
                raise ValueError("worker scratch must be new, empty and owner-private")
            self._scratch = (path, metadata.st_dev, metadata.st_ino)

    def _clear_scratch(self) -> None:
        """Remove only this worker's claimed scratch contents after confirmed reap."""
        if self._scratch is None:
            return
        path, device, inode = self._scratch
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            metadata = os.fstat(descriptor)
            if (metadata.st_dev, metadata.st_ino, metadata.st_uid) != (device, inode, os.getuid()):
                raise RuntimeError("worker scratch directory identity changed")

            def clear(parent):
                for name in os.listdir(parent):
                    child = os.stat(name, dir_fd=parent, follow_symlinks=False)
                    if stat.S_ISDIR(child.st_mode):
                        nested = os.open(
                            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
                        )
                        try:
                            clear(nested)
                        finally:
                            os.close(nested)
                        os.rmdir(name, dir_fd=parent)
                    else:
                        # A link is unlinked, never followed to its target.
                        os.unlink(name, dir_fd=parent)

            clear(descriptor)
        finally:
            os.close(descriptor)

    async def _discard(self) -> None:
        process = self._process
        if process is None:
            return

        async def reap():
            for attempt in range(3):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                except ProcessLookupError:
                    break
                except PermissionError:
                    if sys.platform != "darwin" or attempt == 2:
                        raise
                    # Retry a teardown EPERM only after observing leader exit.
                    # EPERM itself is never proof that the group is absent.
                    await asyncio.wait_for(process.wait(), timeout=1)
                    await asyncio.sleep(0.01)
            if process.stdin is not None:
                process.stdin.close()
            # Drain the bounded pipe buffers after killing the process group.
            # wait() alone can hang if a full stdout pipe paused its transport.
            await asyncio.wait_for(process.communicate(), timeout=5)

        try:
            await _finish(reap())
            self._process = None
            self._clear_scratch()
        except BaseException:
            self._closed = True  # Uncertain cleanup forbids a replacement.
            raise

    async def _start(self) -> None:
        if self._closed:
            raise RuntimeError("model process is closed")
        if self._process is not None:
            return
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *self.command,
                cwd=self.cwd,
                env=self.environment,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
                limit=64 * 1024,
            )
        )
        try:
            self._process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            self._process = await _finish(creation)
            await self._discard()
            raise
        ready = await self._process.stdout.readexactly(len(READY) + 32)
        if ready != READY + self._revision:
            raise RuntimeError("worker readiness revision differs")

    async def _operate(self, video: bytes | None, deadline_ns: int | None):
        async with self._lock:
            try:
                await self._start()
                if video is None:
                    return None
                nonce = secrets.token_bytes(32)
                process = self._process
                process.stdin.write(
                    REQUEST
                    + _REQUEST_FIELDS.pack(
                        nonce, hashlib.sha256(video).digest(), len(video), deadline_ns
                    )
                    + video
                )
                await process.stdin.drain()
                prefix = await process.stdout.readexactly(len(RESPONSE) + _RESPONSE_FIELDS.size)
                if not prefix.startswith(RESPONSE):
                    raise RuntimeError("worker response marker differs")
                echoed, size = _RESPONSE_FIELDS.unpack(prefix[len(RESPONSE) :])
                if echoed != nonce or not 0 < size <= TEXT_LIMIT:
                    raise RuntimeError("worker response binding or length differs")
                text = (await process.stdout.readexactly(size)).decode("utf-8", errors="strict")
                if not text.strip():
                    raise RuntimeError("worker returned empty text")
                return text
            except BaseException:
                await self._discard()
                raise

    async def _run(self, video: bytes | None, seconds: float):
        if self._closed:
            raise RuntimeError("model process is closed")
        deadline_ns = None if video is None else time.time_ns() + math.ceil(seconds * 1e9)
        task = asyncio.create_task(self._operate(video, deadline_ns))
        self._operations.add(task)
        try:
            return await asyncio.wait_for(task, timeout=seconds)
        finally:
            if not task.done():
                task.cancel()
                await _finish(asyncio.gather(task, return_exceptions=True))
            self._operations.discard(task)

    async def startup(self) -> None:
        await self._run(None, self.startup_seconds)

    @property
    def closed(self) -> bool:
        """True after explicit shutdown or cleanup uncertainty; no replacement is allowed."""
        return self._closed

    async def translate(self, video: bytes) -> str:
        if not isinstance(video, bytes) or not 0 < len(video) <= VIDEO_LIMIT:
            raise ValueError("video must be nonempty bytes within the transport ceiling")
        return await self._run(video, self.inference_seconds)

    async def close(self) -> None:
        self._closed = True
        pending = tuple(self._operations)
        for task in pending:
            task.cancel()
        await _finish(asyncio.gather(*pending, return_exceptions=True))
        async with self._lock:
            await self._discard()


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        body = stream.read(size - len(chunks))
        if not body:
            raise EOFError("truncated worker request")
        chunks.extend(body)
    return bytes(chunks)


def serve_worker(
    translate: Callable[[bytes, int], str],
    *,
    verified_model_revision: str,
    source: BinaryIO | None = None,
    destination: BinaryIO | None = None,
) -> None:
    """Serve an already loaded and verified model in the isolated child only.

    The parent enforces the outer deadline, including native code that ignores
    the supplied Unix-nanosecond deadline. A loader must not call this before
    verifying all inference-affecting files against its release manifest.
    """
    revision = _revision(verified_model_revision)
    source = sys.stdin.buffer if source is None else source
    if destination is None:
        # Keep protocol output separate from Python and native-library stdout.
        # The loader must likewise redirect diagnostics before model imports.
        sys.stdout.flush()
        with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as protocol:
            os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
            return serve_worker(
                translate,
                verified_model_revision=verified_model_revision,
                source=source,
                destination=protocol,
            )
    destination.write(READY + revision)
    destination.flush()
    while True:
        first = source.read(1)
        if not first:
            return
        marker = first + _read_exact(source, len(REQUEST) - 1)
        if marker != REQUEST:
            raise ValueError("invalid worker request marker")
        nonce, digest, size, deadline_ns = _REQUEST_FIELDS.unpack(
            _read_exact(source, _REQUEST_FIELDS.size)
        )
        if not 0 < size <= VIDEO_LIMIT or deadline_ns <= time.time_ns():
            raise ValueError("worker request size or deadline is invalid")
        video = _read_exact(source, size)
        if hashlib.sha256(video).digest() != digest or deadline_ns <= time.time_ns():
            raise ValueError("worker video binding or deadline differs")
        with contextlib.redirect_stdout(sys.stderr):
            text = translate(video, deadline_ns)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("model returned no text")
        body = text.encode("utf-8", errors="strict")
        if len(body) > TEXT_LIMIT or deadline_ns <= time.time_ns():
            raise ValueError("model exceeded its output or time limit")
        destination.write(RESPONSE + _RESPONSE_FIELDS.pack(nonce, len(body)) + body)
        destination.flush()
