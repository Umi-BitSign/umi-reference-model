"""Verify a locally reviewed native overlay before importing executable code."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import re
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

F_GETPATH = 50
MAXIMUM_FILE_BYTES = 512 * 1024 * 1024
MAXIMUM_MANIFEST_BYTES = 256 * 1024
MAXIMUM_TOTAL_BYTES = 1024 * 1024 * 1024


def _identity(info) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _canonical_root(descriptor: int, requested: Path) -> None:
    if not requested.is_absolute():
        raise ValueError("overlay path must be absolute without links or aliases")
    if platform.system() == "Darwin":
        raw = fcntl.fcntl(descriptor, F_GETPATH, b"\0" * 1024)
        actual = raw.split(b"\0", 1)[0]
        if not actual or actual != os.fsencode(requested):
            raise ValueError("overlay path must be absolute without links or aliases")
    elif os.fsencode(requested.resolve(strict=True)) != os.fsencode(requested):
        raise ValueError("overlay path must be absolute without links or aliases")


@contextmanager
def _directory(path: Path | str, *, parent: int | None = None, canonical: Path | None = None):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    except OSError as error:
        if canonical is not None:
            raise ValueError("overlay path must be absolute without links or aliases") from error
        raise
    try:
        before = os.fstat(descriptor)
        if canonical is not None:
            _canonical_root(descriptor, canonical)
        if before.st_uid != os.getuid() or stat.S_IMODE(before.st_mode) & 0o222:
            raise ValueError("overlay must be owned and read-only")
        yield descriptor
        after = os.fstat(descriptor)
        linked = os.stat(path, dir_fd=parent, follow_symlinks=False)
        if _identity(before) != _identity(after) or _identity(after) != _identity(linked):
            raise ValueError("overlay directory changed during verification")
    finally:
        os.close(descriptor)


def _file(
    parent: int,
    name: str,
    *,
    limit: int,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
    retain: bool = False,
) -> tuple[str, int, bytes]:
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_mode & 0o222
            or before.st_size > limit
            or (expected_size is not None and before.st_size != expected_size)
        ):
            raise ValueError("overlay file ownership, mode or size invalid")
        digest = hashlib.sha256()
        payload = bytearray()
        size = 0
        while chunk := handle.read(min(1024 * 1024, limit + 1 - size)):
            size += len(chunk)
            if size > limit:
                raise ValueError("overlay file exceeded limit")
            digest.update(chunk)
            if retain:
                payload.extend(chunk)
        after = os.fstat(handle.fileno())
        linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            size != before.st_size
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(linked)
        ):
            raise ValueError("overlay file changed during verification")
    checksum = digest.hexdigest()
    if expected_sha256 is not None and checksum != expected_sha256:
        raise ValueError("overlay file checksum or size mismatch")
    return checksum, size, bytes(payload)


def _manifest(raw: bytes) -> tuple[dict[str, dict], int]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate overlay manifest key")
            result[key] = value
        return result

    document = json.loads(raw, object_pairs_hook=pairs)
    if (
        not isinstance(document, dict)
        or document.get("schema") != "umi-mediapipe-cpu-overlay/1"
        or document.get("platform") != "macos/arm64"
        or document.get("python_abi") != "cp310"
        or document.get("mediapipe_version") != "0.10.14"
        or not isinstance(document.get("files"), list)
        or not 1 <= len(document["files"]) <= 2000
    ):
        raise ValueError("unsupported overlay manifest")
    files = {}
    total = 0
    for entry in document["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
            raise ValueError("invalid overlay file entry")
        name = entry["path"]
        relative = PurePosixPath(name) if isinstance(name, str) else None
        if (
            relative is None
            or relative.is_absolute()
            or relative.as_posix() != name
            or ".." in relative.parts
            or len(relative.parts) < 2
            or relative.parts[0] != "mediapipe"
            or name in files
            or not isinstance(entry["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) is None
            or type(entry["size_bytes"]) is not int
            or not 0 <= entry["size_bytes"] <= MAXIMUM_FILE_BYTES
        ):
            raise ValueError("invalid or duplicate overlay path")
        files[name] = entry
        total += entry["size_bytes"]
    if total > MAXIMUM_TOTAL_BYTES:
        raise ValueError("overlay exceeds total size limit")
    return files, total


def _tree(root: int, files: dict[str, dict]) -> None:
    directories = {
        str(parent)
        for name in files
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    seen = set()

    def walk(descriptor: int, prefix: str) -> None:
        with os.scandir(descriptor) as entries:
            for entry in entries:
                name = prefix + entry.name
                if not prefix and name == "overlay.json" and entry.is_file(follow_symlinks=False):
                    continue
                if name in directories and entry.is_dir(follow_symlinks=False):
                    with _directory(entry.name, parent=descriptor) as child:
                        walk(child, name + "/")
                elif name in files and entry.is_file(follow_symlinks=False):
                    record = files[name]
                    _file(
                        descriptor,
                        entry.name,
                        limit=MAXIMUM_FILE_BYTES,
                        expected_sha256=record["sha256"],
                        expected_size=record["size_bytes"],
                    )
                    seen.add(name)
                else:
                    raise ValueError("overlay contains an undeclared file, link or directory")

    walk(root, "")
    if seen != set(files):
        raise ValueError("declared overlay file missing")


def verify_overlay(root: Path, *, expected_manifest_sha256: str) -> dict[str, object]:
    """Verify a caller-pinned tree; the sandbox must keep it read-only afterward."""
    if re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256 or "") is None:
        raise ValueError("invalid expected overlay digest")
    if not root.is_absolute():
        raise ValueError("overlay path must be absolute without links or aliases")
    with _directory(root, canonical=root) as root_fd:
        checksum, _, raw = _file(root_fd, "overlay.json", limit=MAXIMUM_MANIFEST_BYTES, retain=True)
        if checksum != expected_manifest_sha256:
            raise ValueError("overlay manifest checksum mismatch")
        files, total = _manifest(raw)
        _tree(root_fd, files)
        if _file(root_fd, "overlay.json", limit=MAXIMUM_MANIFEST_BYTES, retain=True)[2] != raw:
            raise ValueError("overlay manifest changed during verification")
    return {"manifest_sha256": expected_manifest_sha256, "files": len(files), "bytes": total}
