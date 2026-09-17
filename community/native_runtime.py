"""Local native-runtime inventory; never imports contributed model code.

This binds the installed environment as well as the loader to a reviewed local
release. It is not an independently reproducible evaluator image or a rights
approval. The launcher must deny worker writes to all inventoried roots.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import stat
from pathlib import Path

DOMAIN = b"umi-community-native-runtime-v1\0"
F_GETPATH = 50
MAXIMUM_ENTRIES = 40_000
MAXIMUM_BYTES = 4 * 1024**3
MAXIMUM_MANIFEST_BYTES = 12 * 1024**2
EXECUTION = {
    "device": "mps",
    "generation_num_beams": 5,
    "generation_max_length": 2048,
    "dino_batch_size": 128,
    "model_execution_concurrency": 1,
    "cpu_threads": 4,
    "mps_high_watermark_ratio": "0.25",
    "mps_low_watermark_ratio": "0.20",
}


def canonical(document) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("expected lowercase SHA-256")
    return value


def file_hash(path: Path, *, maximum_bytes: int) -> tuple[str, int]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum_bytes:
            raise ValueError("runtime input must be a bounded regular file")
        result = hashlib.sha256()
        total = 0
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            total += len(chunk)
            if total > maximum_bytes:
                raise ValueError("runtime input exceeds size limit")
            result.update(chunk)
        after = os.fstat(stream.fileno())
        linked = path.lstat()
        attributes = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if total != before.st_size or any(
            getattr(before, key) != getattr(after, key)
            or getattr(after, key) != getattr(linked, key)
            for key in attributes
        ):
            raise ValueError("runtime input changed while hashing")
    return result.hexdigest(), total


def opened_directory(path: Path) -> tuple[Path, tuple[int, int]]:
    if not path.is_absolute():
        raise ValueError("runtime roots must be absolute directories without aliases")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("runtime roots must be absolute directories without aliases") from error
    try:
        metadata = os.fstat(descriptor)
        if platform.system() == "Darwin":
            raw = fcntl.fcntl(descriptor, F_GETPATH, b"\0" * 1024)
            actual = raw.split(b"\0", 1)[0]
            canonical = actual == os.fsencode(path)
        else:
            canonical = os.fsencode(path.resolve(strict=True)) == os.fsencode(path)
        linked = path.stat(follow_symlinks=False)
        if (
            not canonical
            or not stat.S_ISDIR(metadata.st_mode)
            or (
                metadata.st_dev,
                metadata.st_ino,
            )
            != (linked.st_dev, linked.st_ino)
        ):
            raise ValueError("runtime roots must be absolute directories without aliases")
        return path, (metadata.st_dev, metadata.st_ino)
    finally:
        os.close(descriptor)


def checked_roots(roots: dict[str, Path]) -> dict[str, Path]:
    if set(roots) != {"code", "environment", "python"}:
        raise ValueError("runtime needs exactly code, environment and python roots")
    resolved = {}
    identities = {}
    for name, path in roots.items():
        resolved[name], identities[name] = opened_directory(path)
    for name, path in resolved.items():
        if any(
            identities[name] == identities[key] or path.is_relative_to(other)
            for key, other in resolved.items()
            if key != name
        ):
            raise ValueError("runtime roots must not overlap")
    return resolved


def inventory(roots: dict[str, Path]) -> list[dict]:
    roots = checked_roots(roots)
    entries = []
    total = 0
    for label, root in sorted(roots.items()):
        pending = [root]
        while pending:
            path = pending.pop()
            info = path.lstat()
            if info.st_uid != os.getuid() or (
                not stat.S_ISLNK(info.st_mode) and stat.S_IMODE(info.st_mode) & 0o022
            ):
                raise ValueError("runtime ownership or writable group permissions differ")
            relative = path.relative_to(root).as_posix()
            entry = {"root": label, "path": relative, "mode": stat.S_IMODE(info.st_mode)}
            if stat.S_ISLNK(info.st_mode):
                target = path.resolve(strict=True)
                matches = [
                    (name, target.relative_to(base).as_posix())
                    for name, base in roots.items()
                    if target.is_relative_to(base)
                ]
                if len(matches) != 1:
                    raise ValueError("runtime link escapes inventoried roots")
                entry.update(kind="symlink", target_root=matches[0][0], target_path=matches[0][1])
            elif stat.S_ISDIR(info.st_mode):
                entry.update(kind="directory")
                pending.extend(sorted(path.iterdir(), reverse=True))
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                checksum, size = file_hash(path, maximum_bytes=MAXIMUM_BYTES - total)
                total += size
                entry.update(kind="file", sha256=checksum, size_bytes=size)
            else:
                raise ValueError("runtime contains special or hardlinked files")
            entries.append(entry)
            if len(entries) > MAXIMUM_ENTRIES:
                raise ValueError("runtime contains too many entries")
    return sorted(entries, key=lambda entry: (entry["root"], entry["path"]))


def stage_manifest(
    roots: dict[str, Path], *, bundle_sha256: str, overlay_sha256: str, output: Path
) -> str:
    document = {
        "schema": "umi-community-native-runtime/1",
        "platform": "macos/arm64",
        "python_abi": "cp310",
        "model_bundle_sha256": digest(bundle_sha256),
        "native_overlay_sha256": digest(overlay_sha256),
        "execution": EXECUTION,
        "entries": inventory(roots),
    }
    raw = canonical(document)
    if len(raw) > MAXIMUM_MANIFEST_BYTES:
        raise ValueError("runtime manifest exceeds size limit")
    with output.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o400)
        stream.write(raw)
    return hashlib.sha256(DOMAIN + raw).hexdigest()


def verify_runtime(roots: dict[str, Path], *, manifest: Path, expected_revision: str) -> dict:
    digest(expected_revision)
    checksum, size = file_hash(manifest, maximum_bytes=MAXIMUM_MANIFEST_BYTES)
    raw = manifest.read_bytes()
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != checksum:
        raise ValueError("runtime manifest changed during read")
    if hashlib.sha256(DOMAIN + raw).hexdigest() != expected_revision:
        raise ValueError("runtime manifest differs from reviewed release")
    document = json.loads(raw)
    if (
        not isinstance(document, dict)
        or document.get("schema") != "umi-community-native-runtime/1"
        or document.get("platform") != "macos/arm64"
        or document.get("python_abi") != "cp310"
        or document.get("execution") != EXECUTION
        or canonical(document) != raw
    ):
        raise ValueError("unsupported native runtime manifest")
    digest(document.get("model_bundle_sha256"))
    digest(document.get("native_overlay_sha256"))
    if inventory(roots) != document.get("entries"):
        raise ValueError("installed runtime differs from reviewed release")
    if manifest.read_bytes() != raw:
        raise ValueError("runtime manifest changed during verification")
    return document
