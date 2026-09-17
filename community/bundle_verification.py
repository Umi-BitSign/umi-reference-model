"""Read-only verification of the imported community bundle before model imports.

This verifies the existing offline_bundle/1 artifact identity, not a native
serving runtime, rights approval or model quality. The launcher must keep the
verified tree read-only to the model process for its entire lifetime. A local
checksum check cannot protect against a privileged host changing files later.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

MAXIMUM_FILES = 512
MAXIMUM_BYTES = 4 * 1024**3
MAXIMUM_MANIFEST_BYTES = 256 * 1024
DOMAIN = b"umi-open-competition-v1\0"
F_GETPATH = 50
ROLES = frozenset(
    {
        "weights",
        "inference",
        "config",
        "processor",
        "environment",
        "license",
        "dependency",
        "provenance",
    }
)


def _canonical(document: object) -> bytes:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _digest(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA-256 digest")
    return value


def _name(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 240:
        raise ValueError("invalid artifact path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or str(path) != value
        or not path.parts
        or any(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", p) is None for p in path.parts)
    ):
        raise ValueError("invalid artifact path")
    return value


def _identity(info):
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


def _require_canonical_directory(descriptor: int, requested: Path) -> None:
    """Reject spelling, case, link and normalization aliases for an opened root."""
    if not requested.is_absolute():
        raise ValueError("artifact root must be absolute")
    if platform.system() == "Darwin":
        raw = fcntl.fcntl(descriptor, F_GETPATH, b"\0" * 1024)
        actual = raw.split(b"\0", 1)[0]
        if not actual or actual != os.fsencode(requested):
            raise ValueError(
                "artifact root aliases are forbidden; use its canonical filesystem path"
            )
    elif os.fsencode(requested.resolve(strict=True)) != os.fsencode(requested):
        raise ValueError("artifact root aliases are forbidden; use its canonical filesystem path")


@contextmanager
def _directory(
    path: Path | str,
    *,
    parent: int | None = None,
    private: bool = False,
    canonical: Path | None = None,
):
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
    except OSError as error:
        if canonical is not None:
            raise ValueError(
                "artifact root aliases are forbidden; use its canonical filesystem path"
            ) from error
        raise
    try:
        before = os.fstat(descriptor)
        if canonical is not None:
            _require_canonical_directory(descriptor, canonical)
        forbidden = 0o077 if private else 0o022
        if before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) & forbidden:
            raise ValueError("artifact directory ownership or permissions differ")
        yield descriptor
        after = os.fstat(descriptor)
        linked = os.stat(path, dir_fd=parent, follow_symlinks=False)
        if _identity(before) != _identity(after) or _identity(after) != _identity(linked):
            raise ValueError("artifact directory changed during verification")
    finally:
        os.close(descriptor)


def _file(parent: int, name: str, *, size: int | None = None, expected: str | None = None):
    descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o222
        ):
            raise ValueError("artifact must be an owner-held, read-only, single-link regular file")
        ceiling = MAXIMUM_MANIFEST_BYTES if size is None else size
        if before.st_size > ceiling or (size is not None and before.st_size != size):
            raise ValueError("artifact size differs")
        total = 0
        checksum = hashlib.sha256()
        metadata = bytearray()
        while chunk := stream.read(min(1024 * 1024, ceiling + 1 - total)):
            total += len(chunk)
            if total > ceiling:
                raise ValueError("artifact grew during verification")
            checksum.update(chunk)
            if size is None:
                metadata.extend(chunk)
        after = os.fstat(stream.fileno())
        linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
        if (
            total != before.st_size
            or _identity(before) != _identity(after)
            or _identity(after) != _identity(linked)
        ):
            raise ValueError("artifact changed during verification")
        if expected is not None and checksum.hexdigest() != expected:
            raise ValueError("artifact checksum differs")
        return bytes(metadata)


def _records(raw: bytes) -> list[dict]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate manifest key")
            result[key] = value
        return result

    document = json.loads(raw, object_pairs_hook=pairs)
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "profile", "parent_baseline_sha256", "license_id", "files"}
        or document["schema"] != "umi-model-bundle/1"
        or document["profile"] != "offline_bundle/1"
        or document["parent_baseline_sha256"] is not None
        or document["license_id"] != "MIT"
        or _canonical(document) != raw
    ):
        raise ValueError("not the canonical imported-community bundle manifest")
    records = document["files"]
    if not isinstance(records, list) or not 1 <= len(records) <= MAXIMUM_FILES:
        raise ValueError("invalid artifact inventory size")
    names = []
    total = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "role", "sha256", "size_bytes"}:
            raise ValueError("invalid artifact record")
        names.append(_name(record["path"]))
        _digest(record["sha256"])
        if (
            not isinstance(record["role"], str)
            or record["role"] not in ROLES
            or type(record["size_bytes"]) is not int
            or not 0 <= record["size_bytes"] <= MAXIMUM_BYTES
        ):
            raise ValueError("invalid artifact role or size")
        total += record["size_bytes"]
    if (
        total > MAXIMUM_BYTES
        or names != sorted(names)
        or len(set(n.casefold() for n in names)) != len(names)
    ):
        raise ValueError("artifact inventory is oversized, unsorted or aliased")
    if [r["path"] for r in records if r["role"] == "inference"] != ["umi_inference.py"]:
        raise ValueError("imported-community entrypoint differs")
    folded = {name.casefold() for name in names}
    for name in names:
        if any(str(p).casefold() in folded for p in PurePosixPath(name).parents):
            raise ValueError("artifact path is also a directory")
    return records


def _tree(root: int, records: list[dict]) -> None:
    files = {record["path"]: record for record in records}
    directories = {str(p) for name in files for p in PurePosixPath(name).parents if str(p) != "."}
    seen = set()

    def walk(descriptor, prefix):
        with os.scandir(descriptor) as entries:
            for entry in entries:
                name = prefix + entry.name
                if name in directories and entry.is_dir(follow_symlinks=False):
                    with _directory(entry.name, parent=descriptor) as child:
                        walk(child, name + "/")
                elif name in files and entry.is_file(follow_symlinks=False):
                    record = files[name]
                    _file(
                        descriptor, entry.name, size=record["size_bytes"], expected=record["sha256"]
                    )
                    seen.add(name)
                else:
                    raise ValueError("undeclared artifact, directory, link or special file")

    walk(root, "")
    if seen != set(files):
        raise ValueError("missing declared artifact")


def verify_imported_bundle(root: Path, *, expected_bundle_sha256: str) -> dict:
    """Verify the caller-pinned manifest and exact model tree without imports.

    ``root`` is the trusted operator-selected staging directory. It contains
    manifest.json and model/. Ancestors outside this directory are part of the
    launcher's trust boundary. intake.json is a report, never an authority input.
    No file is created, chmodded, removed or repaired by verification.
    """
    _digest(expected_bundle_sha256)
    if not root.is_absolute():
        raise ValueError("bundle root must be an absolute directory without aliases")
    with _directory(root, private=True, canonical=root) as bundle_fd:
        raw = _file(bundle_fd, "manifest.json")
        revision = hashlib.sha256(DOMAIN + raw).hexdigest()
        if revision != expected_bundle_sha256:
            raise ValueError("bundle manifest does not match the caller-pinned identity")
        records = _records(raw)
        with _directory("model", parent=bundle_fd) as model_fd:
            _tree(model_fd, records)
        # A manifest replaced during the tree scan must not authorize readiness.
        if _file(bundle_fd, "manifest.json") != raw:
            raise ValueError("bundle manifest changed during verification")
    return {
        "model_bundle_sha256": revision,
        "manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
        "files": len(records),
        "bytes": sum(record["size_bytes"] for record in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--expected-bundle-sha256", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_imported_bundle(args.bundle, expected_bundle_sha256=args.expected_bundle_sha256)
        )
    )


if __name__ == "__main__":
    main()
