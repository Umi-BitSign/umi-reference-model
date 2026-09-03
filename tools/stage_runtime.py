from __future__ import annotations

import argparse
import os
import stat
import sys
from pathlib import Path, PurePosixPath

MAXIMUM_SOURCE_BYTES = 4 * 1024 * 1024


class StagingError(RuntimeError):
    pass


def _manifest_entries(path: Path) -> tuple[PurePosixPath, ...]:
    raw = path.read_text(encoding="utf-8")
    entries: list[PurePosixPath] = []
    seen: set[PurePosixPath] = set()
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        entry = PurePosixPath(line)
        if entry.is_absolute() or ".." in entry.parts or "." in entry.parts:
            raise StagingError(f"manifest line {line_number} is not a safe relative path")
        if entry in seen:
            raise StagingError(f"manifest line {line_number} is duplicated")
        seen.add(entry)
        entries.append(entry)
    if not entries:
        raise StagingError("runtime manifest is empty")
    return tuple(entries)


def _regular_payload(root: Path, relative: PurePosixPath) -> bytes:
    parent = root
    for part in relative.parts[:-1]:
        parent = parent / part
        try:
            parent_state = parent.lstat()
        except OSError as exc:
            raise StagingError(f"staging source is unavailable: {relative}") from exc
        if parent.is_symlink() or not stat.S_ISDIR(parent_state.st_mode):
            raise StagingError(f"staging source parent contains a symlink: {relative}")
    path = parent / relative.parts[-1]
    try:
        before = path.lstat()
    except OSError as exc:
        raise StagingError(f"staging source is unavailable: {relative}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or not 1 <= before.st_size <= MAXIMUM_SOURCE_BYTES
    ):
        raise StagingError(f"staging source violates its file contract: {relative}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StagingError(f"staging source escapes its repository: {relative}") from exc
    if resolved != path:
        raise StagingError(f"staging source parent contains a symlink: {relative}")
    payload = path.read_bytes()
    after = path.lstat()
    if len(payload) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise StagingError(f"staging source changed while being read: {relative}")
    return payload


def stage(
    source: Path,
    destination: Path,
    manifest: Path,
    *,
    check_only: bool,
    replace: bool,
) -> int:
    source_direct = Path(os.path.abspath(source))
    destination_direct = Path(os.path.abspath(destination))
    if source_direct.is_symlink() or destination_direct.is_symlink():
        raise StagingError("source and destination repository roots must not be symlinks")
    source_root = source_direct.resolve(strict=True)
    destination_root = destination_direct.resolve(strict=True)
    if source_root == destination_root:
        raise StagingError("source and destination repositories must differ")
    entries = _manifest_entries(manifest)
    payloads = [(entry, _regular_payload(source_root, entry)) for entry in entries]

    if check_only:
        for entry, payload in payloads:
            target = destination_root.joinpath(*entry.parts)
            if target.exists() or target.is_symlink():
                if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
                    raise StagingError(f"existing destination differs: {entry}")
        return len(payloads)

    for entry, payload in payloads:
        target = destination_root.joinpath(*entry.parts)
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        if target.parent.resolve(strict=True) != target.parent:
            raise StagingError(f"destination parent contains a symlink: {entry}")
        if target.exists() or target.is_symlink():
            if not replace:
                raise StagingError(f"destination already exists: {entry}")
            if target.is_symlink() or not target.is_file():
                raise StagingError(f"destination is not a regular file: {entry}")
        temporary = target.with_name(f".{target.name}.stage-{os.getpid()}")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o644,
            )
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise StagingError(f"write made no progress: {entry}")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.replace(temporary, target)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
    return len(payloads)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage the explicit public runtime allowlist")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--replace", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.check_only and arguments.replace:
        _parser().error("--check-only and --replace cannot be combined")
    try:
        count = stage(
            arguments.source,
            arguments.destination,
            arguments.manifest,
            check_only=arguments.check_only,
            replace=arguments.replace,
        )
    except (OSError, UnicodeError, StagingError) as exc:
        print(f"runtime staging failed: {exc}", file=sys.stderr)
        return 2
    action = "checked" if arguments.check_only else "staged"
    print(f"{action} {count} allowlisted runtime files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
