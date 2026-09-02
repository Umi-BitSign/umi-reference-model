from __future__ import annotations

import hashlib
import re
import stat
import subprocess
import sys
from pathlib import Path

MAXIMUM_GIT_FILE_BYTES = 16 * 1024 * 1024
FORBIDDEN_SUFFIXES = {
    ".docker.tar",
    ".docker.tar.zst",
    ".mov",
    ".mp4",
    ".mkv",
    ".npz",
    ".parquet",
    ".safetensors",
    ".task",
    ".tar.gz",
    ".zip",
}
SECRET_NAME = re.compile(
    r"(?:^|[._-])(?:credential|credentials|private[-_]?key|secret|seed|wallet)(?:[._-]|$)",
    re.IGNORECASE,
)
REQUIRED_DIGESTS = {
    "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "licenses/CC-BY-4.0.txt": (
        "9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94"
    ),
    "licenses/CC-BY-SA-4.0.txt": (
        "23ee78c8bae49cf08ea2f0c84945c66b987ebe4520881fb51b3dad4fb43d07c2"
    ),
    "licenses/FSBOARD-SOURCE-NOTICE.txt": (
        "96218feb836f9005261b0346501f349d11e4019deb6bbfce899483d94639c7dd"
    ),
}
FORBIDDEN_TEXT = (
    b"This repository and its contents are private and " + b"proprietary",
    b"BEGIN OPENSSH " + b"PRIVATE KEY",
    b"BEGIN " + b"PRIVATE KEY",
)


class RepositoryGuardError(RuntimeError):
    pass


def _forbidden_suffix(name: str) -> bool:
    lowered = name.lower()
    return any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)


def check_repository(root: Path) -> tuple[int, int]:
    repository = root.resolve(strict=True)
    if repository.is_symlink() or not repository.is_dir():
        raise RepositoryGuardError("repository root must be a direct directory")
    checked_files = 0
    checked_bytes = 0
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RepositoryGuardError("cannot enumerate the candidate Git tree") from exc
    raw_paths = result.stdout.split(b"\0")
    if raw_paths and raw_paths[-1] == b"":
        raw_paths.pop()
    for encoded in raw_paths:
        try:
            relative = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RepositoryGuardError("candidate Git path is not UTF-8") from exc
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RepositoryGuardError("candidate Git path escapes the repository")
        path = repository / candidate
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise RepositoryGuardError(f"repository contains a non-regular file: {relative}")
        if metadata.st_size > MAXIMUM_GIT_FILE_BYTES:
            raise RepositoryGuardError(f"repository file exceeds 16 MiB: {relative}")
        if _forbidden_suffix(path.name):
            raise RepositoryGuardError(f"repository contains a forbidden artifact: {relative}")
        if SECRET_NAME.search(path.name):
            raise RepositoryGuardError(f"repository contains a secret-like filename: {relative}")
        payload = path.read_bytes()
        if any(marker in payload for marker in FORBIDDEN_TEXT):
            raise RepositoryGuardError(
                f"repository contains forbidden private material: {relative}"
            )
        checked_files += 1
        checked_bytes += len(payload)
    for relative, expected in REQUIRED_DIGESTS.items():
        path = repository / relative
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RepositoryGuardError(f"required license file is unavailable: {relative}") from exc
        if hashlib.sha256(payload).hexdigest() != expected:
            raise RepositoryGuardError(f"required license digest differs: {relative}")
    return checked_files, checked_bytes


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) == 2 else Path.cwd()
    if len(sys.argv) > 2:
        print("usage: python tools/repo_guard.py [repository]", file=sys.stderr)
        return 2
    try:
        files, size = check_repository(root)
    except (OSError, RepositoryGuardError) as exc:
        print(f"repository guard failed: {exc}", file=sys.stderr)
        return 2
    print(f"repository guard passed: {files} files, {size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
