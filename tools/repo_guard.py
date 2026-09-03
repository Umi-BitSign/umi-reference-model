from __future__ import annotations

import hashlib
import re
import stat
import subprocess
import sys
from pathlib import Path

MAXIMUM_GIT_FILE_BYTES = 16 * 1024 * 1024
MAXIMUM_RELEASE_MODEL_BYTES = 128 * 1024 * 1024
RELEASE_MODEL_PATHS = frozenset(
    {
        "release/umi-s1-baseline-v0-portable.zip",
        "release/umi-s1-public-finetune-v1-portable.zip",
    }
)
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
    "licenses/CC-BY-4.0.txt": ("9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94"),
    "licenses/CC-BY-SA-4.0.txt": (
        "23ee78c8bae49cf08ea2f0c84945c66b987ebe4520881fb51b3dad4fb43d07c2"
    ),
    "licenses/FSBOARD-SOURCE-NOTICE.txt": (
        "96218feb836f9005261b0346501f349d11e4019deb6bbfce899483d94639c7dd"
    ),
    "NOTICE": "74be8e3f4f30280f77ee026268ef81c6d7acc450100120731235a994e3338935",
    "licenses/FLEURS-ATTRIBUTION.txt": (
        "e9e6b293c5e2058d969561e8ac164add7fe1d4221e049de971ae32056e9b8a51"
    ),
    "licenses/FSBOARD-ATTRIBUTION.txt": (
        "a042e85d40b25f7171c3e4cbd742cf8852361c71e72203f66e0947f0440bff21"
    ),
    "release/LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
    "release/NOTICE": "74be8e3f4f30280f77ee026268ef81c6d7acc450100120731235a994e3338935",
    "release/CC-BY-SA-4.0.txt": (
        "23ee78c8bae49cf08ea2f0c84945c66b987ebe4520881fb51b3dad4fb43d07c2"
    ),
    "release/2M-FLORES-ASL-LICENSE.txt": (
        "23ee78c8bae49cf08ea2f0c84945c66b987ebe4520881fb51b3dad4fb43d07c2"
    ),
    "release/2M-FLORES-ASL-ATTRIBUTION.txt": (
        "541c723652f261dbf9cc8bbe812476a5e821348f7f73d421415ba99d027e7640"
    ),
    "release/FLEURS-ASL-LICENSE.txt": (
        "23ee78c8bae49cf08ea2f0c84945c66b987ebe4520881fb51b3dad4fb43d07c2"
    ),
    "release/FLEURS-ASL-ATTRIBUTION.txt": (
        "df4d82cf63e09d0513ac08ab4ff01141583ea7774b6abfe2bad2a99c6b8bebe3"
    ),
    "release/FSBOARD-LICENSE.txt": (
        "96218feb836f9005261b0346501f349d11e4019deb6bbfce899483d94639c7dd"
    ),
    "release/FSBOARD-ATTRIBUTION.txt": (
        "0cfe5b50a410a6fd9d2a6101de37cc9c0a98e1478761b1bf4e5acf9d0a7cd88f"
    ),
    "release/TASKMASTER-LICENSE.txt": (
        "05e1bc05724716ca7905f567a2ec30d99c8d2f5f58e926c2ca7d9da116002446"
    ),
    "release/TASKMASTER-ATTRIBUTION.txt": (
        "0bff2f85846a14f4c85088ab1f70d3ace952926136c98d7592ac9a10b9bf11e5"
    ),
}
FORBIDDEN_TEXT = (
    b"This repository and its contents are private and " + b"proprietary",
    b"BEGIN OPENSSH " + b"PRIVATE KEY",
    b"BEGIN " + b"PRIVATE KEY",
)


class RepositoryGuardError(RuntimeError):
    pass


def _forbidden_suffix(relative: str) -> bool:
    if relative in RELEASE_MODEL_PATHS:
        return False
    lowered = relative.lower()
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
        maximum_bytes = (
            MAXIMUM_RELEASE_MODEL_BYTES
            if relative in RELEASE_MODEL_PATHS
            else MAXIMUM_GIT_FILE_BYTES
        )
        if metadata.st_size > maximum_bytes:
            raise RepositoryGuardError(
                f"repository file exceeds its {maximum_bytes}-byte ceiling: {relative}"
            )
        if _forbidden_suffix(relative):
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
