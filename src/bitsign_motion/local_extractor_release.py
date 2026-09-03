from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final, cast

from . import amd64_holistic_container as extractor
from .canonical import canonical_json_bytes, canonical_json_sha256

LOCAL_EXTRACTOR_SCHEMA: Final = "umi-local-extractor-build/1"
LOCAL_EXTRACTOR_STATUS: Final = "component_test_no_weight"
LOCAL_EXTRACTOR_PLATFORM: Final = "linux/amd64"

_CONTENT_DOMAIN = b"umi-local-extractor-build-v1\0"
_PACKAGE_LINE = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^ \\]+) \\")
_HASH_LINE = re.compile(r"--hash=sha256:[0-9a-f]{64}")
_MAXIMUM_SOURCE_BYTES: Final = 4 * 1024 * 1024
_MAXIMUM_RECORD_BYTES: Final = 1024 * 1024
_CLAIM_BOUNDARY: Final = (
    "This Linux/AMD64 extractor was built and checked on the operator's machine. Its immutable "
    "image ID is locally bound into a derived model bundle. No byte-equivalence claim is made "
    "against an image built on another machine."
)


class LocalExtractorReleaseError(RuntimeError):
    """Raised when a local extractor build cannot be bound safely."""


def _source_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[2]


def _read_regular(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    try:
        before = path.lstat()
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise LocalExtractorReleaseError(f"{label} is unavailable") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or not 1 <= len(payload) <= maximum_bytes
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise LocalExtractorReleaseError(f"{label} violates its file contract")
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_record() -> dict[str, str]:
    root = _source_root()
    source_paths = {
        "amd64_container_host_source_sha256": root
        / "src"
        / "bitsign_motion"
        / "amd64_holistic_container.py",
        "dockerfile_sha256": root / "docker" / "mediapipe-holistic" / "Dockerfile.amd64",
        "requirements_sha256": root / "docker" / "mediapipe-holistic" / "requirements.amd64.lock",
        "worker_sha256": root / "docker" / "mediapipe-holistic" / "worker.amd64.py",
    }
    return {
        name: _sha256(_read_regular(path, maximum_bytes=_MAXIMUM_SOURCE_BYTES, label=name))
        for name, path in source_paths.items()
    }


def _locked_packages() -> dict[str, str]:
    lock_path = _source_root() / "docker" / "mediapipe-holistic" / "requirements.amd64.lock"
    payload = _read_regular(
        lock_path,
        maximum_bytes=_MAXIMUM_SOURCE_BYTES,
        label="AMD64 requirements lock",
    )
    try:
        lines = payload.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise LocalExtractorReleaseError("AMD64 requirements lock is not ASCII") from exc
    packages: dict[str, str] = {}
    hash_count = 0
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        package_match = _PACKAGE_LINE.fullmatch(line)
        if package_match is not None:
            name, version = package_match.groups()
            normalized = name.lower().replace("_", "-")
            if normalized in packages:
                raise LocalExtractorReleaseError("AMD64 requirements lock repeats a package")
            packages[normalized] = version
            continue
        if _HASH_LINE.fullmatch(line) is not None:
            hash_count += 1
            continue
        raise LocalExtractorReleaseError("AMD64 requirements lock has an unsupported line")
    if not packages or hash_count < len(packages):
        raise LocalExtractorReleaseError("AMD64 requirements lock is incomplete")
    return dict(sorted(packages.items()))


def _docker_executable(path: Path) -> Path:
    direct = Path(os.path.abspath(path))
    try:
        resolved = direct.resolve(strict=True)
    except OSError as exc:
        raise LocalExtractorReleaseError("Docker executable is unavailable") from exc
    resolved_metadata = resolved.stat()
    if not stat.S_ISREG(resolved_metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise LocalExtractorReleaseError("Docker executable must resolve to an executable file")
    return resolved


def _container_check_command(image_id: str, expected_packages: dict[str, str]) -> list[str]:
    package_names = sorted(expected_packages)
    program = (
        "import importlib.metadata as m,json,subprocess;"
        f"names={package_names!r};"
        "packages={name:m.version(name) for name in names};"
        "ffmpeg=subprocess.run(['ffmpeg','-version'],stdin=subprocess.DEVNULL,"
        "capture_output=True,text=True,timeout=15,check=True).stdout.splitlines()[0];"
        "print(json.dumps({'ffmpeg':ffmpeg,'packages':packages},sort_keys=True,"
        "separators=(',',':')))"
    )
    return [
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "64",
        "--memory",
        "1g",
        "--memory-swap",
        "1g",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=64m",
        "--entrypoint",
        "python",
        image_id,
        "-c",
        program,
    ]


def _dependency_check(
    image_id: str,
    *,
    docker_executable: Path,
    runner: Callable[..., Any],
) -> dict[str, Any]:
    expected_packages = _locked_packages()
    command = [str(docker_executable), *_container_check_command(image_id, expected_packages)]
    try:
        result = runner(
            command,
            timeout_seconds=90.0,
            stdout_limit=128 * 1024,
            stderr_limit=extractor.MAXIMUM_DOCKER_DIAGNOSTIC_BYTES,
        )
    except Exception as exc:
        raise LocalExtractorReleaseError("local extractor package check failed") from exc
    if result.return_code != 0:
        raise LocalExtractorReleaseError("local extractor package check failed")
    try:
        observed = json.loads(result.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LocalExtractorReleaseError("local extractor package report is invalid") from exc
    expected_ffmpeg = f"ffmpeg version {extractor.EXPECTED_FFMPEG_VERSION}"
    if (
        not isinstance(observed, dict)
        or set(observed) != {"ffmpeg", "packages"}
        or not isinstance(observed["ffmpeg"], str)
        or not observed["ffmpeg"].startswith(expected_ffmpeg)
        or observed["packages"] != expected_packages
    ):
        raise LocalExtractorReleaseError("local extractor package versions differ")
    pip_command = [*command[:-2], "-m", "pip", "check"]
    try:
        pip_result = runner(
            pip_command,
            timeout_seconds=90.0,
            stdout_limit=128 * 1024,
            stderr_limit=extractor.MAXIMUM_DOCKER_DIAGNOSTIC_BYTES,
        )
    except Exception as exc:
        raise LocalExtractorReleaseError(
            "local extractor package consistency check failed"
        ) from exc
    if pip_result.return_code != 0 or pip_result.stdout.strip() != b"No broken requirements found.":
        raise LocalExtractorReleaseError("local extractor package consistency check failed")
    return {
        "ffmpeg": extractor.EXPECTED_FFMPEG_VERSION,
        "packages": expected_packages,
        "pip_check": "passed",
    }


def validate_local_extractor(
    image_reference: str,
    *,
    docker_executable: Path,
    runner: Callable[..., Any] = extractor._run_bounded_command,
) -> tuple[extractor.HolisticContainerImage, dict[str, Any]]:
    """Inspect an immutable local image and execute its pinned dependency check."""

    docker = _docker_executable(docker_executable)
    try:
        image = extractor.resolve_holistic_container_image(
            image_reference,
            platform=LOCAL_EXTRACTOR_PLATFORM,
            docker_executable=str(docker),
            timeout_seconds=30.0,
            _runner=runner,
        )
    except extractor.HolisticContainerError as exc:
        raise LocalExtractorReleaseError("local extractor image validation failed") from exc
    dependencies = _dependency_check(
        image.image_id,
        docker_executable=docker,
        runner=runner,
    )
    return image, dependencies


def build_local_extractor(
    *,
    docker_executable: Path,
    image_tag: str,
    timeout_seconds: float,
    builder: Callable[..., extractor.HolisticContainerImage] = (
        extractor.build_holistic_container_image
    ),
    runner: Callable[..., Any] = extractor._run_bounded_command,
) -> dict[str, Any]:
    """Build the checked-in Linux/AMD64 source and return its immutable authority."""

    docker = _docker_executable(docker_executable)
    try:
        built = builder(
            image_tag=image_tag,
            platform=LOCAL_EXTRACTOR_PLATFORM,
            docker_executable=str(docker),
            timeout_seconds=timeout_seconds,
            _runner=runner,
        )
    except extractor.HolisticContainerError as exc:
        raise LocalExtractorReleaseError("local extractor source build failed") from exc
    image, dependencies = validate_local_extractor(
        built.image_id,
        docker_executable=docker,
        runner=runner,
    )
    if image.image_id != built.image_id:
        raise LocalExtractorReleaseError("local extractor image changed after its build")
    record: dict[str, Any] = {
        "schema": LOCAL_EXTRACTOR_SCHEMA,
        "status": LOCAL_EXTRACTOR_STATUS,
        "platform": LOCAL_EXTRACTOR_PLATFORM,
        "image_id": image.image_id,
        "dependencies": dependencies,
        "sources": _source_record(),
        "claim_boundary": _CLAIM_BOUNDARY,
    }
    record["content_sha256"] = canonical_json_sha256(record, domain=_CONTENT_DOMAIN)
    return record


def validate_local_extractor_record(
    value: object,
    *,
    docker_executable: Path,
    runner: Callable[..., Any] = extractor._run_bounded_command,
) -> dict[str, Any]:
    """Validate a build record against checked-in sources and the current Docker image."""

    if not isinstance(value, dict) or set(value) != {
        "schema",
        "status",
        "platform",
        "image_id",
        "dependencies",
        "sources",
        "claim_boundary",
        "content_sha256",
    }:
        raise LocalExtractorReleaseError("local extractor record field set differs")
    record = cast(dict[str, Any], value)
    supplied_digest = record["content_sha256"]
    unsigned = dict(record)
    del unsigned["content_sha256"]
    if (
        record["schema"] != LOCAL_EXTRACTOR_SCHEMA
        or record["status"] != LOCAL_EXTRACTOR_STATUS
        or record["platform"] != LOCAL_EXTRACTOR_PLATFORM
        or record["claim_boundary"] != _CLAIM_BOUNDARY
        or not isinstance(supplied_digest, str)
        or supplied_digest != canonical_json_sha256(unsigned, domain=_CONTENT_DOMAIN)
        or record["sources"] != _source_record()
    ):
        raise LocalExtractorReleaseError("local extractor record identity differs")
    image_id = record["image_id"]
    if not isinstance(image_id, str):
        raise LocalExtractorReleaseError("local extractor record has no image ID")
    image, dependencies = validate_local_extractor(
        image_id,
        docker_executable=docker_executable,
        runner=runner,
    )
    if image.image_id != image_id or record["dependencies"] != dependencies:
        raise LocalExtractorReleaseError("local extractor image differs from its build record")
    return record


def _write_record(path: Path, record: dict[str, Any]) -> None:
    destination = Path(os.path.abspath(path))
    parent = destination.parent.resolve(strict=True)
    if destination.parent != parent or destination.exists() or destination.is_symlink():
        raise LocalExtractorReleaseError("build-record destination must be a new direct path")
    payload = canonical_json_bytes(record)
    if len(payload) > _MAXIMUM_RECORD_BYTES:
        raise LocalExtractorReleaseError("local extractor record exceeds its byte ceiling")
    temporary = destination.with_name(f".{destination.name}.build-{os.getpid()}")
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise LocalExtractorReleaseError("build-record write made no progress")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate the component-test Linux/AMD64 extractor locally"
    )
    parser.add_argument("--docker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--image-tag",
        default="umi-reference-mediapipe-holistic:1.0.3-amd64-local",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    arguments = parser.parse_args(argv)
    try:
        if not 60 <= arguments.timeout_seconds <= 7200:
            raise LocalExtractorReleaseError("build timeout must be from 60 through 7200 seconds")
        record = build_local_extractor(
            docker_executable=arguments.docker,
            image_tag=arguments.image_tag,
            timeout_seconds=arguments.timeout_seconds,
        )
        _write_record(arguments.output, record)
    except (OSError, LocalExtractorReleaseError) as exc:
        print(f"local extractor build failed: {exc}", file=sys.stderr)
        return 2
    print(canonical_json_bytes(record).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
