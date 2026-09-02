from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import selectors
import shutil
import stat
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np

from .canonical import canonical_json_bytes
from .holistic_container import HolisticExtractionResult

RAW_SCHEMA: Final = "umi-raw-holistic-landmarks/1"
RECEIPT_SCHEMA: Final = "umi-raw-holistic-extraction-receipt/3"
COMPLETION_SCHEMA: Final = "umi-raw-holistic-extraction-completion/1"
DEFAULT_IMAGE_TAG: Final = "bitsign-motion-mediapipe-holistic:1.0.3"
AMD64_IMAGE_TAG: Final = "bitsign-motion-mediapipe-holistic:1.0.3-amd64"
IMAGE_BUILD_SOURCE_DATE_EPOCH: Final = "1788307200"
EXPECTED_BASE_IMAGE_DIGEST: Final = (
    "sha256:9bb659dc6d5218917236f3711e866a5634bb4c2f208de9d4533aa4863f57c1d3"
)
EXPECTED_AMD64_BASE_IMAGE_DIGEST: Final = (
    "sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49"
)
SUPPORTED_CONTAINER_PLATFORMS: Final = frozenset({"linux/arm64", "linux/amd64"})
EXPECTED_MEDIAPIPE_VERSION: Final = "1.0.1"
EXPECTED_FFMPEG_VERSION: Final = "5.1.9-0+deb12u1"
EXPECTED_MODEL_SHA256: Final = "e2dab61191e2dcd0a15f943d8e3ed1dce13c82dfa597b9dd39f562975a50c3f8"
MAXIMUM_SOURCE_VIDEO_BYTES: Final = 256 * 1024 * 1024
MAXIMUM_UMI_VIDEO_BYTES: Final = 16 * 1024 * 1024
MAXIMUM_MODEL_BYTES: Final = 128 * 1024 * 1024
MAXIMUM_RAW_ARTIFACT_BYTES: Final = 32 * 1024 * 1024
MAXIMUM_RECEIPT_BYTES: Final = 2 * 1024 * 1024
MAXIMUM_COMPLETION_BYTES: Final = 16 * 1024
MAXIMUM_DOCKER_DIAGNOSTIC_BYTES: Final = 256 * 1024
MINIMUM_CLIP_US: Final = 2_000_000
MAXIMUM_CLIP_US: Final = 15_000_000
TARGET_RATE: Final = 8
MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE: Final = Fraction(60, 1)
MAXIMUM_BOOTSTRAP_CANDIDATE_FRAMES: Final = 902
MAXIMUM_SOURCE_LONG_SIDE: Final = 3840
MAXIMUM_SOURCE_SHORT_SIDE: Final = 2160
MAXIMUM_DERIVED_LONG_SIDE: Final = 1280
MAXIMUM_DERIVED_SHORT_SIDE: Final = 720
SCALE_FLAGS: Final = "bilinear+accurate_rnd+full_chroma_int+bitexact"
ALLOWED_MP4_MAJOR_BRANDS: Final = frozenset(
    {"avc1", "dash", "iso2", "iso3", "iso4", "iso5", "iso6", "isom", "M4V ", "mp41", "mp42"}
)
ALLOWED_BOOTSTRAP_VIDEO_PROFILES: Final = frozenset(
    {
        ("h264", "High", "yuv420p"),
        ("h264", "High 10", "yuv420p10le"),
        ("hevc", "Main 10", "yuv420p10le"),
    }
)
ALLOWED_BOOTSTRAP_AUDIO_CODECS: Final = frozenset({"aac"})
MAXIMUM_SAMPLE_ASPECT_RATIO_DEVIATION: Final = Fraction(1, 100)
SAFE_SAMPLE_ID: Final = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
IMAGE_ID: Final = re.compile(r"sha256:[0-9a-f]{64}")
SHA256: Final = re.compile(r"[0-9a-f]{64}")
CONTAINER_NAME: Final = re.compile(r"bitsign-holistic-[0-9a-f]{32}")
RAW_FILENAME: Final = "raw-holistic.npz"
RECEIPT_FILENAME: Final = "receipt.json"
RAW_TRACK_PREFIX_ORDER: Final = (
    "pose_image",
    "observed_hand_image_0",
    "observed_hand_image_1",
    "face_image",
    "pose_world",
    "observed_hand_world_0",
    "observed_hand_world_1",
)
TRACK_SPECS: Final = {
    "pose_image": 33,
    "pose_world": 33,
    "observed_hand_image_0": 21,
    "observed_hand_image_1": 21,
    "observed_hand_world_0": 21,
    "observed_hand_world_1": 21,
    "face_image": 478,
}
TRACK_FIELDS: Final = {
    "coordinates": ("<f4", lambda frames, points: [frames, points, 3]),
    "coordinate_available": ("|b1", lambda frames, points: [frames, points]),
    "presence": ("<f4", lambda frames, points: [frames, points]),
    "presence_available": ("|b1", lambda frames, points: [frames, points]),
    "visibility": ("<f4", lambda frames, points: [frames, points]),
    "visibility_available": ("|b1", lambda frames, points: [frames, points]),
}


class HolisticContainerError(RuntimeError):
    """A contained extraction failure that never includes child-process output."""


class _BoundedCommandFailure(RuntimeError):
    def __init__(self, reason: str, *, return_code: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.return_code = return_code


@dataclass(frozen=True, slots=True)
class _CommandResult:
    return_code: int
    stdout: bytes
    stderr: bytes


CommandRunner = Callable[..., _CommandResult]


@dataclass(frozen=True, slots=True)
class HolisticContainerImage:
    requested_reference: str
    image_id: str
    platform: str


def _run_bounded_command(
    command: list[str],
    *,
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
) -> _CommandResult:
    if (
        not command
        or type(timeout_seconds) not in (int, float)
        or timeout_seconds <= 0
        or type(stdout_limit) is not int
        or stdout_limit < 0
        or type(stderr_limit) is not int
        or stderr_limit < 0
    ):
        raise ValueError("bounded command arguments are invalid")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
        )
    except OSError as exc:
        raise _BoundedCommandFailure("start") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    selected = selectors.DefaultSelector()
    selected.register(process.stdout, selectors.EVENT_READ, ("stdout", stdout_limit))
    selected.register(process.stderr, selectors.EVENT_READ, ("stderr", stderr_limit))
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + float(timeout_seconds)
    try:
        while selected.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _BoundedCommandFailure("timeout")
            events = selected.select(min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = [(key, selectors.EVENT_READ) for key in selected.get_map().values()]
            for key, _ in events:
                name, limit = key.data
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selected.unregister(key.fileobj)
                    continue
                buffers[name].extend(chunk)
                if len(buffers[name]) > limit:
                    raise _BoundedCommandFailure("output-limit")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _BoundedCommandFailure("timeout")
        return_code = process.wait(timeout=remaining)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        raise _BoundedCommandFailure("timeout") from exc
    except _BoundedCommandFailure:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        raise
    except BaseException:
        process.kill()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        raise
    finally:
        selected.close()
        process.stdout.close()
        process.stderr.close()
    return _CommandResult(
        return_code=return_code,
        stdout=bytes(buffers["stdout"]),
        stderr=bytes(buffers["stderr"]),
    )


def _strict_json_bytes(raw: bytes, *, label: str, maximum_bytes: int) -> dict[str, Any]:
    if not raw or len(raw) > maximum_bytes:
        raise HolisticContainerError(f"{label} violates its byte ceiling")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise HolisticContainerError(f"{label} contains duplicate object members")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                HolisticContainerError(f"{label} contains a non-finite number: {token}")
            ),
        )
    except UnicodeDecodeError as exc:
        raise HolisticContainerError(f"{label} is not UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise HolisticContainerError(f"{label} is not JSON") from exc
    if not isinstance(value, dict):
        raise HolisticContainerError(f"{label} must be a JSON object")
    return value


def _file_sha256(path: Path, *, maximum_bytes: int) -> tuple[str, int]:
    try:
        initial = path.lstat()
    except OSError as exc:
        raise HolisticContainerError("required file is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(initial.st_mode):
        raise HolisticContainerError("required file must be a regular non-symlink file")
    if initial.st_size <= 0 or initial.st_size > maximum_bytes:
        raise HolisticContainerError("required file violates its byte ceiling")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HolisticContainerError("required file could not be opened safely") from exc
    digest = hashlib.sha256()
    observed = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev != initial.st_dev or opened.st_ino != initial.st_ino
        ):
            raise HolisticContainerError("required file changed before hashing")
        while True:
            read_size = min(1024 * 1024, maximum_bytes + 1 - observed)
            chunk = os.read(descriptor, read_size)
            if not chunk:
                break
            observed += len(chunk)
            if observed > maximum_bytes:
                raise HolisticContainerError("required file exceeded its byte ceiling")
            digest.update(chunk)
        final = os.fstat(descriptor)
        if final.st_size != observed or final.st_size != opened.st_size:
            raise HolisticContainerError("required file changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), observed


def _read_file_pinned(path: Path, *, maximum_bytes: int) -> tuple[bytes, str]:
    try:
        initial = path.lstat()
    except OSError as exc:
        raise HolisticContainerError("required bounded file is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(initial.st_mode):
        raise HolisticContainerError("required bounded file must be regular and non-symlink")
    if initial.st_size <= 0 or initial.st_size > maximum_bytes:
        raise HolisticContainerError("required bounded file violates its byte ceiling")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HolisticContainerError("required bounded file could not be opened safely") from exc
    payload = bytearray()
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev != initial.st_dev or opened.st_ino != initial.st_ino
        ):
            raise HolisticContainerError("required bounded file changed before reading")
        while True:
            chunk = os.read(descriptor, min(65536, maximum_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
            if len(payload) > maximum_bytes:
                raise HolisticContainerError("required bounded file exceeded its byte ceiling")
            digest.update(chunk)
        final = os.fstat(descriptor)
        if final.st_size != len(payload) or final.st_size != opened.st_size:
            raise HolisticContainerError("required bounded file changed while reading")
    finally:
        os.close(descriptor)
    return bytes(payload), digest.hexdigest()


def _safe_mount_path(path: Path, *, maximum_bytes: int) -> tuple[Path, str, int]:
    try:
        original = Path(path)
        if original.is_symlink():
            raise HolisticContainerError("bind-mounted files must not be symlinks")
        resolved = original.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HolisticContainerError("bind-mounted file cannot be resolved") from exc
    if any(character in str(resolved) for character in (",", "\n", "\r", "\0")):
        raise HolisticContainerError("bind-mounted path contains a Docker mount delimiter")
    digest, byte_count = _file_sha256(resolved, maximum_bytes=maximum_bytes)
    return resolved, digest, byte_count


def _prepare_output_root(
    path: Path, *, sample_id: str, allow_existing_sample: bool = False
) -> tuple[Path, bool]:
    requested = Path(path)
    if any(character in str(requested) for character in (",", "\n", "\r", "\0")):
        raise HolisticContainerError("output path contains a Docker mount delimiter")
    try:
        requested.mkdir(mode=0o700, parents=True, exist_ok=True)
        resolved = requested.resolve(strict=True)
        metadata = resolved.lstat()
    except OSError as exc:
        raise HolisticContainerError("private output directory cannot be prepared") from exc
    if resolved.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise HolisticContainerError("output path must be a non-symlink directory")
    if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
        raise HolisticContainerError("output directory must be owned by the caller and mode 0700")
    final = resolved / sample_id
    try:
        final.lstat()
    except FileNotFoundError:
        exists = False
    except OSError as exc:
        raise HolisticContainerError("sample output cannot be inspected") from exc
    else:
        exists = True
    if exists and not allow_existing_sample:
        raise HolisticContainerError("sample output already exists; extraction is create-once")
    return resolved, exists


def _create_isolated_output_mount(output_root: Path, *, sample_id: str) -> Path:
    invocation_root = output_root / f".{sample_id}.container-{uuid.uuid4().hex}"
    try:
        os.mkdir(invocation_root, 0o700)
        metadata = invocation_root.lstat()
    except OSError as exc:
        raise HolisticContainerError("isolated container output mount cannot be created") from exc
    if (
        invocation_root.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise HolisticContainerError("isolated container output mount is not owner-only")
    return invocation_root


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform == "darwin":
        renamex_np = getattr(library, "renamex_np", None)
        if renamex_np is None:
            raise OSError(errno.ENOTSUP, "renamex_np is unavailable")
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(encoded_source, encoded_destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        renameat2 = getattr(library, "renameat2", None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, "renameat2 is unavailable")
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, encoded_source, -100, encoded_destination, 1)
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unsupported")
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _runner_call(
    runner: CommandRunner,
    command: list[str],
    *,
    timeout_seconds: float,
    stdout_limit: int,
    stderr_limit: int,
) -> _CommandResult:
    return runner(
        command,
        timeout_seconds=timeout_seconds,
        stdout_limit=stdout_limit,
        stderr_limit=stderr_limit,
    )


def _inspect_image(
    image_reference: str,
    *,
    expected_platform: str,
    docker_executable: str,
    timeout_seconds: float,
    runner: CommandRunner,
) -> HolisticContainerImage:
    try:
        result = _runner_call(
            runner,
            [docker_executable, "image", "inspect", "--format", "{{json .}}", image_reference],
            timeout_seconds=timeout_seconds,
            stdout_limit=512 * 1024,
            stderr_limit=MAXIMUM_DOCKER_DIAGNOSTIC_BYTES,
        )
    except _BoundedCommandFailure as exc:
        raise HolisticContainerError("local Holistic image inspection failed") from exc
    if result.return_code != 0:
        raise HolisticContainerError("local Holistic image is unavailable")
    value = _strict_json_bytes(
        result.stdout.strip(), label="Docker image inspection", maximum_bytes=512 * 1024
    )
    image_id = value.get("Id")
    architecture = value.get("Architecture")
    operating_system = value.get("Os")
    config = value.get("Config")
    if not isinstance(image_id, str) or IMAGE_ID.fullmatch(image_id) is None:
        raise HolisticContainerError("Docker image does not expose an immutable local image ID")
    if expected_platform not in SUPPORTED_CONTAINER_PLATFORMS:
        raise HolisticContainerError("Holistic container platform is unsupported")
    expected_architecture = expected_platform.removeprefix("linux/")
    if architecture != expected_architecture or operating_system != "linux":
        raise HolisticContainerError(f"Holistic image must be {expected_platform}")
    if not isinstance(config, dict):
        raise HolisticContainerError("Docker image configuration is missing")
    configured_user = config.get("User")
    if not isinstance(configured_user, str) or configured_user in ("", "0", "root", "0:0"):
        raise HolisticContainerError("Holistic image must declare a non-root user")
    labels = config.get("Labels")
    if not isinstance(labels, dict):
        raise HolisticContainerError("Holistic image labels are missing")
    expected_base_digest = (
        EXPECTED_BASE_IMAGE_DIGEST
        if expected_platform == "linux/arm64"
        else EXPECTED_AMD64_BASE_IMAGE_DIGEST
    )
    expected_labels = {
        "org.opencontainers.image.base.digest": expected_base_digest,
        "org.bitsign.mediapipe.version": EXPECTED_MEDIAPIPE_VERSION,
        "org.bitsign.ffmpeg.version": EXPECTED_FFMPEG_VERSION,
        "org.bitsign.holistic-model.sha256": EXPECTED_MODEL_SHA256,
        "org.bitsign.worker.contract": RAW_SCHEMA,
        "org.bitsign.worker.receipt": RECEIPT_SCHEMA,
    }
    if any(labels.get(name) != expected for name, expected in expected_labels.items()):
        raise HolisticContainerError("Holistic image dependency labels do not match the pin set")
    return HolisticContainerImage(
        requested_reference=image_reference,
        image_id=image_id,
        platform=expected_platform,
    )


def build_holistic_container_image(
    *,
    image_tag: str | None = None,
    platform: str = "linux/arm64",
    docker_executable: str = "docker",
    timeout_seconds: float = 900.0,
    _runner: CommandRunner = _run_bounded_command,
) -> HolisticContainerImage:
    """Build one pinned worker platform and resolve its immutable local image ID."""

    if platform not in SUPPORTED_CONTAINER_PLATFORMS:
        raise HolisticContainerError("Holistic container platform is unsupported")
    if image_tag is None:
        image_tag = DEFAULT_IMAGE_TAG if platform == "linux/arm64" else AMD64_IMAGE_TAG

    context = Path(__file__).resolve().parents[2] / "docker" / "mediapipe-holistic"
    dockerfile = context / ("Dockerfile" if platform == "linux/arm64" else "Dockerfile.amd64")
    if not dockerfile.is_file():
        raise HolisticContainerError("Holistic container build context is missing")
    try:
        result = _runner_call(
            _runner,
            [
                docker_executable,
                "build",
                "--quiet",
                "--provenance=false",
                "--platform",
                platform,
                "--pull=false",
                "--build-arg",
                f"SOURCE_DATE_EPOCH={IMAGE_BUILD_SOURCE_DATE_EPOCH}",
                "--file",
                str(dockerfile),
                "--tag",
                image_tag,
                str(context),
            ],
            timeout_seconds=timeout_seconds,
            stdout_limit=64 * 1024,
            stderr_limit=1024 * 1024,
        )
    except _BoundedCommandFailure as exc:
        raise HolisticContainerError("pinned Holistic image build failed") from exc
    if result.return_code != 0:
        raise HolisticContainerError("pinned Holistic image build failed")
    build_output = result.stdout.strip()
    try:
        built_reference = build_output.decode("ascii")
    except UnicodeDecodeError as exc:
        raise HolisticContainerError("pinned Holistic build returned an invalid image ID") from exc
    if IMAGE_ID.fullmatch(built_reference) is None:
        raise HolisticContainerError("pinned Holistic build did not return one immutable image ID")
    return _inspect_image(
        built_reference,
        expected_platform=platform,
        docker_executable=docker_executable,
        timeout_seconds=30.0,
        runner=_runner,
    )


def resolve_holistic_container_image(
    image_reference: str | None = None,
    *,
    platform: str = "linux/arm64",
    docker_executable: str = "docker",
    timeout_seconds: float = 30.0,
    _runner: CommandRunner = _run_bounded_command,
) -> HolisticContainerImage:
    """Resolve and validate a prebuilt local image without pulling from a registry."""

    if platform not in SUPPORTED_CONTAINER_PLATFORMS:
        raise HolisticContainerError("Holistic container platform is unsupported")
    if image_reference is None:
        image_reference = DEFAULT_IMAGE_TAG if platform == "linux/arm64" else AMD64_IMAGE_TAG

    return _inspect_image(
        image_reference,
        expected_platform=platform,
        docker_executable=docker_executable,
        timeout_seconds=timeout_seconds,
        runner=_runner,
    )


def _docker_mount(source: Path, destination: str, *, read_only: bool) -> str:
    option = f"type=bind,src={source},dst={destination}"
    return f"{option},readonly" if read_only else option


def _cleanup_container_after_runner_failure(
    *, name: str, docker_executable: str, runner: CommandRunner
) -> bool:
    try:
        result = _runner_call(
            runner,
            [docker_executable, "container", "rm", "--force", name],
            timeout_seconds=10.0,
            stdout_limit=32 * 1024,
            stderr_limit=32 * 1024,
        )
    except (OSError, RuntimeError):
        return False
    return result.return_code == 0


def _remove_isolated_output_mount(invocation_root: Path, *, output_root: Path) -> None:
    try:
        relative = invocation_root.relative_to(output_root)
        metadata = invocation_root.lstat()
    except (OSError, ValueError) as exc:
        raise HolisticContainerError("isolated output cleanup target is invalid") from exc
    if (
        len(relative.parts) != 1
        or not relative.name.startswith(".")
        or invocation_root.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or not getattr(shutil.rmtree, "avoids_symlink_attacks", False)
    ):
        raise HolisticContainerError("isolated output cleanup cannot be performed safely")
    try:
        shutil.rmtree(invocation_root)
    except OSError as exc:
        raise HolisticContainerError("isolated output cleanup failed") from exc


def _validate_inventory(receipt: dict[str, Any], *, frame_count: int) -> dict[str, dict[str, Any]]:
    raw = receipt.get("raw_artifact")
    if not isinstance(raw, dict) or raw.get("schema") != RAW_SCHEMA:
        raise HolisticContainerError("worker receipt has the wrong raw artifact contract")
    inventory = raw.get("array_inventory")
    if not isinstance(inventory, list):
        raise HolisticContainerError("worker receipt omits the raw array inventory")
    expected: dict[str, tuple[str, list[int] | None]] = {
        "timestamps_us": ("<i8", [frame_count]),
        "requested_timestamps_us": ("<i8", [frame_count]),
        "source_timestamps_us": ("<i8", [frame_count]),
        "displayed_timestamps_us": ("<i8", [frame_count]),
        "source_pts": ("<i8", [frame_count]),
        "source_time_base": ("<i8", [2]),
        "source_frame_indices": ("<i4", [frame_count]),
        "clip_bounds_us": ("<i8", [2]),
        "displayed_rgb_sha256": ("|u1", [frame_count, 32]),
        "metadata_json_utf8": ("|u1", None),
    }
    for prefix, point_count in TRACK_SPECS.items():
        for field, (dtype, shape_builder) in TRACK_FIELDS.items():
            expected[f"{prefix}_{field}"] = (dtype, shape_builder(frame_count, point_count))
    observed: dict[str, dict[str, Any]] = {}
    observed_order: list[str] = []
    for entry in inventory:
        if not isinstance(entry, dict):
            raise HolisticContainerError("raw array inventory entry is malformed")
        _require_exact_keys(
            entry,
            {"name", "dtype", "shape", "tensor_sha256"},
            label="raw array inventory entry",
        )
        name = entry.get("name")
        if not isinstance(name, str) or name in observed:
            raise HolisticContainerError("raw array inventory has a duplicate or invalid name")
        observed[name] = entry
        observed_order.append(name)
    if observed_order != sorted(observed_order):
        raise HolisticContainerError("raw array inventory is not sorted by array name")
    if set(observed) != set(expected):
        raise HolisticContainerError("raw array inventory does not match the strict contract")
    for name, (dtype, shape) in expected.items():
        entry = observed[name]
        if entry.get("dtype") != dtype:
            raise HolisticContainerError("raw array inventory dtype mismatch")
        if shape is None:
            observed_shape = entry.get("shape")
            if (
                not isinstance(observed_shape, list)
                or len(observed_shape) != 1
                or type(observed_shape[0]) is not int
                or not 1 <= observed_shape[0] <= 4096
            ):
                raise HolisticContainerError("raw metadata array shape is invalid")
        elif entry.get("shape") != shape:
            raise HolisticContainerError("raw array inventory shape mismatch")
        digest = entry.get("tensor_sha256")
        if not isinstance(digest, str) or SHA256.fullmatch(digest) is None:
            raise HolisticContainerError("raw array inventory digest is invalid")
    return observed


def _validate_raw_npz_pinned(
    path: Path,
    *,
    sample_id: str,
    inventory: dict[str, dict[str, Any]],
) -> tuple[str, int, dict[str, np.ndarray]]:
    expected_members = {f"{name}.npy": entry for name, entry in inventory.items()}
    try:
        initial = path.lstat()
    except OSError as exc:
        raise HolisticContainerError("raw Holistic NPZ is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(initial.st_mode):
        raise HolisticContainerError("raw Holistic NPZ must be regular and non-symlink")
    if initial.st_size <= 0 or initial.st_size > MAXIMUM_RAW_ARTIFACT_BYTES:
        raise HolisticContainerError("raw Holistic NPZ violates its byte ceiling")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HolisticContainerError("raw Holistic NPZ could not be opened safely") from exc
    captured_arrays: dict[str, np.ndarray] = {}
    semantic_arrays: dict[str, np.ndarray] = {}
    provenance_names = {
        "timestamps_us",
        "requested_timestamps_us",
        "source_timestamps_us",
        "displayed_timestamps_us",
        "source_pts",
        "source_time_base",
        "source_frame_indices",
        "clip_bounds_us",
        "displayed_rgb_sha256",
        "metadata_json_utf8",
    }
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev != initial.st_dev or opened.st_ino != initial.st_ino
        ):
            raise HolisticContainerError("raw Holistic NPZ changed before validation")
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
            try:
                archive = zipfile.ZipFile(stream, mode="r")
            except (OSError, zipfile.BadZipFile) as exc:
                raise HolisticContainerError("raw Holistic NPZ is not a valid ZIP archive") from exc
            with archive:
                members = archive.infolist()
                names = [member.filename for member in members]
                if (
                    len(names) != len(set(names))
                    or names != sorted(names)
                    or set(names) != set(expected_members)
                ):
                    raise HolisticContainerError("raw Holistic NPZ member set is invalid")
                for member_info in members:
                    entry = expected_members[member_info.filename]
                    if (
                        member_info.compress_type != zipfile.ZIP_STORED
                        or member_info.compress_size != member_info.file_size
                        or member_info.flag_bits & 0x1
                        or member_info.date_time != (1980, 1, 1, 0, 0, 0)
                        or member_info.create_system != 3
                        or (member_info.external_attr >> 16) & 0o777 != 0o600
                    ):
                        raise HolisticContainerError(
                            "raw Holistic NPZ member encoding is not strict"
                        )
                    try:
                        expected_dtype = np.dtype(entry["dtype"])
                        expected_shape = tuple(entry["shape"])
                    except (TypeError, ValueError) as exc:
                        raise HolisticContainerError(
                            "raw array inventory cannot be interpreted"
                        ) from exc
                    expected_data_bytes = math.prod(expected_shape) * expected_dtype.itemsize
                    if expected_data_bytes < 0 or expected_data_bytes > MAXIMUM_RAW_ARTIFACT_BYTES:
                        raise HolisticContainerError("raw array declares an invalid byte count")
                    try:
                        member = archive.open(member_info, mode="r")
                        with member:
                            version = np.lib.format.read_magic(member)
                            if version != (1, 0):
                                raise HolisticContainerError(
                                    "raw Holistic NPY member must use format version 1.0"
                                )
                            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                                member, max_header_size=10_000
                            )
                            if (
                                tuple(shape) != expected_shape
                                or fortran_order
                                or dtype.str != expected_dtype.str
                            ):
                                raise HolisticContainerError(
                                    "raw Holistic NPY header disagrees with its inventory"
                                )
                            digest = hashlib.sha256()
                            remaining = expected_data_bytes
                            captured = bytearray()
                            while remaining:
                                chunk = member.read(min(65536, remaining))
                                if not chunk:
                                    raise HolisticContainerError(
                                        "raw Holistic NPY member ended before its declared tensor"
                                    )
                                remaining -= len(chunk)
                                digest.update(chunk)
                                if expected_dtype.kind == "f":
                                    numeric = np.frombuffer(chunk, dtype=expected_dtype)
                                    if (
                                        not np.isfinite(numeric).all()
                                        or np.signbit(numeric[numeric == 0]).any()
                                    ):
                                        raise HolisticContainerError(
                                            "raw Holistic tensor contains non-finite or "
                                            "negative-zero values"
                                        )
                                elif expected_dtype.kind == "b" and any(
                                    value not in (0, 1) for value in chunk
                                ):
                                    raise HolisticContainerError(
                                        "raw Holistic availability tensor is not canonical "
                                        "boolean data"
                                    )
                                if member_info.filename.removesuffix(".npy") in provenance_names:
                                    captured.extend(chunk)
                                elif any(
                                    member_info.filename.startswith(f"{prefix}_")
                                    for prefix in TRACK_SPECS
                                ):
                                    captured.extend(chunk)
                            if member.read(1):
                                raise HolisticContainerError(
                                    "raw Holistic NPY member contains trailing tensor bytes"
                                )
                    except (OSError, ValueError, zipfile.BadZipFile) as exc:
                        raise HolisticContainerError(
                            "raw Holistic NPY member cannot be decoded safely"
                        ) from exc
                    if digest.hexdigest() != entry["tensor_sha256"]:
                        raise HolisticContainerError(
                            "raw Holistic tensor digest disagrees with its inventory"
                        )
                    array_name = member_info.filename.removesuffix(".npy")
                    if array_name in provenance_names:
                        captured_arrays[array_name] = np.frombuffer(
                            bytes(captured), dtype=expected_dtype
                        ).reshape(expected_shape)
                    elif any(array_name.startswith(f"{prefix}_") for prefix in TRACK_SPECS):
                        semantic_arrays[array_name] = np.frombuffer(
                            bytes(captured), dtype=expected_dtype
                        ).reshape(expected_shape)
        final = os.fstat(descriptor)
        if final.st_size != opened.st_size:
            raise HolisticContainerError("raw Holistic NPZ changed during validation")
    finally:
        os.close(descriptor)
    metadata_array = captured_arrays.get("metadata_json_utf8")
    if metadata_array is None:
        raise HolisticContainerError("raw Holistic NPZ omits its metadata member")
    metadata_payload = metadata_array.tobytes()
    metadata = _strict_json_bytes(
        metadata_payload,
        label="raw Holistic metadata",
        maximum_bytes=4096,
    )
    if canonical_json_bytes(metadata) != metadata_payload:
        raise HolisticContainerError("raw Holistic metadata is not RFC 8785 canonical JSON")
    if metadata != {
        "schema": RAW_SCHEMA,
        "sample_id": sample_id,
        "status": "ex-203-candidate-only",
        "canonical_or_release_quality": False,
        "hand_slot_semantics": "opaque-observation-slots-mapper-assigns-anatomical-side",
        "track_prefixes": list(RAW_TRACK_PREFIX_ORDER),
    }:
        raise HolisticContainerError("raw Holistic metadata contract is invalid")
    for prefix in TRACK_SPECS:
        coordinate_mask = semantic_arrays[f"{prefix}_coordinate_available"]
        coordinates = semantic_arrays[f"{prefix}_coordinates"]
        presence_mask = semantic_arrays[f"{prefix}_presence_available"]
        presence = semantic_arrays[f"{prefix}_presence"]
        visibility_mask = semantic_arrays[f"{prefix}_visibility_available"]
        visibility = semantic_arrays[f"{prefix}_visibility"]
        if (
            np.any(coordinates[~coordinate_mask] != 0)
            or np.any(presence[~presence_mask] != 0)
            or np.any(visibility[~visibility_mask] != 0)
        ):
            raise HolisticContainerError(
                "raw Holistic unavailable numeric slots are not positive zero"
            )
    digest, byte_count = _file_sha256(path, maximum_bytes=MAXIMUM_RAW_ARTIFACT_BYTES)
    return digest, byte_count, captured_arrays


def _require_exact_keys(value: dict[str, Any], expected: set[str], *, label: str) -> None:
    if set(value) != expected:
        raise HolisticContainerError(f"{label} has an unexpected member set")


def _integer(value: object, *, label: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise HolisticContainerError(f"{label} is not a valid integer")
    return value


def _fraction(value: object, *, label: str, positive: bool = True) -> Fraction:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise HolisticContainerError(f"{label} is not a bounded rational string")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise HolisticContainerError(f"{label} is not a rational") from exc
    if positive and parsed <= 0:
        raise HolisticContainerError(f"{label} must be positive")
    if value != f"{parsed.numerator}/{parsed.denominator}":
        raise HolisticContainerError(f"{label} is not a reduced canonical rational")
    return parsed


def _to_microseconds(value: Fraction) -> int:
    scaled = value * 1_000_000
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    if remainder * 2 > scaled.denominator:
        quotient += 1
    return quotient


def _candidate_frame_limit(duration_us: int) -> int:
    scaled = Fraction(duration_us, 1_000_000) * MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    interval_limit = quotient + bool(remainder) + 2
    return min(MAXIMUM_BOOTSTRAP_CANDIDATE_FRAMES, interval_limit)


def _candidate_probe_interval_us(
    *,
    clip_start_us: int,
    clip_end_us: int,
    stream_start_time: Fraction,
    nominal_frame_rate: Fraction,
    has_b_frames: int,
) -> list[int]:
    if nominal_frame_rate <= 0:
        raise HolisticContainerError("candidate probe nominal frame rate must be positive")
    if type(has_b_frames) is not int or not 0 <= has_b_frames <= 2:
        raise HolisticContainerError(
            "candidate probe B-frame depth is outside the bootstrap profile"
        )
    start_guard = Fraction(1, 1) / nominal_frame_rate
    end_guard = Fraction(has_b_frames + 1, 1) / nominal_frame_rate
    start = max(
        Fraction(0),
        stream_start_time + Fraction(clip_start_us, 1_000_000) - start_guard,
    )
    end = stream_start_time + Fraction(clip_end_us, 1_000_000) + end_guard
    values = [_to_microseconds(start), _to_microseconds(end)]
    if values[1] <= values[0]:
        raise HolisticContainerError("candidate probe interval is empty")
    return values


def _effective_display_dimensions(
    coded_width: int,
    coded_height: int,
    rotation: int,
    sample_aspect_ratio: Fraction,
) -> tuple[Fraction, Fraction]:
    if rotation in (0, 180):
        return Fraction(coded_width) * sample_aspect_ratio, Fraction(coded_height)
    return Fraction(coded_height), Fraction(coded_width) * sample_aspect_ratio


def _derived_dimensions(width: int | Fraction, height: int | Fraction) -> tuple[int, int]:
    if width <= 0 or height <= 0:
        raise HolisticContainerError("displayed dimensions must be positive")
    width = Fraction(width)
    height = Fraction(height)
    if width >= height:
        maximum_width, maximum_height = MAXIMUM_DERIVED_LONG_SIDE, MAXIMUM_DERIVED_SHORT_SIDE
    else:
        maximum_width, maximum_height = MAXIMUM_DERIVED_SHORT_SIDE, MAXIMUM_DERIVED_LONG_SIDE
    scale = min(Fraction(1), Fraction(maximum_width, width), Fraction(maximum_height, height))
    if scale == 1 and width.denominator == 1 and height.denominator == 1:
        return int(width), int(height)

    def nearest_bounded_even(value: Fraction, maximum: int) -> int:
        lower = 2 * (value // 2)
        upper = lower + 2
        if upper <= maximum and value - lower > upper - value:
            return upper
        return max(2, lower)

    return (
        nearest_bounded_even(width * scale, maximum_width),
        nearest_bounded_even(height * scale, maximum_height),
    )


def _strict_int_list(value: object, *, length: int, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != length:
        raise HolisticContainerError(f"{label} has the wrong length")
    return [_integer(item, label=label) for item in value]


def _validate_receipt_contract(
    receipt: dict[str, Any],
    *,
    sample_id: str,
    expected_image_id: str | None,
    video_sha256: str,
    video_bytes: int,
    model_sha256: str,
    model_bytes: int,
    clip_start_us: int,
    clip_end_us: int,
    input_mirrored: bool,
    expected_platform: str = "linux/arm64",
) -> tuple[int, str, dict[str, Any], dict[str, Any]]:
    _require_exact_keys(
        receipt,
        {
            "schema",
            "sample_id",
            "status",
            "claim_boundary",
            "container",
            "input_video",
            "model",
            "source_media",
            "clip",
            "derived_view",
            "selection",
            "extractor",
            "raw_artifact",
            "publication",
            "content_sha256",
        },
        label="worker receipt",
    )
    if (
        receipt["schema"] != RECEIPT_SCHEMA
        or receipt["sample_id"] != sample_id
        or receipt["status"] != "ex-203-candidate-only"
    ):
        raise HolisticContainerError("worker receipt identity is invalid")
    content_digest = receipt["content_sha256"]
    if not isinstance(content_digest, str) or SHA256.fullmatch(content_digest) is None:
        raise HolisticContainerError("worker receipt content digest is invalid")
    without_digest = dict(receipt)
    del without_digest["content_sha256"]
    calculated = hashlib.sha256(
        b"umi-raw-holistic-extraction-receipt-v3\0" + canonical_json_bytes(without_digest)
    ).hexdigest()
    if content_digest != calculated:
        raise HolisticContainerError("worker receipt content digest does not reproduce")

    claim = receipt["claim_boundary"]
    if not isinstance(claim, dict):
        raise HolisticContainerError("worker claim boundary is malformed")
    _require_exact_keys(
        claim,
        {
            "canonical_or_release_quality",
            "cross_runtime_equivalence_established",
            "source_is_umi_challenge_media",
            "purpose",
        },
        label="worker claim boundary",
    )
    if claim != {
        "canonical_or_release_quality": False,
        "cross_runtime_equivalence_established": False,
        "source_is_umi_challenge_media": False,
        "purpose": "bootstrap-timed-source-isolated-raw-holistic-research",
    }:
        raise HolisticContainerError("worker overstates the extractor claim boundary")

    container = receipt["container"]
    if not isinstance(container, dict):
        raise HolisticContainerError("worker container binding is malformed")
    _require_exact_keys(
        container,
        {"image_id", "platform", "network", "root_filesystem", "worker_uid"},
        label="worker container binding",
    )
    image_id = container["image_id"]
    if (
        not isinstance(image_id, str)
        or IMAGE_ID.fullmatch(image_id) is None
        or (expected_image_id is not None and image_id != expected_image_id)
        or container["platform"] != expected_platform
        or container["network"] != "none"
        or container["root_filesystem"] != "read-only"
        or container["worker_uid"] != os.geteuid()
    ):
        raise HolisticContainerError("worker container binding does not match the invocation")

    input_video = receipt["input_video"]
    model = receipt["model"]
    if not isinstance(input_video, dict) or not isinstance(model, dict):
        raise HolisticContainerError("worker input binding is malformed")
    _require_exact_keys(
        input_video,
        {"sha256", "byte_count", "mounted_read_only", "source_ceiling_bytes"},
        label="worker source binding",
    )
    _require_exact_keys(
        model,
        {"sha256", "byte_count", "mounted_read_only"},
        label="worker model binding",
    )
    if input_video != {
        "sha256": video_sha256,
        "byte_count": video_bytes,
        "mounted_read_only": True,
        "source_ceiling_bytes": MAXIMUM_SOURCE_VIDEO_BYTES,
    } or model != {
        "sha256": model_sha256,
        "byte_count": model_bytes,
        "mounted_read_only": True,
    }:
        raise HolisticContainerError("worker input binding does not match the invocation")

    clip = receipt["clip"]
    if not isinstance(clip, dict):
        raise HolisticContainerError("worker clip binding is malformed")
    _require_exact_keys(
        clip,
        {"clip_start_us", "clip_end_us", "duration_us", "interval"},
        label="worker clip binding",
    )
    duration_us = clip_end_us - clip_start_us
    if clip != {
        "clip_start_us": clip_start_us,
        "clip_end_us": clip_end_us,
        "duration_us": duration_us,
        "interval": "half-open-[start,end)",
    }:
        raise HolisticContainerError("worker clip bounds do not match the invocation")

    source = receipt["source_media"]
    if not isinstance(source, dict):
        raise HolisticContainerError("worker source media record is malformed")
    _require_exact_keys(
        source,
        {
            "profile",
            "container",
            "format_name",
            "major_brand",
            "codec",
            "codec_profile",
            "pixel_format",
            "has_b_frames",
            "audio_stream_count",
            "audio_codecs",
            "audio_streams",
            "audio_policy",
            "sample_aspect_ratio",
            "sample_aspect_ratio_source",
            "derived_sample_aspect_ratio",
            "coded_dimensions",
            "displayed_dimensions",
            "display_aspect_ratio",
            "display_rotation_degrees",
            "display_autorotation",
            "input_mirrored",
            "canonical_unmirroring",
            "duration",
            "duration_us",
            "nominal_frame_rate",
            "average_frame_rate",
            "clip_end_tolerance_policy",
            "clip_end_overshoot",
            "maximum_clip_end_overshoot",
            "maximum_source_frame_rate",
            "source_time_base",
            "stream_start_pts",
            "stream_start_time",
            "candidate_frame_count",
            "candidate_frame_limit",
            "candidate_frame_pts",
            "candidate_probe_policy",
            "candidate_probe_interval_us",
        },
        label="worker source media record",
    )
    coded = _strict_int_list(source["coded_dimensions"], length=2, label="coded dimensions")
    displayed = _strict_int_list(
        source["displayed_dimensions"], length=2, label="displayed dimensions"
    )
    rotation = _integer(source["display_rotation_degrees"], label="display rotation")
    if rotation not in (0, 90, 180, 270):
        raise HolisticContainerError("worker source rotation is invalid")
    expected_displayed = [coded[1], coded[0]] if rotation in (90, 270) else coded
    source_duration = _fraction(source["duration"], label="source duration")
    nominal_rate = _fraction(source["nominal_frame_rate"], label="nominal frame rate")
    _fraction(source["average_frame_rate"], label="average frame rate")
    clip_end_overshoot = _fraction(
        source["clip_end_overshoot"], label="clip end overshoot", positive=False
    )
    maximum_clip_end_overshoot = _fraction(
        source["maximum_clip_end_overshoot"],
        label="maximum clip end overshoot",
    )
    sample_aspect_ratio = _fraction(
        source["sample_aspect_ratio"], label="source sample aspect ratio"
    )
    display_aspect_ratio = _fraction(
        source["display_aspect_ratio"], label="source display aspect ratio"
    )
    codec_name = source["codec"]
    codec_profile = source["codec_profile"]
    pixel_format = source["pixel_format"]
    has_b_frames = _integer(source["has_b_frames"], label="B-frame depth", minimum=0)
    audio_codecs = source["audio_codecs"]
    audio_streams = source["audio_streams"]
    audio_stream_count = _integer(
        source["audio_stream_count"], label="audio stream count", minimum=0
    )
    maximum_source_rate = _fraction(
        source["maximum_source_frame_rate"], label="maximum source frame rate"
    )
    time_base = _fraction(source["source_time_base"], label="source time base")
    stream_start_pts = _integer(source["stream_start_pts"], label="stream start PTS")
    stream_start_time = _fraction(
        source["stream_start_time"], label="stream start time", positive=False
    )
    candidate_count = _integer(
        source["candidate_frame_count"], label="candidate frame count", minimum=1
    )
    candidate_limit = _integer(
        source["candidate_frame_limit"], label="candidate frame limit", minimum=1
    )
    candidate_pts = _strict_int_list(
        source["candidate_frame_pts"],
        length=candidate_count,
        label="candidate frame PTS",
    )
    candidate_probe_interval_us = _strict_int_list(
        source["candidate_probe_interval_us"],
        length=2,
        label="candidate probe interval",
    )
    expected_candidate_limit = _candidate_frame_limit(duration_us)
    expected_candidate_probe_interval_us = _candidate_probe_interval_us(
        clip_start_us=clip_start_us,
        clip_end_us=clip_end_us,
        stream_start_time=stream_start_time,
        nominal_frame_rate=nominal_rate,
        has_b_frames=has_b_frames,
    )
    expected_clip_end_overshoot = max(
        Fraction(0), Fraction(clip_end_us, 1_000_000) - source_duration
    )
    candidate_times = [point * time_base - stream_start_time for point in candidate_pts]
    expected_audio_streams = [
        {
            "codec": codec,
            "profile": "LC",
            "sample_format": "fltp",
            "sample_rate_hz": 48_000,
            "channels": 2,
        }
        for codec in audio_codecs
    ]
    effective_display_width, effective_display_height = _effective_display_dimensions(
        coded[0],
        coded[1],
        rotation,
        sample_aspect_ratio,
    )
    if (
        source["profile"] != "bounded-bootstrap-source-not-umi-challenge-media"
        or source["container"] != "mp4"
        or not isinstance(source["format_name"], str)
        or "mp4" not in source["format_name"].split(",")
        or source["major_brand"] not in ALLOWED_MP4_MAJOR_BRANDS
        or not all(isinstance(value, str) for value in (codec_name, codec_profile, pixel_format))
        or (codec_name, codec_profile, pixel_format) not in ALLOWED_BOOTSTRAP_VIDEO_PROFILES
        or has_b_frames > 2
        or not isinstance(audio_codecs, list)
        or any(
            not isinstance(codec, str) or codec not in ALLOWED_BOOTSTRAP_AUDIO_CODECS
            for codec in audio_codecs
        )
        or len(audio_codecs) > 1
        or audio_stream_count != len(audio_codecs)
        or not isinstance(audio_streams, list)
        or audio_streams != expected_audio_streams
        or source["audio_policy"] != "ignored-by-ffmpeg-an"
        or source["sample_aspect_ratio_source"]
        not in {"declared-near-square", "unspecified-assumed-square"}
        or abs(sample_aspect_ratio - 1) > MAXIMUM_SAMPLE_ASPECT_RATIO_DEVIATION
        or (
            source["sample_aspect_ratio_source"] == "unspecified-assumed-square"
            and sample_aspect_ratio != 1
        )
        or source["derived_sample_aspect_ratio"] != "1/1"
        or display_aspect_ratio != effective_display_width / effective_display_height
        or displayed != expected_displayed
        or max(displayed) > MAXIMUM_SOURCE_LONG_SIDE
        or min(displayed) > MAXIMUM_SOURCE_SHORT_SIDE
        or source["display_autorotation"] != "ffmpeg-default-enabled"
        or source["input_mirrored"] is not input_mirrored
        or source["canonical_unmirroring"]
        != ("ffmpeg-hflip-before-extraction" if input_mirrored else "not-required")
        or source["duration_us"] != _to_microseconds(source_duration)
        or Fraction(clip_start_us, 1_000_000) >= source_duration
        or source["clip_end_tolerance_policy"]
        != "source-metadata-rounding-at-most-one-nominal-frame-v1"
        or clip_end_overshoot != expected_clip_end_overshoot
        or maximum_clip_end_overshoot != Fraction(1, 1) / nominal_rate
        or clip_end_overshoot > maximum_clip_end_overshoot
        or maximum_source_rate != MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE
        or nominal_rate > MAXIMUM_BOOTSTRAP_SOURCE_FRAME_RATE
        or stream_start_time != stream_start_pts * time_base
        or candidate_limit != expected_candidate_limit
        or candidate_count > candidate_limit
        or candidate_count > MAXIMUM_BOOTSTRAP_CANDIDATE_FRAMES
        or source["candidate_probe_policy"]
        != "absolute-stream-time-with-one-frame-start-and-bframe-plus-one-end-guard-v1"
        or candidate_probe_interval_us != expected_candidate_probe_interval_us
        or any(later <= earlier for earlier, later in pairwise(candidate_pts))
        or any(
            not Fraction(clip_start_us, 1_000_000) <= value < Fraction(clip_end_us, 1_000_000)
            for value in candidate_times
        )
    ):
        raise HolisticContainerError("worker source media record violates the bounded profile")

    derived = receipt["derived_view"]
    if not isinstance(derived, dict):
        raise HolisticContainerError("worker derived view is malformed")
    _require_exact_keys(
        derived,
        {
            "dimensions",
            "orientation_neutral_bound",
            "pixel_format",
            "aspect_policy",
            "pixel_aspect_filter",
            "scale_filter",
        },
        label="worker derived view",
    )
    expected_derived = list(_derived_dimensions(effective_display_width, effective_display_height))
    expected_scale = (
        "identity"
        if expected_derived == displayed
        else f"scale={expected_derived[0]}:{expected_derived[1]}:flags={SCALE_FLAGS}"
    )
    if derived != {
        "dimensions": expected_derived,
        "orientation_neutral_bound": [1280, 720],
        "pixel_format": "rgb24",
        "aspect_policy": "apply-source-sar-then-nearest-bounded-even-per-axis",
        "pixel_aspect_filter": "setsar=1/1",
        "scale_filter": expected_scale,
    }:
        raise HolisticContainerError("worker derived view does not reproduce")

    selection = receipt["selection"]
    if not isinstance(selection, dict):
        raise HolisticContainerError("worker frame selection is malformed")
    _require_exact_keys(
        selection,
        {
            "algorithm",
            "tie_break",
            "clip_bounds_us",
            "frame_count",
            "requested_times",
            "requested_timestamps_us",
            "source_frame_indices",
            "source_frame_index_basis",
            "source_pts",
            "source_times",
            "source_timestamps_us",
            "displayed_rgb_sha256",
        },
        label="worker frame selection",
    )
    frame_count = -(-(duration_us * TARGET_RATE) // 1_000_000)
    if not 16 <= frame_count <= 120 or selection["frame_count"] != frame_count:
        raise HolisticContainerError("worker receipt frame count is outside the bounded profile")
    requested = [
        Fraction(clip_start_us, 1_000_000)
        + Fraction(2 * index + 1, 2 * frame_count) * Fraction(duration_us, 1_000_000)
        for index in range(frame_count)
    ]
    expected_requested_times = [f"{value.numerator}/{value.denominator}" for value in requested]
    expected_requested_us = [_to_microseconds(value) for value in requested]
    source_indices = _strict_int_list(
        selection["source_frame_indices"], length=frame_count, label="source frame indices"
    )
    if any(index < 0 for index in source_indices):
        raise HolisticContainerError("source frame indices must be non-negative")
    source_pts = _strict_int_list(selection["source_pts"], length=frame_count, label="source PTS")
    source_times = [point * time_base - stream_start_time for point in source_pts]
    expected_source_indices = [
        min(
            range(candidate_count),
            key=lambda candidate_index: (
                abs(candidate_times[candidate_index] - requested_time),
                candidate_times[candidate_index],
                candidate_index,
            ),
        )
        for requested_time in requested
    ]
    expected_source_pts = [candidate_pts[index] for index in expected_source_indices]
    expected_source_time_strings = [
        f"{value.numerator}/{value.denominator}" for value in source_times
    ]
    expected_source_us = [_to_microseconds(value) for value in source_times]
    displayed_hashes = selection["displayed_rgb_sha256"]
    if (
        selection["algorithm"] != "8hz-equal-clip-bin-center-nearest-ffprobe-pts-v2"
        or selection["tie_break"]
        != "earlier-presentation-timestamp-then-earlier-interval-frame-index"
        or selection["clip_bounds_us"] != [clip_start_us, clip_end_us]
        or selection["requested_times"] != expected_requested_times
        or selection["requested_timestamps_us"] != expected_requested_us
        or selection["source_frame_index_basis"]
        != "ffprobe-filtered-clip-candidate-sequence-zero-based"
        or source_indices != expected_source_indices
        or source_pts != expected_source_pts
        or selection["source_times"] != expected_source_time_strings
        or selection["source_timestamps_us"] != expected_source_us
        or any(later < earlier for earlier, later in pairwise(source_indices))
        or any(later < earlier for earlier, later in pairwise(source_pts))
        or any(index >= candidate_count for index in source_indices)
        or any(
            candidate_pts[index] != point
            for index, point in zip(source_indices, source_pts, strict=True)
        )
        or any(
            not Fraction(clip_start_us, 1_000_000) <= value < Fraction(clip_end_us, 1_000_000)
            for value in source_times
        )
        or not isinstance(displayed_hashes, list)
        or len(displayed_hashes) != frame_count
        or any(
            not isinstance(value, str) or SHA256.fullmatch(value) is None
            for value in displayed_hashes
        )
        or any(
            source_pts[index] == source_pts[index - 1]
            and (
                source_indices[index] != source_indices[index - 1]
                or displayed_hashes[index] != displayed_hashes[index - 1]
            )
            for index in range(1, frame_count)
        )
    ):
        raise HolisticContainerError("worker frame selection does not reproduce its authority")

    extractor = receipt["extractor"]
    if not isinstance(extractor, dict):
        raise HolisticContainerError("worker extractor record is malformed")
    _require_exact_keys(
        extractor,
        {
            "mediapipe_version",
            "task",
            "running_mode",
            "invocation",
            "delegate",
            "thresholds",
            "ffprobe_version",
            "ffmpeg_version",
        },
        label="worker extractor record",
    )
    expected_thresholds = {
        "min_face_detection_confidence": "1/2",
        "min_face_suppression_threshold": "1/2",
        "min_face_landmarks_confidence": "1/2",
        "min_pose_detection_confidence": "1/2",
        "min_pose_suppression_threshold": "1/2",
        "min_pose_landmarks_confidence": "1/2",
        "min_hand_landmarks_confidence": "1/2",
    }
    if (
        extractor["mediapipe_version"] != EXPECTED_MEDIAPIPE_VERSION
        or extractor["task"] != "HolisticLandmarker"
        or extractor["running_mode"] != "VIDEO"
        or extractor["invocation"] != "synchronous-detect_for_video"
        or extractor["delegate"] != "CPU"
        or extractor["thresholds"] != expected_thresholds
        or not isinstance(extractor["ffprobe_version"], str)
        or not extractor["ffprobe_version"].startswith(f"ffprobe version {EXPECTED_FFMPEG_VERSION}")
        or not isinstance(extractor["ffmpeg_version"], str)
        or not extractor["ffmpeg_version"].startswith(f"ffmpeg version {EXPECTED_FFMPEG_VERSION}")
    ):
        raise HolisticContainerError("worker extractor dependency record is invalid")

    raw = receipt["raw_artifact"]
    publication = receipt["publication"]
    if not isinstance(raw, dict) or not isinstance(publication, dict):
        raise HolisticContainerError("worker output publication record is malformed")
    _require_exact_keys(
        raw,
        {
            "schema",
            "filename",
            "sha256",
            "byte_count",
            "array_inventory",
            "hand_observation_slots",
            "anatomical_side_assignment",
        },
        label="worker raw artifact record",
    )
    _require_exact_keys(
        publication,
        {"directory", "raw_filename", "receipt_filename", "mode", "owner_only"},
        label="worker publication record",
    )
    expected_slots = [
        {
            "slot": 0,
            "image_source_attribute": "left_hand_landmarks",
            "world_source_attribute": "left_hand_world_landmarks",
        },
        {
            "slot": 1,
            "image_source_attribute": "right_hand_landmarks",
            "world_source_attribute": "right_hand_world_landmarks",
        },
    ]
    if (
        raw["schema"] != RAW_SCHEMA
        or raw["filename"] != RAW_FILENAME
        or not isinstance(raw["sha256"], str)
        or SHA256.fullmatch(raw["sha256"]) is None
        or _integer(raw["byte_count"], label="raw byte count", minimum=1)
        > MAXIMUM_RAW_ARTIFACT_BYTES
        or raw["hand_observation_slots"] != expected_slots
        or raw["anatomical_side_assignment"] != "deferred-to-mapping-layer"
        or publication
        != {
            "directory": sample_id,
            "raw_filename": RAW_FILENAME,
            "receipt_filename": RECEIPT_FILENAME,
            "mode": "atomic-directory-renameat2-noreplace",
            "owner_only": True,
        }
    ):
        raise HolisticContainerError("worker publication record is invalid")
    return frame_count, content_digest, selection, raw


def _validate_outputs(
    *,
    output_root: Path,
    sample_id: str,
    expected_image_id: str | None,
    video_sha256: str,
    video_bytes: int,
    model_sha256: str,
    model_bytes: int,
    clip_start_us: int,
    clip_end_us: int,
    input_mirrored: bool,
    completion_stdout: bytes | None,
    expected_platform: str = "linux/arm64",
    require_umi_whole_video: bool = False,
) -> HolisticExtractionResult:
    completion: dict[str, Any] | None = None
    if completion_stdout is not None:
        if not completion_stdout.endswith(b"\n") or completion_stdout.count(b"\n") != 1:
            raise HolisticContainerError("worker completion record framing is invalid")
        completion_raw = completion_stdout[:-1]
        completion = _strict_json_bytes(
            completion_raw,
            label="worker completion record",
            maximum_bytes=MAXIMUM_COMPLETION_BYTES,
        )
        if canonical_json_bytes(completion) != completion_raw:
            raise HolisticContainerError("worker completion record is not RFC 8785 canonical JSON")
    sample_dir = output_root / sample_id
    try:
        directory_metadata = sample_dir.lstat()
    except OSError as exc:
        raise HolisticContainerError("worker did not publish the final sample directory") from exc
    if sample_dir.is_symlink() or not stat.S_ISDIR(directory_metadata.st_mode):
        raise HolisticContainerError("published sample output is not a private directory")
    if directory_metadata.st_uid != os.geteuid() or directory_metadata.st_mode & 0o077:
        raise HolisticContainerError("published sample directory is not owner-only")
    try:
        names = sorted(item.name for item in sample_dir.iterdir())
    except OSError as exc:
        raise HolisticContainerError("published sample directory cannot be inspected") from exc
    if names != [RAW_FILENAME, RECEIPT_FILENAME]:
        raise HolisticContainerError("published sample directory contains unexpected files")
    raw_path = sample_dir / RAW_FILENAME
    receipt_path = sample_dir / RECEIPT_FILENAME
    for path in (raw_path, receipt_path):
        metadata = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise HolisticContainerError("published output must be a regular non-symlink file")
        if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
            raise HolisticContainerError("published output file is not owner-only")
    receipt_raw, receipt_digest = _read_file_pinned(
        receipt_path, maximum_bytes=MAXIMUM_RECEIPT_BYTES
    )
    receipt = _strict_json_bytes(
        receipt_raw,
        label="worker receipt",
        maximum_bytes=MAXIMUM_RECEIPT_BYTES,
    )
    if canonical_json_bytes(receipt) != receipt_raw:
        raise HolisticContainerError("worker receipt is not RFC 8785 canonical JSON")
    frame_count, content_digest, selection, raw = _validate_receipt_contract(
        receipt,
        sample_id=sample_id,
        expected_image_id=expected_image_id,
        video_sha256=video_sha256,
        video_bytes=video_bytes,
        model_sha256=model_sha256,
        model_bytes=model_bytes,
        clip_start_us=clip_start_us,
        clip_end_us=clip_end_us,
        input_mirrored=input_mirrored,
        expected_platform=expected_platform,
    )
    if require_umi_whole_video:
        source = receipt["source_media"]
        if (
            clip_start_us != 0
            or receipt["input_video"]["byte_count"] > MAXIMUM_UMI_VIDEO_BYTES
            or source["codec"] != "h264"
            or source["audio_stream_count"] != 0
            or max(source["displayed_dimensions"]) > 1280
            or min(source["displayed_dimensions"]) > 720
            or _fraction(source["nominal_frame_rate"], label="nominal frame rate") > 30
            or source["duration_us"] != clip_end_us
            or not MINIMUM_CLIP_US
            <= _integer(source["duration_us"], label="source duration")
            <= MAXIMUM_CLIP_US
        ):
            raise HolisticContainerError("worker output violates the UMI whole-video profile")
    inventory = _validate_inventory(receipt, frame_count=frame_count)
    raw_digest, raw_bytes, provenance = _validate_raw_npz_pinned(
        raw_path, sample_id=sample_id, inventory=inventory
    )
    if raw["sha256"] != raw_digest or raw["byte_count"] != raw_bytes:
        raise HolisticContainerError(
            "worker raw artifact digest or size does not match its receipt"
        )
    expected_provenance = {
        "timestamps_us": np.asarray(selection["requested_timestamps_us"], dtype=np.int64),
        "requested_timestamps_us": np.asarray(selection["requested_timestamps_us"], dtype=np.int64),
        "source_timestamps_us": np.asarray(selection["source_timestamps_us"], dtype=np.int64),
        "displayed_timestamps_us": np.asarray(selection["source_timestamps_us"], dtype=np.int64),
        "source_pts": np.asarray(selection["source_pts"], dtype=np.int64),
        "source_time_base": np.asarray(
            [
                _fraction(
                    receipt["source_media"]["source_time_base"], label="source time base"
                ).numerator,
                _fraction(
                    receipt["source_media"]["source_time_base"], label="source time base"
                ).denominator,
            ],
            dtype=np.int64,
        ),
        "source_frame_indices": np.asarray(selection["source_frame_indices"], dtype=np.int32),
        "clip_bounds_us": np.asarray([clip_start_us, clip_end_us], dtype=np.int64),
        "displayed_rgb_sha256": np.frombuffer(
            b"".join(bytes.fromhex(value) for value in selection["displayed_rgb_sha256"]),
            dtype=np.uint8,
        ).reshape(frame_count, 32),
    }
    if any(
        name not in provenance or not np.array_equal(provenance[name], expected)
        for name, expected in expected_provenance.items()
    ):
        raise HolisticContainerError("raw Holistic provenance disagrees with the signed receipt")
    if completion is not None:
        image_id = receipt["container"]["image_id"]
        expected_completion = {
            "schema": COMPLETION_SCHEMA,
            "sample_id": sample_id,
            "container_image_id": image_id,
            "input_video_sha256": video_sha256,
            "model_sha256": model_sha256,
            "clip_start_us": clip_start_us,
            "clip_end_us": clip_end_us,
            "input_mirrored": input_mirrored,
            "raw_artifact_sha256": raw_digest,
            "receipt_sha256": receipt_digest,
            "receipt_content_sha256": content_digest,
            "raw_relative_path": f"{sample_id}/{RAW_FILENAME}",
            "receipt_relative_path": f"{sample_id}/{RECEIPT_FILENAME}",
        }
        if completion != expected_completion:
            raise HolisticContainerError("worker completion record does not bind the invocation")
    return HolisticExtractionResult(
        sample_id=sample_id,
        image_id=receipt["container"]["image_id"],
        raw_artifact_path=raw_path,
        receipt_path=receipt_path,
        raw_artifact_sha256=raw_digest,
        receipt_sha256=receipt_digest,
        receipt_content_sha256=content_digest,
        frame_count=frame_count,
        status="ex-203-candidate-only",
    )


def _publish_validated_output(
    result: HolisticExtractionResult,
    *,
    invocation_root: Path,
    output_root: Path,
) -> HolisticExtractionResult:
    try:
        invocation_metadata = invocation_root.lstat()
        entries = list(invocation_root.iterdir())
    except OSError as exc:
        raise HolisticContainerError("isolated output mount changed after validation") from exc
    source = invocation_root / result.sample_id
    destination = output_root / result.sample_id
    if (
        invocation_root.is_symlink()
        or not stat.S_ISDIR(invocation_metadata.st_mode)
        or entries != [source]
    ):
        raise HolisticContainerError("isolated output mount contains unexpected entries")
    try:
        _rename_directory_no_replace(source, destination)
        root_descriptor = os.open(
            output_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(root_descriptor)
        finally:
            os.close(root_descriptor)
        observed = invocation_root.lstat()
        if (
            observed.st_dev != invocation_metadata.st_dev
            or observed.st_ino != invocation_metadata.st_ino
        ):
            raise OSError(errno.ESTALE, "isolated output mount identity changed")
        os.rmdir(invocation_root)
    except OSError as exc:
        raise HolisticContainerError("validated output could not be published create-once") from exc
    final_raw = destination / RAW_FILENAME
    final_receipt = destination / RECEIPT_FILENAME
    if not final_raw.is_file() or not final_receipt.is_file():
        raise HolisticContainerError("published output disappeared after atomic rename")
    final_raw_digest, _ = _file_sha256(final_raw, maximum_bytes=MAXIMUM_RAW_ARTIFACT_BYTES)
    _, final_receipt_digest = _read_file_pinned(final_receipt, maximum_bytes=MAXIMUM_RECEIPT_BYTES)
    if (
        final_raw_digest != result.raw_artifact_sha256
        or final_receipt_digest != result.receipt_sha256
    ):
        raise HolisticContainerError("published output changed across atomic rename")
    return HolisticExtractionResult(
        sample_id=result.sample_id,
        image_id=result.image_id,
        raw_artifact_path=final_raw,
        receipt_path=final_receipt,
        raw_artifact_sha256=result.raw_artifact_sha256,
        receipt_sha256=result.receipt_sha256,
        receipt_content_sha256=result.receipt_content_sha256,
        frame_count=result.frame_count,
        status=result.status,
    )


def _validate_authority(
    *,
    sample_id: str,
    clip_start_us: int,
    clip_end_us: int,
    input_mirrored: bool,
    expected_video_sha256: str | None,
    expected_video_byte_count: int | None,
) -> None:
    if not isinstance(sample_id, str) or SAFE_SAMPLE_ID.fullmatch(sample_id) is None:
        raise HolisticContainerError("sample_id must be a safe lowercase identifier")
    if type(clip_start_us) is not int or type(clip_end_us) is not int:
        raise HolisticContainerError("clip bounds must be exact integer microseconds")
    if clip_start_us < 0 or not MINIMUM_CLIP_US <= clip_end_us - clip_start_us <= MAXIMUM_CLIP_US:
        raise HolisticContainerError("clip bounds must name a 2-through-15-second interval")
    if type(input_mirrored) is not bool:
        raise HolisticContainerError("input_mirrored must be an explicit boolean")
    if expected_video_sha256 is not None and (
        not isinstance(expected_video_sha256, str)
        or SHA256.fullmatch(expected_video_sha256) is None
    ):
        raise HolisticContainerError("expected video digest must be lowercase SHA-256")
    if expected_video_byte_count is not None and (
        type(expected_video_byte_count) is not int or expected_video_byte_count <= 0
    ):
        raise HolisticContainerError("expected video byte count must be a positive integer")


def _validate_umi_whole_video_authority(
    *,
    sample_id: str,
    input_mirrored: bool,
    expected_video_sha256: str | None,
    expected_video_byte_count: int | None,
) -> None:
    if not isinstance(sample_id, str) or SAFE_SAMPLE_ID.fullmatch(sample_id) is None:
        raise HolisticContainerError("sample_id must be a safe lowercase identifier")
    if type(input_mirrored) is not bool:
        raise HolisticContainerError("input_mirrored must be an explicit boolean")
    if (
        not isinstance(expected_video_sha256, str)
        or SHA256.fullmatch(expected_video_sha256) is None
    ):
        raise HolisticContainerError("UMI whole-video extraction requires the request SHA-256")
    if (
        type(expected_video_byte_count) is not int
        or not 0 < expected_video_byte_count <= MAXIMUM_UMI_VIDEO_BYTES
    ):
        raise HolisticContainerError(
            "UMI whole-video extraction requires a positive request size no greater than 16 MiB"
        )


def _completion_whole_video_bounds(completion_stdout: bytes) -> tuple[int, int]:
    if not completion_stdout.endswith(b"\n") or completion_stdout.count(b"\n") != 1:
        raise HolisticContainerError("worker completion record framing is invalid")
    raw = completion_stdout[:-1]
    completion = _strict_json_bytes(
        raw,
        label="worker completion record",
        maximum_bytes=MAXIMUM_COMPLETION_BYTES,
    )
    if canonical_json_bytes(completion) != raw:
        raise HolisticContainerError("worker completion record is not RFC 8785 canonical JSON")
    start = completion.get("clip_start_us")
    end = completion.get("clip_end_us")
    if (
        type(start) is not int
        or type(end) is not int
        or start != 0
        or not MINIMUM_CLIP_US <= end <= MAXIMUM_CLIP_US
    ):
        raise HolisticContainerError("worker returned invalid UMI whole-video bounds")
    return start, end


def _validated_input_authority(
    *,
    video_path: Path,
    model_path: Path,
    expected_video_sha256: str | None,
    expected_video_byte_count: int | None,
) -> tuple[Path, str, int, Path, str, int]:
    video, video_sha256, video_bytes = _safe_mount_path(
        video_path, maximum_bytes=MAXIMUM_SOURCE_VIDEO_BYTES
    )
    model, model_sha256, model_bytes = _safe_mount_path(
        model_path, maximum_bytes=MAXIMUM_MODEL_BYTES
    )
    if model_sha256 != EXPECTED_MODEL_SHA256:
        raise HolisticContainerError("model is not the pinned Holistic task artifact")
    if expected_video_sha256 is not None and video_sha256 != expected_video_sha256:
        raise HolisticContainerError("mounted source digest disagrees with inventory authority")
    if expected_video_byte_count is not None and video_bytes != expected_video_byte_count:
        raise HolisticContainerError("mounted source size disagrees with inventory authority")
    return video, video_sha256, video_bytes, model, model_sha256, model_bytes


def load_existing_holistic_extraction(
    *,
    video_path: Path,
    model_path: Path,
    output_root: Path,
    sample_id: str,
    clip_start_us: int,
    clip_end_us: int,
    input_mirrored: bool = False,
    expected_video_sha256: str | None = None,
    expected_video_byte_count: int | None = None,
    container_platform: str = "linux/arm64",
) -> HolisticExtractionResult:
    """Verify and load a create-once sample without executing Docker again."""

    _validate_authority(
        sample_id=sample_id,
        clip_start_us=clip_start_us,
        clip_end_us=clip_end_us,
        input_mirrored=input_mirrored,
        expected_video_sha256=expected_video_sha256,
        expected_video_byte_count=expected_video_byte_count,
    )
    video, video_sha256, video_bytes, _, model_sha256, model_bytes = _validated_input_authority(
        video_path=video_path,
        model_path=model_path,
        expected_video_sha256=expected_video_sha256,
        expected_video_byte_count=expected_video_byte_count,
    )
    del video
    output, exists = _prepare_output_root(
        output_root, sample_id=sample_id, allow_existing_sample=True
    )
    if not exists:
        raise HolisticContainerError("existing Holistic extraction is unavailable")
    return _validate_outputs(
        output_root=output,
        sample_id=sample_id,
        expected_image_id=None,
        video_sha256=video_sha256,
        video_bytes=video_bytes,
        model_sha256=model_sha256,
        model_bytes=model_bytes,
        clip_start_us=clip_start_us,
        clip_end_us=clip_end_us,
        input_mirrored=input_mirrored,
        completion_stdout=None,
        expected_platform=container_platform,
    )


def extract_holistic_in_container(
    *,
    video_path: Path,
    model_path: Path,
    output_root: Path,
    sample_id: str,
    clip_start_us: int | None = None,
    clip_end_us: int | None = None,
    input_mirrored: bool = False,
    expected_video_sha256: str | None = None,
    expected_video_byte_count: int | None = None,
    resume: bool = False,
    image_reference: str | None = None,
    container_platform: str = "linux/arm64",
    container_name: str | None = None,
    docker_executable: str = "docker",
    timeout_seconds: float = 180.0,
    _umi_whole_video: bool = False,
    _runner: CommandRunner = _run_bounded_command,
) -> HolisticExtractionResult:
    """Run one local-only extraction without loading MediaPipe into the host process.

    The result exposes paths and digests only. It never deserializes or returns raw
    landmark arrays, and child stdout/stderr is never included in a raised error.
    """

    if _umi_whole_video:
        if clip_start_us is not None or clip_end_us is not None:
            raise HolisticContainerError("UMI whole-video mode refuses caller clip bounds")
        if resume:
            raise HolisticContainerError("UMI whole-video mode does not resume existing output")
        _validate_umi_whole_video_authority(
            sample_id=sample_id,
            input_mirrored=input_mirrored,
            expected_video_sha256=expected_video_sha256,
            expected_video_byte_count=expected_video_byte_count,
        )
    else:
        _validate_authority(
            sample_id=sample_id,
            clip_start_us=clip_start_us,  # type: ignore[arg-type]
            clip_end_us=clip_end_us,  # type: ignore[arg-type]
            input_mirrored=input_mirrored,
            expected_video_sha256=expected_video_sha256,
            expected_video_byte_count=expected_video_byte_count,
        )
    if type(resume) is not bool:
        raise HolisticContainerError("resume must be an explicit boolean")
    if container_name is not None and (
        not isinstance(container_name, str) or CONTAINER_NAME.fullmatch(container_name) is None
    ):
        raise HolisticContainerError(
            "container_name must be bitsign-holistic- followed by 32 lowercase hex digits"
        )
    if type(timeout_seconds) not in (int, float) or not 1 <= timeout_seconds <= 3600:
        raise HolisticContainerError("container timeout must be from 1 through 3600 seconds")
    if os.geteuid() == 0:
        raise HolisticContainerError("host extraction refuses to launch a root container user")
    video, video_sha256, video_bytes, model, model_sha256, model_bytes = _validated_input_authority(
        video_path=video_path,
        model_path=model_path,
        expected_video_sha256=expected_video_sha256,
        expected_video_byte_count=expected_video_byte_count,
    )
    output, exists = _prepare_output_root(
        output_root, sample_id=sample_id, allow_existing_sample=resume
    )
    if exists:
        assert clip_start_us is not None and clip_end_us is not None
        return _validate_outputs(
            output_root=output,
            sample_id=sample_id,
            expected_image_id=None,
            video_sha256=video_sha256,
            video_bytes=video_bytes,
            model_sha256=model_sha256,
            model_bytes=model_bytes,
            clip_start_us=clip_start_us,
            clip_end_us=clip_end_us,
            input_mirrored=input_mirrored,
            completion_stdout=None,
            expected_platform=container_platform,
        )
    image = resolve_holistic_container_image(
        image_reference,
        platform=container_platform,
        docker_executable=docker_executable,
        timeout_seconds=min(float(timeout_seconds), 30.0),
        _runner=_runner,
    )
    invocation_output = _create_isolated_output_mount(output, sample_id=sample_id)
    if container_name is None:
        container_name = f"bitsign-holistic-{uuid.uuid4().hex}"
    user = f"{os.geteuid()}:{os.getegid()}"
    command = [
        docker_executable,
        "run",
        "--rm",
        "--init",
        "--name",
        container_name,
        "--pull",
        "never",
        "--platform",
        container_platform,
        "--network",
        "none",
        "--read-only",
        "--user",
        user,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--pids-limit",
        "128",
        "--memory",
        "4g",
        "--memory-swap",
        "4g",
        "--cpus",
        "4",
        "--ulimit",
        "nofile=256:256",
        "--ipc",
        "none",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=536870912,mode=1777",
        "--env",
        "HOME=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "MEDIAPIPE_DISABLE_GPU=1",
        "--env",
        "LIBGL_ALWAYS_SOFTWARE=1",
        "--mount",
        _docker_mount(video, "/input/video", read_only=True),
        "--mount",
        _docker_mount(model, "/model/holistic.task", read_only=True),
        "--mount",
        _docker_mount(invocation_output, "/output", read_only=False),
        image.image_id,
        "--sample-id",
        sample_id,
        "--expected-video-sha256",
        video_sha256,
        "--expected-model-sha256",
        model_sha256,
        "--container-image-id",
        image.image_id,
    ]
    if _umi_whole_video:
        command.append("--whole-video")
    else:
        assert clip_start_us is not None and clip_end_us is not None
        command.extend(
            [
                "--clip-start-us",
                str(clip_start_us),
                "--clip-end-us",
                str(clip_end_us),
            ]
        )
    command.extend(
        [
            "--input-mirrored",
            "true" if input_mirrored else "false",
        ]
    )
    try:
        result = _runner_call(
            _runner,
            command,
            timeout_seconds=float(timeout_seconds),
            stdout_limit=MAXIMUM_COMPLETION_BYTES,
            stderr_limit=MAXIMUM_DOCKER_DIAGNOSTIC_BYTES,
        )
    except _BoundedCommandFailure as exc:
        container_stopped = _cleanup_container_after_runner_failure(
            name=container_name,
            docker_executable=docker_executable,
            runner=_runner,
        )
        if container_stopped:
            _remove_isolated_output_mount(invocation_output, output_root=output)
        if exc.reason == "timeout":
            raise HolisticContainerError("isolated Holistic extraction timed out") from exc
        if exc.reason == "output-limit":
            raise HolisticContainerError(
                "isolated Holistic extraction exceeded its diagnostic byte ceiling"
            ) from exc
        raise HolisticContainerError("isolated Holistic extraction could not start") from exc
    if result.return_code != 0:
        _remove_isolated_output_mount(invocation_output, output_root=output)
        raise HolisticContainerError(
            f"isolated Holistic extraction exited with status {result.return_code}"
        )
    try:
        observed_video_sha256, observed_video_bytes = _file_sha256(
            video, maximum_bytes=MAXIMUM_SOURCE_VIDEO_BYTES
        )
        observed_model_sha256, observed_model_bytes = _file_sha256(
            model, maximum_bytes=MAXIMUM_MODEL_BYTES
        )
        if (observed_video_sha256, observed_video_bytes) != (video_sha256, video_bytes) or (
            observed_model_sha256,
            observed_model_bytes,
        ) != (model_sha256, model_bytes):
            raise HolisticContainerError("mounted extraction input changed during execution")
        if _umi_whole_video:
            clip_start_us, clip_end_us = _completion_whole_video_bounds(result.stdout)
        assert clip_start_us is not None and clip_end_us is not None
        validated = _validate_outputs(
            output_root=invocation_output,
            sample_id=sample_id,
            expected_image_id=image.image_id,
            video_sha256=video_sha256,
            video_bytes=video_bytes,
            model_sha256=model_sha256,
            model_bytes=model_bytes,
            clip_start_us=clip_start_us,
            clip_end_us=clip_end_us,
            input_mirrored=input_mirrored,
            completion_stdout=result.stdout,
            expected_platform=container_platform,
            require_umi_whole_video=_umi_whole_video,
        )
        return _publish_validated_output(
            validated,
            invocation_root=invocation_output,
            output_root=output,
        )
    except BaseException:
        try:
            invocation_output.lstat()
        except FileNotFoundError:
            pass
        else:
            _remove_isolated_output_mount(invocation_output, output_root=output)
        raise


def extract_umi_whole_video_in_container(
    *,
    video_path: Path,
    model_path: Path,
    output_root: Path,
    sample_id: str,
    expected_video_sha256: str,
    expected_video_byte_count: int,
    input_mirrored: bool = False,
    image_reference: str | None = None,
    container_platform: str = "linux/amd64",
    container_name: str | None = None,
    docker_executable: str = "docker",
    timeout_seconds: float = 180.0,
    _runner: CommandRunner = _run_bounded_command,
) -> HolisticExtractionResult:
    """Extract one complete 2-through-15-second UMI request without caller clip bounds."""

    return extract_holistic_in_container(
        video_path=video_path,
        model_path=model_path,
        output_root=output_root,
        sample_id=sample_id,
        input_mirrored=input_mirrored,
        expected_video_sha256=expected_video_sha256,
        expected_video_byte_count=expected_video_byte_count,
        image_reference=image_reference,
        container_platform=container_platform,
        container_name=container_name,
        docker_executable=docker_executable,
        timeout_seconds=timeout_seconds,
        _umi_whole_video=True,
        _runner=_runner,
    )
