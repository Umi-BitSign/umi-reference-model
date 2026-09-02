from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import signal
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

import numpy as np

from . import amd64_holistic_container
from . import holistic_container as arm64_holistic_container
from .canonical import canonical_json_bytes
from .holistic_container import EXPECTED_MODEL_SHA256
from .holistic_motion import (
    HolisticMotionIntegrationError,
    convert_holistic_raw_to_motion,
)
from .s1_portable_runtime import S1PortableError, load_s1_portable_bundle

UMI_REFERENCE_BACKEND_STATUS: Final = "component_test_no_weight"
UMI_REFERENCE_JOB_SCHEMA: Final = "umi-s1-reference-miner-job/1"
UMI_REFERENCE_RESULT_SCHEMA: Final = "umi-s1-reference-miner-result/1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_DEVICE = re.compile(r"(?:cpu|mps|cuda(?::[0-9]+)?)")
_CONTAINER_NAME = re.compile(r"bitsign-holistic-[0-9a-f]{32}")
_JOB_DIRECTORY = re.compile(r"umi-s1-job-[0-9a-f]{32}")
_PLATFORMS: Final = frozenset({"linux/arm64", "linux/amd64"})
_MAXIMUM_JOB_BYTES: Final = 64 * 1024
_MAXIMUM_RESULT_BYTES: Final = 16 * 1024
_MAXIMUM_HYPOTHESIS_BYTES: Final = 4 * 1024
_TERMINATION_GRACE_SECONDS: Final = 2.0
_DOCKER_CONTROL_TIMEOUT_SECONDS: Final = 5.0
_WORKER_MODULE: Final = "bitsign_motion.umi_reference_backend"


class UmiReferenceBackendError(RuntimeError):
    """Raised when the component-test reference backend cannot translate safely."""


class _VideoAuthority(Protocol):
    sha256: str
    size_bytes: int


class TranslationRequestCompatible(Protocol):
    video: _VideoAuthority


@dataclass(frozen=True, slots=True)
class ReferenceBackendConfig:
    bundle: Path
    inference_revision: str
    extractor_image: str
    extractor_model: Path
    extractor_platform: str
    docker_executable: Path
    temporary_root: Path
    device: str
    hard_deadline_seconds: float


@dataclass(frozen=True, slots=True)
class _RequestVideo:
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _CliRequest:
    video: _RequestVideo


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise UmiReferenceBackendError(f"{label} must be lowercase hexadecimal SHA-256")
    return value


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        raise UmiReferenceBackendError(f"required environment setting is missing: {name}")
    return value


def _absolute_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise UmiReferenceBackendError(f"{label} must be an absolute path")
    return path


def _validate_private_directory(path: Path, *, create: bool) -> Path:
    absolute = Path(os.path.abspath(path))
    if create:
        try:
            absolute.mkdir(mode=0o700, parents=False, exist_ok=True)
        except OSError as exc:
            raise UmiReferenceBackendError("temporary root cannot be created") from exc
    try:
        initial = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise UmiReferenceBackendError("temporary root is unavailable") from exc
    if (
        resolved != absolute
        or absolute.is_symlink()
        or not stat.S_ISDIR(initial.st_mode)
        or initial.st_uid != os.geteuid()
        or initial.st_mode & 0o077
    ):
        raise UmiReferenceBackendError(
            "temporary root must be a direct owner-only non-symlink directory"
        )
    return resolved


def _configuration_from_environment() -> ReferenceBackendConfig:
    revision = _require_sha256(
        _required_environment("UMI_S1_INFERENCE_REVISION"), "inference revision"
    )
    image = _required_environment("UMI_S1_EXTRACTOR_IMAGE")
    if _IMAGE_ID.fullmatch(image) is None:
        raise UmiReferenceBackendError("extractor image must be an immutable local image ID")
    platform_id = _required_environment("UMI_S1_EXTRACTOR_PLATFORM")
    if platform_id != "linux/amd64":
        raise UmiReferenceBackendError(
            "live UMI extraction currently requires the linux/amd64 whole-video worker"
        )
    device = _required_environment("UMI_S1_DEVICE")
    if _DEVICE.fullmatch(device) is None:
        raise UmiReferenceBackendError("inference device is unsupported")
    raw_deadline = _required_environment("UMI_S1_HARD_DEADLINE_SECONDS")
    try:
        deadline = float(raw_deadline)
    except ValueError as exc:
        raise UmiReferenceBackendError("hard deadline must be numeric") from exc
    if not 1.0 <= deadline <= 3_600.0 or not deadline < float("inf"):
        raise UmiReferenceBackendError("hard deadline must be from 1 through 3600 seconds")

    bundle = _absolute_path(_required_environment("UMI_S1_BUNDLE"), "bundle")
    extractor_model = _absolute_path(
        _required_environment("UMI_S1_EXTRACTOR_MODEL"), "extractor model"
    )
    temporary_root_value = _absolute_path(
        _required_environment("UMI_S1_TEMP_ROOT"), "temporary root"
    )
    docker = _absolute_path(_required_environment("UMI_S1_DOCKER_EXECUTABLE"), "Docker executable")
    try:
        docker_metadata = docker.stat()
    except OSError as exc:
        raise UmiReferenceBackendError("Docker executable is unavailable") from exc
    if not stat.S_ISREG(docker_metadata.st_mode) or not os.access(docker, os.X_OK):
        raise UmiReferenceBackendError("Docker executable is not executable")
    return ReferenceBackendConfig(
        bundle=bundle,
        inference_revision=revision,
        extractor_image=image,
        extractor_model=extractor_model,
        extractor_platform=platform_id,
        docker_executable=docker,
        temporary_root=_validate_private_directory(temporary_root_value, create=True),
        device=device,
        hard_deadline_seconds=deadline,
    )


def _strict_json(payload: bytes, *, maximum_bytes: int, label: str) -> dict[str, Any]:
    if not 1 <= len(payload) <= maximum_bytes:
        raise UmiReferenceBackendError(f"{label} violates its byte ceiling")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise UmiReferenceBackendError(f"{label} contains a duplicate member")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                UmiReferenceBackendError(f"{label} contains {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UmiReferenceBackendError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
        raise UmiReferenceBackendError(f"{label} is not a canonical JSON object")
    return value


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:  # pragma: no cover
                raise OSError("write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _read_bounded_regular_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise UmiReferenceBackendError(f"{label} cannot be opened safely") from exc
    try:
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or initial.st_nlink != 1
            or not 1 <= initial.st_size <= maximum_bytes
        ):
            raise UmiReferenceBackendError(f"{label} violates its input contract")
        payload = bytearray()
        while len(payload) <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        final = os.fstat(descriptor)
        if (
            len(payload) != initial.st_size
            or len(payload) > maximum_bytes
            or (initial.st_dev, initial.st_ino, initial.st_size, initial.st_mtime_ns)
            != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
        ):
            raise UmiReferenceBackendError(f"{label} changed while being read")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _remove_private_job_root(job_root: Path, *, temporary_root: Path) -> None:
    """Remove exactly one generated job directory and verify that it is gone."""

    root = Path(os.path.abspath(temporary_root))
    target = Path(os.path.abspath(job_root))
    if (
        target.parent != root
        or _JOB_DIRECTORY.fullmatch(target.name) is None
        or not getattr(shutil.rmtree, "avoids_symlink_attacks", False)
    ):
        raise UmiReferenceBackendError("temporary cleanup target is not a pinned job directory")
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise UmiReferenceBackendError("temporary job directory cannot be inspected") from exc
    if target.is_symlink() or not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise UmiReferenceBackendError("temporary cleanup target changed identity")
    try:
        shutil.rmtree(target)
    except OSError as exc:
        raise UmiReferenceBackendError("temporary job directory cleanup failed") from exc
    try:
        target.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise UmiReferenceBackendError("temporary cleanup result cannot be verified") from exc
    raise UmiReferenceBackendError("temporary job directory remained after cleanup")


def _job_record(
    config: ReferenceBackendConfig,
    *,
    operation: str,
    job_root: Path,
    video_sha256: str | None = None,
    video_size_bytes: int | None = None,
) -> dict[str, Any]:
    if operation not in {"probe", "translate"}:
        raise ValueError("worker operation is invalid")
    record: dict[str, Any] = {
        "schema": UMI_REFERENCE_JOB_SCHEMA,
        "operation": operation,
        "status": UMI_REFERENCE_BACKEND_STATUS,
        "bundle": str(config.bundle),
        "inference_revision": config.inference_revision,
        "extractor_image": config.extractor_image,
        "extractor_model": str(config.extractor_model),
        "extractor_platform": config.extractor_platform,
        "docker_executable": str(config.docker_executable),
        "device": config.device,
        "timeout_seconds": config.hard_deadline_seconds,
        "result": str(job_root / "result.json"),
    }
    if operation == "translate":
        assert video_sha256 is not None and video_size_bytes is not None
        sample_id = job_root.name.removeprefix("umi-s1-job-")
        record.update(
            {
                "sample_id": sample_id,
                "container_name": f"bitsign-holistic-{sample_id}",
                "video": str(job_root / "video.mp4"),
                "video_sha256": video_sha256,
                "video_size_bytes": video_size_bytes,
                "extraction_root": str(job_root / "extraction"),
                "motion": str(job_root / f"{sample_id}.npz"),
            }
        )
    return record


def _validate_job(record: object) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise UmiReferenceBackendError("worker job must be an object")
    operation = record.get("operation")
    common = {
        "schema",
        "operation",
        "status",
        "bundle",
        "inference_revision",
        "extractor_image",
        "extractor_model",
        "extractor_platform",
        "docker_executable",
        "device",
        "timeout_seconds",
        "result",
    }
    translated = {
        "sample_id",
        "container_name",
        "video",
        "video_sha256",
        "video_size_bytes",
        "extraction_root",
        "motion",
    }
    expected = common if operation == "probe" else common | translated
    if operation not in {"probe", "translate"} or set(record) != expected:
        raise UmiReferenceBackendError("worker job field set is invalid")
    if (
        record["schema"] != UMI_REFERENCE_JOB_SCHEMA
        or record["status"] != UMI_REFERENCE_BACKEND_STATUS
    ):
        raise UmiReferenceBackendError("worker job schema or status is invalid")
    _require_sha256(record["inference_revision"], "worker inference revision")
    if (
        not isinstance(record["extractor_image"], str)
        or _IMAGE_ID.fullmatch(record["extractor_image"]) is None
    ):
        raise UmiReferenceBackendError("worker extractor image is invalid")
    if record["extractor_platform"] not in _PLATFORMS:
        raise UmiReferenceBackendError("worker extractor platform is invalid")
    if not isinstance(record["device"], str) or _DEVICE.fullmatch(record["device"]) is None:
        raise UmiReferenceBackendError("worker device is invalid")
    if (
        type(record["timeout_seconds"]) not in (int, float)
        or not 1 <= record["timeout_seconds"] <= 3_600
    ):
        raise UmiReferenceBackendError("worker timeout is invalid")
    for field in ("bundle", "extractor_model", "docker_executable", "result"):
        if not isinstance(record[field], str) or not Path(record[field]).is_absolute():
            raise UmiReferenceBackendError(f"worker path is invalid: {field}")
    if operation == "translate":
        _require_sha256(record["video_sha256"], "worker video digest")
        if (
            type(record["video_size_bytes"]) is not int
            or not 1
            <= record["video_size_bytes"]
            <= amd64_holistic_container.MAXIMUM_UMI_VIDEO_BYTES
        ):
            raise UmiReferenceBackendError("worker video size is invalid")
        if not isinstance(record["sample_id"], str) or not re.fullmatch(
            r"[0-9a-f]{32}", record["sample_id"]
        ):
            raise UmiReferenceBackendError("worker sample ID is invalid")
        if record["container_name"] != f"bitsign-holistic-{record['sample_id']}":
            raise UmiReferenceBackendError("worker container name is not bound to its sample")
        for field in ("video", "extraction_root", "motion"):
            if not isinstance(record[field], str) or not Path(record[field]).is_absolute():
                raise UmiReferenceBackendError(f"worker path is invalid: {field}")
    return record


def _verify_model_file(path: Path) -> None:
    payload = _read_bounded_regular_file(
        path, maximum_bytes=32 * 1024 * 1024, label="extractor model"
    )
    if hashlib.sha256(payload).hexdigest() != EXPECTED_MODEL_SHA256:
        raise UmiReferenceBackendError("extractor model digest differs")


def _load_worker_runtime(job: Mapping[str, Any]):
    runtime = load_s1_portable_bundle(
        Path(cast(str, job["bundle"])),
        expected_inference_revision=cast(str, job["inference_revision"]),
        device=cast(str, job["device"]),
    )
    required_image = runtime.require_preprocessing_platform(cast(str, job["extractor_platform"]))
    if required_image != job["extractor_image"]:
        raise UmiReferenceBackendError("configured extractor image differs from the bundle")
    _verify_model_file(Path(cast(str, job["extractor_model"])))
    container_module = (
        amd64_holistic_container
        if job["extractor_platform"] == "linux/amd64"
        else arm64_holistic_container
    )
    resolved = container_module.resolve_holistic_container_image(
        cast(str, job["extractor_image"]),
        platform=cast(str, job["extractor_platform"]),
        docker_executable=cast(str, job["docker_executable"]),
        timeout_seconds=min(float(job["timeout_seconds"]), 30.0),
    )
    if resolved.image_id != job["extractor_image"]:
        raise UmiReferenceBackendError("resolved extractor image differs")
    return runtime


def _load_motion(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = _read_bounded_regular_file(
        path, maximum_bytes=2 * 1024 * 1024, label="motion artifact"
    )
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if sorted(archive.files) != ["frame_mask", "metadata", "motion"]:
                raise UmiReferenceBackendError("motion artifact field set is invalid")
            motion = np.array(archive["motion"], copy=True)
            frame_mask = np.array(archive["frame_mask"], copy=True)
    except (KeyError, OSError, ValueError) as exc:
        raise UmiReferenceBackendError("motion artifact cannot be decoded") from exc
    return motion, frame_mask


def _execute_worker_job(job: Mapping[str, Any]) -> dict[str, Any]:
    runtime = _load_worker_runtime(job)
    if job["operation"] == "probe":
        return {
            "schema": UMI_REFERENCE_RESULT_SCHEMA,
            "status": "ready",
            "claim_status": UMI_REFERENCE_BACKEND_STATUS,
            "inference_revision": runtime.inference_revision,
        }
    if job["extractor_platform"] != "linux/amd64":
        raise UmiReferenceBackendError(
            "live UMI extraction requires the linux/amd64 whole-video worker"
        )
    extraction = amd64_holistic_container.extract_umi_whole_video_in_container(
        video_path=Path(cast(str, job["video"])),
        model_path=Path(cast(str, job["extractor_model"])),
        output_root=Path(cast(str, job["extraction_root"])),
        sample_id=cast(str, job["sample_id"]),
        expected_video_sha256=cast(str, job["video_sha256"]),
        expected_video_byte_count=cast(int, job["video_size_bytes"]),
        input_mirrored=False,
        image_reference=cast(str, job["extractor_image"]),
        container_platform=cast(str, job["extractor_platform"]),
        docker_executable=cast(str, job["docker_executable"]),
        timeout_seconds=float(job["timeout_seconds"]),
        container_name=cast(str, job["container_name"]),
    )
    conversion = convert_holistic_raw_to_motion(
        raw_path=extraction.raw_artifact_path,
        receipt_path=extraction.receipt_path,
        output_path=Path(cast(str, job["motion"])),
        sample_id=extraction.sample_id,
        expected_raw_sha256=extraction.raw_artifact_sha256,
        expected_receipt_sha256=extraction.receipt_sha256,
        expected_receipt_content_sha256=extraction.receipt_content_sha256,
        expected_frame_count=extraction.frame_count,
    )
    motion, frame_mask = _load_motion(conversion.motion_artifact.path)
    inference = runtime.infer_motion(motion, frame_mask)
    encoded = inference.text.encode("utf-8")
    if len(encoded) > _MAXIMUM_HYPOTHESIS_BYTES:
        raise UmiReferenceBackendError("inference text exceeds the UMI byte ceiling")
    return {
        "schema": UMI_REFERENCE_RESULT_SCHEMA,
        "status": "ok",
        "claim_status": UMI_REFERENCE_BACKEND_STATUS,
        "inference_revision": runtime.inference_revision,
        "hypothesis": inference.text,
    }


def _worker(job_path: Path) -> int:
    try:
        payload = _read_bounded_regular_file(
            job_path, maximum_bytes=_MAXIMUM_JOB_BYTES, label="worker job"
        )
        job = _validate_job(
            _strict_json(payload, maximum_bytes=_MAXIMUM_JOB_BYTES, label="worker job")
        )
        result = _execute_worker_job(job)
        _write_exclusive(Path(cast(str, job["result"])), canonical_json_bytes(result))
        return 0
    except (
        amd64_holistic_container.HolisticContainerError,
        arm64_holistic_container.HolisticContainerError,
        HolisticMotionIntegrationError,
        OSError,
        S1PortableError,
        UmiReferenceBackendError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return 2


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        await process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(process.wait(), timeout=_TERMINATION_GRACE_SECONDS)
        return
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await process.wait()


def _child_environment() -> dict[str, str]:
    environment = {
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PATH": "/usr/bin:/bin",
    }
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_FAST_MATH",
        "PYTORCH_MPS_PREFER_METAL",
    ):
        value = os.environ.get(name)
        if value is not None:
            environment[name] = value
    return environment


async def _run_killable_process(
    command: list[str],
    *,
    deadline_seconds: float,
) -> int:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env=_child_environment(),
        close_fds=True,
        start_new_session=True,
    )
    try:
        return await asyncio.wait_for(process.wait(), timeout=deadline_seconds)
    except BaseException:
        await asyncio.shield(_terminate_process_group(process))
        raise


async def _docker_daemon_is_available(docker_executable: Path) -> bool:
    return_code = await _run_killable_process(
        [str(docker_executable), "version", "--format", "{{.Server.Version}}"],
        deadline_seconds=_DOCKER_CONTROL_TIMEOUT_SECONDS,
    )
    return return_code == 0


async def _docker_container_exists(docker_executable: Path, container_name: str) -> bool:
    if _CONTAINER_NAME.fullmatch(container_name) is None:
        raise UmiReferenceBackendError("Docker cleanup container name is invalid")
    return_code = await _run_killable_process(
        [str(docker_executable), "container", "inspect", container_name],
        deadline_seconds=_DOCKER_CONTROL_TIMEOUT_SECONDS,
    )
    if return_code == 0:
        return True
    if not await _docker_daemon_is_available(docker_executable):
        raise UmiReferenceBackendError("Docker daemon is unavailable during container control")
    return False


async def _require_docker_container_absent(
    docker_executable: Path, container_name: str
) -> None:
    if await _docker_container_exists(docker_executable, container_name):
        raise UmiReferenceBackendError("isolated translation container name already exists")


async def _force_remove_docker_container(
    docker_executable: Path, container_name: str
) -> None:
    if not await _docker_container_exists(docker_executable, container_name):
        return
    await _run_killable_process(
        [str(docker_executable), "container", "rm", "--force", container_name],
        deadline_seconds=_DOCKER_CONTROL_TIMEOUT_SECONDS,
    )
    if await _docker_container_exists(docker_executable, container_name):
        raise UmiReferenceBackendError("isolated translation container survived forced removal")


async def _finish_container_cleanup(
    docker_executable: Path, container_name: str
) -> tuple[BaseException | None, asyncio.CancelledError | None]:
    """Finish daemon-side cleanup even if the caller is canceled again."""

    task = asyncio.create_task(
        _force_remove_docker_container(docker_executable, container_name)
    )
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = cancellation or exc
    try:
        task.result()
    except BaseException as exc:
        return exc, cancellation
    return None, cancellation


async def _launch_worker_job(
    job_path: Path,
    *,
    deadline_seconds: float,
    docker_executable: Path,
    container_name: str | None,
) -> None:
    if container_name is None:
        return_code = await _run_killable_process(
            [sys.executable, "-m", _WORKER_MODULE, "--_worker", str(job_path)],
            deadline_seconds=deadline_seconds,
        )
        if return_code != 0:
            raise UmiReferenceBackendError("isolated translation job failed")
        return
    await _require_docker_container_absent(docker_executable, container_name)
    primary_error: BaseException | None = None
    return_code: int | None = None
    try:
        return_code = await _run_killable_process(
            [sys.executable, "-m", _WORKER_MODULE, "--_worker", str(job_path)],
            deadline_seconds=deadline_seconds,
        )
    except BaseException as exc:
        primary_error = exc
    cleanup_error, cleanup_cancellation = await _finish_container_cleanup(
        docker_executable, container_name
    )
    if primary_error is None and cleanup_cancellation is not None:
        primary_error = cleanup_cancellation
    if primary_error is not None:
        if cleanup_error is not None:
            primary_error.add_note(
                f"daemon-side container cleanup also failed: {type(cleanup_error).__name__}"
            )
            primary_error.container_cleanup_error = cleanup_error  # type: ignore[attr-defined]
            raise primary_error.with_traceback(primary_error.__traceback__) from cleanup_error
        raise primary_error.with_traceback(primary_error.__traceback__)
    if cleanup_error is not None:
        raise cleanup_error
    assert return_code is not None
    if return_code != 0:
        raise UmiReferenceBackendError("isolated translation job failed")


def _validate_worker_result(
    result: object,
    *,
    expected_revision: str,
    operation: str,
) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise UmiReferenceBackendError("worker result must be an object")
    common = {"schema", "status", "claim_status", "inference_revision"}
    expected = common if operation == "probe" else common | {"hypothesis"}
    if set(result) != expected:
        raise UmiReferenceBackendError("worker result field set is invalid")
    if (
        result["schema"] != UMI_REFERENCE_RESULT_SCHEMA
        or result["claim_status"] != UMI_REFERENCE_BACKEND_STATUS
        or result["inference_revision"] != expected_revision
        or result["status"] != ("ready" if operation == "probe" else "ok")
    ):
        raise UmiReferenceBackendError("worker result identity differs")
    if operation == "translate":
        hypothesis = result["hypothesis"]
        if not isinstance(hypothesis, str):
            raise UmiReferenceBackendError("worker hypothesis is not text")
        try:
            encoded = hypothesis.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise UmiReferenceBackendError("worker hypothesis is not valid UTF-8") from exc
        if len(encoded) > _MAXIMUM_HYPOTHESIS_BYTES:
            raise UmiReferenceBackendError("worker hypothesis exceeds the UMI byte ceiling")
    return result


class UmiS1Translator:
    """Async UMI plugin whose native work is confined to killable child jobs."""

    def __init__(self) -> None:
        self._config: ReferenceBackendConfig | None = None
        self._declared_revision: str | None = None
        self._startup_lock = asyncio.Lock()
        self._jobs: set[asyncio.Task[None]] = set()
        self._closing = False

    @property
    def model_revision(self) -> str:
        configured = _require_sha256(
            _required_environment("UMI_S1_INFERENCE_REVISION"), "inference revision"
        )
        if self._declared_revision is None:
            self._declared_revision = configured
        elif configured != self._declared_revision:
            raise UmiReferenceBackendError(
                "inference revision changed after the backend identity was declared"
            )
        return self._declared_revision

    async def _run_job(
        self,
        config: ReferenceBackendConfig,
        *,
        operation: str,
        video: bytes | None = None,
        video_sha256: str | None = None,
    ) -> dict[str, Any]:
        sample_id = secrets.token_hex(16)
        job_root = config.temporary_root / f"umi-s1-job-{sample_id}"
        job_root.mkdir(mode=0o700)
        result: dict[str, Any] | None = None
        primary_error: BaseException | None = None
        try:
            try:
                if operation == "translate":
                    assert video is not None and video_sha256 is not None
                    _write_exclusive(job_root / "video.mp4", video)
                record = _job_record(
                    config,
                    operation=operation,
                    job_root=job_root,
                    video_sha256=video_sha256,
                    video_size_bytes=None if video is None else len(video),
                )
                job_path = job_root / "job.json"
                _write_exclusive(job_path, canonical_json_bytes(record))
                task = asyncio.create_task(
                    _launch_worker_job(
                        job_path,
                        deadline_seconds=config.hard_deadline_seconds,
                        docker_executable=config.docker_executable,
                        container_name=cast(str | None, record.get("container_name")),
                    )
                )
                self._jobs.add(task)
                try:
                    await task
                finally:
                    self._jobs.discard(task)
                result_payload = _read_bounded_regular_file(
                    job_root / "result.json",
                    maximum_bytes=_MAXIMUM_RESULT_BYTES,
                    label="worker result",
                )
                result = _validate_worker_result(
                    _strict_json(
                        result_payload,
                        maximum_bytes=_MAXIMUM_RESULT_BYTES,
                        label="worker result",
                    ),
                    expected_revision=config.inference_revision,
                    operation=operation,
                )
            except TimeoutError as exc:
                raise UmiReferenceBackendError(
                    "isolated translation job exceeded its deadline"
                ) from exc
        except BaseException as exc:
            primary_error = exc

        cleanup_error: BaseException | None = None
        try:
            _remove_private_job_root(job_root, temporary_root=config.temporary_root)
        except BaseException as exc:
            cleanup_error = exc
        if primary_error is not None:
            if cleanup_error is not None:
                if isinstance(primary_error, asyncio.CancelledError):
                    primary_error.add_note(
                        f"temporary cleanup also failed: {type(cleanup_error).__name__}"
                    )
                    primary_error.temporary_cleanup_error = cleanup_error  # type: ignore[attr-defined]
                    raise primary_error.with_traceback(primary_error.__traceback__)
                raise BaseExceptionGroup(
                    "translation job and temporary cleanup both failed",
                    [primary_error, cleanup_error],
                )
            raise primary_error.with_traceback(primary_error.__traceback__)
        if cleanup_error is not None:
            raise cleanup_error
        assert result is not None
        return result

    async def startup(self) -> None:
        async with self._startup_lock:
            if self._config is not None:
                return
            self._closing = False
            config = _configuration_from_environment()
            if config.inference_revision != self.model_revision:
                raise UmiReferenceBackendError(
                    "startup inference revision differs from the declared backend identity"
                )
            await self._run_job(config, operation="probe")
            self._config = config

    async def shutdown(self) -> None:
        async with self._startup_lock:
            self._closing = True
            jobs = tuple(self._jobs)
            for task in jobs:
                task.cancel()
            if jobs:
                await asyncio.gather(*jobs, return_exceptions=True)
            self._jobs.clear()
            self._config = None

    async def __call__(self, video: bytes, request: TranslationRequestCompatible) -> str:
        if self._config is None or self._closing:
            raise UmiReferenceBackendError("reference translator has not completed startup")
        if (
            type(video) is not bytes
            or not 1 <= len(video) <= amd64_holistic_container.MAXIMUM_UMI_VIDEO_BYTES
        ):
            raise UmiReferenceBackendError("verified video bytes violate the UMI ceiling")
        authority = getattr(request, "video", None)
        expected_sha256 = _require_sha256(
            getattr(authority, "sha256", None), "request video digest"
        )
        expected_size = getattr(authority, "size_bytes", None)
        if type(expected_size) is not int or expected_size != len(video):
            raise UmiReferenceBackendError("request video size differs from verified bytes")
        observed_sha256 = hashlib.sha256(video).hexdigest()
        if observed_sha256 != expected_sha256:
            raise UmiReferenceBackendError("request video digest differs from verified bytes")
        result = await self._run_job(
            self._config,
            operation="translate",
            video=video,
            video_sha256=observed_sha256,
        )
        return cast(str, result["hypothesis"])


# UMI loads this object as ``bitsign_motion.umi_reference_backend:translator``.
translator = UmiS1Translator()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bitsign_motion.umi_reference_backend",
        description=(
            "Probe or run the component-test-only S1 UMI reference backend. Translation "
            "writes its hypothesis to an owner-only result file instead of logs."
        ),
    )
    parser.add_argument("command", nargs="?", choices=("probe", "translate"))
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_worker", type=Path)
    return parser


async def _cli(arguments: argparse.Namespace) -> int:
    backend = UmiS1Translator()
    await backend.startup()
    try:
        if arguments.command == "probe":
            summary = {
                "schema": "umi-s1-reference-miner-cli/1",
                "status": "ready",
                "claim_status": UMI_REFERENCE_BACKEND_STATUS,
                "inference_revision": backend.model_revision,
            }
        else:
            if arguments.video is None or arguments.output is None:
                raise UmiReferenceBackendError("translate requires --video and --output")
            video = _read_bounded_regular_file(
                arguments.video,
                maximum_bytes=amd64_holistic_container.MAXIMUM_UMI_VIDEO_BYTES,
                label="CLI video",
            )
            digest = _require_sha256(arguments.video_sha256, "CLI video digest")
            if hashlib.sha256(video).hexdigest() != digest:
                raise UmiReferenceBackendError("CLI video digest differs")
            hypothesis = await backend(
                video,
                _CliRequest(video=_RequestVideo(sha256=digest, size_bytes=len(video))),
            )
            output = {
                "schema": "umi-s1-reference-miner-cli-result/1",
                "claim_status": UMI_REFERENCE_BACKEND_STATUS,
                "inference_revision": backend.model_revision,
                "hypothesis": hypothesis,
            }
            _write_exclusive(arguments.output, canonical_json_bytes(output))
            summary = {
                "schema": "umi-s1-reference-miner-cli/1",
                "status": "written",
                "claim_status": UMI_REFERENCE_BACKEND_STATUS,
                "inference_revision": backend.model_revision,
                "output": str(arguments.output),
            }
        sys.stdout.buffer.write(canonical_json_bytes(summary) + b"\n")
        return 0
    finally:
        await backend.shutdown()


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments._worker is not None:
        if (
            arguments.command is not None
            or arguments.video is not None
            or arguments.output is not None
        ):
            return 2
        return _worker(arguments._worker)
    if arguments.command is None:
        _parser().error("a command is required")
    try:
        return asyncio.run(_cli(arguments))
    except (OSError, UmiReferenceBackendError):
        print("S1 reference backend command failed", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
