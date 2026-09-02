from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

import bitsign_motion.umi_reference_backend as backend_module
from bitsign_motion.canonical import canonical_json_bytes
from bitsign_motion.portable_model import PortableS1, PortableS1Config
from bitsign_motion.s1_portable_runtime import (
    S1PortableError,
    export_s1_portable_bundle,
)
from bitsign_motion.umi_reference_backend import (
    UMI_REFERENCE_BACKEND_STATUS,
    UMI_REFERENCE_JOB_SCHEMA,
    UMI_REFERENCE_RESULT_SCHEMA,
    ReferenceBackendConfig,
    UmiReferenceBackendError,
    UmiS1Translator,
)
from tests.test_s1_portable_runtime import _preprocessing, _rights, _synthetic_tokenizer


@pytest.fixture(scope="module")
def portable_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    root = tmp_path_factory.mktemp("reference-backend-portable")
    tokenizer_model, tokenizer_record = _synthetic_tokenizer()
    model_path = root / "tokenizer.model"
    record_path = root / "tokenizer.json"
    model_path.write_bytes(tokenizer_model)
    record_path.write_bytes(tokenizer_record)
    torch.manual_seed(17)
    report = export_s1_portable_bundle(
        PortableS1(PortableS1Config()).eval(),
        root / "bundle",
        tokenizer_model_path=model_path,
        tokenizer_record_path=record_path,
        preprocessing=_preprocessing(),
        rights=_rights(),
        upstream_release_identity_sha256="71" * 32,
    )
    return root / "bundle", report


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\nexit 1\n")
    path.chmod(0o700)
    return path


def test_path_file_keeps_backend_importable_after_child_environment_cleanup(
    tmp_path: Path,
) -> None:
    environment = tmp_path / "clean-venv"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / "bin" / "python"
    purelib = Path(
        subprocess.run(
            [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    project_root = Path(__file__).resolve().parents[1]
    (purelib / "umi-reference-model-source.pth").write_text(
        f"{project_root / 'src'}\n", encoding="utf-8"
    )
    (purelib / "test-dependencies.pth").write_text(
        f"{Path(sysconfig.get_paths()['purelib'])}\n", encoding="utf-8"
    )
    working_directory = tmp_path / "outside-source"
    working_directory.mkdir()
    script = """
import subprocess
import sys
from bitsign_motion.umi_reference_backend import _child_environment

environment = _child_environment()
assert "PYTHONPATH" not in environment
subprocess.run(
    [sys.executable, "-c", "import bitsign_motion.umi_reference_backend"],
    check=True,
    cwd=sys.argv[1],
    env=environment,
)
"""
    subprocess.run(
        [str(python), "-c", script, str(working_directory)],
        check=True,
        cwd=working_directory,
        env={"PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1"},
    )


def _environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    portable_bundle: tuple[Path, dict[str, Any]],
) -> tuple[Path, str]:
    bundle, report = portable_bundle
    temporary_root = tmp_path / "jobs"
    temporary_root.mkdir(mode=0o700)
    docker = _executable(tmp_path / "docker")
    values = {
        "UMI_S1_BUNDLE": str(bundle),
        "UMI_S1_INFERENCE_REVISION": report["inference_revision"],
        "UMI_S1_EXTRACTOR_IMAGE": _preprocessing()["supported_oci_images"]["linux/amd64"],
        "UMI_S1_EXTRACTOR_MODEL": str(
            Path("artifacts/mediapipe/holistic_landmarker-float16-2023-12-21.task").resolve()
        ),
        "UMI_S1_EXTRACTOR_PLATFORM": "linux/amd64",
        "UMI_S1_DOCKER_EXECUTABLE": str(docker),
        "UMI_S1_TEMP_ROOT": str(temporary_root),
        "UMI_S1_DEVICE": "cpu",
        "UMI_S1_HARD_DEADLINE_SECONDS": "5",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))
    return temporary_root, report["inference_revision"]


def _read_job(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _write_result(job: dict[str, Any], *, hypothesis: str | None = None) -> None:
    result = {
        "schema": UMI_REFERENCE_RESULT_SCHEMA,
        "status": "ready" if hypothesis is None else "ok",
        "claim_status": UMI_REFERENCE_BACKEND_STATUS,
        "inference_revision": job["inference_revision"],
    }
    if hypothesis is not None:
        result["hypothesis"] = hypothesis
    backend_module._write_exclusive(Path(job["result"]), canonical_json_bytes(result))


def test_translation_job_uses_the_motion_converter_filename_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    portable_bundle: tuple[Path, dict[str, Any]],
) -> None:
    temporary_root, _ = _environment(monkeypatch, tmp_path, portable_bundle)
    sample_id = "ab" * 16
    record = backend_module._job_record(
        backend_module._configuration_from_environment(),
        operation="translate",
        job_root=temporary_root / f"umi-s1-job-{sample_id}",
        video_sha256="cd" * 32,
        video_size_bytes=123,
    )
    assert record["sample_id"] == sample_id
    assert Path(record["motion"]).name == f"{sample_id}.npz"


def test_module_translator_exposes_strict_revision_and_async_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    portable_bundle: tuple[Path, dict[str, Any]],
) -> None:
    temporary_root, revision = _environment(monkeypatch, tmp_path, portable_bundle)
    observed: list[tuple[str, float]] = []

    async def launch(
        path: Path,
        *,
        deadline_seconds: float,
        docker_executable: Path,
        container_name: str | None,
    ) -> None:
        del docker_executable
        job = _read_job(path)
        assert container_name is None
        observed.append((job["operation"], deadline_seconds))
        _write_result(job)

    monkeypatch.setattr(backend_module, "_launch_worker_job", launch)

    async def scenario() -> None:
        backend = UmiS1Translator()
        assert backend.model_revision == revision
        await backend.startup()
        await backend.startup()
        await backend.shutdown()

    asyncio.run(scenario())
    assert observed == [("probe", 5.0)]
    assert list(temporary_root.iterdir()) == []


def test_translation_rechecks_request_authority_cleans_jobs_and_emits_no_sensitive_logs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    portable_bundle: tuple[Path, dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    temporary_root, _ = _environment(monkeypatch, tmp_path, portable_bundle)
    video = b"private-video-payload-that-must-not-enter-logs"
    hypothesis = "private translated sentence"
    digest = hashlib.sha256(video).hexdigest()

    async def launch(
        path: Path,
        *,
        deadline_seconds: float,
        docker_executable: Path,
        container_name: str | None,
    ) -> None:
        del docker_executable
        assert deadline_seconds == 5.0
        job = _read_job(path)
        if job["operation"] == "probe":
            assert container_name is None
            _write_result(job)
            return
        assert container_name == job["container_name"]
        assert Path(job["video"]).read_bytes() == video
        assert stat.S_IMODE(Path(job["video"]).stat().st_mode) == 0o600
        _write_result(job, hypothesis=hypothesis)

    monkeypatch.setattr(backend_module, "_launch_worker_job", launch)

    async def scenario() -> None:
        backend = UmiS1Translator()
        await backend.startup()
        request = SimpleNamespace(video=SimpleNamespace(sha256=digest, size_bytes=len(video)))
        assert await backend(video, request) == hypothesis
        await backend.shutdown()

    asyncio.run(scenario())
    assert list(temporary_root.iterdir()) == []
    captured = capsys.readouterr()
    combined = captured.out + captured.err + caplog.text
    assert video.decode() not in combined
    assert hypothesis not in combined


def test_translation_rejects_mismatched_video_before_child_launch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    portable_bundle: tuple[Path, dict[str, Any]],
) -> None:
    _environment(monkeypatch, tmp_path, portable_bundle)
    operations: list[str] = []

    async def launch(
        path: Path,
        *,
        deadline_seconds: float,
        docker_executable: Path,
        container_name: str | None,
    ) -> None:
        del deadline_seconds, docker_executable
        job = _read_job(path)
        assert container_name is None
        operations.append(job["operation"])
        _write_result(job)

    monkeypatch.setattr(backend_module, "_launch_worker_job", launch)

    async def scenario() -> None:
        backend = UmiS1Translator()
        await backend.startup()
        request = SimpleNamespace(video=SimpleNamespace(sha256="00" * 32, size_bytes=3))
        with pytest.raises(UmiReferenceBackendError, match="digest differs"):
            await backend(b"abc", request)
        await backend.shutdown()

    asyncio.run(scenario())
    assert operations == ["probe"]


def test_worker_probe_strictly_loads_bundle_and_rejects_corruption_before_docker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    portable_bundle: tuple[Path, dict[str, Any]],
) -> None:
    bundle, report = portable_bundle
    model = Path("artifacts/mediapipe/holistic_landmarker-float16-2023-12-21.task").resolve()
    image = _preprocessing()["supported_oci_images"]["linux/arm64"]
    monkeypatch.setattr(
        backend_module.arm64_holistic_container,
        "resolve_holistic_container_image",
        lambda *_args, **_kwargs: SimpleNamespace(image_id=image),
    )
    # The public repository deliberately excludes the separately downloaded task
    # model. This test exercises portable-bundle validation before Docker access.
    monkeypatch.setattr(backend_module, "_verify_model_file", lambda _path: None)
    job = {
        "schema": UMI_REFERENCE_JOB_SCHEMA,
        "operation": "probe",
        "status": UMI_REFERENCE_BACKEND_STATUS,
        "bundle": str(bundle),
        "inference_revision": report["inference_revision"],
        "extractor_image": image,
        "extractor_model": str(model),
        "extractor_platform": "linux/arm64",
        "docker_executable": str(_executable(tmp_path / "docker")),
        "device": "cpu",
        "timeout_seconds": 5.0,
        "result": str(tmp_path / "result.json"),
    }
    result = backend_module._execute_worker_job(job)
    assert result["status"] == "ready"
    assert result["inference_revision"] == report["inference_revision"]

    corrupted = tmp_path / "corrupted"
    shutil.copytree(bundle, corrupted)
    checkpoint = corrupted / "model.safetensors"
    payload = bytearray(checkpoint.read_bytes())
    payload[-1] ^= 1
    checkpoint.write_bytes(payload)
    job["bundle"] = str(corrupted)
    with pytest.raises(S1PortableError, match="inventory"):
        backend_module._execute_worker_job(job)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _wait_for_file(path: Path) -> None:
    for _ in range(200):
        if path.exists():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("child PID file was not created")


async def _wait_for_exit(pid: int) -> None:
    for _ in range(300):
        if not _pid_exists(pid):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"process remained after group termination: {pid}")


def test_cancellation_kills_and_reaps_the_native_process_group(tmp_path: Path) -> None:
    async def scenario() -> None:
        pid_path = tmp_path / "pids"
        script = (
            "import os,subprocess,sys,time;"
            "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
            "fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);"
            "os.write(fd,f'{os.getpid()} {child.pid}'.encode());os.close(fd);time.sleep(60)"
        )
        task = asyncio.create_task(
            backend_module._run_killable_process(
                [sys.executable, "-c", script, str(pid_path)],
                deadline_seconds=30,
            )
        )
        await _wait_for_file(pid_path)
        parent_pid, child_pid = (int(value) for value in pid_path.read_text().split())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await _wait_for_exit(parent_pid)
        await _wait_for_exit(child_pid)

    asyncio.run(scenario())


def test_hard_timeout_kills_and_reaps_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        pid_path = tmp_path / "pid"
        script = (
            "import os,sys,time;"
            "fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);"
            "os.write(fd,str(os.getpid()).encode());os.close(fd);time.sleep(60)"
        )
        with pytest.raises(TimeoutError):
            await backend_module._run_killable_process(
                [sys.executable, "-c", script, str(pid_path)],
                deadline_seconds=0.2,
            )
        await _wait_for_file(pid_path)
        await _wait_for_exit(int(pid_path.read_text()))

    asyncio.run(scenario())


def test_worker_cancellation_force_removes_the_exact_daemon_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    container_name = "bitsign-holistic-" + "a" * 32
    docker = tmp_path / "docker"
    job = tmp_path / "job.json"
    state = {"container_exists": False}
    worker_started = asyncio.Event()
    commands: list[list[str]] = []

    async def run(command: list[str], *, deadline_seconds: float) -> int:
        del deadline_seconds
        commands.append(command)
        if command[0] == sys.executable:
            state["container_exists"] = True
            worker_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")
        if command[1:3] == ["container", "inspect"]:
            return 0 if state["container_exists"] else 1
        if command[1] == "version":
            return 0
        if command[1:4] == ["container", "rm", "--force"]:
            assert command[4] == container_name
            state["container_exists"] = False
            return 0
        raise AssertionError(command)

    monkeypatch.setattr(backend_module, "_run_killable_process", run)

    async def scenario() -> None:
        task = asyncio.create_task(
            backend_module._launch_worker_job(
                job,
                deadline_seconds=30,
                docker_executable=docker,
                container_name=container_name,
            )
        )
        await worker_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert state["container_exists"] is False

    asyncio.run(scenario())
    assert [str(docker), "container", "rm", "--force", container_name] in commands


def test_worker_deadline_force_removes_the_exact_daemon_container(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    container_name = "bitsign-holistic-" + "d" * 32
    docker = tmp_path / "docker"
    state = {"container_exists": False}

    async def run(command: list[str], *, deadline_seconds: float) -> int:
        del deadline_seconds
        if command[0] == sys.executable:
            state["container_exists"] = True
            raise TimeoutError
        if command[1:3] == ["container", "inspect"]:
            return 0 if state["container_exists"] else 1
        if command[1] == "version":
            return 0
        if command[1:4] == ["container", "rm", "--force"]:
            state["container_exists"] = False
            return 0
        raise AssertionError(command)

    monkeypatch.setattr(backend_module, "_run_killable_process", run)

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await backend_module._launch_worker_job(
                tmp_path / "job.json",
                deadline_seconds=1,
                docker_executable=docker,
                container_name=container_name,
            )
        assert state["container_exists"] is False

    asyncio.run(scenario())


def test_private_job_cleanup_is_target_pinned_and_failure_is_observable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "jobs"
    root.mkdir(mode=0o700)
    job = root / ("umi-s1-job-" + "b" * 32)
    job.mkdir(mode=0o700)
    (job / "video.mp4").write_bytes(b"private")

    def retain_job(_path: Path) -> None:
        return None

    retain_job.avoids_symlink_attacks = True  # type: ignore[attr-defined]
    monkeypatch.setattr(backend_module.shutil, "rmtree", retain_job)
    with pytest.raises(UmiReferenceBackendError, match="remained after cleanup"):
        backend_module._remove_private_job_root(job, temporary_root=root)
    assert job.is_dir()

    outside = tmp_path / ("umi-s1-job-" + "c" * 32)
    outside.mkdir()
    with pytest.raises(UmiReferenceBackendError, match="not a pinned job"):
        backend_module._remove_private_job_root(outside, temporary_root=root)
    assert outside.is_dir()


def test_job_failure_and_cleanup_failure_are_both_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    portable_bundle: tuple[Path, dict[str, Any]],
) -> None:
    temporary_root, revision = _environment(monkeypatch, tmp_path, portable_bundle)
    config = ReferenceBackendConfig(
        bundle=portable_bundle[0],
        inference_revision=revision,
        extractor_image=_preprocessing()["supported_oci_images"]["linux/amd64"],
        extractor_model=Path(
            "artifacts/mediapipe/holistic_landmarker-float16-2023-12-21.task"
        ).resolve(),
        extractor_platform="linux/amd64",
        docker_executable=Path(os.environ["UMI_S1_DOCKER_EXECUTABLE"]),
        temporary_root=temporary_root,
        device="cpu",
        hard_deadline_seconds=5,
    )

    async def fail_launch(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("inference sentinel")

    def fail_cleanup(*_args: object, **_kwargs: object) -> None:
        raise UmiReferenceBackendError("cleanup sentinel")

    monkeypatch.setattr(backend_module, "_launch_worker_job", fail_launch)
    monkeypatch.setattr(backend_module, "_remove_private_job_root", fail_cleanup)

    async def scenario() -> None:
        with pytest.raises(BaseExceptionGroup) as captured:
            await UmiS1Translator()._run_job(config, operation="probe")
        assert [type(item) for item in captured.value.exceptions] == [
            RuntimeError,
            UmiReferenceBackendError,
        ]

    asyncio.run(scenario())


@pytest.mark.skipif(
    os.environ.get("BITSIGN_RUN_DOCKER_CLEANUP_INTEGRATION") != "1",
    reason="set BITSIGN_RUN_DOCKER_CLEANUP_INTEGRATION=1 for real Docker cleanup",
)
def test_real_docker_daemon_container_is_removed_after_process_cancellation() -> None:
    docker_value = shutil.which("docker")
    if docker_value is None:
        pytest.skip("Docker executable is unavailable")
    docker = Path(docker_value).resolve()
    container_name = "bitsign-holistic-" + secrets.token_hex(16)

    async def scenario() -> None:
        process = asyncio.create_task(
            backend_module._run_killable_process(
                [
                    str(docker),
                    "run",
                    "--pull",
                    "never",
                    "--rm",
                    "--name",
                    container_name,
                    "python:3.12-slim",
                    "sleep",
                    "60",
                ],
                deadline_seconds=30,
            )
        )
        for _ in range(100):
            if await backend_module._docker_container_exists(docker, container_name):
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("audit Docker container did not start")
        process.cancel()
        with pytest.raises(asyncio.CancelledError):
            await process
        await backend_module._force_remove_docker_container(docker, container_name)
        assert not await backend_module._docker_container_exists(docker, container_name)

    asyncio.run(scenario())


def test_environment_and_worker_result_reject_ambiguous_or_unbound_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UMI_S1_INFERENCE_REVISION", "not-a-digest")
    with pytest.raises(UmiReferenceBackendError, match="SHA-256"):
        _ = UmiS1Translator().model_revision
    with pytest.raises(UmiReferenceBackendError, match="field set"):
        backend_module._validate_worker_result(
            {
                "schema": UMI_REFERENCE_RESULT_SCHEMA,
                "status": "ready",
                "claim_status": UMI_REFERENCE_BACKEND_STATUS,
                "inference_revision": "11" * 32,
                "unexpected": True,
            },
            expected_revision="11" * 32,
            operation="probe",
        )


def test_declared_plugin_revision_cannot_change_before_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = "12" * 32
    monkeypatch.setenv("UMI_S1_INFERENCE_REVISION", first)
    backend = UmiS1Translator()
    assert backend.model_revision == first
    monkeypatch.setenv("UMI_S1_INFERENCE_REVISION", "34" * 32)
    with pytest.raises(UmiReferenceBackendError, match="changed after"):
        _ = backend.model_revision


def test_cli_surface_requires_an_owner_selected_output_file() -> None:
    parser = backend_module._parser()
    parsed = parser.parse_args(["translate", "--video", "/tmp/input.mp4"])
    assert parsed.command == "translate"
    assert parsed.output is None


def test_reference_config_is_immutable_and_binds_deadline() -> None:
    fields = ReferenceBackendConfig.__dataclass_fields__
    assert "hard_deadline_seconds" in fields
    assert ReferenceBackendConfig.__dataclass_params__.frozen is True
