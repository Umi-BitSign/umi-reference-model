from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from bitsign_motion import amd64_holistic_container as ahc
from bitsign_motion.holistic_container import HolisticExtractionResult

IMAGE_ID = "sha256:" + "a" * 64


def _load_worker_module() -> ModuleType:
    worker_path = (
        Path(__file__).resolve().parents[1] / "docker" / "mediapipe-holistic" / "worker.amd64.py"
    )
    spec = importlib.util.spec_from_file_location("bitsign_test_amd64_holistic_worker", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _image_inspection() -> bytes:
    return json.dumps(
        {
            "Id": IMAGE_ID,
            "Architecture": "amd64",
            "Os": "linux",
            "Config": {
                "User": "65532:65532",
                "Labels": {
                    "org.opencontainers.image.base.digest": (ahc.EXPECTED_AMD64_BASE_IMAGE_DIGEST),
                    "org.bitsign.mediapipe.version": ahc.EXPECTED_MEDIAPIPE_VERSION,
                    "org.bitsign.ffmpeg.version": ahc.EXPECTED_FFMPEG_VERSION,
                    "org.bitsign.holistic-model.sha256": ahc.EXPECTED_MODEL_SHA256,
                    "org.bitsign.worker.contract": ahc.RAW_SCHEMA,
                    "org.bitsign.worker.receipt": ahc.RECEIPT_SCHEMA,
                },
            },
        },
        separators=(",", ":"),
    ).encode()


class InspectAndBuildRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        command: list[str],
        *,
        timeout_seconds: float,
        stdout_limit: int,
        stderr_limit: int,
    ) -> ahc._CommandResult:
        del timeout_seconds, stdout_limit, stderr_limit
        self.calls.append(command)
        if command[1:3] == ["image", "inspect"]:
            return ahc._CommandResult(0, _image_inspection(), b"")
        if command[1] == "build":
            return ahc._CommandResult(0, (IMAGE_ID + "\n").encode(), b"")
        raise AssertionError(command)


def _option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_build_uses_separate_amd64_context_and_pins() -> None:
    runner = InspectAndBuildRunner()
    image = ahc.build_holistic_container_image(platform="linux/amd64", _runner=runner)
    build = next(command for command in runner.calls if command[1] == "build")
    assert "--provenance=false" in build
    assert _option_value(build, "--platform") == "linux/amd64"
    assert _option_value(build, "--build-arg") == (
        f"SOURCE_DATE_EPOCH={ahc.IMAGE_BUILD_SOURCE_DATE_EPOCH}"
    )
    assert Path(_option_value(build, "--file")).name == "Dockerfile.amd64"
    assert _option_value(build, "--tag") == ahc.AMD64_IMAGE_TAG
    assert image.platform == "linux/amd64"


def test_amd64_result_type_enters_existing_motion_pipeline() -> None:
    assert ahc.HolisticExtractionResult is HolisticExtractionResult


def test_whole_video_worker_refuses_caller_clip_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _load_worker_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker.amd64.py",
            "--sample-id",
            "fixture",
            "--expected-video-sha256",
            "1" * 64,
            "--expected-model-sha256",
            ahc.EXPECTED_MODEL_SHA256,
            "--container-image-id",
            IMAGE_ID,
            "--whole-video",
            "--clip-start-us",
            "0",
            "--clip-end-us",
            "2000000",
            "--input-mirrored",
            "false",
        ],
    )
    with pytest.raises(worker.WorkerError, match="refuses caller-supplied clip bounds"):
        worker._arguments()


def test_whole_video_requires_request_digest_and_size_before_docker(tmp_path: Path) -> None:
    model = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "mediapipe"
        / "holistic_landmarker-float16-2023-12-21.task"
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"not reached")
    with pytest.raises(ahc.HolisticContainerError, match="requires the request SHA-256"):
        ahc.extract_holistic_in_container(
            video_path=video,
            model_path=model,
            output_root=tmp_path / "output",
            sample_id="whole-video-no-authority",
            expected_video_byte_count=video.stat().st_size,
            container_platform="linux/amd64",
            _umi_whole_video=True,
        )


def test_whole_video_rejects_unsafe_parent_container_name_before_docker(tmp_path: Path) -> None:
    model = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "mediapipe"
        / "holistic_landmarker-float16-2023-12-21.task"
    )
    video = tmp_path / "video.mp4"
    video.write_bytes(b"not reached")
    with pytest.raises(ahc.HolisticContainerError, match="container_name must be"):
        ahc.extract_umi_whole_video_in_container(
            video_path=video,
            model_path=model,
            output_root=tmp_path / "output",
            sample_id="whole-video-container-name",
            expected_video_sha256="1" * 64,
            expected_video_byte_count=video.stat().st_size,
            container_name="--network=host",
        )


def test_amd64_build_context_pins_base_and_arch_specific_wheels() -> None:
    context = Path(__file__).resolve().parents[1] / "docker" / "mediapipe-holistic"
    dockerfile = (context / "Dockerfile.amd64").read_text()
    requirements = (context / "requirements.amd64.lock").read_text()
    assert f"python@{ahc.EXPECTED_AMD64_BASE_IMAGE_DIGEST}" in dockerfile
    assert "FROM --platform=linux/amd64" in dockerfile
    assert "COPY requirements.amd64.lock" in dockerfile
    assert "COPY worker.amd64.py" in dockerfile
    assert "--require-hashes" in dockerfile
    assert (
        "mediapipe==1.0.1 \\\n"
        "    --hash=sha256:121522251afc3c135e4b7b0c341dd5e050ad1ec87631127484f3c389ae385044"
        in requirements
    )
    assert (
        "numpy==2.5.2 \\\n"
        "    --hash=sha256:3cdec01fa790a186d430433fdd4d4ffb70eed6f0eeb4bf05c8dbe2dce0a9bcb8"
        in requirements
    )


@pytest.mark.skipif(
    os.environ.get("BITSIGN_RUN_AMD64_MEDIAPIPE_INTEGRATION") != "1",
    reason="set BITSIGN_RUN_AMD64_MEDIAPIPE_INTEGRATION=1 for the real amd64 smoke",
)
def test_real_amd64_whole_video_fleurs_smoke(tmp_path: Path) -> None:
    raw_video = os.environ.get("BITSIGN_AMD64_MEDIAPIPE_VIDEO")
    raw_model = os.environ.get("BITSIGN_AMD64_MEDIAPIPE_TASK")
    if raw_video is None or raw_model is None:
        pytest.fail(
            "integration requires BITSIGN_AMD64_MEDIAPIPE_VIDEO and "
            "BITSIGN_AMD64_MEDIAPIPE_TASK"
        )
    video = Path(raw_video)
    model = Path(raw_model)
    if (
        not video.is_absolute()
        or not video.is_file()
        or not model.is_absolute()
        or not model.is_file()
    ):
        pytest.fail("integration video and task must be existing absolute files")
    image = ahc.build_holistic_container_image(platform="linux/amd64", timeout_seconds=900)
    result = ahc.extract_umi_whole_video_in_container(
        video_path=video,
        model_path=model,
        output_root=tmp_path / "raw",
        sample_id="fleurs-whole-amd64",
        expected_video_sha256=("98a8e249307030eedcff884e45dc94cc00fcc2f3ca3f3acd9ed57469a09e7056"),
        expected_video_byte_count=1_153_872,
        image_reference=image.image_id,
        timeout_seconds=300,
    )
    receipt = json.loads(result.receipt_path.read_bytes())
    assert receipt["container"]["platform"] == "linux/amd64"
    assert receipt["clip"] == {
        "clip_start_us": 0,
        "clip_end_us": 2_400_000,
        "duration_us": 2_400_000,
        "interval": "half-open-[start,end)",
    }
    assert result.frame_count == 20
