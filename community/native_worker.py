"""Warm MPS model worker, launched inside an independently enforced sandbox.

Only the local supervisor may invoke this entrypoint. The model receives exact
video bytes and a deadline through the existing bounded process protocol.
"""

from __future__ import annotations

import argparse
import ctypes
import fcntl
import os
import platform
import socket
import stat
import sys
import tempfile
from itertools import combinations
from pathlib import Path

from bundle_verification import verify_imported_bundle
from native_runtime import EXECUTION, verify_runtime
from verify_macos_overlay import verify_overlay
from worker_transport import serve_worker

F_GETPATH = 50


def _opened_directory(path: Path) -> tuple[Path, tuple[int, int]]:
    if not path.is_absolute():
        raise ValueError("worker directory must be absolute without aliases")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("worker directory must be absolute without aliases") from error
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
            raise ValueError("worker directory must be absolute without aliases")
        return path, (metadata.st_dev, metadata.st_ino)
    finally:
        os.close(descriptor)


def checked_directories(paths: dict[str, Path]) -> dict[str, Path]:
    """Require canonical, distinct directory roots before sandbox probing."""
    checked = {}
    identities = {}
    for name, path in paths.items():
        try:
            checked[name], identities[name] = _opened_directory(path)
        except ValueError as error:
            raise ValueError(f"{name} must be an absolute directory without aliases") from error
    for (left_name, left), (right_name, right) in combinations(checked.items(), 2):
        if (
            identities[left_name] == identities[right_name]
            or left.is_relative_to(right)
            or right.is_relative_to(left)
        ):
            raise ValueError("worker directories must be distinct and non-overlapping")
    return checked


def private_metal_cache(suffix: str) -> Path:
    if not suffix.startswith("org.umi.") or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for c in suffix
    ):
        raise ValueError("invalid private Metal cache suffix")
    libc = ctypes.CDLL(None, use_errno=True)
    select = libc._set_user_dir_suffix
    select.argtypes, select.restype = [ctypes.c_char_p], ctypes.c_int
    if not select(suffix.encode("ascii")):
        raise RuntimeError("private Metal cache selection failed")
    confstr = libc.confstr
    confstr.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
    confstr.restype = ctypes.c_size_t
    value = ctypes.create_string_buffer(4096)
    size = confstr(65538, value, len(value))
    if not 0 < size <= len(value):
        raise RuntimeError("private Metal cache lookup failed")
    result = Path(os.fsdecode(value.value)).resolve(strict=True)
    if result.name != suffix or "/var/folders/" not in str(result):
        raise RuntimeError("private Metal cache path differs")
    return result


def sandbox_probes(read_only_roots: list[Path], deny_read_marker: Path) -> None:
    for root in read_only_roots:
        try:
            # A no-op chmod tests sandbox denial, without changing mode on failure.
            os.chmod(root, root.stat().st_mode & 0o777)
        except PermissionError:
            pass
        else:
            raise RuntimeError("sandbox must deny writes to runtime and model roots")
    try:
        with deny_read_marker.open("rb"):
            pass
    except PermissionError:
        pass
    else:
        raise RuntimeError("sandbox must deny the harmless out-of-scope marker")
    with socket.socket() as connection:
        connection.settimeout(2)
        try:
            connection.connect(("1.1.1.1", 443))
        except PermissionError:
            pass
        else:
            raise RuntimeError("sandbox must deny outbound connections")


def load_model(bundle: Path, overlay: Path, scratch: Path):
    model_root = bundle / "model"
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(scratch / "hf"),
        "TORCH_HOME": str(scratch / "torch"),
        "MPLCONFIGDIR": str(scratch / "matplotlib"),
        "XDG_CACHE_HOME": str(scratch / "cache"),
        "SHUBERT_DINOV2_SOURCE": str(model_root / "vendor/dinov2-source"),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO": EXECUTION["mps_high_watermark_ratio"],
        "PYTORCH_MPS_LOW_WATERMARK_RATIO": EXECUTION["mps_low_watermark_ratio"],
        "OMP_NUM_THREADS": str(EXECUTION["cpu_threads"]),
        "OPENBLAS_NUM_THREADS": str(EXECUTION["cpu_threads"]),
        "MKL_NUM_THREADS": str(EXECUTION["cpu_threads"]),
    }.items():
        os.environ[name] = value
    sys.path[:0] = [str(model_root), str(overlay)]
    import torch
    from mediapipe.python import _framework_bindings
    from runtime import SHuBERTInferenceRuntime

    expected_binding = overlay / "mediapipe/python/_framework_bindings.cpython-310-darwin.so"
    if Path(_framework_bindings.__file__).resolve() != expected_binding:
        raise RuntimeError("CPU-only MediaPipe overlay was not loaded")
    torch.set_num_threads(EXECUTION["cpu_threads"])
    torch.set_num_interop_threads(1)
    model = SHuBERTInferenceRuntime(
        model_root,
        device=EXECUTION["device"],
        generation_num_beams=EXECUTION["generation_num_beams"],
        generation_max_length=EXECUTION["generation_max_length"],
        dino_batch_size=EXECUTION["dino_batch_size"],
        model_execution_concurrency=EXECUTION["model_execution_concurrency"],
        verify_assets=True,
    )
    from umi_landmarks import video_holistic
    from umi_video_reader import VideoReader

    model._video_reader = VideoReader
    model._video_holistic = video_holistic
    return model


def translate_video(model, scratch: Path, video: bytes, deadline_ns: int) -> str:
    descriptor, name = tempfile.mkstemp(prefix="request-", suffix=".mp4", dir=scratch)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(video)
        # The upstream translate_bytes writes inside the model tree. Preserve
        # that tree and use a private input file owned by this worker instead.
        return model.translate_path(path, deadline_unix_ns=deadline_ns)
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "bundle",
        "overlay",
        "runtime-manifest",
        "environment-root",
        "python-root",
        "scratch",
        "deny-read-marker",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--metal-cache-suffix", required=True)
    args = parser.parse_args()
    if (
        platform.system() != "Darwin"
        or platform.machine() != "arm64"
        or sys.version_info[:2] != (3, 10)
    ):
        raise RuntimeError("native worker requires macOS ARM64 and Python 3.10")
    os.umask(0o077)
    directories = checked_directories(
        {
            "code": Path(__file__).resolve().parent,
            "environment": args.environment_root,
            "python": args.python_root,
            "bundle": args.bundle,
            "overlay": args.overlay,
            "scratch": args.scratch,
        }
    )
    roots = {name: directories[name] for name in ("code", "environment", "python")}
    bundle = directories["bundle"]
    overlay = directories["overlay"]
    scratch = directories["scratch"]
    if Path(sys.executable).resolve() != (args.environment_root / "bin/python").resolve(
        strict=True
    ):
        raise RuntimeError("worker interpreter differs from configured environment")
    if Path(sys.prefix).resolve(strict=True) != args.environment_root:
        raise RuntimeError("worker packages are not loaded from configured environment")
    info = scratch.stat()
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("scratch ownership or permissions differ")
    sandbox_probes([*roots.values(), bundle, overlay], args.deny_read_marker)
    document = verify_runtime(
        roots, manifest=args.runtime_manifest, expected_revision=args.model_revision
    )
    verify_imported_bundle(bundle, expected_bundle_sha256=document["model_bundle_sha256"])
    verify_overlay(overlay, expected_manifest_sha256=document["native_overlay_sha256"])
    private_metal_cache(args.metal_cache_suffix)
    # Reserve the protocol descriptor before native libraries can print.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        model = load_model(bundle, overlay, scratch)
        # No readiness frame is emitted until verification and real loading pass.
        serve_worker(
            lambda video, deadline: translate_video(model, scratch, video, deadline),
            verified_model_revision=args.model_revision,
            destination=protocol,
        )


if __name__ == "__main__":
    main()
