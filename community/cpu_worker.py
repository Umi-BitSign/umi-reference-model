"""Warm Linux x86_64 CPU worker for service.py's bounded process protocol.

Run inside an independently enforced, wallet-free sandbox. The supervisor owns
process cancellation; the launcher owns network, filesystem and resource limits.
"""

from __future__ import annotations

import argparse
import errno
import os
import platform
import socket
import stat
import sys
import tempfile
from itertools import combinations
from pathlib import Path

from bundle_verification import verify_imported_bundle
from cpu_runtime import EXECUTION, verify_runtime
from native_runtime import opened_directory
from worker_transport import serve_worker


def checked_directories(paths: dict[str, Path]) -> dict[str, Path]:
    checked = {name: opened_directory(path)[0] for name, path in paths.items()}
    for left, right in combinations(checked.values(), 2):
        if left.is_relative_to(right) or right.is_relative_to(left) or left.samefile(right):
            raise ValueError("worker directories must be distinct and non-overlapping")
    scratch = checked["scratch"].stat()
    if scratch.st_uid != os.getuid() or stat.S_IMODE(scratch.st_mode) != 0o700:
        raise ValueError("scratch must be owner-private")
    return checked


def sandbox_probes(read_only_roots: list[Path], deny_read_marker: Path) -> None:
    """Catch a missing sandbox before loading model code; not a full sandbox audit."""
    if not deny_read_marker.is_absolute():
        raise ValueError("sandbox marker must be absolute")
    for root in read_only_roots:
        try:
            os.chmod(root, stat.S_IMODE(root.stat().st_mode))
        except OSError as error:
            if error.errno not in {errno.EROFS, errno.EPERM, errno.EACCES}:
                raise
        else:
            raise RuntimeError("sandbox must deny writes to runtime and model roots")
    try:
        with deny_read_marker.open("rb"):
            pass
    except (PermissionError, FileNotFoundError):
        pass  # Mount-namespace isolation hides the pre-created host marker.
    else:
        raise RuntimeError("sandbox must hide the out-of-scope marker")
    with socket.socket() as connection:
        connection.settimeout(2)
        try:
            connection.connect(("1.1.1.1", 443))
        except OSError as error:
            if error.errno not in {errno.ENETUNREACH, errno.EPERM, errno.EACCES}:
                raise RuntimeError("sandbox network isolation is unverified") from error
        else:
            raise RuntimeError("sandbox must deny outbound connections")


def load_model(bundle: Path, scratch: Path):
    model_root = bundle / "model"
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(scratch / "hf"),
        "TORCH_HOME": str(scratch / "torch"),
        "MPLCONFIGDIR": str(scratch / "matplotlib"),
        "XDG_CACHE_HOME": str(scratch / "cache"),
        "TMPDIR": str(scratch),
        "SHUBERT_DINOV2_SOURCE": str(model_root / "vendor/dinov2-source"),
        "SHUBERT_DEVICE": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": str(EXECUTION["cpu_threads"]),
        "OPENBLAS_NUM_THREADS": str(EXECUTION["cpu_threads"]),
        "MKL_NUM_THREADS": str(EXECUTION["cpu_threads"]),
    }.items():
        os.environ[name] = value
    sys.path.insert(0, str(model_root))
    import torch
    from runtime import SHuBERTInferenceRuntime

    torch.set_num_threads(EXECUTION["cpu_threads"])
    torch.set_num_interop_threads(1)
    return SHuBERTInferenceRuntime(
        model_root,
        device=EXECUTION["device"],
        generation_num_beams=EXECUTION["generation_num_beams"],
        generation_max_length=EXECUTION["generation_max_length"],
        dino_batch_size=EXECUTION["dino_batch_size"],
        model_execution_concurrency=EXECUTION["model_execution_concurrency"],
        verify_assets=True,
    )


def translate_video(model, scratch: Path, video: bytes, deadline_ns: int) -> str:
    descriptor, name = tempfile.mkstemp(prefix="request-", suffix=".mp4", dir=scratch)
    path = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            os.fchmod(output.fileno(), 0o600)
            output.write(video)
        return model.translate_path(path, deadline_unix_ns=deadline_ns)
    finally:
        path.unlink(missing_ok=True)


def run_worker(bundle: Path, scratch: Path, model_revision: str) -> None:
    # Reserve stdout before Torch/model imports, including native-library prints.
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "wb") as protocol:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        model = load_model(bundle, scratch)
        serve_worker(
            lambda video, deadline: translate_video(model, scratch, video, deadline),
            verified_model_revision=model_revision,
            destination=protocol,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "bundle",
        "runtime-manifest",
        "environment-root",
        "python-root",
        "scratch",
        "deny-read-marker",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()
    if (platform.system(), platform.machine(), sys.version_info[:2]) != (
        "Linux",
        "x86_64",
        (3, 10),
    ):
        raise RuntimeError("CPU worker requires Linux x86_64 and Python 3.10")
    if not sys.flags.no_user_site or not sys.dont_write_bytecode:
        raise RuntimeError("CPU worker requires Python -B -s")
    os.umask(0o077)
    directories = checked_directories(
        {
            "code": Path(__file__).resolve().parent,
            "environment": args.environment_root,
            "python": args.python_root,
            "bundle": args.bundle,
            "scratch": args.scratch,
        }
    )
    if (
        Path(sys.executable).resolve()
        != (args.environment_root / "bin/python").resolve(strict=True)
        or Path(sys.prefix).resolve(strict=True) != args.environment_root
        or Path(sys.base_prefix).resolve(strict=True) != args.python_root
    ):
        raise RuntimeError("worker interpreter differs from configured runtime")
    roots = {name: directories[name] for name in ("code", "environment", "python")}
    sandbox_probes([*roots.values(), args.bundle], args.deny_read_marker)
    document = verify_runtime(
        roots, manifest=args.runtime_manifest, expected_revision=args.model_revision
    )
    verify_imported_bundle(args.bundle, expected_bundle_sha256=document["model_bundle_sha256"])
    run_worker(args.bundle, args.scratch, args.model_revision)


if __name__ == "__main__":
    main()
