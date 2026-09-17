"""Offline CPU entrypoint for the preserved community baseline.

Run only inside the UMI evaluator's wallet-free, network-disabled container.
The evaluator owns the wall-clock, memory, process and output limits.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path


def configure_environment(root: Path) -> None:
    for name, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": "/tmp/hf",
        "TORCH_HOME": "/tmp/torch",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "XDG_CACHE_HOME": "/tmp/cache",
        "SHUBERT_DINOV2_SOURCE": str(root / "vendor/dinov2-source"),
        "SHUBERT_DEVICE": "cpu",
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "PYTHONDONTWRITEBYTECODE": "1",
    }.items():
        os.environ[name] = value


def translate(video: Path, root: Path) -> str:
    configure_environment(root)
    # The inspected bundle code is imported only after the container starts.
    # Never load this module's model runtime in a wallet-bearing process.
    with contextlib.redirect_stdout(sys.stderr):
        from runtime import SHuBERTInferenceRuntime

        model = SHuBERTInferenceRuntime(
            root,
            device="cpu",
            generation_num_beams=5,
            generation_max_length=2048,
            model_execution_concurrency=1,
            verify_assets=True,
        )
        # The supplied runtime installs vendor/demo paths during construction.
        # The landmark adapter imports that vendor's video_reader module.
        from umi_landmarks import video_holistic
        from umi_video_reader import VideoReader

        # Retain the supplied archive unchanged. This separately hashed adapter
        # reader avoids redundant seeks without changing model weights or frames.
        model._video_reader = VideoReader
        # This adapter keeps face/hand task outputs and uses the pose-only graph
        # for the pose output. The supplied archive remains unchanged.
        model._video_holistic = video_holistic
        hypothesis = model.translate_path(video)
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("community baseline returned no hypothesis")
    if len(hypothesis.encode("utf-8")) > 4096:
        raise ValueError("community baseline hypothesis exceeds the protocol ceiling")
    return hypothesis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    # Plain text is the offline_bundle/1 contract, not the upstream JSON CLI.
    sys.stdout.write(translate(args.video.resolve(strict=True), Path(__file__).resolve().parent))


if __name__ == "__main__":
    main()
