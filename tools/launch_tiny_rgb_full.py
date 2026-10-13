"""Set native Metal fallback before importing the full-data training dependencies."""

from __future__ import annotations

import os


def main() -> None:
    for key, value in {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "OPENCV_FFMPEG_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[key] = value
    from train_tiny_rgb_full import main as train_main

    train_main()


if __name__ == "__main__":
    main()
