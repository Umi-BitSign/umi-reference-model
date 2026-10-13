"""Set Metal fallback before importing the stationary replay trainer."""

from __future__ import annotations

import os


def main() -> None:
    for key, value in {
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    from train_c5_stationary_mps import main as train_main

    train_main()


if __name__ == "__main__":
    main()
