"""Apply Metal fallback and offline bounds before importing any winner trainer dependencies."""

from __future__ import annotations

import os
import sys


def main():
    for key, value in {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        os.environ[key] = value
    sys.dont_write_bytecode = True
    from train_c5_winner_full import main as train_main

    train_main()


if __name__ == "__main__":
    main()
