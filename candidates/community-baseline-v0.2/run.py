"""Standalone, offline ASL video inference. No network or miner integration."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args()
    for name, value in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(ROOT / "run/hf"),
        "TORCH_HOME": str(ROOT / "run/torch"),
        "MPLCONFIGDIR": str(ROOT / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(ROOT / "vendor/dinov2-source"),
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
    }.items():
        os.environ[name] = value
    started = time.monotonic()
    with contextlib.redirect_stdout(sys.stderr):
        from runtime import SHuBERTInferenceRuntime
        model = SHuBERTInferenceRuntime(ROOT, device=args.device)
        result = model.translate_path(args.video.resolve(strict=True))
    print(json.dumps({"text": result, "seconds": time.monotonic() - started}))

if __name__ == "__main__":
    main()
