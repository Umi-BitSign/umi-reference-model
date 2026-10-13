"""Verify sequential decoding reproduces a previously cached encoder state."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import miner_revision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    for key, value in {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HOME": str(baseline / "run/hf"),
        "TORCH_HOME": str(baseline / "run/torch"),
        "MPLCONFIGDIR": str(baseline / "run/matplotlib"),
        "SHUBERT_DINOV2_SOURCE": str(baseline / "vendor/dinov2-source"),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
    }.items():
        os.environ[key] = value
    sys.path.insert(0, str(baseline))
    import numpy as np
    from cache_fleurs_encoder import _encoder_state
    from runtime import SHuBERTInferenceRuntime
    from sequential_video_reader import make_sequential_reader

    revision = miner_revision(args.miner_health_url, None)
    model = SHuBERTInferenceRuntime(baseline, device="cpu")
    model._video_reader = make_sequential_reader(model._video_reader)
    state, diagnostics = _encoder_state(model, args.clip, args.miner_health_url, revision)
    with np.load(args.cache, allow_pickle=False) as payload:
        expected = payload["encoder_hidden_state"]
    if not np.array_equal(state, expected):
        maximum = float(np.max(np.abs(state - expected)))
        raise RuntimeError(f"sequential encoder state differs from baseline cache: max={maximum}")
    miner_revision(args.miner_health_url, revision)
    print(
        f"encoder state byte-identical: shape={state.shape} "
        f"timings={diagnostics['timings_seconds']}"
    )


if __name__ == "__main__":
    main()
