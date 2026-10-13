"""Compare exact-pixel CPU decoders on fixed training clips with equal thread bounds."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def main() -> None:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OPENCV_FFMPEG_THREADS"):
        os.environ[name] = "2"
    import cv2
    import numpy as np
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_data import decode_rgb_video
    from tiny_rgb_fit import ROOT, SIDES, checked_plan
    from tiny_rgb_full_training_tools import decode_rgb_video_fast

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("RGB decoder comparison needs a new private receipt")
    plan = checked_plan(args.fit_plan)
    ledger = [
        json.loads(s) for s in args.fit_plan.with_name("pixels.jsonl").read_text().splitlines()
    ]
    binding = {(r["source"], r["example_id"]): r for r in ledger}
    selected, shapes = [], set()
    for row in plan["examples"]:
        shape = tuple(
            binding[(row["source"], row["example_id"])]["views"]["128"]["metadata"]["source_shape"]
        )
        if shape not in shapes:
            selected.append(row)
            shapes.add(shape)
    records = []
    for row in selected:
        capture = cv2.VideoCapture(str(ROOT / row["video"]))
        try:
            original_threads = int(capture.get(cv2.CAP_PROP_N_THREADS))
        finally:
            capture.release()
        if original_threads != 2:
            raise RuntimeError("reference decoder did not honor the matched CPU2 setting")
        for side in SIDES:
            reference = decode_rgb_video(row["video"], side)
            optimized = decode_rgb_video_fast(row["video"], side)
            if (
                not all(np.array_equal(reference[i], optimized[i]) for i in (0, 1))
                or reference[2] != optimized[2]
            ):
                raise RuntimeError("optimized decoder changed pixels, timing or metadata")
            del reference, optimized
            times = {"reference": [], "optimized": []}
            for repeat in range(4):
                order = (
                    ("reference", "optimized") if repeat % 2 == 0 else ("optimized", "reference")
                )
                for name in order:
                    function = decode_rgb_video if name == "reference" else decode_rgb_video_fast
                    started = time.perf_counter()
                    pixels, _, _ = function(row["video"], side)
                    times[name].append(time.perf_counter() - started)
                    del pixels
            before, after = (statistics.median(times[n]) for n in ("reference", "optimized"))
            record = {
                "example_id": row["example_id"],
                "short_side": side,
                "times": times,
                "reference_median_seconds": before,
                "optimized_median_seconds": after,
                "relative_time_reduction": 1 - after / before,
                "exact_pixels_indices_and_metadata": True,
                "cpu_threads": 2,
            }
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "times"}), flush=True)
    receipt = {
        "schema": "umi-studio-tiny-rgb-streaming-benchmark/1",
        "fit_plan_sha256": sha256(args.fit_plan),
        "cases": records,
        "source_sha256": {
            f"tools/{name}": sha256(ROOT / "tools" / name)
            for name in (
                "benchmark_tiny_rgb_streaming.py",
                "tiny_rgb_full_training_tools.py",
                "tiny_rgb_data.py",
            )
        },
        "gpu_used": False,
        "optimizer_updates": 0,
        "quality_verified": False,
        "qualification": (
            "Alternating paired CPU2 decoder timing only; no inference, ASL quality, "
            "end-to-end training-speed or generalization claim."
        ),
    }
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
