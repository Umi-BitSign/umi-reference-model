"""Measure a lossless temporal pixel codec on exactly the existing training-fit views."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    import numpy as np
    from benchmark_fleurs_dev import sha256
    from tiny_rgb_fit import ROOT, SIDES, checked_plan, storage
    from tiny_rgb_full_training_tools import decode_pixel_delta, encode_pixel_delta

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    plan = checked_plan(args.fit_plan)
    output, data_dir = args.output_dir.resolve(), args.data_dir.resolve()
    if output.exists() or not output.is_relative_to(ROOT / "runs"):
        raise ValueError("pixel codec probe needs a new private runs/ directory")
    if data_dir.exists() or not data_dir.is_relative_to(ROOT / "data"):
        raise ValueError("pixel codec probe needs a new private data/ directory")
    output.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    bindings = [
        json.loads(s) for s in args.fit_plan.with_name("pixels.jsonl").read_text().splitlines()
    ]
    records = []
    for index, binding in enumerate(bindings):
        for side in SIDES:
            view = binding["views"][str(side)]
            source = ROOT / view["path"]
            if sha256(source) != view["sha256"]:
                raise ValueError("pixel codec probe original pixels changed")
            with np.load(source, allow_pickle=False) as archive:
                pixels, indices = archive["pixels"], archive["frame_indices"]
            storage(data_dir, pixels.nbytes + indices.nbytes + 4096)
            target = data_dir / f"case-{index:02d}-cap{side}-delta.npz"
            started = time.perf_counter()
            with target.open("xb") as stream:
                np.savez_compressed(
                    stream, pixel_delta=encode_pixel_delta(pixels), frame_indices=indices
                )
            encode_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with np.load(target, allow_pickle=False) as archive:
                reconstructed = decode_pixel_delta(archive["pixel_delta"])
                new_indices = archive["frame_indices"]
            decode_seconds = time.perf_counter() - started
            if not np.array_equal(reconstructed, pixels) or not np.array_equal(
                new_indices, indices
            ):
                raise RuntimeError("temporal pixel codec changed a pixel or frame index")
            records.append(
                {
                    "source": binding["source"],
                    "example_id": binding["example_id"],
                    "short_side": side,
                    "reference_bytes": source.stat().st_size,
                    "delta_bytes": target.stat().st_size,
                    "encode_seconds": encode_seconds,
                    "decode_seconds": decode_seconds,
                    "exact_pixels_and_indices": True,
                    "path": str(target.relative_to(ROOT)),
                    "sha256": sha256(target),
                }
            )
        print(json.dumps({"verified_cases": index + 1, "total_cases": len(bindings)}), flush=True)
    receipt = {
        "schema": "umi-studio-tiny-rgb-temporal-pixel-codec/1",
        "fit_plan_sha256": sha256(args.fit_plan),
        "cases": records,
        "reference_bytes": sum(r["reference_bytes"] for r in records),
        "delta_bytes": sum(r["delta_bytes"] for r in records),
        "decoder_source_sha256": sha256(ROOT / "tools/tiny_rgb_full_training_tools.py"),
        "probe_source_sha256": sha256(Path(__file__)),
        "training_cases": len(plan["examples"]),
        "gpu_used": False,
        "learned_features": False,
        "quality_verified": False,
        "qualification": (
            "Lossless pixel storage only. All RGB bytes and complete temporal indices are "
            "preserved; no learned features, augmentation, model update or generalization claim."
        ),
    }
    output.joinpath("receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    main()
