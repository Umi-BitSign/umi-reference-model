"""Cache baseline visual streams for selected ASL STEM Wiki training clips."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import extract_clip, miner_revision, sha256
from cache_fleurs_encoder import _encoder_state
from fetch_asl_stem_wiki_videos import verified_receipts

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"


def checked_completed(
    index: Path,
    selection_hash: str,
    receipt_hash: str,
    cache_dir: Path,
) -> dict[str, dict]:
    if not index.exists():
        return {}
    completed = {}
    for line in index.read_text().splitlines():
        row = json.loads(line)
        example_id = row.get("example_id")
        cache = cache_dir / f"{example_id}.npz"
        if (
            row.get("schema")
            != "umi-studio-asl-stem-wiki-encoder-cache-example/1"
            or row.get("split") != "training"
            or row.get("selection_sha256") != selection_hash
            or row.get("source_receipts_sha256") != receipt_hash
            or row.get("clip_fps") != 25
            or row.get("encoder_dtype") != "float32"
            or row.get("stream_feature_dtype") != "float32"
            or example_id in completed
            or not cache.is_file()
            or sha256(cache) != row.get("encoder_cache_sha256")
        ):
            raise ValueError(f"ASL STEM Wiki encoder cache differs: {example_id}")
        completed[example_id] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-receipts", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--mps-idle-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--miner-health-url")
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if args.miner_maintenance_mode == bool(args.miner_health_url):
        parser.error(
            "choose exactly one of --miner-health-url or --miner-maintenance-mode"
        )
    if not 1 <= args.limit <= 32:
        parser.error("visual extraction batch must contain 1-32 clips")
    if args.device == "mps" and args.mps_idle_fraction != 0.5:
        parser.error("MPS extraction must reserve half of average GPU capacity")
    if args.device == "cpu" and args.mps_idle_fraction != 0:
        parser.error("CPU extraction requires --mps-idle-fraction 0")
    selection_hash = sha256(args.selection)
    receipt_hash = sha256(args.source_receipts)
    selection = json.loads(args.selection.read_text())
    receipt_lines = args.source_receipts.read_text().splitlines()
    if (
        selection.get("schema")
        != "umi-studio-asl-stem-wiki-training-selection/1"
        or len(receipt_lines) != selection.get("selected_examples")
    ):
        raise ValueError("ASL STEM Wiki selection or receipts are incomplete")
    verified_receipts(
        selection, selection_hash, receipt_lines, args.source_dir
    )
    receipts = {row["example_id"]: row for row in map(json.loads, receipt_lines)}
    completed = checked_completed(
        args.index, selection_hash, receipt_hash, args.cache_dir
    )

    baseline = BASELINE.resolve()
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
    from runtime import SHuBERTInferenceRuntime
    from sequential_video_reader import make_sequential_reader

    revision = (
        None
        if args.miner_maintenance_mode
        else miner_revision(args.miner_health_url, None)
    )
    model = SHuBERTInferenceRuntime(baseline, device=args.device)
    model._video_reader = make_sequential_reader(model._video_reader)
    args.clip_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    baseline_identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    baseline_runtime_hash = sha256(baseline / "runtime.py")

    def check_miner() -> None:
        nonlocal revision
        if not args.miner_maintenance_mode:
            revision = miner_revision(args.miner_health_url, revision)

    new_count = 0
    for row in selection["records"]:
        example_id = row["example_id"]
        if example_id in completed:
            continue
        if new_count >= args.limit:
            break
        check_miner()
        receipt = receipts[example_id]
        source = args.source_dir / f"{example_id}.mp4"
        if sha256(source) != receipt["video_sha256"]:
            raise ValueError(f"ASL STEM Wiki source video changed: {example_id}")
        clip = args.clip_dir / f"{example_id}.mp4"
        extract_clip(
            source,
            clip,
            0,
            receipt["video_probe"]["duration_sec"],
            "25",
        )
        clip_hash = sha256(clip)
        active_started = time.perf_counter()
        state, diagnostics, streams = _encoder_state(
            model,
            clip,
            None if args.miner_maintenance_mode else args.miner_health_url,
            revision,
            return_streams=True,
        )
        active_seconds = time.perf_counter() - active_started
        check_miner()
        cache = args.cache_dir / f"{example_id}.npz"
        temporary = cache.with_suffix(".partial.npz")
        if cache.exists() or temporary.exists():
            raise RuntimeError(f"unindexed cache needs inspection: {cache}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, encoder_hidden_state=state, **streams)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "schema": "umi-studio-asl-stem-wiki-encoder-cache-example/1",
            "example_id": example_id,
            "participant": row["participant"],
            "article": row["article"],
            "topic": row["topic"],
            "source_group": row["source_group"],
            "split": "training",
            "selection_sha256": selection_hash,
            "source_receipts_sha256": receipt_hash,
            "source_video_sha256": receipt["video_sha256"],
            "resampled_clip_sha256": clip_hash,
            "clip_fps": 25,
            "encoder_cache_sha256": sha256(cache),
            "encoder_shape": list(state.shape),
            "encoder_dtype": str(state.dtype),
            "stream_feature_shapes": {
                name: list(value.shape) for name, value in streams.items()
            },
            "stream_feature_dtype": "float32",
            "video_decode_strategy": "opencv_sequential_full_range",
            "baseline_identity_sha256": baseline_identity_hash,
            "baseline_runtime_sha256": baseline_runtime_hash,
            "reference": row["reference"],
            "miner_model_revision": revision,
            "miner_health_policy": (
                "user_authorized_maintenance"
                if args.miner_maintenance_mode
                else "required_stable_revision"
            ),
            "device": args.device,
            "mps_idle_fraction": args.mps_idle_fraction,
            "diagnostics": diagnostics,
        }
        with args.index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        new_count += 1
        print(
            f"{len(completed) + new_count}/{selection['selected_examples']}: "
            f"cached {example_id}",
            flush=True,
        )
        if args.device == "mps" and args.mps_idle_fraction:
            time.sleep(
                active_seconds
                * args.mps_idle_fraction
                / (1 - args.mps_idle_fraction)
            )
        del state, streams
        gc.collect()
    check_miner()


if __name__ == "__main__":
    main()
