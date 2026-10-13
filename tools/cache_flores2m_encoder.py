"""Cache frozen SHuBERT states for protected 2M-Flores-ASL dev pilot clips.

The selection excludes all source articles containing FLEURS validation text.
Only its pinned dev videos are accepted; devtest is outside this workflow.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import urllib.parse
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import miner_revision, sha256
from cache_fleurs_encoder import _encoder_state
from cache_how2sign_encoder import resample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clip-receipts", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--resampled-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("use the independent frozen baseline in this workspace")
    selection_hash = sha256(args.selection)
    selection = json.loads(args.selection.read_text())
    if (
        selection.get("schema") != "umi-studio-flores2m-training-pilot/1"
        or selection.get("source_split") != "dev"
        or selection.get("devtest_used") is not False
    ):
        raise ValueError("unexpected 2M-Flores pilot selection or protected split")
    rows = selection["rows"]
    if len({row["row_idx"] for row in rows}) != len(rows):
        raise ValueError("pilot selection repeats a source row")
    receipt_hash = sha256(args.clip_receipts)
    receipts = {}
    for line in args.clip_receipts.read_text().splitlines():
        record = json.loads(line)
        if (
            record["schema"] != "umi-studio-flores2m-training-clip/1"
            or record["selection_sha256"] != selection_hash
            or record["row_idx"] in receipts
        ):
            raise ValueError("download receipt is invalid or duplicated")
        receipts[record["row_idx"]] = record
    if set(receipts) != {row["row_idx"] for row in rows}:
        raise ValueError("all selected training clips must be downloaded first")
    completed = {}
    if args.index.exists():
        for line in args.index.read_text().splitlines():
            record = json.loads(line)
            if (
                record["schema"] != "umi-studio-flores2m-encoder-cache-example/1"
                or record["selection_sha256"] != selection_hash
                or record["clip_receipts_sha256"] != receipt_hash
                or record["example_id"] in completed
            ):
                raise ValueError("encoder cache index differs from the frozen pilot")
            cache = args.cache_dir / f"{record['example_id']}.npz"
            if not cache.is_file() or sha256(cache) != record["encoder_cache_sha256"]:
                raise ValueError(f"encoder cache missing or altered: {cache}")
            completed[record["example_id"]] = record
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

    revision = miner_revision(args.miner_health_url, None)
    model = SHuBERTInferenceRuntime(baseline, device="cpu")
    model._video_reader = make_sequential_reader(model._video_reader)
    revision = miner_revision(args.miner_health_url, revision)
    baseline_identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    baseline_runtime_hash = sha256(baseline / "runtime.py")
    args.resampled_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    for row in rows:
        identifier = f"flores2m_dev_row_{row['row_idx']:04d}"
        if identifier in completed:
            continue
        if new_count >= args.limit:
            break
        revision = miner_revision(args.miner_health_url, revision)
        name = Path(urllib.parse.urlparse(row["video_url"]).path).name
        source = args.source_dir / name
        source_receipt = receipts[row["row_idx"]]
        if (
            source_receipt["filename"] != name
            or source_receipt["video_sha256"] != row["video_sha256"]
            or not source.is_file()
            or sha256(source) != row["video_sha256"]
        ):
            raise ValueError(f"source clip differs from publisher file: {source}")
        clip = args.resampled_dir / f"{identifier}.mp4"
        resample(source, clip)
        clip_hash = sha256(clip)
        state, diagnostics = _encoder_state(model, clip, args.miner_health_url, revision)
        revision = miner_revision(args.miner_health_url, revision)
        cache = args.cache_dir / f"{identifier}.npz"
        temporary = cache.with_suffix(".partial.npz")
        if cache.exists() or temporary.exists():
            raise RuntimeError(f"unindexed encoder cache must be inspected: {cache}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, encoder_hidden_state=state)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "schema": "umi-studio-flores2m-encoder-cache-example/1",
            "example_id": identifier,
            "row_idx": row["row_idx"],
            "source_page_sha256": sha256_text(row["source_page"]),
            "split": "training",
            "selection_sha256": selection_hash,
            "clip_receipts_sha256": receipt_hash,
            "source_clip_sha256": row["video_sha256"],
            "resampled_clip_sha256": clip_hash,
            "clip_fps": 25,
            "encoder_cache_sha256": sha256(cache),
            "encoder_shape": list(state.shape),
            "encoder_dtype": str(state.dtype),
            "video_decode_strategy": "opencv_sequential_full_range",
            "baseline_identity_sha256": baseline_identity_hash,
            "baseline_runtime_sha256": baseline_runtime_hash,
            "reference": row["sentence"],
            "miner_model_revision": revision,
            "diagnostics": diagnostics,
        }
        with args.index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        new_count += 1
        print(f"{len(completed) + new_count}: cached {identifier}", flush=True)
        del state
        gc.collect()


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


if __name__ == "__main__":
    main()
