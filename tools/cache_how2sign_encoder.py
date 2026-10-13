"""Cache frozen SHuBERT states for verified How2Sign sentence clips.

Inputs are either publisher pre-cut clips or locally cropped publisher-realigned
intervals from a verified raw training video. The live miner has priority.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from benchmark_fleurs_dev import miner_revision, sha256
from cache_fleurs_encoder import _encoder_state


def checked_receipts(selection: dict, selection_hash: str, lines: list[str]) -> dict[str, dict]:
    rows = selection["rows"]
    selected = {row["sentence_name"] for row in rows}
    if len(selected) != len(rows):
        raise ValueError("How2Sign selection repeats a sentence")
    receipts = {}
    validation = selection["schema"] == "umi-studio-how2sign-validation-selection/1"
    raw_realigned = selection["schema"] == "umi-studio-how2sign-training-raw-realigned/1"
    expected_schema = (
        "umi-studio-how2sign-raw-realigned-clip/1"
        if raw_realigned
        else (
            "umi-studio-how2sign-validation-clip/1"
            if validation
            else "umi-studio-how2sign-training-clip/1"
        )
    )
    if raw_realigned and selection.get("training_only") is not True:
        raise ValueError("raw realigned selection must be training-only")
    selected_rows = {row["sentence_name"]: row for row in rows}
    for line in lines:
        record = json.loads(line)
        name = record["sentence_name"]
        if (
            record["schema"] != expected_schema
            or record["selection_sha256"] != selection_hash
            or name not in selected
            or name in receipts
            or (validation and record.get("split") != "validation")
        ):
            raise ValueError("download receipt is invalid or duplicated")
        if raw_realigned:
            row = selected_rows[name]
            if (
                record.get("source_video_sha256") != selection["source_video_sha256"]
                or record.get("source_video_receipt_sha256")
                != selection["source_video_receipt_sha256"]
                or record.get("video_name") != row["video_name"]
                or record.get("signer") != row["signer"]
                or record.get("start_sec") != row["start_sec"]
                or record.get("end_sec") != row["end_sec"]
                or record.get("reference") != row["reference"]
            ):
                raise ValueError("raw crop receipt differs from source interval")
        receipts[name] = record
    if selection["schema"] == "umi-studio-how2sign-training-pilot/1" and set(receipts) != selected:
        raise ValueError("all pilot training clips must be downloaded first")
    return receipts


def selected_window(rows: list[dict], start: int, stop: int | None) -> list[dict]:
    end = len(rows) if stop is None else stop
    if start < 0 or end > len(rows) or start >= end:
        raise ValueError("encoder row window is outside the frozen selection")
    return rows[start:end]


def verified_index(
    path: Path,
    selection_hash: str,
    receipt_hash: str | None,
    receipts: dict[str, dict],
    cache_dir: Path,
) -> dict[str, dict]:
    records = {}
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        record = json.loads(line)
        name = record["sentence_name"]
        if (
            record["schema"] != "umi-studio-how2sign-encoder-cache-example/1"
            or record["selection_sha256"] != selection_hash
            or record["clip_receipts_sha256"] != receipt_hash
            or name in records
            or name not in receipts
            or record["source_clip_sha256"] != receipts[name]["clip_sha256"]
        ):
            raise ValueError("encoder cache index differs from the frozen selection")
        cache = cache_dir / f"{name}.npz"
        if not cache.is_file() or sha256(cache) != record["encoder_cache_sha256"]:
            raise ValueError(f"encoder cache missing or altered: {cache}")
        records[name] = record
    return records


def pending_rows(
    window: list[dict],
    receipts: dict[str, dict],
    completed: dict[str, dict],
    skipped: dict[str, dict],
) -> list[dict]:
    if set(completed) & set(skipped):
        raise ValueError("encoder indexes overlap")
    return [
        row
        for row in window
        if row["sentence_name"] in receipts
        and row["sentence_name"] not in completed
        and row["sentence_name"] not in skipped
    ]


def resample(source: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError(f"unindexed resampled clip exists: {destination}")
    temporary = destination.with_suffix(".partial.mp4")
    if temporary.exists():
        raise ValueError(f"unfinished resampled clip must be inspected: {temporary}")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-vf",
                "fps=25",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if temporary.stat().st_size == 0:
            raise RuntimeError("resampled clip is empty")
        temporary.rename(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clip-receipts", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--resampled-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--skip-index", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--stop-row", type=int)
    parser.add_argument("--save-stream-features", action="store_true")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("use the independent frozen baseline in this workspace")
    selection_hash = sha256(args.selection)
    selection = json.loads(args.selection.read_text())
    if selection.get("schema") not in {
        "umi-studio-how2sign-training-pilot/1",
        "umi-studio-how2sign-training-scale/1",
        "umi-studio-how2sign-training-aligned/1",
        "umi-studio-how2sign-training-raw-realigned/1",
        "umi-studio-how2sign-validation-selection/1",
    }:
        raise ValueError("unexpected How2Sign training selection schema")
    rows = selection["rows"]
    window = selected_window(rows, args.start_row, args.stop_row)
    if args.skip_index is not None and args.skip_index.resolve() == args.index.resolve():
        parser.error("skip index must differ from the output index")
    if args.skip_index is not None and not args.skip_index.is_file():
        parser.error("skip index must be an existing verified index")
    validation = selection["schema"] == "umi-studio-how2sign-validation-selection/1"
    if validation and (
        selection.get("split") != "validation" or selection.get("training_excluded") is not True
    ):
        raise ValueError("How2Sign validation selection lacks split protection")
    scale = selection["schema"] != "umi-studio-how2sign-training-pilot/1"
    receipt_hash = None if scale else sha256(args.clip_receipts)
    receipts = checked_receipts(
        selection, selection_hash, args.clip_receipts.read_text().splitlines()
    )
    completed = verified_index(args.index, selection_hash, receipt_hash, receipts, args.cache_dir)
    skipped = (
        verified_index(args.skip_index, selection_hash, receipt_hash, receipts, args.cache_dir)
        if args.skip_index is not None
        else {}
    )
    todo = pending_rows(window, receipts, completed, skipped)
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
    for row in todo:
        name = row["sentence_name"]
        if new_count >= args.limit:
            break
        revision = miner_revision(args.miner_health_url, revision)
        source = args.source_dir / f"{name}.mp4"
        source_receipt = receipts[name]
        if not source.is_file() or sha256(source) != source_receipt["clip_sha256"]:
            raise ValueError(f"source clip differs from publisher download: {source}")
        clip = args.resampled_dir / f"{name}.mp4"
        resample(source, clip)
        clip_hash = sha256(clip)
        if args.save_stream_features:
            state, diagnostics, streams = _encoder_state(
                model, clip, args.miner_health_url, revision, return_streams=True
            )
        else:
            state, diagnostics = _encoder_state(model, clip, args.miner_health_url, revision)
            streams = {}
        revision = miner_revision(args.miner_health_url, revision)
        cache = args.cache_dir / f"{name}.npz"
        temporary = cache.with_suffix(".partial.npz")
        if cache.exists() or temporary.exists():
            raise RuntimeError(f"unindexed encoder cache must be inspected: {cache}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, encoder_hidden_state=state, **streams)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(cache)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "schema": "umi-studio-how2sign-encoder-cache-example/1",
            "sentence_name": name,
            "signer": row["signer"],
            "split": "validation" if validation else "training",
            "selection_sha256": selection_hash,
            "clip_receipts_sha256": receipt_hash,
            "source_clip_sha256": source_receipt["clip_sha256"],
            "resampled_clip_sha256": clip_hash,
            "clip_fps": 25,
            "encoder_cache_sha256": sha256(cache),
            "encoder_shape": list(state.shape),
            "encoder_dtype": str(state.dtype),
            "video_decode_strategy": "opencv_sequential_full_range",
            "baseline_identity_sha256": baseline_identity_hash,
            "baseline_runtime_sha256": baseline_runtime_hash,
            "reference": row["reference"],
            "miner_model_revision": revision,
            "diagnostics": diagnostics,
        }
        if streams:
            record["stream_feature_shapes"] = {
                name: list(value.shape) for name, value in streams.items()
            }
            record["stream_feature_dtype"] = "float32"
        with args.index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        new_count += 1
        print(f"{len(completed) + new_count}: cached {name}", flush=True)
        del state
        gc.collect()


if __name__ == "__main__":
    main()
