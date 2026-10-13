"""Recover visual streams for immutable How2Sign training encoder caches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from cache_fleurs_encoder import _encoder_state


def embedded_streams(row: dict) -> bool:
    dtype = row.get("stream_feature_dtype")
    shapes = row.get("stream_feature_shapes")
    if dtype is None and shapes is None:
        return False
    frames = row.get("encoder_shape", [None])[0]
    expected = {
        "face_features": [frames, 384],
        "left_hand_features": [frames, 384],
        "right_hand_features": [frames, 384],
        "pose_features": [frames, 14],
    }
    if dtype != "float32" or shapes != expected:
        raise ValueError("backfill index has invalid embedded visual streams")
    return True


def check_training_rows(rows: list[dict], baseline: Path) -> None:
    identity = sha256(baseline / "MODEL_IDENTITY.json")
    runtime = sha256(baseline / "runtime.py")
    names = [row["sentence_name"] for row in rows]
    if not rows or len(names) != len(set(names)):
        raise ValueError("backfill training index is empty or duplicated")
    if any(
        row.get("schema") != "umi-studio-how2sign-encoder-cache-example/1"
        or row.get("split") != "training"
        or row.get("signer") not in (1, 2, 3, 4, 5, 8, 9, 11)
        or row.get("clip_fps") != 25
        or row.get("video_decode_strategy") != "opencv_sequential_full_range"
        or row.get("baseline_identity_sha256") != identity
        or row.get("baseline_runtime_sha256") != runtime
        or row.get("encoder_dtype") != "float32"
        or not isinstance(row.get("encoder_shape"), list)
        or len(row["encoder_shape"]) != 2
        or row["encoder_shape"][0] < 1
        or row["encoder_shape"][1] != 1536
        for row in rows
    ):
        raise ValueError("backfill index is not frozen How2Sign training-only data")
    for row in rows:
        embedded_streams(row)


def checked_completed(
    path: Path,
    rows: list[dict],
    index_hash: str,
    sidecar_dir: Path,
) -> dict[str, dict]:
    if not path.exists():
        return {}
    by_name = {row["sentence_name"]: row for row in rows}
    completed = {}
    for line in path.read_text().splitlines():
        record = json.loads(line)
        name = record["sentence_name"]
        source = by_name.get(name)
        if (
            record.get("schema") != "umi-studio-how2sign-visual-stream-sidecar/1"
            or source is None
            or embedded_streams(source)
            or name in completed
            or record.get("original_index_sha256") != index_hash
            or record.get("selection_sha256") != source["selection_sha256"]
            or record.get("encoder_cache_sha256") != source["encoder_cache_sha256"]
            or record.get("resampled_clip_sha256") != source["resampled_clip_sha256"]
            or record.get("stream_feature_dtype") != "float32"
            or sha256(sidecar_dir / f"{name}.npz") != record.get("stream_sha256")
        ):
            raise ValueError(f"existing visual-stream sidecar differs: {name}")
        completed[name] = record
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--sidecar-index", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if (
        baseline != workspace / "artifacts/baseline-6fe8df59"
        or not args.sidecar_dir.resolve().is_relative_to(workspace / "data")
        or not args.sidecar_index.resolve().is_relative_to(workspace / "runs")
        or args.sidecar_dir.resolve() == args.cache_dir.resolve()
        or args.sidecar_index.resolve() == args.index.resolve()
        or args.limit < 1
    ):
        parser.error("backfill requires independent, new sidecar paths and a positive limit")
    index_hash = sha256(args.index)
    rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    check_training_rows(rows, baseline)
    completed = checked_completed(args.sidecar_index, rows, index_hash, args.sidecar_dir)
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
    from runtime import SHuBERTInferenceRuntime
    from sequential_video_reader import make_sequential_reader

    revision = miner_revision(args.miner_health_url, None)
    model = SHuBERTInferenceRuntime(baseline, device="cpu")
    model._video_reader = make_sequential_reader(model._video_reader)
    revision = miner_revision(args.miner_health_url, revision)
    args.sidecar_dir.mkdir(parents=True, exist_ok=True)
    args.sidecar_index.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    for row in rows:
        if new_count >= args.limit:
            break
        name = row["sentence_name"]
        if name in completed or embedded_streams(row):
            continue
        revision = miner_revision(args.miner_health_url, revision)
        clip = args.clip_dir / f"{name}.mp4"
        cache = args.cache_dir / f"{name}.npz"
        if (
            sha256(clip) != row["resampled_clip_sha256"]
            or sha256(cache) != row["encoder_cache_sha256"]
        ):
            raise ValueError(f"backfill source clip or encoder cache differs: {name}")
        with np.load(cache, allow_pickle=False) as payload:
            if set(payload.files) != {"encoder_hidden_state"}:
                raise ValueError(f"backfill input already has visual streams: {name}")
            old_state = payload["encoder_hidden_state"]
        state, _, streams = _encoder_state(
            model, clip, args.miner_health_url, revision, return_streams=True
        )
        if (
            old_state.shape != tuple(row["encoder_shape"])
            or old_state.dtype != np.float32
            or not np.array_equal(old_state, state)
        ):
            raise ValueError(f"backfill changed the frozen encoder state: {name}")
        revision = miner_revision(args.miner_health_url, revision)
        target = args.sidecar_dir / f"{name}.npz"
        temporary = target.with_suffix(".partial.npz")
        if target.exists() or temporary.exists():
            raise RuntimeError(f"unindexed stream sidecar needs inspection: {target}")
        try:
            with temporary.open("xb") as output:
                np.savez(output, **streams)
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(target)
        finally:
            temporary.unlink(missing_ok=True)
        record = {
            "schema": "umi-studio-how2sign-visual-stream-sidecar/1",
            "sentence_name": name,
            "selection_sha256": row["selection_sha256"],
            "original_index_sha256": index_hash,
            "encoder_cache_sha256": row["encoder_cache_sha256"],
            "resampled_clip_sha256": row["resampled_clip_sha256"],
            "stream_sha256": sha256(target),
            "stream_feature_shapes": {key: list(value.shape) for key, value in streams.items()},
            "stream_feature_dtype": "float32",
            "baseline_identity_sha256": row["baseline_identity_sha256"],
            "baseline_runtime_sha256": row["baseline_runtime_sha256"],
            "miner_model_revision": revision,
        }
        with args.sidecar_index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        new_count += 1
        print(f"{len(completed) + new_count}: recovered visual streams for {name}", flush=True)
    miner_revision(args.miner_health_url, revision)


if __name__ == "__main__":
    main()
