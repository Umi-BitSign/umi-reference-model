"""Reuse verified pilot encoder states in the aligned How2Sign training cache.

Run only while no other process writes the aligned cache index. Reuse requires
matching source bytes, reference, transform, frozen model and cache checksums.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from cache_how2sign_encoder import checked_receipts


def reusable_record(
    row: dict,
    receipt: dict,
    prior: dict,
    selection_hash: str,
    baseline_identity_hash: str,
    baseline_runtime_hash: str,
    revision: str,
    prior_index_hash: str,
) -> dict:
    if (
        prior["schema"] != "umi-studio-how2sign-encoder-cache-example/1"
        or prior["split"] != "training"
        or prior["sentence_name"] != row["sentence_name"]
        or prior["signer"] != row["signer"]
        or prior["reference"] != row["reference"]
        or prior["source_clip_sha256"] != receipt["clip_sha256"]
        or prior["clip_fps"] != 25
        or prior["video_decode_strategy"] != "opencv_sequential_full_range"
        or prior["baseline_identity_sha256"] != baseline_identity_hash
        or prior["baseline_runtime_sha256"] != baseline_runtime_hash
    ):
        raise ValueError("prior encoder state differs from aligned source or frozen model")
    return {
        **prior,
        "selection_sha256": selection_hash,
        "clip_receipts_sha256": None,
        "miner_model_revision": revision,
        "reuse_source": {
            "prior_index_sha256": prior_index_hash,
            "prior_selection_sha256": prior["selection_sha256"],
            "prior_encoder_cache_sha256": prior["encoder_cache_sha256"],
        },
    }


def copy_verified(source: Path, destination: Path, expected_hash: str) -> None:
    if not source.is_file() or sha256(source) != expected_hash:
        raise ValueError(f"prior artifact differs from recorded checksum: {source}")
    if destination.exists():
        raise ValueError(f"unindexed aligned artifact needs inspection: {destination}")
    temporary = destination.with_suffix(".partial" + destination.suffix)
    if temporary.exists():
        raise ValueError(f"unfinished aligned artifact needs inspection: {temporary}")
    try:
        with source.open("rb") as source_file, temporary.open("xb") as target_file:
            shutil.copyfileobj(source_file, target_file, length=1024 * 1024)
            target_file.flush()
            os.fsync(target_file.fileno())
        if sha256(temporary) != expected_hash:
            raise ValueError("copied aligned artifact differs from prior checksum")
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
    parser.add_argument("--prior-index", type=Path, required=True)
    parser.add_argument("--prior-resampled-dir", type=Path, required=True)
    parser.add_argument("--prior-cache-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=128)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("reuse requires the independent frozen baseline")
    selection = json.loads(args.selection.read_text())
    if selection.get("schema") != "umi-studio-how2sign-training-aligned/1":
        raise ValueError("target selection is not alignment-audited training")
    selection_hash = sha256(args.selection)
    receipts = checked_receipts(
        selection, selection_hash, args.clip_receipts.read_text().splitlines()
    )
    prior_index_hash = sha256(args.prior_index)
    prior_rows = [json.loads(line) for line in args.prior_index.read_text().splitlines()]
    prior_by_name = {row["sentence_name"]: row for row in prior_rows}
    if len(prior_by_name) != len(prior_rows):
        raise ValueError("prior encoder index repeats a sentence")
    current_rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    current_by_name = {row["sentence_name"]: row for row in current_rows}
    if len(current_by_name) != len(current_rows):
        raise ValueError("aligned encoder index repeats a sentence")
    for row in current_rows:
        name = row["sentence_name"]
        if (
            row["schema"] != "umi-studio-how2sign-encoder-cache-example/1"
            or row["split"] != "training"
            or row["selection_sha256"] != selection_hash
            or row["source_clip_sha256"] != receipts[name]["clip_sha256"]
            or row["clip_receipts_sha256"] is not None
            or sha256(args.cache_dir / f"{name}.npz") != row["encoder_cache_sha256"]
        ):
            raise ValueError("existing aligned cache index differs from verified inputs")
    identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    runtime_hash = sha256(baseline / "runtime.py")
    revision = miner_revision(args.miner_health_url, None)
    args.resampled_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    added = 0
    for row in selection["rows"]:
        name = row["sentence_name"]
        if name in current_by_name or name not in prior_by_name or name not in receipts:
            continue
        if added >= args.limit:
            break
        revision = miner_revision(args.miner_health_url, revision)
        receipt = receipts[name]
        source_clip = args.source_dir / f"{name}.mp4"
        if not source_clip.is_file() or sha256(source_clip) != receipt["clip_sha256"]:
            raise ValueError(f"aligned source clip differs from publisher receipt: {source_clip}")
        prior = prior_by_name[name]
        record = reusable_record(
            row,
            receipt,
            prior,
            selection_hash,
            identity_hash,
            runtime_hash,
            revision,
            prior_index_hash,
        )
        copy_verified(
            args.prior_resampled_dir / f"{name}.mp4",
            args.resampled_dir / f"{name}.mp4",
            prior["resampled_clip_sha256"],
        )
        copy_verified(
            args.prior_cache_dir / f"{name}.npz",
            args.cache_dir / f"{name}.npz",
            prior["encoder_cache_sha256"],
        )
        revision = miner_revision(args.miner_health_url, revision)
        record["miner_model_revision"] = revision
        with args.index.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        added += 1
        print(f"{len(current_rows) + added}: reused {name}", flush=True)


if __name__ == "__main__":
    main()
