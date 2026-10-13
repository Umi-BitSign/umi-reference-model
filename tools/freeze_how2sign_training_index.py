"""Freeze a complete aligned How2Sign encoder index in selection order.

The incremental cache worker may append verified reused states out of order.
This command checks every source, transform and cache checksum before writing
the immutable index used by a full-data training plan.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from cache_how2sign_encoder import checked_receipts


def canonical_rows(
    selection: dict,
    selection_hash: str,
    receipts: dict[str, dict],
    encoder_rows: list[dict],
    source_dir: Path,
    resampled_dir: Path,
    cache_dir: Path,
    identity_hash: str,
    runtime_hash: str,
    health_url: str | None = None,
) -> list[dict]:
    selected = selection["rows"]
    names = [row["sentence_name"] for row in selected]
    videos = Counter(row["video_name"] for row in selected)
    signers = Counter(str(row["signer"]) for row in selected)
    quotas = selection.get("per_signer_quotas")
    if (
        selection.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or not isinstance(quotas, dict)
        or any(type(count) is not int or count < 0 for count in quotas.values())
        or not selected
        or len(selected) != sum(quotas.values())
        or len(set(names)) != len(names)
        or max(videos.values()) > selection.get("max_clips_per_source_video", 0)
        or dict(signers) != quotas
        or any(
            row["original_interval_overlap_fraction"]
            < selection.get("min_bidirectional_overlap", 1)
            or row["realigned_interval_overlap_fraction"]
            < selection.get("min_bidirectional_overlap", 1)
            for row in selected
        )
        or set(receipts) != set(names)
        or len(encoder_rows) != len(names)
    ):
        raise ValueError("aligned training selection or cache is incomplete")
    by_name = {row["sentence_name"]: row for row in encoder_rows}
    if len(by_name) != len(encoder_rows) or set(by_name) != set(names):
        raise ValueError("encoder index is duplicated or differs from selection")
    revision = miner_revision(health_url, None) if health_url else None
    ordered = []
    for position, selected_row in enumerate(selected):
        if health_url and position % 32 == 0:
            revision = miner_revision(health_url, revision)
        name = selected_row["sentence_name"]
        receipt = receipts[name]
        record = by_name[name]
        if (
            record["schema"] != "umi-studio-how2sign-encoder-cache-example/1"
            or record["split"] != "training"
            or record["sentence_name"] != name
            or record["signer"] != selected_row["signer"]
            or record["reference"] != selected_row["reference"]
            or record["selection_sha256"] != selection_hash
            or record["clip_receipts_sha256"] is not None
            or record["source_clip_sha256"] != receipt["clip_sha256"]
            or record["baseline_identity_sha256"] != identity_hash
            or record["baseline_runtime_sha256"] != runtime_hash
            or record["clip_fps"] != 25
            or record["video_decode_strategy"] != "opencv_sequential_full_range"
            or record["encoder_dtype"] != "float32"
            or len(record["encoder_shape"]) != 2
            or record["encoder_shape"][0] < 1
            or record["encoder_shape"][1] != 1536
        ):
            raise ValueError(f"encoder record differs from aligned selection: {name}")
        paths = (
            (source_dir / f"{name}.mp4", record["source_clip_sha256"]),
            (resampled_dir / f"{name}.mp4", record["resampled_clip_sha256"]),
            (cache_dir / f"{name}.npz", record["encoder_cache_sha256"]),
        )
        for path, digest in paths:
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"training artifact differs from its checksum: {path}")
        ordered.append(record)
    if health_url:
        miner_revision(health_url, revision)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clip-receipts", type=Path, required=True)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--resampled-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("requires the independent frozen baseline")
    if args.output.resolve() in {index.resolve() for index in args.index}:
        parser.error("output must not replace an incremental encoder index")
    selection = json.loads(args.selection.read_text())
    selection_hash = sha256(args.selection)
    receipts = checked_receipts(
        selection, selection_hash, args.clip_receipts.read_text().splitlines()
    )
    encoder_rows = [
        json.loads(line) for index in args.index for line in index.read_text().splitlines()
    ]
    ordered = canonical_rows(
        selection,
        selection_hash,
        receipts,
        encoder_rows,
        args.source_dir,
        args.resampled_dir,
        args.cache_dir,
        sha256(baseline / "MODEL_IDENTITY.json"),
        sha256(baseline / "runtime.py"),
        args.miner_health_url,
    )
    if args.output.exists():
        if [json.loads(line) for line in args.output.read_text().splitlines()] != ordered:
            raise ValueError("frozen training index differs from current inputs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".partial.jsonl")
        with temporary.open("x", encoding="utf-8") as output:
            for record in ordered:
                output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(args.output)
    print(f"frozen {len(ordered)} aligned training states sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
