"""Freeze complete publisher-realigned How2Sign training states for one raw video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from itertools import pairwise
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from cache_how2sign_encoder import checked_receipts
from crop_how2sign_raw_video import read_aligned, select_rows


def receipt_sha256(receipt: dict) -> str:
    serialized = json.dumps(receipt, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(serialized).hexdigest()


def checked_source_rows(
    selection: dict,
    video_receipt: dict,
    aligned_rows: list[dict[str, str]],
    pre_cut_selection: dict,
) -> None:
    if pre_cut_selection.get("schema") != "umi-studio-how2sign-training-aligned/1":
        raise ValueError("pre-cut exclusion is not a training selection")
    if any(row["video_id"] == selection["video_id"] for row in pre_cut_selection["rows"]):
        raise ValueError("raw source video is already in pre-cut training selection")
    excluded_names = {row["sentence_name"] for row in pre_cut_selection["rows"]}
    policy = selection.get("source_duration_policy")
    if policy not in {None, "drop_intervals_past_verified_video_end"}:
        raise ValueError("raw training selection has an unknown source duration policy")
    expected = select_rows(
        aligned_rows,
        video_receipt,
        excluded_names,
        skip_unavailable=policy is not None,
    )
    if selection["rows"] != expected:
        raise ValueError("raw training rows differ from publisher realignment")


def canonical_rows(
    selection: dict,
    selection_hash: str,
    video_receipt: dict,
    clip_receipts: dict[str, dict],
    encoder_rows: list[dict],
    video_dir: Path,
    source_dir: Path,
    resampled_dir: Path,
    cache_dir: Path,
    identity_hash: str,
    runtime_hash: str,
    health_url: str | None = None,
) -> list[dict]:
    rows = selection["rows"]
    names = [row["sentence_name"] for row in rows]
    ordered_intervals = sorted(rows, key=lambda row: (row["start_sec"], row["sentence_name"]))
    duration = video_receipt["video_probe"]["duration_sec"]
    if (
        selection.get("schema") != "umi-studio-how2sign-training-raw-realigned/1"
        or selection.get("training_only") is not True
        or video_receipt.get("schema") != "umi-studio-how2sign-raw-training-video/1"
        or selection["video_name"] != video_receipt["video_name"]
        or selection["video_id"] != video_receipt["video_id"]
        or selection["signer"] != video_receipt["signer"]
        or selection["source_video_sha256"] != video_receipt["video_sha256"]
        or selection["source_video_receipt_sha256"] != receipt_sha256(video_receipt)
        or not rows
        or len(names) != len(set(names))
        or set(clip_receipts) != set(names)
        or len(encoder_rows) != len(names)
        or any(
            row["video_name"] != selection["video_name"]
            or row["video_id"] != selection["video_id"]
            or row["signer"] != selection["signer"]
            or row["source_video_sha256"] != selection["source_video_sha256"]
            or not math.isfinite(row["start_sec"])
            or not math.isfinite(row["end_sec"])
            or row["start_sec"] < 0
            or not 2 <= row["end_sec"] - row["start_sec"] <= 12
            or row["end_sec"] > duration + 0.01
            or len(row["reference"].split()) < 3
            or Path(row["sentence_name"]).name != row["sentence_name"]
            for row in rows
        )
        or any(
            left["end_sec"] > right["start_sec"] + 0.001
            for left, right in pairwise(ordered_intervals)
        )
    ):
        raise ValueError("raw training selection, source video or cache is incomplete")
    source_video = video_dir / f"{selection['video_name']}.mp4"
    if not source_video.is_file() or sha256(source_video) != selection["source_video_sha256"]:
        raise ValueError("raw source video differs from verified publisher bytes")
    by_name = {row["sentence_name"]: row for row in encoder_rows}
    if len(by_name) != len(encoder_rows) or set(by_name) != set(names):
        raise ValueError("raw encoder index is duplicated or differs from selection")
    revision = miner_revision(health_url, None) if health_url else None
    ordered = []
    for position, selected in enumerate(rows):
        if health_url and position % 16 == 0:
            revision = miner_revision(health_url, revision)
        name = selected["sentence_name"]
        clip_receipt = clip_receipts[name]
        record = by_name[name]
        if (
            record["schema"] != "umi-studio-how2sign-encoder-cache-example/1"
            or record["split"] != "training"
            or record["sentence_name"] != name
            or record["signer"] != selected["signer"]
            or record["reference"] != selected["reference"]
            or record["selection_sha256"] != selection_hash
            or record["clip_receipts_sha256"] is not None
            or record["source_clip_sha256"] != clip_receipt["clip_sha256"]
            or record["baseline_identity_sha256"] != identity_hash
            or record["baseline_runtime_sha256"] != runtime_hash
            or record["clip_fps"] != 25
            or record["video_decode_strategy"] != "opencv_sequential_full_range"
            or record["encoder_dtype"] != "float32"
            or len(record["encoder_shape"]) != 2
            or record["encoder_shape"][0] < 1
            or record["encoder_shape"][1] != 1536
        ):
            raise ValueError(f"raw encoder record differs from selection: {name}")
        paths = (
            (source_dir / f"{name}.mp4", record["source_clip_sha256"]),
            (resampled_dir / f"{name}.mp4", record["resampled_clip_sha256"]),
            (cache_dir / f"{name}.npz", record["encoder_cache_sha256"]),
        )
        for path, digest in paths:
            if not path.is_file() or sha256(path) != digest:
                raise ValueError(f"raw training artifact differs from its checksum: {path}")
        ordered.append(record)
    if health_url:
        miner_revision(health_url, revision)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--video-receipts", type=Path, required=True)
    parser.add_argument("--alignment-metadata", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, required=True)
    parser.add_argument("--clip-receipts", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
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
    if args.output.resolve() == args.index.resolve():
        parser.error("output must not replace the incremental encoder index")
    selection = json.loads(args.selection.read_text())
    if selection["alignment_metadata_sha256"] != sha256(args.alignment_metadata) or selection[
        "excluded_pre_cut_selection_sha256"
    ] != sha256(args.exclude_selection):
        raise ValueError("raw training selection differs from source metadata")
    video_receipts = [json.loads(line) for line in args.video_receipts.read_text().splitlines()]
    matching = [row for row in video_receipts if row["video_name"] == selection["video_name"]]
    if len(matching) != 1:
        raise ValueError("raw training selection lacks one source video receipt")
    checked_source_rows(
        selection,
        matching[0],
        read_aligned(args.alignment_metadata),
        json.loads(args.exclude_selection.read_text()),
    )
    selection_hash = sha256(args.selection)
    clip_receipts = checked_receipts(
        selection, selection_hash, args.clip_receipts.read_text().splitlines()
    )
    encoder_rows = [json.loads(line) for line in args.index.read_text().splitlines()]
    ordered = canonical_rows(
        selection,
        selection_hash,
        matching[0],
        clip_receipts,
        encoder_rows,
        args.video_dir,
        args.source_dir,
        args.resampled_dir,
        args.cache_dir,
        sha256(baseline / "MODEL_IDENTITY.json"),
        sha256(baseline / "runtime.py"),
        args.miner_health_url,
    )
    if args.output.exists():
        if [json.loads(line) for line in args.output.read_text().splitlines()] != ordered:
            raise ValueError("frozen raw training index differs from current inputs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".partial.jsonl")
        if temporary.exists():
            parser.error("unfinished frozen raw index needs inspection")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                for record in ordered:
                    output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"frozen {len(ordered)} raw training states sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
