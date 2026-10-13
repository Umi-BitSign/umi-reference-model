"""Freeze all source-verified raw How2Sign training states in plan order."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from cache_how2sign_encoder import checked_receipts, verified_index
from cache_how2sign_raw_expansion import existing_indexes
from crop_how2sign_raw_expansion import receipt_paths, verified_crops
from crop_how2sign_raw_video import read_aligned
from fetch_how2sign_raw_expansion import verified_downloads
from freeze_how2sign_raw_training_index import canonical_rows, checked_source_rows
from revise_how2sign_raw_expansion import revised_selection


def ordered_expansion_rows(plan: dict, by_video: dict[str, list[dict]]) -> list[dict]:
    videos = plan["videos"]
    names = [video["video_name"] for video in videos]
    ids = [video["video_id"] for video in videos]
    if (
        plan.get("schema") != "umi-studio-how2sign-raw-training-expansion/2"
        or plan.get("training_only") is not True
        or len(videos) != plan["selected_video_count"]
        or len(names) != len(set(names))
        or len(ids) != len(set(ids))
        or set(by_video) != set(names)
    ):
        raise ValueError("raw expansion video coverage is incomplete or duplicated")
    ordered = []
    sentence_names = set()
    for video in videos:
        records = by_video[video["video_name"]]
        clips = video["clips"]
        if len(records) != video["eligible_clip_count"] or len(clips) != len(records):
            raise ValueError("raw expansion video has incomplete encoder coverage")
        for clip, record in zip(clips, records, strict=True):
            name = clip["sentence_name"]
            if (
                name in sentence_names
                or record["sentence_name"] != name
                or record["signer"] != video["signer"]
                or record["reference"] != clip["reference"]
            ):
                raise ValueError("raw expansion row differs from the frozen selection")
            sentence_names.add(name)
            ordered.append(record)
    if len(ordered) != plan["selected_clip_count"]:
        raise ValueError("raw expansion does not cover every planned clip")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--previous-selection", type=Path, required=True)
    parser.add_argument("--video-receipts", type=Path, required=True)
    parser.add_argument("--alignment-metadata", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--resampled-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    baseline = args.baseline.resolve()
    if baseline != workspace / "artifacts/baseline-6fe8df59":
        parser.error("requires the independent frozen baseline")
    plan = json.loads(args.selection.read_text())
    previous = json.loads(args.previous_selection.read_text())
    video_lines = args.video_receipts.read_text().splitlines()
    video_receipts = [json.loads(line) for line in video_lines]
    expected = revised_selection(
        previous,
        video_receipts,
        sha256(args.previous_selection),
        sha256(args.video_receipts),
    )
    if (
        plan != expected
        or plan["realigned_metadata_sha256"] != sha256(args.alignment_metadata)
        or plan["precut_selection_sha256"] != sha256(args.exclude_selection)
    ):
        raise ValueError("raw expansion differs from the frozen publisher sources")
    revision = miner_revision(args.miner_health_url, None)
    if len(verified_downloads(plan, video_lines, args.video_dir)) != len(plan["videos"]):
        raise ValueError("raw expansion lacks verified source videos")
    revision = miner_revision(args.miner_health_url, revision)
    receipts_by_video = {row["video_name"]: row for row in video_receipts}
    selections = receipt_paths(args.ledger_dir)
    if verified_crops(plan["videos"], receipts_by_video, selections, args.source_dir) != set(
        receipts_by_video
    ):
        raise ValueError("raw expansion lacks verified crops")
    revision = miner_revision(args.miner_health_url, revision)
    indexes = existing_indexes(args.ledger_dir, selections)
    if set(indexes) != set(receipts_by_video) or args.output.resolve() in {
        path.resolve() for path in indexes.values()
    }:
        raise ValueError("raw expansion lacks an encoder index for each source")
    aligned = read_aligned(args.alignment_metadata)
    pre_cut = json.loads(args.exclude_selection.read_text())
    identity_hash = sha256(baseline / "MODEL_IDENTITY.json")
    runtime_hash = sha256(baseline / "runtime.py")
    by_video = {}
    for video in plan["videos"]:
        name = video["video_name"]
        selection_path, receipt_path = selections[name]
        selection = json.loads(selection_path.read_text())
        if (
            selection.get("alignment_metadata_sha256")
            != plan["realigned_metadata_sha256"]
            or selection.get("excluded_pre_cut_selection_sha256")
            != plan["precut_selection_sha256"]
        ):
            raise ValueError(f"raw crop selection uses another training source: {name}")
        checked_source_rows(selection, receipts_by_video[name], aligned, pre_cut)
        selection_hash = sha256(selection_path)
        clip_receipts = checked_receipts(
            selection, selection_hash, receipt_path.read_text().splitlines()
        )
        encoder_rows = list(
            verified_index(
                indexes[name], selection_hash, None, clip_receipts, args.cache_dir
            ).values()
        )
        revision = miner_revision(args.miner_health_url, revision)
        by_video[name] = canonical_rows(
            selection,
            selection_hash,
            receipts_by_video[name],
            clip_receipts,
            encoder_rows,
            args.video_dir,
            args.source_dir,
            args.resampled_dir,
            args.cache_dir,
            identity_hash,
            runtime_hash,
            args.miner_health_url,
        )
    ordered = ordered_expansion_rows(plan, by_video)
    if args.output.exists():
        if [json.loads(line) for line in args.output.read_text().splitlines()] != ordered:
            raise ValueError("frozen raw expansion index differs from current inputs")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".partial.jsonl")
        if temporary.exists():
            parser.error("unfinished raw expansion index needs inspection")
        try:
            with temporary.open("x", encoding="utf-8") as output:
                for record in ordered:
                    output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.rename(args.output)
        finally:
            temporary.unlink(missing_ok=True)
    miner_revision(args.miner_health_url, revision)
    print(f"frozen {len(ordered)} raw expansion states sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()
