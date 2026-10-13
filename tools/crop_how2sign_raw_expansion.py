"""Crop downloaded videos from the frozen raw How2Sign training expansion."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from crop_how2sign_raw_video import read_aligned, select_rows
from fetch_how2sign_raw_expansion import verified_downloads


def planned_clips(video: dict) -> list[dict]:
    return [
        {
            "sentence_name": clip["sentence_name"],
            "start_sec": clip["start_sec"],
            "end_sec": clip["end_sec"],
            "reference": clip["reference"],
        }
        for clip in video["clips"]
    ]


def receipt_paths(ledger_dir: Path) -> dict[str, tuple[Path, Path]]:
    selections = {}
    for path in ledger_dir.glob("*selection.json"):
        selection = json.loads(path.read_text())
        if selection.get("schema") == "umi-studio-how2sign-training-raw-realigned/1":
            selections[sha256(path)] = path
    paths = {}
    for receipt in ledger_dir.glob("*clip-receipts.jsonl"):
        lines = receipt.read_text().splitlines()
        if not lines:
            continue
        rows = [json.loads(line) for line in lines]
        names = {row["video_name"] for row in rows}
        hashes = {row["selection_sha256"] for row in rows}
        if len(names) != 1 or len(hashes) != 1 or next(iter(hashes)) not in selections:
            raise ValueError(f"raw crop ledger has mixed or missing selection: {receipt}")
        name = next(iter(names))
        if name in paths:
            raise ValueError(f"multiple raw crop ledgers for {name}")
        paths[name] = (selections[next(iter(hashes))], receipt)
    return paths


def verified_crops(
    videos: list[dict],
    video_receipts: dict[str, dict],
    ledger_paths: dict[str, tuple[Path, Path]],
    output_dir: Path,
) -> set[str]:
    by_name = {video["video_name"]: video for video in videos}
    complete = set()
    for name, (selection_path, receipt_path) in ledger_paths.items():
        video = by_name.get(name)
        source = video_receipts.get(name)
        if video is None or source is None:
            raise ValueError(f"crop ledger lacks a selected, downloaded source: {name}")
        selection = json.loads(selection_path.read_text())
        if (
            selection["video_name"] != name
            or selection["source_video_sha256"] != source["video_sha256"]
            or planned_clips(video)
            != [
                {key: row[key] for key in ("sentence_name", "start_sec", "end_sec", "reference")}
                for row in selection["rows"]
            ]
        ):
            raise ValueError(f"crop selection differs from expansion plan: {name}")
        expected = {clip["sentence_name"]: clip for clip in video["clips"]}
        seen = set()
        for line in receipt_path.read_text().splitlines():
            row = json.loads(line)
            clip = expected.get(row["sentence_name"])
            clip_path = output_dir / f"{row['sentence_name']}.mp4"
            if (
                clip is None
                or row["sentence_name"] in seen
                or row["schema"] != "umi-studio-how2sign-raw-realigned-clip/1"
                or row["selection_sha256"] != sha256(selection_path)
                or row["source_video_sha256"] != source["video_sha256"]
                or row["signer"] != video["signer"]
                or any(row[key] != clip[key] for key in ("start_sec", "end_sec", "reference"))
                or not clip_path.is_file()
                or sha256(clip_path) != row["clip_sha256"]
            ):
                raise ValueError(f"raw crop receipt differs from expansion plan: {name}")
            seen.add(row["sentence_name"])
        if seen == set(expected):
            complete.add(name)
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--video-receipts", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--alignment-metadata", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 8:
        parser.error("a raw crop batch must contain 1-8 videos")
    selection = json.loads(args.selection.read_text())
    if (
        selection.get("schema")
        not in {
            "umi-studio-how2sign-raw-training-expansion/1",
            "umi-studio-how2sign-raw-training-expansion/2",
        }
        or selection.get("training_only") is not True
        or selection["realigned_metadata_sha256"] != sha256(args.alignment_metadata)
        or selection["precut_selection_sha256"] != sha256(args.exclude_selection)
    ):
        raise ValueError("raw crop inputs differ from frozen expansion")
    if selection["schema"] == "umi-studio-how2sign-raw-training-expansion/2" and selection.get(
        "video_receipts_sha256"
    ) != sha256(args.video_receipts):
        raise ValueError("source-duration selection uses another video receipt ledger")
    video_lines = args.video_receipts.read_text().splitlines()
    downloaded = verified_downloads(selection, video_lines, args.source_dir)
    video_receipts = {row["video_name"]: row for row in map(json.loads, video_lines)}
    aligned = read_aligned(args.alignment_metadata)
    excluded = json.loads(args.exclude_selection.read_text())
    if excluded.get("schema") != "umi-studio-how2sign-training-aligned/1":
        raise ValueError("pre-cut selection has an unexpected schema")
    excluded_names = {row["sentence_name"] for row in excluded["rows"]}
    ledger_paths = receipt_paths(args.ledger_dir)
    complete = verified_crops(selection["videos"], video_receipts, ledger_paths, args.output_dir)
    pending = [
        video
        for video in selection["videos"]
        if video["video_name"] in downloaded and video["video_name"] not in complete
    ][: args.limit]
    revision = miner_revision(args.miner_health_url, None)
    selection_hash = sha256(args.selection)
    for video in pending:
        name = video["video_name"]
        checked_duration = selection["schema"].endswith("/2")
        rows = select_rows(
            aligned, video_receipts[name], excluded_names, skip_unavailable=checked_duration
        )
        actual = [
            {key: row[key] for key in ("sentence_name", "start_sec", "end_sec", "reference")}
            for row in rows
        ]
        if actual != planned_clips(video):
            raise ValueError(f"realigned rows differ from expansion plan: {name}")
        selection_path, receipt_path = ledger_paths.get(
            name,
            (
                args.ledger_dir / f"{name}-selection.json",
                args.ledger_dir / f"{name}-clip-receipts.jsonl",
            ),
        )
        command = [
            sys.executable,
            str(Path(__file__).with_name("crop_how2sign_raw_video.py")),
            f"--video-name={name}",
            "--video-receipts",
            str(args.video_receipts),
            "--source-dir",
            str(args.source_dir),
            "--alignment-metadata",
            str(args.alignment_metadata),
            "--exclude-selection",
            str(args.exclude_selection),
            "--selection",
            str(selection_path),
            "--output-dir",
            str(args.output_dir),
            "--receipt",
            str(receipt_path),
            "--miner-health-url",
            args.miner_health_url,
            "--limit",
            str(len(video["clips"])),
        ]
        if checked_duration:
            command.append("--skip-unavailable-by-source-duration")
        revision = miner_revision(args.miner_health_url, revision)
        subprocess.run(command, check=True)
        revision = miner_revision(args.miner_health_url, revision)
        ledger_paths[name] = selection_path, receipt_path
        complete = verified_crops(
            selection["videos"], video_receipts, ledger_paths, args.output_dir
        )
        if name not in complete or sha256(args.selection) != selection_hash:
            raise RuntimeError("raw crop batch changed its frozen inputs or failed verification")
    print(f"verified {len(complete)}/{selection['selected_video_count']} cropped raw videos")


if __name__ == "__main__":
    main()
