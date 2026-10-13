"""Crop publisher-realigned training sentences from one verified raw video."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from itertools import pairwise
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from fetch_how2sign_raw_video import probe_video


def select_rows(
    aligned_rows: list[dict[str, str]],
    video_receipt: dict,
    excluded_sentences: set[str],
    skip_unavailable: bool = False,
) -> list[dict]:
    video_name = video_receipt["video_name"]
    video_id = video_receipt["video_id"]
    duration = video_receipt["video_probe"]["duration_sec"]
    selected = []
    seen = set()
    for row in aligned_rows:
        if row["VIDEO_NAME"] != video_name:
            continue
        name = row["SENTENCE_NAME"]
        if (
            Path(name).name != name
            or name in seen
            or row["VIDEO_ID"] != video_id
            or row["VIDEO_NAME"] != video_name
        ):
            raise ValueError("realigned row has invalid source identity")
        seen.add(name)
        start = float(row["START_REALIGNED"])
        end = float(row["END_REALIGNED"])
        if start < 0 or end <= start:
            raise ValueError("realigned interval is outside the raw video")
        if end > duration + 0.01:
            if skip_unavailable:
                continue
            raise ValueError("realigned interval is outside the raw video")
        if name in excluded_sentences or not 2 <= end - start <= 12:
            continue
        if len(row["SENTENCE"].split()) < 3:
            continue
        selected.append(
            {
                "sentence_name": name,
                "video_name": video_name,
                "video_id": video_id,
                "signer": video_receipt["signer"],
                "start_sec": start,
                "end_sec": end,
                "reference": row["SENTENCE"],
                "source_video_sha256": video_receipt["video_sha256"],
            }
        )
    selected.sort(key=lambda row: (row["start_sec"], row["sentence_name"]))
    if not selected or any(
        left["end_sec"] > right["start_sec"] + 0.001 for left, right in pairwise(selected)
    ):
        raise ValueError("selected realigned intervals are empty or overlap")
    return selected


def read_aligned(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != [
            "VIDEO_ID",
            "VIDEO_NAME",
            "SENTENCE_ID",
            "SENTENCE_NAME",
            "START_REALIGNED",
            "END_REALIGNED",
            "SENTENCE",
        ]:
            raise ValueError("unexpected publisher-realigned transcript columns")
        return list(reader)


def crop(source: Path, destination: Path, start: float, end: float) -> dict:
    temporary = destination.with_suffix(".partial.mp4")
    if destination.exists() or temporary.exists():
        raise ValueError("unreceipted crop or partial output needs inspection")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(source),
                "-ss",
                f"{start:.6f}",
                "-t",
                f"{end - start:.6f}",
                "-map",
                "0:v:0",
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
        result = probe_video(temporary)
        if abs(result["duration_sec"] - (end - start)) > 0.08:
            raise ValueError("cropped clip duration differs from realigned interval")
        temporary.rename(destination)
        return result
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-name", required=True)
    parser.add_argument("--video-receipts", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--alignment-metadata", type=Path, required=True)
    parser.add_argument("--exclude-selection", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--skip-unavailable-by-source-duration", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    receipts = [json.loads(line) for line in args.video_receipts.read_text().splitlines()]
    matches = [row for row in receipts if row["video_name"] == args.video_name]
    if len(matches) != 1 or matches[0]["schema"] != "umi-studio-how2sign-raw-training-video/1":
        raise ValueError("source video lacks one verified training receipt")
    video_receipt = matches[0]
    source = args.source_dir / f"{args.video_name}.mp4"
    if not source.is_file() or sha256(source) != video_receipt["video_sha256"]:
        raise ValueError("source raw video differs from publisher receipt")
    excluded = json.loads(args.exclude_selection.read_text())
    if excluded.get("schema") != "umi-studio-how2sign-training-aligned/1":
        raise ValueError("excluded pre-cut selection has an unexpected schema")
    excluded_sentences = {row["sentence_name"] for row in excluded["rows"]}
    rows = select_rows(
        read_aligned(args.alignment_metadata),
        video_receipt,
        excluded_sentences,
        skip_unavailable=args.skip_unavailable_by_source_duration,
    )
    receipt_hash = hashlib.sha256(
        json.dumps(video_receipt, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    selection = {
        "schema": "umi-studio-how2sign-training-raw-realigned/1",
        "source_url": video_receipt["source_url"],
        "training_only": True,
        "video_name": args.video_name,
        "video_id": video_receipt["video_id"],
        "signer": video_receipt["signer"],
        "source_video_receipt_sha256": receipt_hash,
        "source_video_sha256": video_receipt["video_sha256"],
        "alignment_metadata_sha256": sha256(args.alignment_metadata),
        "excluded_pre_cut_selection_sha256": sha256(args.exclude_selection),
        "rows": rows,
    }
    if args.skip_unavailable_by_source_duration:
        selection["source_duration_policy"] = "drop_intervals_past_verified_video_end"
    revision = miner_revision(args.miner_health_url, None)
    if args.selection.exists():
        if json.loads(args.selection.read_text()) != selection:
            raise ValueError("frozen raw-video clip selection differs from sources")
    else:
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        with args.selection.open("x", encoding="utf-8") as output:
            json.dump(selection, output, sort_keys=True, ensure_ascii=False, indent=2)
            output.write("\n")
    if args.limit == 0:
        print(f"selected {len(rows)} publisher-realigned training intervals")
        return
    selection_hash = sha256(args.selection)
    completed = {}
    if args.receipt.exists():
        for line in args.receipt.read_text().splitlines():
            record = json.loads(line)
            name = record["sentence_name"]
            clip = args.output_dir / f"{name}.mp4"
            if (
                record["schema"] != "umi-studio-how2sign-raw-realigned-clip/1"
                or record["selection_sha256"] != selection_hash
                or record["source_video_sha256"] != video_receipt["video_sha256"]
                or name in completed
                or name not in {row["sentence_name"] for row in rows}
                or not clip.is_file()
                or sha256(clip) != record["clip_sha256"]
            ):
                raise ValueError("previous realigned crop differs from frozen selection")
            completed[name] = record
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    for row in rows:
        name = row["sentence_name"]
        if name in completed:
            continue
        if added >= args.limit:
            break
        revision = miner_revision(args.miner_health_url, revision)
        clip = args.output_dir / f"{name}.mp4"
        probe = crop(source, clip, row["start_sec"], row["end_sec"])
        revision = miner_revision(args.miner_health_url, revision)
        record = {
            "schema": "umi-studio-how2sign-raw-realigned-clip/1",
            "selection_sha256": selection_hash,
            "source_video_receipt_sha256": receipt_hash,
            "source_video_sha256": video_receipt["video_sha256"],
            "sentence_name": name,
            "video_name": args.video_name,
            "signer": row["signer"],
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"],
            "reference": row["reference"],
            "clip_sha256": sha256(clip),
            "clip_size_bytes": clip.stat().st_size,
            "clip_probe": probe,
            "miner_model_revision": revision,
        }
        with args.receipt.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        added += 1
        print(f"{len(completed) + added}: cropped {name}", flush=True)


if __name__ == "__main__":
    main()
