"""Select aligned How2Sign validation clips from the publisher's frontal ZIP.

This is evaluation-only data. Transfer selected ZIP members with byte ranges,
preserve the publisher transcripts, and check the active miner between clips.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from fetch_how2sign_pilot import MAX_CLIP_BYTES, SEED, archive_fingerprint, check_video
from how2sign_remote_zip import How2SignRemoteZip

VAL_METADATA_FILE_ID = "1aBQUClTlZB504JtDISJ0DJlbuYUZCGu3"
VAL_ARCHIVE_FILE_ID = "1DhLH8tIBn9HsTzUJUfsEOGcP4l9EvOiO"
VAL_ARCHIVE_NAME = "val_rgb_front_clips.zip"
VAL_ARCHIVE_SIZE = 1_778_074_855
ORIGINAL_COLUMNS = [
    "VIDEO_ID",
    "VIDEO_NAME",
    "SENTENCE_ID",
    "SENTENCE_NAME",
    "START",
    "END",
    "SENTENCE",
]
REALIGNED_COLUMNS = [
    "VIDEO_ID",
    "VIDEO_NAME",
    "SENTENCE_ID",
    "SENTENCE_NAME",
    "START_REALIGNED",
    "END_REALIGNED",
    "SENTENCE",
]


def metadata_rows(data: bytes, columns: list[str]) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")), delimiter="\t")
    if reader.fieldnames != columns:
        raise ValueError("How2Sign validation transcript columns differ from the publisher format")
    rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError("How2Sign validation transcript is empty or malformed")
    return rows


def load_original_metadata(path: Path) -> list[dict[str, str]]:
    if path.is_file():
        return metadata_rows(path.read_bytes(), ORIGINAL_COLUMNS)
    url = f"https://drive.google.com/uc?export=download&id={VAL_METADATA_FILE_ID}"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = response.read(1_000_001)
    if len(data) > 1_000_000:
        raise ValueError("original validation transcript exceeds its expected size")
    rows = metadata_rows(data, ORIGINAL_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    return rows


def select_rows(
    original: list[dict[str, str]],
    realigned: list[dict[str, str]],
    info_by_name: dict[str, zipfile.ZipInfo],
    train_video_ids: set[str],
    target: int,
    min_bidirectional_overlap: float,
) -> list[dict]:
    if type(target) is not int or target < 1:
        raise ValueError("validation target must be positive")
    if not math.isfinite(min_bidirectional_overlap) or not 0 < min_bidirectional_overlap <= 1:
        raise ValueError("validation overlap threshold must be in (0, 1]")
    by_name = {row["SENTENCE_NAME"]: row for row in realigned}
    if (
        len(by_name) != len(realigned)
        or len({row["SENTENCE_NAME"] for row in original}) != len(original)
        or set(by_name) != {row["SENTENCE_NAME"] for row in original}
    ):
        raise ValueError("original and realigned validation sentence identities differ")
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in original:
        aligned = by_name[row["SENTENCE_NAME"]]
        if any(
            row[key] != aligned[key]
            for key in ("VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME", "SENTENCE")
        ):
            raise ValueError("original and realigned validation row identities differ")
        if row["VIDEO_ID"] in train_video_ids:
            raise ValueError("validation source video overlaps official training metadata")
        name = row["SENTENCE_NAME"]
        if Path(name).name != name:
            raise ValueError("validation sentence name contains a path separator")
        match = re.search(r"-(\d+)-rgb_front$", row["VIDEO_NAME"])
        if match is None:
            raise ValueError("validation camera or signer label is unexpected")
        start, end = float(row["START"]), float(row["END"])
        aligned_start = float(aligned["START_REALIGNED"])
        aligned_end = float(aligned["END_REALIGNED"])
        if not all(math.isfinite(value) for value in (start, end, aligned_start, aligned_end)):
            raise ValueError("validation timestamp is not finite")
        if min(start, aligned_start) < 0 or end <= start or aligned_end <= aligned_start:
            raise ValueError("validation interval is invalid")
        member = f"raw_videos/{name}.mp4"
        info = info_by_name.get(member)
        duration = end - start
        if (
            info is None
            or not 2 <= duration <= 12
            or len(row["SENTENCE"].split()) < 3
            or not 0 < info.file_size <= MAX_CLIP_BYTES
        ):
            continue
        overlap = max(0.0, min(end, aligned_end) - max(start, aligned_start))
        original_fraction = overlap / duration
        realigned_fraction = overlap / (aligned_end - aligned_start)
        if (
            original_fraction < min_bidirectional_overlap
            or realigned_fraction < min_bidirectional_overlap
        ):
            continue
        groups[int(match.group(1))].append(
            {
                "sentence_name": name,
                "video_name": row["VIDEO_NAME"],
                "video_id": row["VIDEO_ID"],
                "signer": int(match.group(1)),
                "start_sec": start,
                "end_sec": end,
                "realigned_start_sec": aligned_start,
                "realigned_end_sec": aligned_end,
                "original_interval_overlap_fraction": original_fraction,
                "realigned_interval_overlap_fraction": realigned_fraction,
                "reference": row["SENTENCE"],
                "archive_member": member,
                "archive_crc32": info.CRC,
                "archive_file_size": info.file_size,
                "archive_compress_size": info.compress_size,
            }
        )
    if not groups:
        raise ValueError("no validation clips pass the alignment gate")
    for rows in groups.values():
        rows.sort(
            key=lambda item: (
                hashlib.sha256(f"{SEED}:{item['sentence_name']}".encode()).hexdigest(),
                item["sentence_name"],
            )
        )
    selected = []
    seen_videos = set()
    positions = {signer: 0 for signer in groups}
    while len(selected) < target:
        progress = False
        for signer in sorted(groups):
            rows = groups[signer]
            while positions[signer] < len(rows):
                item = rows[positions[signer]]
                positions[signer] += 1
                if item["video_id"] not in seen_videos:
                    selected.append(item)
                    seen_videos.add(item["video_id"])
                    progress = True
                    break
            if len(selected) == target:
                break
        if not progress:
            raise ValueError(f"only {len(selected)} distinct aligned validation videos qualify")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--alignment-metadata", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--target", type=int, default=64)
    parser.add_argument("--min-bidirectional-overlap", type=float, default=0.8)
    parser.add_argument(
        "--limit", type=int, default=1, help="maximum new downloads, or 0 to select"
    )
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be nonnegative")
    if not args.alignment_metadata.is_file() or not args.training_metadata.is_file():
        parser.error("source validation alignment and training metadata must exist")
    if not args.license.is_file():
        parser.error("source license must exist")
    revision = miner_revision(args.miner_health_url, None)
    original = load_original_metadata(args.metadata)
    realigned = metadata_rows(args.alignment_metadata.read_bytes(), REALIGNED_COLUMNS)
    train = metadata_rows(args.training_metadata.read_bytes(), ORIGINAL_COLUMNS)
    train_video_ids = {row["VIDEO_ID"] for row in train}
    remote = How2SignRemoteZip(
        file_id=VAL_ARCHIVE_FILE_ID,
        archive_name=VAL_ARCHIVE_NAME,
        archive_size=VAL_ARCHIVE_SIZE,
    )
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        info_by_name = {info.filename: info for info in infos}
        if len(info_by_name) != len(infos):
            raise ValueError("validation archive repeats a member name")
        rows = select_rows(
            original,
            realigned,
            info_by_name,
            train_video_ids,
            args.target,
            args.min_bidirectional_overlap,
        )
        selection = {
            "schema": "umi-studio-how2sign-validation-selection/1",
            "source_url": "https://how2sign.github.io/",
            "split": "validation",
            "training_excluded": True,
            "archive_google_drive_file_id": VAL_ARCHIVE_FILE_ID,
            "archive_size_bytes": VAL_ARCHIVE_SIZE,
            "archive_fingerprint_sha256": archive_fingerprint(infos),
            "metadata_google_drive_file_id": VAL_METADATA_FILE_ID,
            "metadata_sha256": sha256(args.metadata),
            "alignment_metadata_sha256": sha256(args.alignment_metadata),
            "training_metadata_sha256": sha256(args.training_metadata),
            "license_sha256": sha256(args.license),
            "seed": SEED,
            "min_bidirectional_overlap": args.min_bidirectional_overlap,
            "one_clip_per_source_video": True,
            "original_validation_rows": len(original),
            "archive_members": len(infos),
            "rows": rows,
        }
        if args.selection.exists():
            if json.loads(args.selection.read_text()) != selection:
                raise ValueError("frozen validation selection differs from publisher sources")
        else:
            args.selection.parent.mkdir(parents=True, exist_ok=True)
            with args.selection.open("x", encoding="utf-8") as output:
                json.dump(selection, output, sort_keys=True, ensure_ascii=False, indent=2)
                output.write("\n")
        print(f"selected {len(rows)} validation clips", flush=True)
        print(f"archive index used {remote.bytes_downloaded} bytes", flush=True)
        if args.limit == 0:
            return
        selection_hash = sha256(args.selection)
        completed = {}
        if args.receipt.exists():
            for line in args.receipt.read_text().splitlines():
                record = json.loads(line)
                name = record["sentence_name"]
                if (
                    record["schema"] != "umi-studio-how2sign-validation-clip/1"
                    or record["selection_sha256"] != selection_hash
                    or name in completed
                    or name not in {row["sentence_name"] for row in rows}
                ):
                    raise ValueError("validation receipt is invalid or duplicated")
                clip = args.output_dir / f"{name}.mp4"
                if not clip.is_file() or sha256(clip) != record["clip_sha256"]:
                    raise ValueError(f"previous validation clip differs: {clip}")
                completed[name] = record
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        new_count = 0
        for row in rows:
            name = row["sentence_name"]
            if name in completed:
                continue
            if new_count >= args.limit:
                break
            revision = miner_revision(args.miner_health_url, revision)
            clip = args.output_dir / f"{name}.mp4"
            temporary = clip.with_suffix(".partial.mp4")
            if clip.exists() or temporary.exists():
                raise ValueError(f"unreceipted validation clip needs inspection: {clip}")
            info = info_by_name[row["archive_member"]]
            digest = hashlib.sha256()
            size = 0
            try:
                with archive.open(info) as source, temporary.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                if size != row["archive_file_size"]:
                    raise ValueError("validation clip size differs from archive index")
                video_stream = check_video(temporary)
                temporary.rename(clip)
            finally:
                temporary.unlink(missing_ok=True)
            revision = miner_revision(args.miner_health_url, revision)
            record = {
                "schema": "umi-studio-how2sign-validation-clip/1",
                "split": "validation",
                "selection_sha256": selection_hash,
                "sentence_name": name,
                "archive_member": row["archive_member"],
                "archive_crc32": row["archive_crc32"],
                "clip_sha256": digest.hexdigest(),
                "clip_size_bytes": size,
                "video_stream": video_stream,
                "miner_model_revision": revision,
            }
            with args.receipt.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
            new_count += 1
            print(f"{len(completed) + new_count}: fetched {name}", flush=True)


if __name__ == "__main__":
    main()
