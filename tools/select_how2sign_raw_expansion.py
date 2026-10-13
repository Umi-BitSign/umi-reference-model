"""Freeze a bounded, signer-diverse raw How2Sign training-video expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from crop_how2sign_raw_video import read_aligned, select_rows
from fetch_how2sign_raw_video import read_metadata
from how2sign_split_remote_zip import SplitRemoteZip, SplitZipEntry

QUOTAS = {1: 5, 2: 5, 3: 8, 4: 6, 5: 8, 8: 8, 9: 5, 11: 8}
MAX_COMPRESSED_BYTES = 256 * 1024 * 1024
MAX_TOTAL_COMPRESSED_BYTES = 12 * 1024 * 1024 * 1024
MIN_CLIPS_PER_VIDEO = 4
SEED = 17


def select_videos(
    training_rows: list[dict[str, str]],
    realigned_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    pre_cut: dict,
    entries: dict[str, SplitZipEntry],
    included_video_names: set[str],
    quotas: dict[int, int] = QUOTAS,
    max_compressed_bytes: int = MAX_COMPRESSED_BYTES,
    min_clips: int = MIN_CLIPS_PER_VIDEO,
) -> list[dict]:
    if pre_cut.get("schema") != "umi-studio-how2sign-training-aligned/1":
        raise ValueError("pre-cut exclusion is not a frozen training selection")
    if not quotas or any(type(value) is not int or value < 1 for value in quotas.values()):
        raise ValueError("raw expansion signer quotas must be positive integers")
    training_id_by_name = {}
    for row in training_rows:
        name = row["VIDEO_NAME"]
        video_id = row["VIDEO_ID"]
        if name in training_id_by_name and training_id_by_name[name] != video_id:
            raise ValueError("official training video has inconsistent source IDs")
        training_id_by_name[name] = video_id
    validation_ids = {row["VIDEO_ID"] for row in validation_rows}
    pre_cut_ids = {row["video_id"] for row in pre_cut["rows"]}
    if validation_ids & set(training_id_by_name.values()):
        raise ValueError("official training and validation source video IDs overlap")
    grouped = defaultdict(list)
    for row in realigned_rows:
        grouped[row["VIDEO_NAME"]].append(row)
    candidates = defaultdict(list)
    for video_name, aligned in grouped.items():
        match = re.fullmatch(r"(.+)-(\d+)-rgb_front", video_name)
        if match is None or Path(video_name).name != video_name:
            raise ValueError("publisher realignment has an invalid frontal video name")
        video_id, signer_text = match.groups()
        signer = int(signer_text)
        if training_id_by_name.get(video_name) != video_id or {
            row["VIDEO_ID"] for row in aligned
        } != {video_id}:
            raise ValueError("publisher realignment differs from training source identity")
        if video_id in pre_cut_ids or signer not in quotas:
            continue
        member = f"raw_videos/{video_name}.mp4"
        entry = entries.get(member)
        if (
            entry is None
            or not 0 < entry.compressed_size <= max_compressed_bytes
            or not 0 < entry.file_size <= 2 * max_compressed_bytes
        ):
            continue
        planning_receipt = {
            "video_name": video_name,
            "video_id": video_id,
            "signer": signer,
            "video_sha256": "pending-source-download",
            "video_probe": {"duration_sec": float("inf")},
        }
        try:
            selected = select_rows(aligned, planning_receipt, set())
        except ValueError:
            continue
        if len(selected) < min_clips:
            continue
        clips = [
            {
                "sentence_name": row["sentence_name"],
                "start_sec": row["start_sec"],
                "end_sec": row["end_sec"],
                "reference": row["reference"],
            }
            for row in selected
        ]
        candidates[signer].append(
            {
                "video_name": video_name,
                "video_id": video_id,
                "signer": signer,
                "archive_member": member,
                "archive_disk": entry.disk,
                "archive_header_offset": entry.header_offset,
                "archive_compressed_size": entry.compressed_size,
                "archive_file_size": entry.file_size,
                "archive_crc32": entry.crc32,
                "eligible_clip_count": len(clips),
                "eligible_duration_sec": sum(clip["end_sec"] - clip["start_sec"] for clip in clips),
                "clips": clips,
            }
        )
    by_name = {row["video_name"]: row for videos in candidates.values() for row in videos}
    if not included_video_names <= by_name.keys():
        raise ValueError("included raw video is not an eligible training source")
    selected = []
    chosen_ids = set()
    for signer in sorted(quotas, key=lambda value: (len(candidates[value]) - quotas[value], value)):
        forced = [row for row in candidates[signer] if row["video_name"] in included_video_names]
        if len(forced) > quotas[signer]:
            raise ValueError("included raw videos exceed a signer quota")
        ranked = sorted(
            candidates[signer],
            key=lambda row: (
                -row["eligible_clip_count"],
                -row["eligible_duration_sec"],
                row["archive_compressed_size"],
                hashlib.sha256(f"{SEED}:{row['video_name']}".encode()).hexdigest(),
            ),
        )
        chosen = sorted(forced, key=lambda row: row["video_name"])
        if any(row["video_id"] in chosen_ids for row in chosen):
            raise ValueError("included raw source ID collides across signers")
        chosen_ids.update(row["video_id"] for row in chosen)
        for row in ranked:
            if len(chosen) >= quotas[signer]:
                break
            if row["video_name"] in included_video_names or row["video_id"] in chosen_ids:
                continue
            chosen.append(row)
            chosen_ids.add(row["video_id"])
        if len(chosen) != quotas[signer]:
            raise ValueError(f"too few raw training videos qualify for signer {signer}")
        selected.extend(chosen)
    if sum(row["archive_compressed_size"] for row in selected) > MAX_TOTAL_COMPRESSED_BYTES:
        raise ValueError("raw video expansion exceeds its compressed-download budget")
    return sorted(selected, key=lambda row: (row["signer"], row["video_name"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--realigned-metadata", type=Path, required=True)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--precut-selection", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--include-video", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if not args.license.is_file() or len(args.include_video) != len(set(args.include_video)):
        parser.error("source license is missing or included raw videos repeat")
    manifest = json.loads(args.manifest.read_text())
    revision = miner_revision(args.miner_health_url, None)
    archive = SplitRemoteZip(manifest)
    entries = archive.load_directory()
    videos = select_videos(
        read_metadata(args.training_metadata),
        read_aligned(args.realigned_metadata),
        read_metadata(args.validation_metadata),
        json.loads(args.precut_selection.read_text()),
        entries,
        set(args.include_video),
    )
    revision = miner_revision(args.miner_health_url, revision)
    selection = {
        "schema": "umi-studio-how2sign-raw-training-expansion/1",
        "source_url": manifest["source_url"],
        "training_only": True,
        "archive_manifest_sha256": sha256(args.manifest),
        "archive_central_directory_sha256": manifest["zip64_central_directory"]["sha256"],
        "training_metadata_sha256": sha256(args.training_metadata),
        "realigned_metadata_sha256": sha256(args.realigned_metadata),
        "validation_metadata_sha256": sha256(args.validation_metadata),
        "precut_selection_sha256": sha256(args.precut_selection),
        "license_sha256": sha256(args.license),
        "seed": SEED,
        "per_signer_video_quotas": {str(key): value for key, value in sorted(QUOTAS.items())},
        "max_compressed_bytes_per_video": MAX_COMPRESSED_BYTES,
        "max_total_compressed_bytes": MAX_TOTAL_COMPRESSED_BYTES,
        "min_clips_per_video": MIN_CLIPS_PER_VIDEO,
        "included_video_names": sorted(args.include_video),
        "selected_video_count": len(videos),
        "selected_clip_count": sum(row["eligible_clip_count"] for row in videos),
        "selected_duration_sec": sum(row["eligible_duration_sec"] for row in videos),
        "selected_compressed_bytes": sum(row["archive_compressed_size"] for row in videos),
        "videos": videos,
    }
    if args.output.exists():
        if json.loads(args.output.read_text()) != selection:
            raise ValueError("frozen raw expansion differs from publisher sources")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(selection, output, sort_keys=True, ensure_ascii=False, indent=2)
            output.write("\n")
    print(
        f"selected {len(videos)} raw training videos, {selection['selected_clip_count']} clips, "
        f"{selection['selected_compressed_bytes'] / 1024**3:.2f} GiB compressed"
    )


if __name__ == "__main__":
    main()
