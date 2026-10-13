"""Freeze a signer-balanced How2Sign confirmation supplement from full videos.

The publisher's pre-cut validation clips leave signer 2 out of the sealed
confirmation partition. This selection uses other official validation source
videos and manually realigned timestamps. It does not fetch or score clips.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from fetch_how2sign_pilot import SEED, archive_fingerprint
from fetch_how2sign_validation import ORIGINAL_COLUMNS, REALIGNED_COLUMNS
from how2sign_remote_zip import How2SignRemoteZip

ARCHIVE_FILE_ID = "1fCkyuKSsc7gauljuL9sx_jBomf3N6i0g"
ARCHIVE_NAME = "val_raw_videos.zip"
ARCHIVE_SIZE = 17_528_664_599
MAX_VIDEO_BYTES = 300 * 1024 * 1024
MIN_SENTENCE_GAP_SEC = 15.0
QUOTAS = {1: 4, 2: 4, 3: 4, 5: 4, 8: 1}


def read_metadata(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != columns:
            raise ValueError(f"unexpected How2Sign metadata columns: {path}")
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError(f"empty or malformed How2Sign metadata: {path}")
    return rows


def select_videos(
    validation_rows: list[dict[str, str]],
    info_by_name: dict[str, zipfile.ZipInfo],
    excluded_video_ids: set[str],
    training_video_ids: set[str],
    training_references: set[str],
    quotas: dict[int, int],
) -> list[dict]:
    if not quotas or any(type(count) is not int or count < 1 for count in quotas.values()):
        raise ValueError("confirmation signer quotas must be positive integers")
    grouped: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    identities = {}
    for row in validation_rows:
        video_id = row["VIDEO_ID"]
        if video_id in training_video_ids:
            raise ValueError("validation source video overlaps official training metadata")
        if video_id in excluded_video_ids:
            continue
        video_name = row["VIDEO_NAME"]
        match = re.search(r"-(\d+)-rgb_front$", video_name)
        if match is None or Path(video_name).name != video_name:
            raise ValueError("unexpected validation source-video name")
        signer = int(match.group(1))
        identity = (signer, video_name)
        if (signer, video_id) in identities and identities[(signer, video_id)] != identity:
            raise ValueError("validation source-video ID has inconsistent signer or name")
        identities[(signer, video_id)] = identity
        if signer not in quotas:
            continue
        member = f"raw_videos/{video_name}.mp4"
        info = info_by_name.get(member)
        if info is None or not 0 < info.file_size <= MAX_VIDEO_BYTES:
            continue
        start = float(row["START_REALIGNED"])
        end = float(row["END_REALIGNED"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("invalid manually realigned validation interval")
        reference = row["SENTENCE"]
        if (
            not 2 <= end - start <= 12
            or len(reference.split()) < 3
            or reference.casefold().strip() in training_references
        ):
            continue
        name = row["SENTENCE_NAME"]
        if Path(name).name != name:
            raise ValueError("validation sentence name contains a path separator")
        grouped[signer][video_id].append(
            {
                "sentence_name": name,
                "sentence_id": row["SENTENCE_ID"],
                "start_sec": start,
                "end_sec": end,
                "reference": reference,
            }
        )
    candidates_by_signer = {}
    for signer, quota in sorted(quotas.items()):
        candidates = []
        for video_id, clips in grouped[signer].items():
            ordered = sorted(
                clips,
                key=lambda clip: (
                    hashlib.sha256(f"{SEED}:{clip['sentence_name']}".encode()).hexdigest(),
                    clip["sentence_name"],
                ),
            )
            pair = None
            for first in ordered:
                for second in ordered:
                    if first is second:
                        continue
                    gap = max(
                        second["start_sec"] - first["end_sec"],
                        first["start_sec"] - second["end_sec"],
                    )
                    if gap >= MIN_SENTENCE_GAP_SEC:
                        pair = sorted((first, second), key=lambda item: item["start_sec"])
                        break
                if pair is not None:
                    break
            if pair is None:
                continue
            video_name = identities[(signer, video_id)][1]
            info = info_by_name[f"raw_videos/{video_name}.mp4"]
            candidates.append(
                {
                    "video_id": video_id,
                    "video_name": video_name,
                    "signer": signer,
                    "archive_member": info.filename,
                    "archive_crc32": info.CRC,
                    "archive_file_size": info.file_size,
                    "archive_compress_size": info.compress_size,
                    "clips": pair,
                }
            )
        candidates.sort(
            key=lambda item: (
                hashlib.sha256(f"{SEED}:{item['video_id']}".encode()).hexdigest(),
                item["video_id"],
            )
        )
        if len(candidates) < quota:
            raise ValueError(
                f"only {len(candidates)} full validation videos qualify for signer {signer}"
            )
        candidates_by_signer[signer] = candidates
    selected = []
    chosen_video_ids = set()
    scarce_first = sorted(
        quotas, key=lambda key: (len(candidates_by_signer[key]) - quotas[key], key)
    )
    for signer in scarce_first:
        available = [
            row for row in candidates_by_signer[signer] if row["video_id"] not in chosen_video_ids
        ]
        if len(available) < quotas[signer]:
            raise ValueError(
                f"shared video IDs leave too few confirmation videos for signer {signer}"
            )
        chosen = available[: quotas[signer]]
        selected.extend(chosen)
        chosen_video_ids.update(row["video_id"] for row in chosen)
    return sorted(selected, key=lambda row: (row["signer"], row["video_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-metadata", type=Path, required=True)
    parser.add_argument("--training-metadata", type=Path, required=True)
    parser.add_argument("--precut-selection", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    args = parser.parse_args()
    if not args.license.is_file():
        parser.error("source license must exist")
    validation = read_metadata(args.validation_metadata, REALIGNED_COLUMNS)
    training = read_metadata(args.training_metadata, ORIGINAL_COLUMNS)
    precut = json.loads(args.precut_selection.read_text())
    if (
        precut.get("schema") != "umi-studio-how2sign-validation-selection/1"
        or precut.get("split") != "validation"
        or precut.get("alignment_metadata_sha256") != sha256(args.validation_metadata)
        or precut.get("training_metadata_sha256") != sha256(args.training_metadata)
    ):
        raise ValueError("pre-cut validation selection differs from official metadata")
    excluded = {row["video_id"] for row in precut["rows"]}
    if len(excluded) != len(precut["rows"]):
        raise ValueError("pre-cut validation source videos are duplicated")
    revision = miner_revision(args.miner_health_url, None)
    remote = How2SignRemoteZip(
        file_id=ARCHIVE_FILE_ID, archive_name=ARCHIVE_NAME, archive_size=ARCHIVE_SIZE
    )
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        info_by_name = {info.filename: info for info in infos}
        if len(info_by_name) != len(infos):
            raise ValueError("full validation archive repeats a member name")
        videos = select_videos(
            validation,
            info_by_name,
            excluded,
            {row["VIDEO_ID"] for row in training},
            {row["SENTENCE"].casefold().strip() for row in training},
            QUOTAS,
        )
        revision = miner_revision(args.miner_health_url, revision)
        selection = {
            "schema": "umi-studio-how2sign-full-video-confirmation-selection/1",
            "source_url": "https://how2sign.github.io/",
            "split": "validation",
            "confirmation_only": True,
            "training_excluded": True,
            "precut_selection_sha256": sha256(args.precut_selection),
            "validation_metadata_sha256": sha256(args.validation_metadata),
            "training_metadata_sha256": sha256(args.training_metadata),
            "license_sha256": sha256(args.license),
            "archive_google_drive_file_id": ARCHIVE_FILE_ID,
            "archive_size_bytes": ARCHIVE_SIZE,
            "archive_fingerprint_sha256": archive_fingerprint(infos),
            "seed": SEED,
            "min_sentence_gap_sec": MIN_SENTENCE_GAP_SEC,
            "max_source_video_bytes": MAX_VIDEO_BYTES,
            "per_signer_video_quotas": {str(k): v for k, v in sorted(QUOTAS.items())},
            "selection_rule": (
                "Use two 2-12 second manually realigned validation sentences per source video, "
                "separated by at least 15 seconds. Exclude official training video IDs, all "
                "pre-cut validation selection video IDs and exact training references."
            ),
            "videos": videos,
        }
        if args.output.exists():
            if json.loads(args.output.read_text()) != selection:
                raise ValueError("frozen full-video confirmation selection differs")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as output:
                json.dump(selection, output, sort_keys=True, ensure_ascii=False, indent=2)
                output.write("\n")
        print(
            f"selected {len(videos)} full validation videos and "
            f"{sum(len(video['clips']) for video in videos)} sealed clips; "
            f"archive index used {remote.bytes_downloaded} bytes"
        )


if __name__ == "__main__":
    main()
