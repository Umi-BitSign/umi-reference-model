"""Fetch a signer-balanced How2Sign training pilot from the official clip ZIP.

Only selected ZIP members are transferred. The source is the publisher's frontal
training archive and its matching original-timestamp transcript, never a held-out
split. The live miner is checked before and after each transfer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256
from how2sign_remote_zip import ARCHIVE_FILE_ID, ARCHIVE_SIZE, How2SignRemoteZip

SEED = 17
SIGNERS = (1, 2, 3, 4, 5, 8, 9, 11)
MAX_CLIP_BYTES = 8 * 1024 * 1024


def archive_fingerprint(infos: list[zipfile.ZipInfo]) -> str:
    digest = hashlib.sha256()
    for info in sorted(infos, key=lambda item: item.filename):
        record = (
            f"{info.filename}\t{info.CRC}\t{info.file_size}\t"
            f"{info.compress_size}\t{info.header_offset}\n"
        )
        digest.update(record.encode())
    return digest.hexdigest()


def select_rows(
    metadata: list[dict[str, str]],
    info_by_name: dict[str, zipfile.ZipInfo],
    per_signer: int | None = None,
    *,
    quotas: dict[int, int] | None = None,
    excluded_videos: set[str] | None = None,
    alignment_by_name: dict[str, dict[str, str]] | None = None,
    min_bidirectional_overlap: float | None = None,
    max_clips_per_video: int = 1,
) -> list[dict]:
    if (per_signer is None) == (quotas is None):
        raise ValueError("supply either one per-signer count or signer quotas")
    if quotas is None:
        quotas = {signer: per_signer for signer in SIGNERS}
    if set(quotas) != set(SIGNERS) or any(
        type(count) is not int or count < 0 for count in quotas.values()
    ):
        raise ValueError("signer quotas must cover the known training signers")
    if not any(quotas.values()):
        raise ValueError("at least one signer quota must be positive")
    if type(max_clips_per_video) is not int or not 1 <= max_clips_per_video <= 4:
        raise ValueError("source-video clip cap must be between one and four")
    if (alignment_by_name is None) != (min_bidirectional_overlap is None):
        raise ValueError("alignment metadata and overlap threshold must be supplied together")
    if min_bidirectional_overlap is not None and (
        isinstance(min_bidirectional_overlap, bool)
        or not isinstance(min_bidirectional_overlap, int | float)
        or not math.isfinite(min_bidirectional_overlap)
        or not 0 < min_bidirectional_overlap <= 1
    ):
        raise ValueError("alignment overlap threshold must be in (0, 1]")
    excluded_videos = excluded_videos or set()
    groups: dict[int, list[dict]] = defaultdict(list)
    seen_sentence_names = set()
    for row in metadata:
        sentence_name = row["SENTENCE_NAME"]
        if sentence_name in seen_sentence_names:
            raise ValueError(f"duplicate transcript sentence name: {sentence_name}")
        seen_sentence_names.add(sentence_name)
        if Path(sentence_name).name != sentence_name:
            raise ValueError("transcript sentence name contains a path separator")
        match = re.search(r"-(\d+)-rgb_front$", row["VIDEO_NAME"])
        if match is None:
            raise ValueError(f"unexpected camera or signer label: {row['VIDEO_NAME']}")
        signer = int(match.group(1))
        duration = float(row["END"]) - float(row["START"])
        member = f"raw_videos/{sentence_name}.mp4"
        info = info_by_name.get(member)
        if (
            info is None
            or not 2 <= duration <= 12
            or len(row["SENTENCE"].split()) < 3
            or not 0 < info.file_size <= MAX_CLIP_BYTES
        ):
            continue
        candidate = {
            "sentence_name": sentence_name,
            "video_name": row["VIDEO_NAME"],
            "video_id": row["VIDEO_ID"],
            "signer": signer,
            "start_sec": float(row["START"]),
            "end_sec": float(row["END"]),
            "reference": row["SENTENCE"],
            "archive_member": member,
            "archive_crc32": info.CRC,
            "archive_file_size": info.file_size,
            "archive_compress_size": info.compress_size,
        }
        if alignment_by_name is not None:
            aligned = alignment_by_name.get(sentence_name)
            if (
                aligned is None
                or aligned["VIDEO_ID"] != row["VIDEO_ID"]
                or aligned["VIDEO_NAME"] != row["VIDEO_NAME"]
                or aligned["SENTENCE_ID"] != row["SENTENCE_ID"]
                or aligned["SENTENCE"] != row["SENTENCE"]
            ):
                raise ValueError("realigned transcript identity differs from original")
            aligned_start = float(aligned["START_REALIGNED"])
            aligned_end = float(aligned["END_REALIGNED"])
            if (
                not math.isfinite(aligned_start)
                or not math.isfinite(aligned_end)
                or aligned_start < 0
                or aligned_end <= aligned_start
            ):
                raise ValueError("realigned transcript has an invalid interval")
            overlap = max(
                0.0,
                min(candidate["end_sec"], aligned_end) - max(candidate["start_sec"], aligned_start),
            )
            original_fraction = overlap / duration
            aligned_fraction = overlap / (aligned_end - aligned_start)
            if (
                original_fraction < min_bidirectional_overlap
                or aligned_fraction < min_bidirectional_overlap
            ):
                continue
            candidate.update(
                realigned_start_sec=aligned_start,
                realigned_end_sec=aligned_end,
                original_interval_overlap_fraction=original_fraction,
                realigned_interval_overlap_fraction=aligned_fraction,
            )
        groups[signer].append(candidate)
    if set(groups) != set(SIGNERS):
        raise ValueError(f"unexpected training signer set: {sorted(groups)}")
    selected = []
    for signer in SIGNERS:
        target = quotas[signer]
        if target == 0:
            continue
        chosen_videos: Counter[str] = Counter()
        chosen_sentences = set()
        count = 0
        ordered = sorted(
            groups[signer],
            key=lambda row: (
                hashlib.sha256(f"{SEED}:{row['sentence_name']}".encode()).hexdigest(),
                row["sentence_name"],
            ),
        )
        for pass_number in range(1, max_clips_per_video + 1):
            for row in ordered:
                video = row["video_name"]
                if (
                    video in excluded_videos
                    or row["sentence_name"] in chosen_sentences
                    or chosen_videos[video] >= pass_number
                ):
                    continue
                selected.append(row)
                chosen_videos[video] += 1
                chosen_sentences.add(row["sentence_name"])
                count += 1
                if count == target:
                    break
            if count == target:
                break
        if count != target:
            raise ValueError(f"only {count} qualifying clips for signer {signer}")
    return selected


def check_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1 or streams[0]["width"] <= 0 or streams[0]["height"] <= 0:
        raise ValueError(f"missing valid video stream: {path}")
    return streams[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--per-signer", type=int, default=16)
    parser.add_argument("--quota-plan", type=Path)
    parser.add_argument("--exclude-selection", type=Path)
    parser.add_argument("--aligned-plan", type=Path)
    parser.add_argument("--alignment-metadata", type=Path)
    parser.add_argument(
        "--limit", type=int, default=1, help="maximum new downloads, or 0 to select"
    )
    args = parser.parse_args()
    if args.per_signer < 1 or args.limit < 0:
        parser.error("per-signer must be positive and limit must be nonnegative")
    if (args.quota_plan is None) != (args.exclude_selection is None):
        parser.error("a quota plan and prior selection must be supplied together")
    if (args.aligned_plan is None) != (args.alignment_metadata is None):
        parser.error("an aligned plan and realigned transcript must be supplied together")
    if args.aligned_plan is not None and args.quota_plan is not None:
        parser.error("aligned and prior-exclusion plans are separate selection modes")
    if not args.metadata.is_file() or not args.license.is_file():
        parser.error("original training transcript and license must exist")
    with args.metadata.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != [
            "VIDEO_ID",
            "VIDEO_NAME",
            "SENTENCE_ID",
            "SENTENCE_NAME",
            "START",
            "END",
            "SENTENCE",
        ]:
            raise ValueError("unexpected original-timestamp transcript columns")
        metadata = list(reader)
    revision = miner_revision(args.miner_health_url, None)
    remote = How2SignRemoteZip()
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        info_by_name = {info.filename: info for info in infos}
        if len(info_by_name) != len(infos):
            raise ValueError("archive has duplicate member names")
        fingerprint = archive_fingerprint(infos)
        extra = {}
        if args.aligned_plan is not None:
            plan = json.loads(args.aligned_plan.read_text())
            raw_quotas = plan.get("per_signer_quotas")
            if (
                plan.get("schema") != "umi-studio-how2sign-aligned-plan/1"
                or plan.get("source_metadata_sha256") != sha256(args.metadata)
                or plan.get("alignment_metadata_sha256") != sha256(args.alignment_metadata)
                or plan.get("license_sha256") != sha256(args.license)
                or plan.get("archive_fingerprint_sha256") != fingerprint
                or plan.get("training_only") is not True
                or not isinstance(raw_quotas, dict)
                or set(raw_quotas) != {str(signer) for signer in SIGNERS}
            ):
                raise ValueError("aligned plan differs from the official training sources")
            quotas = {int(signer): count for signer, count in raw_quotas.items()}
            if plan.get("training_examples") != sum(quotas.values()):
                raise ValueError("aligned training quota count differs")
            with args.alignment_metadata.open(newline="", encoding="utf-8") as stream:
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
                    raise ValueError("unexpected realigned transcript columns")
                aligned_rows = list(reader)
            alignment_by_name = {row["SENTENCE_NAME"]: row for row in aligned_rows}
            if len(alignment_by_name) != len(aligned_rows) or set(alignment_by_name) != {
                row["SENTENCE_NAME"] for row in metadata
            }:
                raise ValueError("realigned transcript does not match original row count")
            selected = select_rows(
                metadata,
                info_by_name,
                quotas=quotas,
                alignment_by_name=alignment_by_name,
                min_bidirectional_overlap=plan["min_bidirectional_overlap"],
                max_clips_per_video=plan["max_clips_per_source_video"],
            )
            extra = {
                "aligned_plan_sha256": sha256(args.aligned_plan),
                "alignment_metadata_sha256": sha256(args.alignment_metadata),
                "min_bidirectional_overlap": plan["min_bidirectional_overlap"],
                "max_clips_per_source_video": plan["max_clips_per_source_video"],
                "per_signer_quotas": raw_quotas,
            }
            selection_schema = "umi-studio-how2sign-training-aligned/1"
        elif args.quota_plan is None:
            selected = select_rows(metadata, info_by_name, args.per_signer)
            selection_schema = "umi-studio-how2sign-training-pilot/1"
        else:
            plan = json.loads(args.quota_plan.read_text())
            prior = json.loads(args.exclude_selection.read_text())
            raw_quotas = plan.get("additional_per_signer")
            if (
                plan.get("schema") != "umi-studio-how2sign-scale-plan/1"
                or plan.get("prior_selection_sha256") != sha256(args.exclude_selection)
                or prior.get("schema") != "umi-studio-how2sign-training-pilot/1"
                or prior.get("metadata_sha256") != sha256(args.metadata)
                or prior.get("license_sha256") != sha256(args.license)
                or prior.get("archive_fingerprint_sha256") != fingerprint
                or not isinstance(raw_quotas, dict)
                or set(raw_quotas) != {str(signer) for signer in SIGNERS}
            ):
                raise ValueError("scale plan or prior selection differs from source")
            quotas = {int(signer): count for signer, count in raw_quotas.items()}
            prior_rows = prior["rows"]
            prior_videos = {row["video_name"] for row in prior_rows}
            if (
                len(prior_videos) != len(prior_rows)
                or plan.get("training_only") is not True
                or plan.get("additional_examples") != sum(quotas.values())
                or plan.get("total_with_prior_pilot") != len(prior_rows) + sum(quotas.values())
            ):
                raise ValueError("scale plan counts or prior source videos differ")
            selected = select_rows(
                metadata, info_by_name, quotas=quotas, excluded_videos=prior_videos
            )
            extra = {
                "quota_plan_sha256": sha256(args.quota_plan),
                "prior_selection_sha256": sha256(args.exclude_selection),
                "additional_per_signer": raw_quotas,
            }
            selection_schema = "umi-studio-how2sign-training-scale/1"
        selection = {
            "schema": selection_schema,
            "source_url": "https://how2sign.github.io/",
            "archive_google_drive_file_id": ARCHIVE_FILE_ID,
            "archive_size_bytes": ARCHIVE_SIZE,
            "archive_fingerprint_sha256": fingerprint,
            "metadata_sha256": sha256(args.metadata),
            "license_sha256": sha256(args.license),
            "seed": SEED,
            "per_signer": args.per_signer if not extra else None,
            "original_training_rows": len(metadata),
            "archive_members": len(infos),
            "rows": selected,
            **extra,
        }
        if args.selection.exists():
            if json.loads(args.selection.read_text()) != selection:
                raise ValueError("frozen pilot selection differs from current sources")
        else:
            args.selection.parent.mkdir(parents=True, exist_ok=True)
            with args.selection.open("x", encoding="utf-8") as stream:
                json.dump(selection, stream, sort_keys=True, ensure_ascii=False, indent=2)
                stream.write("\n")
        print(
            f"selected {len(selected)} clips; archive index used {remote.bytes_downloaded} bytes",
            flush=True,
        )
        if args.limit == 0:
            return
        selection_hash = sha256(args.selection)
        completed = {}
        if args.receipt.exists():
            for line in args.receipt.read_text().splitlines():
                record = json.loads(line)
                if record["selection_sha256"] != selection_hash:
                    raise ValueError("download receipt belongs to another pilot selection")
                if record["sentence_name"] in completed:
                    raise ValueError("duplicate download receipt")
                clip = args.output_dir / f"{record['sentence_name']}.mp4"
                if not clip.is_file() or sha256(clip) != record["clip_sha256"]:
                    raise ValueError(f"previously downloaded clip differs: {clip}")
                completed[record["sentence_name"]] = record
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        new_count = 0
        for row in selected:
            if row["sentence_name"] in completed:
                continue
            if new_count >= args.limit:
                break
            revision = miner_revision(args.miner_health_url, revision)
            clip = args.output_dir / f"{row['sentence_name']}.mp4"
            temporary = clip.with_suffix(".partial.mp4")
            if clip.exists() or temporary.exists():
                raise RuntimeError(f"unreceipted clip must be inspected: {clip}")
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
                    raise ValueError("extracted clip size differs from archive index")
                video_stream = check_video(temporary)
                temporary.rename(clip)
            finally:
                temporary.unlink(missing_ok=True)
            revision = miner_revision(args.miner_health_url, revision)
            record = {
                "schema": "umi-studio-how2sign-training-clip/1",
                "selection_sha256": selection_hash,
                "sentence_name": row["sentence_name"],
                "archive_member": row["archive_member"],
                "archive_crc32": row["archive_crc32"],
                "clip_sha256": digest.hexdigest(),
                "clip_size_bytes": size,
                "video_stream": video_stream,
                "miner_model_revision": revision,
            }
            with args.receipt.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            new_count += 1
            print(f"{len(completed) + new_count}: fetched {row['sentence_name']}", flush=True)


if __name__ == "__main__":
    main()
