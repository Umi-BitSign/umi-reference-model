"""Freeze and fetch every unused aligned clip from the official How2Sign ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path

from audit_how2sign_precut_capacity import aligned_eligible, read_rows
from benchmark_fleurs_dev import sha256
from fetch_how2sign_pilot import MAX_CLIP_BYTES, archive_fingerprint, check_video
from how2sign_remote_zip import ARCHIVE_FILE_ID, ARCHIVE_SIZE, How2SignRemoteZip

ROOT = Path(__file__).resolve().parents[1]
SEED = 83
SCHEMA = "umi-studio-how2sign-precut-expansion/1"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise ValueError(f"stale partial selection exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(json.dumps(payload, sort_keys=True, indent=2) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.rename(path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_selection(
    metadata: dict[str, dict[str, str]],
    realigned: dict[str, dict[str, str]],
    prior: dict,
    infos: dict[str, zipfile.ZipInfo],
    fingerprint: str,
    metadata_path: Path,
    realigned_path: Path,
    license_path: Path,
    prior_path: Path,
) -> dict:
    if (
        prior.get("schema") != "umi-studio-how2sign-training-aligned/1"
        or prior.get("metadata_sha256") != sha256(metadata_path)
        or prior.get("alignment_metadata_sha256") != sha256(realigned_path)
        or prior.get("license_sha256") != sha256(license_path)
        or prior.get("archive_fingerprint_sha256") != fingerprint
    ):
        raise ValueError("prior aligned How2Sign selection differs")
    used = {row["sentence_name"] for row in prior["rows"]}
    if len(used) != len(prior["rows"]):
        raise ValueError("prior aligned How2Sign selection repeats a clip")
    records = []
    for name, row in metadata.items():
        if name in used or not aligned_eligible(row, realigned[name]):
            continue
        member = f"raw_videos/{name}.mp4"
        info = infos.get(member)
        if info is None or not 0 < info.file_size <= MAX_CLIP_BYTES:
            continue
        signer = int(row["VIDEO_NAME"].rsplit("-", 2)[1])
        records.append({
            "sentence_name": name,
            "video_id": row["VIDEO_ID"],
            "video_name": row["VIDEO_NAME"],
            "signer": signer,
            "reference": row["SENTENCE"],
            "start_sec": float(row["START"]),
            "end_sec": float(row["END"]),
            "realigned_start_sec": float(realigned[name]["START_REALIGNED"]),
            "realigned_end_sec": float(realigned[name]["END_REALIGNED"]),
            "archive_member": member,
            "archive_crc32": info.CRC,
            "archive_file_size": info.file_size,
            "archive_compress_size": info.compress_size,
        })
    records.sort(key=lambda row: (
        hashlib.sha256(f"{SEED}:{row['sentence_name']}".encode()).hexdigest(),
        row["sentence_name"],
    ))
    counts = Counter(str(row["signer"]) for row in records)
    if len(records) != 2233 or counts != Counter({"8": 1601, "5": 197, "1": 126, "3": 99, "4": 93, "11": 67, "2": 25, "9": 25}):
        raise ValueError("unused aligned How2Sign capacity differs")
    return {
        "schema": SCHEMA,
        "training_only": True,
        "source_url": "https://how2sign.github.io/",
        "archive_google_drive_file_id": ARCHIVE_FILE_ID,
        "archive_size_bytes": ARCHIVE_SIZE,
        "archive_fingerprint_sha256": fingerprint,
        "metadata_sha256": sha256(metadata_path),
        "alignment_metadata_sha256": sha256(realigned_path),
        "license_sha256": sha256(license_path),
        "prior_selection_sha256": sha256(prior_path),
        "seed": SEED,
        "clips": len(records),
        "download_bytes": sum(row["archive_file_size"] for row in records),
        "per_signer": dict(sorted(counts.items())),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--alignment-metadata", type=Path, required=True)
    parser.add_argument("--license", type=Path, required=True)
    parser.add_argument("--prior-selection", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--miner-maintenance-mode", action="store_true")
    args = parser.parse_args()
    if (
        args.limit < 0 or not args.miner_maintenance_mode
        or not all(path.is_file() for path in (
            args.metadata, args.alignment_metadata, args.license, args.prior_selection
        ))
        or not args.selection.resolve().is_relative_to((ROOT / "runs").resolve())
        or not args.output_dir.resolve().is_relative_to((ROOT / "data").resolve())
        or not args.receipts.resolve().is_relative_to((ROOT / "runs").resolve())
    ):
        parser.error("invalid expansion inputs or miner policy")
    metadata = read_rows(args.metadata)
    realigned = read_rows(args.alignment_metadata)
    if set(metadata) != set(realigned):
        raise ValueError("How2Sign metadata sets differ")
    prior = json.loads(args.prior_selection.read_text())
    remote = How2SignRemoteZip()
    with zipfile.ZipFile(remote) as archive:
        info_list = archive.infolist()
        infos = {info.filename: info for info in info_list}
        if len(infos) != len(info_list):
            raise ValueError("How2Sign archive repeats a member")
        selection = freeze_selection(
            metadata, realigned, prior, infos, archive_fingerprint(info_list),
            args.metadata, args.alignment_metadata, args.license, args.prior_selection,
        )
        if args.selection.exists():
            if json.loads(args.selection.read_text()) != selection:
                raise ValueError("frozen pre-cut expansion differs")
        else:
            write_json_atomic(args.selection, selection)
        print(json.dumps({"clips": selection["clips"], "bytes": selection["download_bytes"], "archive_index_bytes": remote.bytes_downloaded}, sort_keys=True), flush=True)
        if args.limit == 0:
            return
        selection_hash = sha256(args.selection)
        completed = {}
        if args.receipts.exists():
            for line in args.receipts.read_text().splitlines():
                receipt = json.loads(line)
                clip = args.output_dir / f"{receipt.get('sentence_name')}.mp4"
                if (
                    receipt.get("schema") != "umi-studio-how2sign-precut-expansion-clip/1"
                    or receipt.get("selection_sha256") != selection_hash
                    or receipt.get("sentence_name") in completed
                    or not clip.is_file() or sha256(clip) != receipt.get("clip_sha256")
                ):
                    raise ValueError("pre-cut expansion receipt differs")
                completed[receipt["sentence_name"]] = receipt
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.receipts.parent.mkdir(parents=True, exist_ok=True)
        new_count = 0
        for row in selection["records"]:
            name = row["sentence_name"]
            if name in completed:
                continue
            if new_count >= args.limit:
                break
            clip = args.output_dir / f"{name}.mp4"
            temporary = clip.with_suffix(".partial.mp4")
            if clip.exists() or temporary.exists():
                raise ValueError(f"unreceipted expansion clip exists: {clip}")
            digest = hashlib.sha256(); size = 0
            try:
                with archive.open(infos[row["archive_member"]]) as source, temporary.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk); digest.update(chunk); size += len(chunk)
                    output.flush(); os.fsync(output.fileno())
                if size != row["archive_file_size"]:
                    raise ValueError("pre-cut expansion clip size differs")
                probe = check_video(temporary)
                temporary.rename(clip)
            finally:
                temporary.unlink(missing_ok=True)
            receipt = {
                "schema": "umi-studio-how2sign-precut-expansion-clip/1",
                "selection_sha256": selection_hash,
                "sentence_name": name,
                "clip_sha256": digest.hexdigest(),
                "clip_size_bytes": size,
                "archive_member": row["archive_member"],
                "archive_crc32": row["archive_crc32"],
                "video_stream": probe,
                "miner_health_policy": "user_authorized_maintenance",
            }
            with args.receipts.open("a", encoding="utf-8") as output:
                output.write(json.dumps(receipt, sort_keys=True) + "\n"); output.flush(); os.fsync(output.fileno())
            new_count += 1
            print(f"fetched={len(completed)+new_count}/{selection['clips']} name={name}", flush=True)


if __name__ == "__main__":
    main()
