"""Bind and fetch complete word clips, preserving signer-disjoint diagnostic roles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path

from benchmark_fleurs_dev import normalize, sha256
from fetch_asl_stem_wiki_metadata import (
    ARCHIVE_NAME,
    ARCHIVE_SIZE,
    AslStemWikiRemoteZip,
    inventory_record,
    validate_inventory,
)
from launch_terra_visual_job import write

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "runs/terra-word-curriculum-capacity-v1"
SOURCE = ROOT / "runs/asl-stem-wiki-v1/source-manifest.json"
CSV = ROOT / "data/asl-stem-wiki/source/videos.csv"
LICENSE = ROOT / "data/asl-stem-wiki/source/use.txt"
POLICY = {
    "physical_free_floor_bytes": 80 * 1024**3,
    "temporary_headroom_bytes": 128 * 1024**2,
    "checkpoint_reserve_bytes": 215000000,
    "maximum_total_raw_bytes": 2 * 1024**3,
    "maximum_clip_bytes": 32 * 1024**2,
    "maximum_batch_clips": 16,
    "maximum_probe_seconds": 120,
    "cpu_threads": 2,
    "GPU_use": False,
    "miner_or_codex_lb_inspection": False,
}
CSV_FIELDS = (
    "participant",
    "article",
    "section index",
    "sentence index",
    "video filename",
    "video length",
    "sentence text",
)


def guard_space(prospective=0):
    required = (
        POLICY["physical_free_floor_bytes"]
        + POLICY["temporary_headroom_bytes"]
        + POLICY["checkpoint_reserve_bytes"]
        + prospective
    )
    if shutil.disk_usage(ROOT).free < required:
        raise RuntimeError("word acquisition would breach unchanged 80GiB floor/headroom")


def validate_rows(draft, metadata, source_rows, protected):
    by_name = {r["video filename"]: r for r in source_rows}
    if len(by_name) != len(source_rows):
        raise ValueError("publisher metadata repeats a video filename")
    identities = Counter(tuple(r[k] for k in CSV_FIELDS[:4]) for r in source_rows)
    heldout = set(metadata["heldout_participants"])
    seen, classes, result = set(), defaultdict(lambda: defaultdict(set)), []
    for row in draft:
        name, participant, role = row["video filename"], row["participant"], row["role"]
        original = by_name.get(name)
        reference = row["sentence text"]
        plain = reference.strip().rstrip(".")
        if (
            original is None
            or any(original[k] != row[k] for k in CSV_FIELDS)
            or re.fullmatch(r"[A-Za-z0-9_-]+\.mp4", name) is None
            or re.fullmatch(r"P\d+", participant) is None
            or name in seen
            or identities[tuple(row[k] for k in CSV_FIELDS[:4])] != 1
            or re.fullmatch(r"[A-Za-z]{3,24}", plain) is None
            or plain.isupper()
            or row["label"] != plain.lower()
            or tuple(normalize(reference)) in protected
            or not 0.75 <= float(row["video length"]) <= 8
            or role not in ("training", "development")
            or row["training_allowed"] is not (role == "training")
            or (participant in heldout) != (role == "development")
        ):
            raise ValueError("word source identity, full label, protection or split differs")
        seen.add(name)
        classes[row["label"]][role].add(participant)
        result.append(
            {
                "example_id": Path(name).stem,
                "filename": name,
                "reference": reference,
                "label": row["label"],
                "participant": participant,
                "article": row["article"],
                "source_group": f"asl_stem:{participant}:{row['article']}",
                "duration_sec": float(row["video length"]),
                "role": role,
                "training_allowed": role == "training",
            }
        )
    if (
        not result
        or dict(Counter(r["role"] for r in result)) != metadata["selected_by_role"]
        or set(classes) != set(metadata["selected_labels"])
        or any(len(v["training"]) < 3 or not v["development"] for v in classes.values())
    ):
        raise ValueError("word pilot class/signer coverage differs from its metadata selection")
    return result


def inventory_sha(infos):
    payload = json.dumps(
        [inventory_record(i) for i in infos], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def sources():
    return {
        str(p.relative_to(ROOT)): sha256(p)
        for p in (
            Path(__file__),
            ROOT / "tools/fetch_asl_stem_wiki_metadata.py",
            ROOT / "tools/how2sign_remote_zip.py",
            ROOT / "tools/benchmark_fleurs_dev.py",
            ROOT / "tools/audit_c5_all_source_corpus.py",
            ROOT / "tools/launch_terra_visual_job.py",
        )
    }


@contextmanager
def remote_archive(source):
    reader = AslStemWikiRemoteZip(
        file_id="direct-download", archive_name=ARCHIVE_NAME, archive_size=ARCHIVE_SIZE
    )
    try:
        with zipfile.ZipFile(reader) as archive:
            infos = archive.infolist()
            validate_inventory(infos)
            if inventory_sha(infos) != source["archive_inventory_sha256"]:
                raise ValueError("word pilot archive differs from the approved publisher inventory")
            yield archive, reader
    finally:
        reader.close()


def metadata():
    from audit_c5_all_source_corpus import PROTECTED, protected_sets

    spec = json.loads(SOURCE.read_text())
    expected = {r["local_name"]: r["sha256"] for r in spec["metadata_members"]}
    draft = json.loads((DRAFT / "draft-records.json").read_text())
    meta = json.loads((DRAFT / "metadata-only.json").read_text())
    if (
        sha256(CSV) != expected["videos.csv"]
        or sha256(LICENSE) != expected["use.txt"]
        or meta["videos_csv_sha256"] != expected["videos.csv"]
        or meta["license_sha256"] != expected["use.txt"]
        or meta["auditor_sha256"] != sha256(DRAFT / "audit.py")
        or meta["complete_metadata_analysis"] is not True
    ):
        raise ValueError("word pilot source metadata or license changed")
    with CSV.open(newline="", encoding="utf-8-sig") as stream:
        raw = list(csv.DictReader(stream))
    _, protected = protected_sets()
    rows = validate_rows(draft, meta, raw, protected)
    paths = [
        SOURCE,
        CSV,
        LICENSE,
        DRAFT / "draft-records.json",
        DRAFT / "metadata-only.json",
        DRAFT / "audit.py",
        *PROTECTED.values(),
    ]
    return spec, meta, rows, {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def bind_members(rows, archive):
    result = []
    for row in rows:
        info = archive.getinfo("ASL_STEM_Wiki/videos/" + row["filename"])
        if (
            info.flag_bits & 1
            or not 0 < info.file_size <= POLICY["maximum_clip_bytes"]
            or not 0 < info.compress_size <= POLICY["maximum_clip_bytes"]
            or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)
        ):
            raise ValueError("word member encryption or size exceeds its bounded profile")
        result.append({**row, "archive": inventory_record(info)})
    if sum(r["archive"]["file_size"] for r in result) > POLICY["maximum_total_raw_bytes"]:
        raise ValueError("word pilot would exceed its 2GiB raw-data budget")
    return result


def freeze(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT / "runs") or path.parent.exists():
        raise ValueError("word data plan needs a fresh owned runs/ directory")
    spec, meta, rows, assets = metadata()
    guard_space()
    with remote_archive(spec) as (archive, reader):
        records = bind_members(rows, archive)
        network = {"requests": reader.requests, "bytes": reader.bytes_downloaded}
    plan = {
        "schema": "umi-studio-word-curriculum-data/1",
        "policy": POLICY,
        "source_sha256": sources(),
        "assets": assets,
        "records": records,
        "heldout_participants": meta["heldout_participants"],
        "label_overlap_train_development_intentional": True,
        "prepared_training_pairs": 0,
        "source_video_downloads": 0,
        "source_index_network": network,
        "primary_metric": "Terra semantic quality 0-10",
        "WER_is_diagnostic_only": True,
        "promotion_authorized": False,
    }
    path.parent.mkdir(parents=True)
    write(path, plan)
    print(
        json.dumps(
            {
                "plan_sha256": sha256(path),
                "clips": len(records),
                "raw_bytes": sum(r["archive"]["file_size"] for r in records),
            }
        ),
        flush=True,
    )


def checked(path):
    plan = json.loads(path.read_text())
    spec, meta, rows, assets = metadata()
    if (
        plan["schema"] != "umi-studio-word-curriculum-data/1"
        or plan["policy"] != POLICY
        or plan["source_sha256"] != sources()
        or plan["assets"] != assets
        or plan["heldout_participants"] != meta["heldout_participants"]
        or [{k: v for k, v in r.items() if k != "archive"} for r in plan["records"]] != rows
    ):
        raise ValueError("word data plan/source/role binding changed")
    return plan, spec


def validate_probe(payload, declared):
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError("word clip requires one complete video stream")
    stream = streams[0]
    try:
        duration = float(payload.get("format", {}).get("duration", 0))
        frames = int(stream.get("nb_read_frames", 0))
        fps = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, TypeError, ZeroDivisionError, OverflowError) as error:
        raise ValueError("word clip frame/duration metadata is invalid") from error
    if (
        not all(math.isfinite(v) for v in (duration, declared, fps))
        or not 32 <= stream.get("width", 0) <= 3840
        or not 32 <= stream.get("height", 0) <= 2160
        or not 1 <= fps <= 120
        or not 1 <= frames <= 2000
        or abs(duration - declared) > max(0.25, declared * 0.05)
        or abs(frames - duration * fps) > max(2, duration * fps * 0.05)
    ):
        raise ValueError("word clip decoded frames, duration, dimensions or frame rate differ")
    return {
        "duration_sec": duration,
        "decoded_frames": frames,
        "fps": fps,
        "width": stream["width"],
        "height": stream["height"],
    }


def probe(path, declared):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads",
            "2",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=POLICY["maximum_probe_seconds"],
    )
    return validate_probe(json.loads(result.stdout), declared)


def paths_for(row, data, receipts):
    return data / row["role"] / (row["example_id"] + ".mp4"), receipts / row["role"] / (
        row["example_id"] + ".json"
    )


def verified(row, data, receipts, plan_sha):
    target, receipt = paths_for(row, data, receipts)
    if (
        target.is_symlink()
        or receipt.is_symlink()
        or target.parent.is_symlink()
        or receipt.parent.is_symlink()
    ):
        raise ValueError("word clip paths cannot be role aliases or links")
    if not target.exists() and not receipt.exists():
        if target.with_suffix(".partial.mp4").exists():
            raise ValueError("word clip has an unapproved partial; inspect it before recovery")
        return False
    if not target.is_file() or not receipt.is_file():
        raise ValueError("word clip or receipt is orphaned; no silent adoption or replay")
    saved = json.loads(receipt.read_text())
    if (
        saved["plan_sha256"] != plan_sha
        or saved["row"] != row
        or saved["video_sha256"] != sha256(target)
        or target.stat().st_size != row["archive"]["file_size"]
    ):
        raise ValueError("word receipt bytes or source/role binding differ")
    return True


def fetch(path, data, receipts, limit):
    path, data, receipts = (p.resolve() for p in (path, data, receipts))
    if (
        not data.is_relative_to(ROOT / "data")
        or not receipts.is_relative_to(ROOT / "runs")
        or not 1 <= limit <= POLICY["maximum_batch_clips"]
    ):
        raise ValueError("word fetch needs owned data/runs paths and a batch of 1-16")
    plan, spec = checked(path)
    plan_sha = sha256(path)
    pending = [r for r in plan["records"] if not verified(r, data, receipts, plan_sha)]
    expected_paths = {paths_for(r, data, receipts)[0] for r in plan["records"]}
    if data.exists() and any(
        p.is_symlink() or (p.is_file() and p not in expected_paths) for p in data.rglob("*")
    ):
        raise ValueError("word raw-data directory contains an unapproved path")
    if not pending:
        print(json.dumps({"complete_raw_acquisition": True, "clips": len(plan["records"])}))
        return
    guard_space(max(r["archive"]["file_size"] for r in pending[:limit]))
    with remote_archive(spec) as (archive, _reader):
        if (
            bind_members(
                [{k: v for k, v in r.items() if k != "archive"} for r in plan["records"]], archive
            )
            != plan["records"]
        ):
            raise ValueError("word archive member binding changed")
        for row in pending[:limit]:
            guard_space(row["archive"]["file_size"])
            payload = archive.read(row["archive"]["name"])
            if len(payload) != row["archive"]["file_size"]:
                raise ValueError("downloaded word member size differs")
            target, receipt = paths_for(row, data, receipts)
            target.parent.mkdir(parents=True, exist_ok=True)
            receipt.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(".partial.mp4")
            with partial.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            info = probe(partial, row["duration_sec"])
            os.link(partial, target)  # Fails atomically if an unexpected target exists.
            partial.unlink()
            write(
                receipt,
                {
                    "schema": "umi-studio-word-curriculum-raw-video/1",
                    "plan_sha256": plan_sha,
                    "row": row,
                    "video_sha256": sha256(target),
                    "probe": info,
                    "model_training_ready": False,
                    "label_semantics_reviewed": False,
                    "miner_or_codex_lb_inspected_or_modified": False,
                },
            )
            print(
                json.dumps(
                    {
                        "example_id": row["example_id"],
                        "role": row["role"],
                        "complete_source_clip_verified": True,
                    }
                ),
                flush=True,
            )
    checked(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.plan and args.data and args.receipts:
        fetch(args.plan, args.data, args.receipts, args.limit)
    else:
        parser.error("provide --freeze or --plan with --data and --receipts")


if __name__ == "__main__":
    main()
