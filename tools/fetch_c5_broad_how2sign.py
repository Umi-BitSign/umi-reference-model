"""Acquire the reserved How2Sign raw recordings without decoding model inputs.

Uses public bounded archive ranges and existing CRC verification. This entry
point performs no model/scorer operation or service-health request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

from benchmark_fleurs_dev import sha256
from fetch_how2sign_pilot import archive_fingerprint
from how2sign_remote_zip import How2SignRemoteZip
from launch_terra_visual_job import identity, now
from prepare_c5_fresh_development import iter_member

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 4 * 1024**3
CHUNK = 8 * 1024**2
SOURCES = (
    "fetch_c5_broad_how2sign.py",
    "prepare_c5_fresh_development.py",
    "how2sign_remote_zip.py",
    "fetch_how2sign_pilot.py",
    "benchmark_fleurs_dev.py",
    "launch_terra_visual_job.py",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def artifact(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def admit(reservation_path):
    require(
        reservation_path.resolve() == reservation_path
        and reservation_path.is_relative_to(ROOT / "runs")
        and reservation_path.is_file(),
        "raw acquisition requires an owned reservation",
    )
    p = read(reservation_path)
    validation = read(reservation_path.with_name("validation.json"))
    require(
        p.get("schema") == "umi-studio-broad-semantic-confirmation-metadata-reservation/1"
        and p.get("training_allowed") is False
        and p.get("confirmation_consumed") is False
        and p.get("model_predictions_inspected") is False
        and p.get("candidate_bound") is False
        and validation["plan_sha256"] == sha256(reservation_path)
        and validation["review_sha256"] == sha256(reservation_path.with_name("review.json")),
        "raw acquisition lost the reviewed unconsumed confirmation reservation",
    )
    for name, digest in {**p["input_sha256"], **p["source_sha256"]}.items():
        require(sha256(ROOT / name) == digest, "confirmation input or selector bytes changed")
    a = ROOT / "runs/c5-reassessment-v1/raw-validation-archive-v2.json"
    require(p["input_sha256"][str(a.relative_to(ROOT))] == sha256(a), "archive inventory changed")
    return p, read(a)


def specification(reservation_path, data):
    p, a = admit(reservation_path)
    require(
        data.resolve() == data and data.is_relative_to(ROOT / "data") and data != ROOT / "data",
        "raw acquisition requires an owned private data directory",
    )
    sources = p["how2sign_primary_recordings"]
    clips = [r["metadata"] for r in p["primary_records"] if r["family"] == "how2sign"]
    require(
        len(sources) == 12 and len(clips) == 24 and len({r["video_id"] for r in sources}) == 12,
        "raw acquisition changed primary source/clip coverage",
    )
    require(
        Counter(r["signer"] for r in sources) == {1: 3, 2: 3, 3: 3, 5: 3},
        "raw acquisition changed signer coverage",
    )
    require(
        sum(r["file_size"] for r in sources) == p["planned_How2Sign_raw_source_bytes"] <= BUDGET,
        "raw acquisition exceeds the frozen source-byte budget",
    )
    for source in sources:
        name = source["video_name"]
        require(
            isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name),
            "raw recording name is unsafe",
        )
        selected = [r for r in clips if r["video_name"] == name]
        require(
            len(selected) == 2
            and all(
                r["video_id"] == source["video_id"]
                and r["signer"] == source["signer"]
                and r["sentence_name"] in source["candidate_clips"]
                for r in selected
            ),
            "raw source/clip identity join changed",
        )
    return {
        "schema": "umi-studio-broad-how2sign-raw-acquisition/1",
        "reservation": artifact(reservation_path),
        "reservation_review": artifact(reservation_path.with_name("review.json")),
        "sources": sources,
        "clips": clips,
        "data_dir": str(data.relative_to(ROOT)),
        "archive": {
            k: a[k]
            for k in ("file_id", "archive_name", "archive_size_bytes", "archive_fingerprint_sha256")
        },
        "source_sha256": {f"tools/{name}": sha256(ROOT / "tools" / name) for name in SOURCES},
        "policy": {
            "data_budget_bytes": BUDGET,
            "chunk_bytes": CHUNK,
            "CPU_threads": 1,
            "active_training_reserve_bytes": 4 * 1024**3,
            "physical_floor_bytes": 80 * 1024**3,
            "headroom_bytes": 128 * 1024**2,
            "range_maximum_attempts": 4,
            "job_or_source_retries": 0,
            "training_allowed": False,
            "model_predictions_performed": False,
            "GPU_or_scorer_used": False,
            "scope": "CRC-checked raw recordings and timestamp probes; no frame decode/crop",
        },
    }


def guard(data, prospective=0):
    paths = list(data.rglob("*")) if data.exists() else []
    require(all(not p.is_symlink() for p in paths), "raw acquisition cannot follow output aliases")
    used = sum(p.stat().st_size for p in paths if p.is_file())
    require(
        0 <= prospective <= BUDGET - used
        and shutil.disk_usage(ROOT).free
        >= 80 * 1024**3 + 4 * 1024**3 + 128 * 1024**2 + (BUDGET - used),
        "raw acquisition storage reserve or physical floor failed",
    )


def freeze(path, reservation_path, data):
    path, reservation_path, data = (p.absolute() for p in (path, reservation_path, data))
    require(
        path.resolve() == path
        and path.is_relative_to(ROOT / "runs")
        and not path.parent.exists()
        and not data.exists(),
        "raw acquisition needs a fresh owned job/data output",
    )
    plan = specification(reservation_path, data)
    guard(data, sum(r["file_size"] for r in plan["sources"]))
    path.parent.mkdir(parents=True)
    write(path, plan)
    for name, digest in plan["source_sha256"].items():
        target = path.parent / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write((ROOT / name).read_bytes())
        require(sha256(target) == digest, "raw acquisition source archive changed")
    print(
        json.dumps(
            {
                "plan": artifact(path),
                "sources": 12,
                "selected_clips": 24,
                "planned_raw_bytes": sum(r["file_size"] for r in plan["sources"]),
            }
        ),
        flush=True,
    )


def checked(path):
    p = read(path)
    require(
        p == specification(ROOT / p["reservation"]["path"], ROOT / p["data_dir"]),
        "raw acquisition plan/source changed",
    )
    require(
        all(
            sha256(path.parent / "source-snapshot" / name) == digest
            for name, digest in p["source_sha256"].items()
        ),
        "raw acquisition source archive changed",
    )
    return p


def probe_source(path, clips):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-threads",
            "1",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    probe = json.loads(result.stdout)
    require(len(probe.get("streams", [])) == 1, "raw source lacks one video stream")
    stream = probe["streams"][0]
    duration = None
    for value in (stream.get("duration"), probe.get("format", {}).get("duration")):
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            duration = v
            break
    require(
        duration is not None and stream.get("width", 0) > 0 and stream.get("height", 0) > 0,
        "raw source has invalid duration or dimensions",
    )
    require(
        all(0 <= r["start_sec"] < r["end_sec"] <= duration + 0.05 for r in clips),
        "reserved realigned interval exceeds its actual raw recording",
    )
    return probe


def run(path):
    path = path.resolve()
    p = checked(path)
    job = path.parent
    data = ROOT / p["data_dir"]
    require(
        not data.exists() and not (job / "worker.json").exists(),
        "raw acquisition cannot replay a prior job or overwrite its data",
    )
    guard(data, sum(r["file_size"] for r in p["sources"]))
    write(job / "worker.json", identity(os.getpid(), [sys.executable, *sys.argv]))
    code = 1
    try:
        data.mkdir(parents=True)
        sources_dir = data / "sources"
        sources_dir.mkdir()
        a = p["archive"]
        receipts = []
        print(json.dumps({"stage": "raw_archive_directory_verification"}), flush=True)
        with How2SignRemoteZip(
            file_id=a["file_id"],
            archive_name=a["archive_name"],
            archive_size=a["archive_size_bytes"],
        ) as remote:
            with zipfile.ZipFile(remote) as archive:
                infos = archive.infolist()
                by_name = {info.filename: info for info in infos}
                require(
                    archive_fingerprint(infos) == a["archive_fingerprint_sha256"],
                    "raw public archive directory identity changed",
                )
            for index, row in enumerate(p["sources"]):
                info = by_name[row["archive_member"]]
                require(
                    (
                        info.CRC,
                        info.file_size,
                        info.compress_size,
                        info.header_offset,
                        info.compress_type,
                    )
                    == (
                        row["crc32"],
                        row["file_size"],
                        row["compressed_size"],
                        row["header_offset"],
                        row["compression_method"],
                    ),
                    "raw member identity changed",
                )
                guard(data, row["file_size"])
                partial = sources_dir / (row["video_name"] + ".partial.mp4")
                target = sources_dir / (row["video_name"] + ".mp4")
                digest = hashlib.sha256()
                written = 0
                print(
                    json.dumps(
                        {"stage": "raw_source_download", "source": index, "of": len(p["sources"])}
                    ),
                    flush=True,
                )
                with partial.open("xb") as f:
                    for block in iter_member(remote, row, chunk_bytes=CHUNK):
                        guard(data, len(block))
                        f.write(block)
                        digest.update(block)
                        written += len(block)
                    f.flush()
                    os.fsync(f.fileno())
                require(written == row["file_size"], "raw source decoded byte count changed")
                clips = [r for r in p["clips"] if r["video_name"] == row["video_name"]]
                probe = probe_source(partial, clips)
                write(job / f"probe-{index:02d}.json", probe)
                require(
                    sha256(partial) == digest.hexdigest(),
                    "raw file bytes differ from verified download",
                )
                partial.rename(target)
                receipt = {
                    "plan": artifact(path),
                    "source": row,
                    "video": artifact(target),
                    "probe": artifact(job / f"probe-{index:02d}.json"),
                    "selected_clips": [r["sentence_name"] for r in clips],
                    "CRC_and_bytes_verified": True,
                    "timestamps_fit_actual_recording": True,
                    "training_allowed": False,
                    "model_predictions_performed": False,
                }
                receipt_path = job / f"source-{index:02d}.json"
                write(receipt_path, receipt)
                receipts.append(artifact(receipt_path))
                print(
                    json.dumps(
                        {
                            "stage": "raw_source_verified",
                            "sources": index + 1,
                            "bytes": written,
                            "selected_clips": len(clips),
                        }
                    ),
                    flush=True,
                )
            checked(path)
            write(
                job / "complete.json",
                {
                    "complete": True,
                    "plan": artifact(path),
                    "source_receipts": receipts,
                    "sources_verified": 12,
                    "selected_intervals_verified": 24,
                    "successful_range_requests": remote.requests,
                    "successful_range_bytes": remote.bytes_downloaded,
                    "confirmation_consumed": False,
                    "training_allowed": False,
                    "model_predictions_performed": False,
                    "scope": "How2Sign raw integrity/timestamps; crops/landmarks/OpenASL pending",
                },
            )
        code = 0
    except BaseException as e:
        write(job / "failure.json", {"error": str(e), "job_retries": 0, "partials_retained": True})
        raise
    finally:
        write(job / "exit.json", {"pid": os.getpid(), "exit_code": code, "observed_at": now()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("freeze", "reservation", "data-dir", "plan"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.freeze and args.reservation and args.data_dir:
        freeze(args.freeze, args.reservation, args.data_dir)
        if args.run:
            run(args.freeze)
    elif args.plan and args.run:
        run(args.plan)
    else:
        parser.error("freeze one metadata-bound acquisition, or execute its fresh plan")


if __name__ == "__main__":
    main()
