"""Download selected 2M-Flores-ASL dev clips with hashes and miner checks.

The selection excludes FLEURS validation source articles. Only its pinned URLs
are fetched, at a bounded transfer rate, into the independent data directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

from benchmark_fleurs_dev import miner_revision, sha256


def check_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration",
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
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-bytes-per-second", type=int, default=4 * 1024 * 1024)
    args = parser.parse_args()
    if args.limit < 1 or args.max_bytes_per_second < 1:
        parser.error("limit and max transfer rate must be positive")
    selection = json.loads(args.selection.read_text())
    if (
        selection["schema"] != "umi-studio-flores2m-training-pilot/1"
        or selection["source_split"] != "dev"
        or selection["devtest_used"] is not False
    ):
        raise ValueError("unexpected training selection or protected split")
    revision = selection["repository_revision"]
    selection_hash = sha256(args.selection)
    rows = selection["rows"]
    if len({row["video_url"] for row in rows}) != len(rows):
        raise ValueError("selection contains duplicate video URLs")
    completed = {}
    if args.receipt.exists():
        for line in args.receipt.read_text().splitlines():
            record = json.loads(line)
            if (
                record["schema"] != "umi-studio-flores2m-training-clip/1"
                or record["selection_sha256"] != selection_hash
                or record["row_idx"] in completed
            ):
                raise ValueError("download receipt differs from the frozen selection")
            clip = args.output_dir / record["filename"]
            if not clip.is_file() or sha256(clip) != record["video_sha256"]:
                raise ValueError(f"downloaded clip missing or altered: {clip}")
            completed[record["row_idx"]] = record
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    miner_model_revision = miner_revision(args.miner_health_url, None)
    new_count = 0
    for row in rows:
        if row["row_idx"] in completed:
            continue
        if new_count >= args.limit:
            break
        url = row["video_url"]
        prefix = (
            f"https://huggingface.co/datasets/facebook/2M-Flores-ASL/resolve/{revision}/data/dev/"
        )
        if not url.startswith(prefix):
            raise ValueError("selection URL left the pinned dev split")
        filename = Path(urllib.parse.urlparse(url).path).name
        if filename != url.rsplit("/", 1)[1]:
            raise ValueError("selection URL has an unexpected filename")
        clip = args.output_dir / filename
        temporary = args.output_dir / f"{filename}.partial"
        if clip.exists() or temporary.exists():
            raise ValueError(f"unreceipted or unfinished clip must be inspected: {clip}")
        miner_model_revision = miner_revision(args.miner_health_url, miner_model_revision)
        expected_size = row["video_size_bytes"]
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(url)
            with urllib.request.urlopen(request, timeout=60) as response:
                if (
                    response.status != 200
                    or int(response.headers["Content-Length"]) != expected_size
                ):
                    raise RuntimeError("source video size differs from pinned file metadata")
                with temporary.open("xb") as output:
                    while True:
                        read_started = time.monotonic()
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        size += len(chunk)
                        if size > expected_size:
                            raise RuntimeError("source video exceeds pinned size")
                        output.write(chunk)
                        digest.update(chunk)
                        if size % (8 * 1024 * 1024) < len(chunk):
                            miner_model_revision = miner_revision(
                                args.miner_health_url, miner_model_revision
                            )
                        wait = len(chunk) / args.max_bytes_per_second - (
                            time.monotonic() - read_started
                        )
                        if wait > 0:
                            time.sleep(wait)
                    output.flush()
                    os.fsync(output.fileno())
            if size != expected_size or digest.hexdigest() != row["video_sha256"]:
                raise ValueError("source video differs from its published SHA-256")
            video_stream = check_video(temporary)
            temporary.rename(clip)
        finally:
            temporary.unlink(missing_ok=True)
        miner_model_revision = miner_revision(args.miner_health_url, miner_model_revision)
        record = {
            "schema": "umi-studio-flores2m-training-clip/1",
            "selection_sha256": selection_hash,
            "repository_revision": revision,
            "row_idx": row["row_idx"],
            "filename": filename,
            "video_sha256": digest.hexdigest(),
            "size_bytes": size,
            "video_stream": video_stream,
            "miner_model_revision": miner_model_revision,
        }
        with args.receipt.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
        new_count += 1
        print(f"{len(completed) + new_count}: fetched {filename} ({size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
