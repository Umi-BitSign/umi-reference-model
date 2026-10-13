"""Run short public-data extraction batches with miner and memory guards."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from benchmark_fleurs_dev import miner_revision
from verify_fleurs_training_stream_fidelity import verify_training_rows

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "artifacts/baseline-6fe8df59"
RUNTIME_PYTHON = BASELINE / ".runtime/bin/python"
MIN_FREE_GIB = 80
POLL_SECONDS = 10
MAX_IDLE_SECONDS = 1800
SOURCES = {
    "how2sign_new": {
        "index": "runs/how2sign-expanded-v1/encoder-cache-new.jsonl",
        "target": 2335,
        "batch": 128,
        "max_rss_gib": 40,
        "worker": "tools/cache_how2sign_encoder.py",
        "args": [
            "--selection", "runs/how2sign-expanded-v1/selection.json",
            "--clip-receipts", "runs/how2sign-expanded-v1/clip-receipts.jsonl",
            "--source-dir", "data/how2sign/aligned-clips",
            "--resampled-dir", "data/how2sign/aligned-clips-25fps",
            "--cache-dir", "data/how2sign/aligned-encoder-cache-25fps",
            "--index", "runs/how2sign-expanded-v1/encoder-cache-new.jsonl",
            "--skip-index", "runs/how2sign-expanded-v1/encoder-cache-rebound.jsonl",
            "--save-stream-features",
        ],
    },
    "how2sign_original_sidecars": {
        "index": "runs/how2sign-expanded-v1/visual-stream-sidecars-rebound.jsonl",
        "target": 1024,
        "batch": 64,
        "max_rss_gib": 32,
        "worker": "tools/backfill_visual_streams.py",
        "args": [
            "--index", "runs/how2sign-expanded-v1/encoder-cache-rebound.jsonl",
            "--cache-dir", "data/how2sign/aligned-encoder-cache-25fps",
            "--clip-dir", "data/how2sign/aligned-clips-25fps",
            "--sidecar-dir", "data/how2sign/aligned-visual-stream-sidecars-25fps",
            "--sidecar-index", "runs/how2sign-expanded-v1/visual-stream-sidecars-rebound.jsonl",
        ],
    },
    "how2sign_raw_sidecars": {
        "index": "runs/how2sign-raw-v1/expansion-visual-stream-sidecars-v2.jsonl",
        "target": 391,
        "batch": 64,
        "max_rss_gib": 32,
        "worker": "tools/backfill_visual_streams.py",
        "args": [
            "--index", "runs/how2sign-raw-v1/expansion-training-index-v2.jsonl",
            "--cache-dir", "data/how2sign/raw-encoder-cache-25fps",
            "--clip-dir", "data/how2sign/raw-aligned-clips-25fps",
            "--sidecar-dir", "data/how2sign/raw-visual-stream-sidecars-25fps",
            "--sidecar-index", "runs/how2sign-raw-v1/expansion-visual-stream-sidecars-v2.jsonl",
        ],
    },
    "fleurs": {
        "index": "runs/fleurs-dev-v1/training-stream-encoder-cache.jsonl",
        "target": 313,
        "batch": 32,
        "max_rss_gib": 32,
        "worker": "tools/cache_fleurs_encoder.py",
        "args": [
            "--selection", "runs/fleurs-dev-v1/selection.json",
            "--source-receipt", "runs/fleurs-dev-v1/training-source-files.json",
            "--source-dir", "data/fleurs-asl/source",
            "--clip-dir", "data/fleurs-asl/training-stream-clips-25fps",
            "--cache-dir", "data/fleurs-asl/encoder-cache-25fps/training-stream",
            "--index", "runs/fleurs-dev-v1/training-stream-encoder-cache.jsonl",
            "--split", "training",
            "--save-stream-features",
        ],
    },
}


def index_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def worker_command(source: str, health_url: str, limit: int) -> list[str]:
    spec = SOURCES[source]
    if limit < 1 or limit > spec["batch"]:
        raise ValueError("extraction limit exceeds the safe batch size")
    return [
        "nice", "-n", "20", str(RUNTIME_PYTHON), spec["worker"],
        *spec["args"],
        "--baseline", str(BASELINE),
        "--miner-health-url", health_url,
        "--limit", str(limit),
    ]


def rss_kib(pid: int) -> int:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "rss="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return 0
    return int(result.stdout.strip())


def active_worker_pids(spec: dict) -> list[int]:
    if "worker" not in spec:
        return []
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    matches = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid, command = parts
        if spec["worker"] in command and spec["index"] in command and int(pid) != os.getpid():
            matches.append(int(pid))
    return matches


def stop_worker(worker: subprocess.Popen) -> None:
    if worker.poll() is not None:
        return
    worker.send_signal(signal.SIGINT)
    try:
        worker.wait(timeout=60)
    except subprocess.TimeoutExpired:
        worker.terminate()
        try:
            worker.wait(timeout=15)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait(timeout=15)


def await_existing_worker(
    source: str,
    pid: int,
    expected_count: int,
    health_url: str,
    expected_revision: str,
) -> None:
    spec = SOURCES[source]
    index = ROOT / spec["index"]
    if pid < 1 or not 0 < expected_count <= spec["target"]:
        raise ValueError("handoff needs an exact worker PID and batch-end count")
    observed = index_count(index)
    if observed > expected_count:
        raise ValueError("handoff index already exceeds the expected batch count")
    last_progress = time.monotonic()
    while True:
        existing = active_worker_pids(spec)
        if pid not in existing:
            if existing:
                raise RuntimeError(f"{source} has another writer during handoff: {existing}")
            if index_count(index) != expected_count:
                raise RuntimeError(f"{source} existing batch stopped short of {expected_count}")
            miner_revision(health_url, expected_revision)
            print(f"{source}: accepted existing batch {expected_count}", flush=True)
            return
        if existing != [pid]:
            raise RuntimeError(f"{source} has multiple writers during handoff: {existing}")
        try:
            miner_revision(health_url, expected_revision)
            if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 1024**3:
                raise RuntimeError("free disk is below the extraction guard")
            if rss_kib(pid) > spec["max_rss_gib"] * 1024**2:
                raise RuntimeError(f"{source} existing worker exceeded its RSS guard")
            current = index_count(index)
            if current > expected_count:
                raise RuntimeError(f"{source} existing worker exceeded its batch count")
            if current > observed:
                observed = current
                last_progress = time.monotonic()
            elif time.monotonic() - last_progress > MAX_IDLE_SECONDS:
                raise RuntimeError(f"{source} existing worker made no indexed progress")
        except Exception:
            if pid in active_worker_pids(spec):
                os.kill(pid, signal.SIGINT)
            raise
        time.sleep(POLL_SECONDS)


def verify_fleurs_prefix(health_url: str, expected: int) -> None:
    old_path = ROOT / "runs/fleurs-dev-v1/training-encoder-cache.jsonl"
    new_path = ROOT / SOURCES["fleurs"]["index"]
    old = [json.loads(line) for line in old_path.read_text().splitlines()]
    new = [json.loads(line) for line in new_path.read_text().splitlines()]
    result = verify_training_rows(
        old,
        new,
        ROOT / "data/fleurs-asl/encoder-cache-25fps/training",
        ROOT / "data/fleurs-asl/encoder-cache-25fps/training-stream",
        json.loads((ROOT / "runs/fleurs-dev-v1/selection.json").read_text()),
        health_url,
    )
    if result["examples_verified"] != expected:
        raise ValueError("FLEURS fidelity check covered another index count")


def run_source(
    source: str,
    health_url: str,
    expected_revision: str,
    *,
    handoff_pid: int | None = None,
    handoff_count: int | None = None,
) -> None:
    spec = SOURCES[source]
    index = ROOT / spec["index"]
    ledger = ROOT / "runs/visual-extraction-supervisor-v1"
    ledger.mkdir(parents=True, exist_ok=True)
    with (ledger / f"{source}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (ledger / f"{source}.log").open("a", encoding="utf-8") as output:
            if handoff_pid is not None and handoff_count is not None:
                await_existing_worker(
                    source, handoff_pid, handoff_count, health_url, expected_revision
                )
                output.write(f"{time.time():.0f} accepted handoff {handoff_pid}: {handoff_count}\n")
                output.flush()
            while True:
                before = index_count(index)
                if before == spec["target"]:
                    print(f"{source}: complete {before}/{spec['target']}", flush=True)
                    return
                if before > spec["target"]:
                    raise ValueError(f"{source} index exceeds the frozen target")
                if shutil.disk_usage(ROOT).free < MIN_FREE_GIB * 1024**3:
                    raise RuntimeError("free disk is below the extraction guard")
                miner_revision(health_url, expected_revision)
                existing = active_worker_pids(spec)
                if existing:
                    raise RuntimeError(f"{source} already has a writer: {existing}")
                limit = min(spec["batch"], spec["target"] - before)
                command = worker_command(source, health_url, limit)
                print(f"{source}: starting {before}+{limit}", flush=True)
                output.write(f"\n{time.time():.0f} starting {before}+{limit}: {command!r}\n")
                output.flush()
                worker = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                )
                last_progress = time.monotonic()
                observed = before
                try:
                    while True:
                        try:
                            exit_code = worker.wait(timeout=POLL_SECONDS)
                            break
                        except subprocess.TimeoutExpired:
                            miner_revision(health_url, expected_revision)
                            if rss_kib(worker.pid) > spec["max_rss_gib"] * 1024**2:
                                raise RuntimeError(f"{source} worker exceeded its RSS guard")
                            current = index_count(index)
                            if current > before + limit:
                                raise RuntimeError(f"{source} index advanced beyond its batch limit")
                            if current > observed:
                                observed = current
                                last_progress = time.monotonic()
                            elif time.monotonic() - last_progress > MAX_IDLE_SECONDS:
                                raise RuntimeError(f"{source} worker made no indexed progress")
                finally:
                    stop_worker(worker)
                after = index_count(index)
                output.write(f"{time.time():.0f} exited {exit_code}: {after}/{spec['target']}\n")
                output.flush()
                if exit_code != 0 or after != before + limit:
                    raise RuntimeError(
                        f"{source} batch failed or stopped early: exit={exit_code}, {before}->{after}"
                    )
                miner_revision(health_url, expected_revision)
                if source == "fleurs":
                    verify_fleurs_prefix(health_url, after)
                print(f"{source}: verified batch {after}/{spec['target']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES), required=True)
    parser.add_argument("--miner-health-url", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--handoff-pid", type=int)
    parser.add_argument("--handoff-count", type=int)
    args = parser.parse_args()
    if (args.handoff_pid is None) != (args.handoff_count is None):
        parser.error("handoff PID and batch-end count must be supplied together")
    if args.inspect_only and args.handoff_pid is not None:
        parser.error("inspection cannot wait for a handoff")
    if not RUNTIME_PYTHON.is_file():
        parser.error("independent baseline runtime is missing")
    revision = miner_revision(args.miner_health_url, args.expected_revision)
    spec = SOURCES[args.source]
    count = index_count(ROOT / spec["index"])
    if args.inspect_only:
        print(json.dumps({
            "source": args.source,
            "count": count,
            "target": spec["target"],
            "batch": spec["batch"],
            "max_rss_gib": spec["max_rss_gib"],
            "miner_model_revision": revision,
        }, sort_keys=True))
        return
    run_source(
        args.source,
        args.miner_health_url,
        revision,
        handoff_pid=args.handoff_pid,
        handoff_count=args.handoff_count,
    )


if __name__ == "__main__":
    main()
